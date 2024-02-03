import os
import glob
import argparse
import json
import torch
from torchvision import transforms
from data.dwidataset import DWIDataset3D
from train_2d import (
    load_from_dataset,
    construct_dataset,
    split_labeled_unlabeled,
    get_net,
    construct_dataset_for_fold,
    collate_fn,
    test,
    CrossEntropyLossWithMask,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_dir", required=True)
    parser.add_argument("--test_target", required=True)
    parser.add_argument("--output", default="eval_output")
    parser.add_argument("--epoch", default=100, type=int)
    return parser


def construct_test_dataset(args):
    print("loading data")
    all_labels, data_path_list, data_origin_list = load_from_dataset(args.test_target)
    data_origin_set = sorted(list(set(data_origin_list)))
    print("nb origin : {}".format(len(data_origin_set)))
    print("nb alldata : {}".format(len(data_path_list)))

    def deform_class(all_labels, oldlabel2newlabel):
        all_labels = [oldlabel2newlabel[label] for label in all_labels]
        return all_labels

    all_labels = deform_class(all_labels, {"N": 0, "B": 1, "M": 2})
    (
        all_labels,
        labeled_paths,
        labeled_origins,
        unlabeled_paths,
        unlabeled_origins,
    ) = split_labeled_unlabeled(all_labels, data_path_list, data_origin_list)
    assert len(labeled_origins) == len(all_labels)
    return (
        all_labels,
        labeled_paths,
        labeled_origins,
        unlabeled_paths,
        unlabeled_origins,
    )


def eval(test_args, train_args, weight_path, log_dir, weight_fold):
    # initialize settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes_list = [2]
    save_json_path = os.path.join(log_dir, "cm.json")
    criterion = CrossEntropyLossWithMask()

    # construct dataset
    (
        all_labels_test,
        labeled_paths_test,
        labeled_origins_test,
        unlabeled_paths_test,
        unlabeled_origins_test,
    ) = construct_test_dataset(test_args)
    (
        all_labels,
        labeled_paths,
        labeled_origins,
        unlabeled_paths,
        unlabeled_origins,
    ) = construct_dataset(train_args)
    origin_set = sorted(list(set(labeled_origins)))

    # calcutate number of test paths and folds
    nb_test_path = train_args.nb_test_paths
    nb_folds = len(origin_set) // nb_test_path
    nb_folds = nb_folds if len(origin_set) % nb_test_path == 0 else nb_folds + 1
    print("nb_folds", nb_folds)
    print("nb origin of labeled data : {}".format(len(origin_set)))

    # initialize result_dict
    result_dict = {}
    with open(save_json_path, "w") as f:
        json.dump(result_dict, f)
    result_dict["N_fold_train_results"] = []

    for fold in range(nb_folds):
        if fold != weight_fold:
            continue
        print("--------------------------------------")
        print("---------------fold:[{}]---------------".format(fold))
        print("--------------------------------------")

        # get initial network for this fold
        net = get_net(train_args, n_classes_list)
        net.load_state_dict(torch.load(weight_path))
        net.to(device)

        # initialize result_dict for this fold
        result_dict_in_this_fold = {}
        result_dict_in_this_fold["fold"] = fold
        result_dict_in_this_fold["cms"] = []

        # construct dataset for this fold
        (
            annotated_train_data_for_fold,
            annotated_train_labels_for_fold,
            annotated_train_origins_for_fold,
            _,
            _,
            _,
            test_paths_for_fold,
        ) = construct_dataset_for_fold(
            fold, nb_test_path, all_labels, labeled_paths, labeled_origins, origin_set
        )

        # get mean and std of train data
        tmp_data_set = DWIDataset3D(
            annotated_train_data_for_fold, train_args, transform=transforms.Compose([])
        )
        annotated_mean, annotated_std = tmp_data_set.set_stats_from_data()

        # make test data loader
        # we assume mean and std of test dataset == mean and std of annotated train data
        test_data_for_fold = labeled_paths_test
        test_data_set = DWIDataset3D(
            test_data_for_fold, train_args, transform=transforms.Compose([])
        )
        test_data_set.set_stats(annotated_mean, annotated_std)
        testloader = torch.utils.data.DataLoader(
            test_data_set, batch_size=1, shuffle=False, collate_fn=collate_fn
        )

        epoch = test_args.epoch
        gt_predict_matrix_list, result_dict_in_this_fold = test(
            net,
            testloader,
            criterion,
            device,
            n_classes_list,
            fold,
            epoch,
            result_dict_in_this_fold,
        )
        result_dict["N_fold_train_results"].append(result_dict_in_this_fold)
        with open(save_json_path, "w") as f:
            json.dump(result_dict, f)

    print("Training was finished")
    print("--------------------------------------")
    print("--------------[[summary]]--------------")
    print("--------------------------------------")
    with open(save_json_path, "w") as f:
        json.dump(result_dict, f)


class TrainArgs(object):
    def __init__(self, name: str, age: int, language: list[str]):
        self.name = name
        self.age = age
        self.language = language


def train_args(d: dict) -> TrainArgs:
    u = TrainArgs.__new__(TrainArgs)
    u.__dict__.update(d)
    return u


if __name__ == "__main__":
    test_args = get_parser().parse_args()
    for weight_subdir in os.listdir(test_args.weight_dir):
        train_setting = json.load(
            open(
                os.path.join(test_args.weight_dir, weight_subdir, "train_setting.json")
            )
        )
        args = train_args(train_setting)
        for weight_path in glob.glob(
            "{}/{}/*_*_{}.pth".format(
                test_args.weight_dir, weight_subdir, test_args.epoch
            )
        ):
            print("weight_path", weight_path)
            weight_fold = int(weight_path.split("_")[-2])
            log_dir = os.path.join(
                test_args.output, "{}fold_{}".format(weight_fold, weight_subdir)
            )
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            train_setting_json_path = os.path.join(log_dir, "train_setting.json")
            with open(train_setting_json_path, "w") as f:
                json.dump(train_setting, f)
            eval(test_args, args, weight_path, log_dir, weight_fold)
