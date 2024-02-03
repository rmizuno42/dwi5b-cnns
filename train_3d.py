from datetime import datetime
import argparse
import json
import pickle
import os
from tqdm import tqdm
import random
import gc
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import cv2
from tensorboardX import SummaryWriter
from elastic_transform import RandomElasticTransforms
from data.dwidataset import DWIDataset3D5Slices


def test(
    net,
    testloader,
    criterion,
    device,
    n_classes,
    fold,
    epoch,
    result_dict_in_this_fold,
    writer=None,
):
    gt_predict_matrix = np.zeros((n_classes, n_classes))
    net.eval()
    total = 0
    running_loss = 0.0
    tmp_dict = {}
    tmp_dict["epoch"] = epoch
    tmp_dict["outputs"] = []
    tmp_dict["labels"] = []
    tmp_dict["paths"] = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels, paths = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            total += labels.size(0)
            loss = 0
            loss += criterion(outputs, labels)
            total += labels.size(0)
            running_loss += loss.item() * labels.size()[0]
            predicted = torch.max(outputs.data, 1)[1]
            for label, predict in zip(labels, predicted):
                gt_predict_matrix[label, predict] += 1
            tmp_dict["outputs"] += outputs.cpu().detach().numpy().tolist()
            tmp_dict["labels"] += labels.cpu().detach().numpy().tolist()
            tmp_dict["paths"] += [paths[0][2]]
    confusion_matrix = gt_predict_matrix / gt_predict_matrix.sum(axis=1)[:, None]
    print("Test confusion matrix: \n {}".format(confusion_matrix))
    if writer is not None:
        writer.add_scalar(
            "/fold{}/test/accuracy/TPR".format(fold + 1), confusion_matrix[1, 1], epoch
        )
        writer.add_scalar(
            "/fold{}/test/accuracy/TNR".format(fold + 1), confusion_matrix[0, 0], epoch
        )
    print("Test loss: {}".format(running_loss / total))
    if writer is not None:
        writer.add_scalar(
            "/fold{}/test/loss".format(fold + 1), running_loss / total, epoch
        )
    result_dict_in_this_fold["cms"].append(tmp_dict)
    return gt_predict_matrix, result_dict_in_this_fold


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.03)",
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="mixup alpha")
    parser.add_argument(
        "--noise_scale", type=float, default=1.0, help="noise scale"
    )
    parser.add_argument(
        "--affine_scale", type=float, default=1.0, help="affine transform scale parameter"
    )
    parser.add_argument(
        "--remove_ncf", action="store_true", default=False, help="remove RCF"
    )
    parser.add_argument(
        "--adc", action="store_true", default=False, help="input ADC maps"
    )
    parser.add_argument("--mixup", action="store_true", default=False, help="use mixup")
    parser.add_argument(
        "--finetune", action="store_true", default=False, help="use pretrained model"
    )
    parser.add_argument(
        "--affine_noise",
        action="store_true",
        default=False,
        help="use random affine and noise data augmentation",
    )
    parser.add_argument("--logdir", default="logs_3d")
    parser.add_argument("--net", default="simple", help="network type")
    parser.add_argument(
        "--elastic", action="store_true", default=False, help="use elastic deformation"
    )
    parser.add_argument("--nb_test_paths", type=int, default=21)
    parser.add_argument("--target", default="dataset_3d", help="dataset path")
    return parser


def load_from_dataset(dataset_path):
    # loading uint16 inputs
    # inputs are normalized in getitem in dataset
    all_labels = []
    data_origin_set = []
    data_path_list = []
    for origin in os.listdir(dataset_path):
        slice_nb2data_path = {}
        class2slice_nb_dict = {"N": [], "B": [], "M": []}
        for slice_name in os.listdir(os.path.join(dataset_path, origin)):
            slice_nb = int(slice_name.split(".")[0])
            data_path = os.path.join(dataset_path, origin, slice_name)
            f = open(data_path, "rb")
            _, tumor_class = pickle.load(f)
            class2slice_nb_dict[tumor_class].append(slice_nb)
            slice_nb2data_path[slice_nb] = data_path
        if len(class2slice_nb_dict["B"]) > 0:
            sorted_slice_nbs = sorted(class2slice_nb_dict["B"])
            if len(sorted_slice_nbs) < 5:
                add_slice_list = [
                    sorted_slice_nbs[0] - 1,
                    sorted_slice_nbs[-1] + 1,
                    sorted_slice_nbs[0] - 2,
                    sorted_slice_nbs[-1] + 2,
                ]
                sorted_slice_nbs += add_slice_list[: 5 - len(sorted_slice_nbs)]
                sorted_slice_nbs = sorted(sorted_slice_nbs)
            data_paths = [slice_nb2data_path[slice_nb] for slice_nb in sorted_slice_nbs]
            data_origin_set.append(origin)
            all_labels.append("B")
            data_path_list.append(data_paths)
        elif len(class2slice_nb_dict["M"]) > 0:
            sorted_slice_nbs = sorted(class2slice_nb_dict["M"])
            if len(sorted_slice_nbs) < 5:
                add_slice_list = [
                    sorted_slice_nbs[0] - 1,
                    sorted_slice_nbs[-1] + 1,
                    sorted_slice_nbs[0] - 2,
                    sorted_slice_nbs[-1] + 2,
                ]
                sorted_slice_nbs += add_slice_list[: 5 - len(sorted_slice_nbs)]
                sorted_slice_nbs = sorted(sorted_slice_nbs)
            data_paths = [slice_nb2data_path[slice_nb] for slice_nb in sorted_slice_nbs]
            data_origin_set.append(origin)
            all_labels.append("M")
            data_path_list.append(data_paths)
        else:
            sorted_slice_nbs = [20, 21, 22, 23, 24]
            for slice_nb in sorted_slice_nbs:
                assert slice_nb in class2slice_nb_dict["N"]
            data_paths = [slice_nb2data_path[slice_nb] for slice_nb in sorted_slice_nbs]
            data_origin_set.append(origin)
            all_labels.append("N")
            data_path_list.append(data_paths)
    return all_labels, data_path_list, data_origin_set


def split_labeled_unlabeled(all_labels, data_path_list, data_origin_list):
    tmp_labels = []
    labeled_paths = []
    labeled_origins = []
    unlabeled_paths = []
    unlabeled_origins = []
    for index in range(len(all_labels)):
        label = all_labels[index]
        if label == 0:
            unlabeled_paths.append(data_path_list[index])
            unlabeled_origins.append(data_origin_list[index])
        else:
            tmp_labels.append(all_labels[index] - 1)
            labeled_paths.append(data_path_list[index])
            labeled_origins.append(data_origin_list[index])
    return (
        tmp_labels,
        labeled_paths,
        labeled_origins,
        unlabeled_paths,
        unlabeled_origins,
    )


def construct_3d_dataset(args):
    all_labels, data_path_list, data_origin_list = load_from_dataset(args.target)
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
    return (
        all_labels,
        labeled_paths,
        labeled_origins,
        unlabeled_paths,
        unlabeled_origins,
    )


def construct_dataset_for_fold(
    fold, nb_test_path, all_labels, labeled_paths, labeled_origins, origin_set
):
    test_paths = origin_set[fold * nb_test_path : (fold + 1) * nb_test_path]
    test_data = []
    test_labels = []
    test_origins = []
    annotated_train_data = []
    annotated_train_labels = []
    annotated_train_origins = []
    for index in range(len(all_labels)):
        if labeled_origins[index] in test_paths:
            test_data.append(labeled_paths[index])
            test_labels.append(all_labels[index])
            test_origins.append(labeled_origins[index])
        else:
            annotated_train_data.append(labeled_paths[index])
            annotated_train_labels.append(all_labels[index])
            annotated_train_origins.append(labeled_origins[index])
    assert len(set(annotated_train_origins) & set(test_origins)) == 0
    print("nb_traindata : {}".format(len(annotated_train_data)))
    print("nb_testdata  : {}".format(len(test_data)))
    return (
        annotated_train_data,
        annotated_train_labels,
        annotated_train_origins,
        test_data,
        test_labels,
        test_origins,
        test_paths,
    )


def collate_fn(batch):
    images, targets, paths = list(zip(*batch))
    images = torch.stack(images)
    targets = torch.LongTensor(targets)
    return images, targets, paths


def get_3d_net(args, n_classes):
    nb_channel = 5
    if args.adc:
        nb_channel = 4
    if args.net == "simple":
        from network.net_3d_simple import Net
    else:
        print("no {} network is defined".format(args.net))
        raise
    net = Net(nb_channel=nb_channel, nb_class=n_classes)
    return net


def construct_dataset_for_epoch(
    annotated_train_data_for_fold,
    annotated_train_labels_for_fold,
    annotated_train_origins_for_fold,
    unlabeled_paths,
    unlabeled_origins,
    test_paths_for_fold,
    diff_class,
):
    train_data, train_labels, train_origins = (
        annotated_train_data_for_fold.copy(),
        annotated_train_labels_for_fold.copy(),
        annotated_train_origins_for_fold.copy(),
    )
    if diff_class > 0:
        all_indexes = [i for i in range(len(unlabeled_origins))]
        indexes = random.sample(all_indexes, len(all_indexes))
        init_size = len(train_labels)
        idx = 0
        while init_size + diff_class > len(train_labels) and idx < len(indexes):
            i = indexes[idx]
            if unlabeled_origins[i] in test_paths_for_fold:
                pass
            else:
                train_data.append(unlabeled_paths[i])
                train_labels.append(0)
                train_origins.append(unlabeled_origins[i])
            idx += 1
    return train_data, train_labels, train_origins


class Trainer:
    def __init__(self, args, fold):
        self.args = args
        self.log_dir = args.logdir
        self.total_epoch = args.epochs
        self.lr = args.lr
        self.criterion = nn.CrossEntropyLoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_classes = 2
        self.transform = self._construct_trainsforms(args)
        self.alpha = args.alpha
        self.net = get_3d_net(args, self.n_classes)
        self.net.to(self.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.fold = fold
        self.batch_size = args.batch_size
        self.mixup = args.mixup

    def _construct_trainsforms(self, args):
        transform_list = []
        noise_scale = args.noise_scale
        affine_scale = args.affine_scale
        if args.affine_noise:
            transform_list = (
                [
                    transforms.RandomAffine(
                        (-10 * affine_scale, 10 * affine_scale),
                        translate=(0.1 * affine_scale, 0.1 * affine_scale),
                        scale=(1 - 0.1 * affine_scale, 1 + 0.1 * affine_scale),
                        shear=10 * affine_scale,
                    ),
                    transforms.RandomHorizontalFlip(),
                ]
                + transform_list
                + [
                    AddRandomNoise(mean=0.0, std=noise_scale),
                ]
            )
        if args.elastic:
            transform_list = [
                RandomElasticTransforms(deform_scale=1, control_points=(3, 3, 3)),
                RandomElasticTransforms(
                    deform_scale=5, control_points=(3, 3), axis=(2, 3)
                ),
            ] + transform_list
        transform = transforms.Compose(transform_list)
        return transform

    def _construct_trainloader(
        self,
        annotated_train_data_for_fold,
        annotated_train_labels_for_fold,
        annotated_train_origins_for_fold,
        unlabeled_paths,
        unlabeled_origins,
        test_paths_for_fold,
        test_origins_for_fold,
        annotated_mean,
        annotated_std,
        batch_size,
    ):
        train_data, train_labels, train_origins = construct_dataset_for_epoch(
            annotated_train_data_for_fold,
            annotated_train_labels_for_fold,
            annotated_train_origins_for_fold,
            unlabeled_paths,
            unlabeled_origins,
            test_paths_for_fold,
            self.diff_class,
        )
        assert len(set(train_origins) & set(test_origins_for_fold)) == 0
        data_set = DWIDataset3D5Slices(train_data, self.args, transform=self.transform)
        data_set.set_stats(annotated_mean, annotated_std)
        trainloader = torch.utils.data.DataLoader(
            data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        return trainloader

    def run(
        self,
        annotated_train_data_for_fold,
        annotated_train_labels_for_fold,
        annotated_train_origins_for_fold,
        test_data_for_fold,
        test_labels_for_fold,
        test_origins_for_fold,
        unlabeled_paths,
        unlabeled_origins,
        test_paths_for_fold,
        result_dict_in_this_fold,
        gt_predict_matrix_ave,
        writer,
    ):
        # get mean and std of train data for this fold
        tmp_data_set = DWIDataset3D5Slices(
            annotated_train_data_for_fold, self.args, transform=transforms.Compose([])
        )
        annotated_mean, annotated_std = tmp_data_set.set_stats_from_data()

        # make test data loader for this fold
        # the test dataset is standarized by the mean and std of the annotated train data
        test_data_set = DWIDataset3D5Slices(
            test_data_for_fold, self.args, transform=transforms.Compose([])
        )
        test_data_set.set_stats(annotated_mean, annotated_std)
        testloader = torch.utils.data.DataLoader(
            test_data_set, batch_size=1, shuffle=False, collate_fn=collate_fn
        )

        # count label distribution
        self.class_dist = count_classes(annotated_train_labels_for_fold, self.n_classes)
        self.diff_class = self.class_dist[1] - self.class_dist[0]
        print("flat class distribution in this fold: ", self.class_dist)

        for epoch in range(self.total_epoch):
            print("--------------------------------------")
            print("--------------epoch:[{}]--------------".format(epoch))
            print("--------------------------------------")
            # construct train dataset for this epoch
            trainloader = self._construct_trainloader(
                annotated_train_data_for_fold,
                annotated_train_labels_for_fold,
                annotated_train_origins_for_fold,
                unlabeled_paths,
                unlabeled_origins,
                test_paths_for_fold,
                test_origins_for_fold,
                annotated_mean,
                annotated_std,
                self.batch_size,
            )

            # initialize variables to log
            running_loss = 0.0
            correct = 0
            total = 0
            confusion_matrix = np.zeros((self.n_classes, self.n_classes))

            # set network to train
            self.net.train()
            print("####### train labeled data ######")
            for i, data in tqdm(enumerate(trainloader, 0)):
                inputs, labels, _ = data
                if args.mixup:
                    mixup_inputs, targets_a, targets_b, lam = mixup_data(
                        inputs, labels, args.alpha
                    )
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if args.mixup:
                    mixup_inputs = mixup_inputs.to(self.device)
                    targets_a = targets_a.to(self.device)
                    targets_b = targets_b.to(self.device)
                self.optimizer.zero_grad()

                # forward + backward + optimize
                if args.mixup:
                    outputs = self.net(mixup_inputs)
                else:
                    outputs = self.net(inputs)
                if args.mixup:
                    loss = mixup_criterion(
                        self.criterion, outputs, targets_a, targets_b, lam
                    )
                else:
                    loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    if args.mixup:
                        outputs = self.net(inputs)
                    running_loss += loss.item() * labels.size()[0]
                    _, predicted = torch.max(outputs.data, 1)
                    for label, predict in zip(labels, predicted):
                        confusion_matrix[label, predict] += 1
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            print("Train samples: {}".format(confusion_matrix.sum(axis=1)))
            confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, None]
            print("Train confusion matrix : \n {}".format(confusion_matrix))
            writer.add_scalar(
                "/fold{}/train/accuracy/TPR".format(self.fold + 1),
                confusion_matrix[1, 1],
                epoch,
            )
            writer.add_scalar(
                "/fold{}/train/accuracy/TNR".format(self.fold + 1),
                confusion_matrix[0, 0],
                epoch,
            )
            print("Train Loss: {}".format(running_loss / total))
            writer.add_scalar(
                "/fold{}/train/loss".format(self.fold + 1), running_loss / total, epoch
            )
            gt_predict_matrix, result_dict_in_this_fold = test(
                self.net,
                testloader,
                self.criterion,
                self.device,
                self.n_classes,
                self.fold,
                epoch,
                result_dict_in_this_fold,
                writer,
            )
            confusion_matrix = (
                gt_predict_matrix / gt_predict_matrix.sum(axis=1)[:, None]
            )
            del inputs, labels, data, outputs, loss, correct, total
            del trainloader
            gc.collect()
            if (epoch + 1) % 100 == 0:
                torch.save(
                    self.net.state_dict(),
                    "{}/{}_{}_{}.pth".format(
                        self.log_dir, args.net, self.fold, epoch + 1
                    ),
                )
        gt_predict_matrix_ave += gt_predict_matrix


class CrossValidator:
    def __init__(self, args):
        self.args = args
        self.n_classes = 2
        self.log_dir = args.logdir
        self.log_dir = os.path.join(
            self.log_dir, "{}".format(datetime.now().strftime("%b%d_%H-%M-%S"))
        )
        args.logdir = self.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        train_setting_json_path = os.path.join(self.log_dir, "train_setting.json")
        setting_dict = vars(args)
        with open(train_setting_json_path, "w") as f:
            json.dump(setting_dict, f)
        self.writer = SummaryWriter(self.log_dir)
        self.save_json_path = os.path.join(self.log_dir, "cm.json")
        self._construct_data()
        self._initialize_result_container()
        self._initialize_result_dict()

    def _construct_data(self):
        (
            all_labels,
            labeled_paths,
            labeled_origins,
            unlabeled_paths,
            unlabeled_origins,
        ) = construct_3d_dataset(args)
        self.origin_set = sorted(list(set(labeled_origins)))
        self.nb_test_path = self.args.nb_test_paths
        nb_folds = len(self.origin_set) // self.nb_test_path
        self.nb_folds = (
            nb_folds if len(self.origin_set) % self.nb_test_path == 0 else nb_folds + 1
        )
        print("nb_folds", self.nb_folds)
        print("batch size", args.batch_size)
        print("nb origin of labeled data : {}".format(len(self.origin_set)))
        self.data = [
            all_labels,
            labeled_paths,
            labeled_origins,
            unlabeled_paths,
            unlabeled_origins,
        ]

    def _initialize_result_container(self):
        # initialize result container
        self.gt_predict_matrix_ave = np.zeros((self.n_classes, self.n_classes))

    def _initialize_result_dict(self):
        # initialize result_dict
        self.result_dict = {}
        with open(self.save_json_path, "w") as f:
            json.dump(self.result_dict, f)
        self.result_dict["N_fold_train_results"] = []

    def run(self):
        (
            all_labels,
            labeled_paths,
            labeled_origins,
            unlabeled_paths,
            unlabeled_origins,
        ) = self.data
        origin_set = self.origin_set
        for fold in range(self.nb_folds):
            print("--------------------------------------")
            print("---------------fold:[{}]---------------".format(fold))
            print("--------------------------------------")

            # initialize result_dict for this fold
            result_dict_in_this_fold = {}
            result_dict_in_this_fold["fold"] = fold
            result_dict_in_this_fold["cms"] = []

            # construct dataset for this fold
            (
                annotated_train_data_for_fold,
                annotated_train_labels_for_fold,
                annotated_train_origins_for_fold,
                test_data_for_fold,
                test_labels_for_fold,
                test_origins_for_fold,
                test_paths_for_fold,
            ) = construct_dataset_for_fold(
                fold,
                self.nb_test_path,
                all_labels,
                labeled_paths,
                labeled_origins,
                origin_set,
            )

            # initialize trainer for this fold
            trainer = Trainer(self.args, fold)

            # run training for this fold
            trainer.run(
                annotated_train_data_for_fold,
                annotated_train_labels_for_fold,
                annotated_train_origins_for_fold,
                test_data_for_fold,
                test_labels_for_fold,
                test_origins_for_fold,
                unlabeled_paths,
                unlabeled_origins,
                test_paths_for_fold,
                result_dict_in_this_fold,
                self.gt_predict_matrix_ave,
                self.writer,
            )
            self.result_dict["N_fold_train_results"].append(result_dict_in_this_fold)
            with open(self.save_json_path, "w") as f:
                json.dump(self.result_dict, f)

        print("Cross validation was finished")
        print("--------------------------------------")
        print("--------------[[summary]]--------------")
        print("--------------------------------------")
        confusion_matrix = (
            self.gt_predict_matrix_ave / self.gt_predict_matrix_ave.sum(axis=1)[:, None]
        )
        print("Test confusion matrix in all fold: \n {}".format(confusion_matrix))
        with open(self.save_json_path, "w") as f:
            json.dump(self.result_dict, f)


def count_classes(all_labels, n_classes):
    class_dist = np.zeros(n_classes)
    for label in all_labels:
        class_dist[label] += 1
    return class_dist


def mixup_data(x, y, alpha):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class AddRandomNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if np.random.randint(0, 2) == 0:
            return tensor
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


if __name__ == "__main__":
    args = get_parser().parse_args()
    cross_validator = CrossValidator(args)
    cross_validator.run()
