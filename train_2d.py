import pickle
import os
import random
import gc
import numpy as np
from datetime import datetime
import argparse
import json
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from elastic_transform import RandomElasticTransforms
from efficientnet_pytorch import EfficientNet
from data.dwidataset import DWIDataset3D


class CrossEntropyLossWithMask(nn.Module):
    def __init__(self):
        super(CrossEntropyLossWithMask, self).__init__()

    def forward(self, output, label):
        assert output.size()[0] == label.size()[0]
        batch_size = label.size()[0]
        used_labels = 0
        loss = 0
        for i in range(batch_size):
            if label[i] == -1:
                continue
            used_labels += 1
            loss += nn.CrossEntropyLoss(reduction="none")(output[i], label[i])
        if used_labels == 0:
            return torch.tensor(0.0)
        return loss / used_labels


class MultiOutputLinear(nn.Module):
    def __init__(self, in_features, out_features_list):
        super().__init__()
        self.out_features_list = out_features_list
        self.linears = nn.ModuleList(
            [nn.Linear(in_features, out_features) for out_features in out_features_list]
        )

    def forward(self, x):
        return [linear(x) for linear in self.linears]


class ResNet5b(nn.Module):
    def __init__(self, nb_channel=5, n_classes_list=[2], pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        self.net = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=pretrained
        )
        num_ftrs = self.net.fc.in_features
        self.net.fc = MultiOutputLinear(num_ftrs, n_classes_list)
        self.bottom = torch.nn.Conv2d(nb_channel, 3, (3, 3), (1, 1), (1, 1), bias=False)
        if not pretrained:
            self.net.conv1 = torch.nn.Conv2d(
                nb_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    def forward(self, x):
        if self.pretrained:
            x = self.bottom(x)
        return self.net(x)


class EfficientNet5b(nn.Module):
    def __init__(
        self, nb_channel=5, n_classes=1000, dropout_rate=0.2, pretrained=False
    ):
        super().__init__()
        self.pretrained = pretrained
        if pretrained:
            self.eff_net = EfficientNet.from_pretrained(
                "efficientnet-b0",
                num_classes=n_classes,
                dropout_rate=dropout_rate,
                drop_connect_rate=0.2,
            )
            self.bottom = nn.Conv2d(
                nb_channel, 3, kernel_size=3, stride=1, padding=1, bias=False
            )
        else:
            self.eff_net = EfficientNet.from_name(
                "efficientnet-b0",
                in_channels=nb_channel,
                num_classes=n_classes,
                dropout_rate=dropout_rate,
                drop_connect_rate=0.2,
            )

    def forward(self, x):
        if self.pretrained:
            x = self.bottom(x)
        x = self.eff_net(x)
        return [x]


def test(
    net,
    testloader,
    criterion,
    device,
    n_classes_list,
    fold,
    epoch,
    result_dict_in_this_fold,
    writer=None,
):
    gt_predict_matrix_list = [
        np.zeros((n_classes, n_classes)) for n_classes in n_classes_list
    ]
    net.eval()
    total = 0
    running_loss = 0
    tmp_dict = {}
    tmp_dict["epoch"] = epoch
    tmp_dict["outputs"] = {}
    for i in range(len(n_classes_list)):
        tmp_dict["outputs"][str(i)] = []
    tmp_dict["labels"] = []
    tmp_dict["paths"] = []

    with torch.no_grad():
        for data in testloader:
            inputs, labels, paths = data
            tmp_labels_3d = []
            for label, input in zip(labels, inputs):
                _, d, _, _ = input.size()
                tmp_y = torch.zeros(d, len(n_classes_list)).long()
                tmp_y[:] = label
                tmp_labels_3d.append(tmp_y)
            labels = torch.cat(tmp_labels_3d, dim=0)

            inputs = torch.cat(inputs, dim=1)
            inputs = inputs.permute(1, 0, 2, 3)

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = 0
            for class_index in range(len(n_classes_list)):
                loss += criterion(outputs[class_index], labels[:, class_index])
            total += labels.size(0)
            running_loss += loss.item() * labels.size()[0]
            predicted_list = [torch.max(output.data, 1)[1] for output in outputs]
            for class_index, predict_batch in enumerate(predicted_list):
                labels_batch = labels[:, class_index]
                for label, predict in zip(labels_batch, predict_batch):
                    gt_predict_matrix_list[class_index][label, predict] += 1
            for i in range(len(n_classes_list)):
                tmp_dict["outputs"][str(i)] += (
                    outputs[i].cpu().detach().numpy().tolist()
                )
            tmp_dict["labels"] += labels.cpu().detach().numpy().tolist()
            tmp_dict["paths"] += list(np.array(paths[0]))

    for index, gt_predict_matrix in enumerate(gt_predict_matrix_list):
        confusion_matrix = gt_predict_matrix / gt_predict_matrix.sum(axis=1)[:, None]
        print(
            "Test confusion matrix of {}th prediction: \n {}".format(
                index, confusion_matrix
            )
        )
        if writer is not None:
            writer.add_scalar(
                "/fold{}/test/accuracy/class{}/TPR".format(fold + 1, index),
                confusion_matrix[1, 1],
                epoch,
            )
            writer.add_scalar(
                "/fold{}/test/accuracy/class{}/TNR".format(fold + 1, index),
                confusion_matrix[0, 0],
                epoch,
            )

    print("Test loss: {}".format(running_loss / total))
    if writer is not None:
        writer.add_scalar(
            "/fold{}/test/loss".format(fold + 1), running_loss / total, epoch
        )
    result_dict_in_this_fold["cms"].append(tmp_dict)

    return gt_predict_matrix_list, result_dict_in_this_fold


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
        default=0.03,
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
        "--remove_ncf", action="store_true", default=False, help="remove NCF"
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
    parser.add_argument("--logdir", default="logs_2d")
    parser.add_argument("--net", default="simple", help="network type")
    parser.add_argument(
        "--elastic", action="store_true", default=False, help="use elastic deformation"
    )
    parser.add_argument("--nb_test_paths", type=int, default=21)
    parser.add_argument("--target", default="dataset", help="dataset path")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="dropout rate")
    return parser


def load_from_dataset(dataset_path):
    # loading uint16 inputs
    # inputs are normalized in getitem in dataset
    all_labels = []
    data_origin_set = []
    data_path_list = []
    for origin in tqdm(os.listdir(dataset_path)):
        if not os.path.isdir(os.path.join(dataset_path, origin)):
            continue
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
            data_paths = [slice_nb2data_path[slice_nb] for slice_nb in sorted_slice_nbs]
            data_origin_set.append(origin)
            all_labels.append("B")
            data_path_list.append(data_paths)

        elif len(class2slice_nb_dict["M"]) > 0:
            sorted_slice_nbs = sorted(class2slice_nb_dict["M"])
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
        assert len(all_labels) == len(data_path_list) == len(data_origin_set)

    return all_labels, data_path_list, data_origin_set


def split_labeled_unlabeled(all_labels, data_path_list, data_origin_list):
    assert len(all_labels) == len(data_path_list) == len(data_origin_list)
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
    assert len(labeled_paths) + len(unlabeled_paths) == len(data_path_list)
    assert len(tmp_labels) == len(labeled_paths) == len(labeled_origins)
    assert len(unlabeled_paths) == len(unlabeled_origins)
    return (
        tmp_labels,
        labeled_paths,
        labeled_origins,
        unlabeled_paths,
        unlabeled_origins,
    )


def collate_fn(batch):
    images, targets, paths = list(zip(*batch))
    targets = torch.LongTensor(targets)
    return images, targets, paths


def count_classes(all_labels, n_classes):
    class_dist = np.zeros(n_classes)
    for label in all_labels:
        class_dist[label] += 1
    return class_dist


def get_net(args, n_classes_list):
    nb_channel = 5
    if args.adc:
        nb_channel = 4
    if args.net == "resnet":
        net = ResNet5b(
            nb_channel=nb_channel,
            pretrained=args.finetune,
            n_classes_list=n_classes_list,
        )
    elif args.net == "efficient":
        net = EfficientNet5b(
            nb_channel=nb_channel,
            n_classes=2,
            pretrained=args.finetune,
            dropout_rate=args.dropout_rate,
        )
    elif args.net == "simple":
        from network.net_simple import Net as NetSimple

        net = NetSimple(nb_channel=nb_channel, nb_class_list=n_classes_list)
    return net


def construct_dataset(args):
    print("loading data")
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
    assert len(labeled_origins) == len(all_labels)

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
    train_data_for_fold = []
    train_labels_for_fold = []
    annotated_train_origins = []
    for index in range(len(all_labels)):
        if labeled_origins[index] in test_paths:
            test_data.append(labeled_paths[index])
            test_labels.append(all_labels[index])
            test_origins.append(labeled_origins[index])
        else:
            train_data_for_fold.append(labeled_paths[index])
            train_labels_for_fold.append(all_labels[index])
            annotated_train_origins.append(labeled_origins[index])
    assert len(set(annotated_train_origins) & set(test_origins)) == 0
    print("nb_traindata : {}".format(len(train_data_for_fold)))
    print("nb_testdata  : {}".format(len(test_data)))
    return (
        train_data_for_fold,
        train_labels_for_fold,
        annotated_train_origins,
        test_data,
        test_labels,
        test_origins,
        test_paths,
    )


def construct_trainsforms(args):
    transform_list = []
    noise_scale = args.noise_scale
    affine_scale = args.affine_scale
    if args.affine_noise:
        transform_list = (
            transform_list
            + [
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
            RandomElasticTransforms(deform_scale=5, control_points=(3, 3), axis=(2, 3))
        ] + transform_list
    transform = transforms.Compose(transform_list)
    return transform


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
        init_size = sum([len(tmp_train_data) for tmp_train_data in train_data])
        idx = 0
        test_ids_for_fold = [
            int(test_path.split("_")[0]) for test_path in test_paths_for_fold
        ]
        while init_size + diff_class > sum(
            [len(tmp_train_data) for tmp_train_data in train_data]
        ) and idx < len(indexes):
            unlabeled_origin_id = int(unlabeled_origins[indexes[idx]].split("_")[0])
            if unlabeled_origin_id in test_ids_for_fold:
                pass
            else:
                train_data.append(unlabeled_paths[indexes[idx]])
                train_labels.append(0)
                train_origins.append(unlabeled_origins[indexes[idx]])
            idx += 1
    return train_data, train_labels, train_origins


class Trainer:
    def __init__(self, args, fold):
        if args.finetune:
            assert args.net == "resnet" or args.net == "efficient"
        self.args = args
        self.log_dir = args.logdir
        self.total_epoch = args.epochs
        self.lr = args.lr
        self.criterion = CrossEntropyLossWithMask()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_classes_list = [2]
        self.transform = construct_trainsforms(args)
        self.alpha = args.alpha
        self.net = get_net(args, self.n_classes_list)
        self.net.to(self.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.fold = fold
        self.batch_size = args.batch_size
        self.mixup = args.mixup

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
        data_set = DWIDataset3D(train_data, self.args, transform=self.transform)
        data_set.set_stats(annotated_mean, annotated_std)
        trainloader = torch.utils.data.DataLoader(
            data_set, batch_size=900, shuffle=True, collate_fn=collate_fn
        )
        return trainloader

    def _construct_inputs_and_labels(self, data):
        inputs_3d, labels_3d, _ = data
        tmp_labels_3d = []
        for label, inputs in zip(labels_3d, inputs_3d):
            c, d, h, w = inputs.size()
            tmp_y = torch.zeros(d, len(self.n_classes_list)).long()
            tmp_y[:] = label
            tmp_labels_3d.append(tmp_y)
        labels_3d = torch.cat(tmp_labels_3d, dim=0)
        inputs_3d = torch.cat(inputs_3d, dim=1)
        inputs_3d = inputs_3d.permute(1, 0, 2, 3)
        if self.diff_class > 0:
            labels_3d = labels_3d[: int(2 * self.class_dist[1])]
            inputs_3d = inputs_3d[: int(2 * self.class_dist[1])]
        print(
            "flat class distribution in this epoch: ",
            count_classes(labels_3d, self.n_classes_list[0]),
        )
        batch_size_of_3d, _, _, _ = inputs_3d.size()
        assert len(inputs_3d) == len(labels_3d)
        perm_indexes = torch.randperm(batch_size_of_3d)
        inputs_3d = inputs_3d[perm_indexes]
        labels_3d = labels_3d[perm_indexes]
        return inputs_3d, labels_3d

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
        gt_predict_matrix_ave_list,
        writer,
    ):
        # get mean and std of train data for this fold
        tmp_data_set = DWIDataset3D(
            annotated_train_data_for_fold, self.args, transform=transforms.Compose([])
        )
        annotated_mean, annotated_std = tmp_data_set.set_stats_from_data()

        # make test data loader for this fold
        # the test dataset is standarized by the mean and std of the annotated train data
        test_data_set = DWIDataset3D(
            test_data_for_fold, self.args, transform=transforms.Compose([])
        )
        test_data_set.set_stats(annotated_mean, annotated_std)
        testloader = torch.utils.data.DataLoader(
            test_data_set, batch_size=1, shuffle=False, collate_fn=collate_fn
        )

        # count label distribution
        label_distribution = []
        for tmp_train_data, tmp_label in zip(
            annotated_train_data_for_fold, annotated_train_labels_for_fold
        ):
            label_distribution += len(tmp_train_data) * [tmp_label]
        self.class_dist = count_classes(label_distribution, self.n_classes_list[0])
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
            )

            # initialize variables to log
            running_loss = 0.0
            correct = 0
            total = 0
            confusion_matrix_list = [
                np.zeros((n_classes, n_classes)) for n_classes in self.n_classes_list
            ]

            # set network to train
            self.net.train()
            print("####### train labeled data ######")
            for i, data in tqdm(enumerate(trainloader, 0)):
                inputs_3d, labels_3d = self._construct_inputs_and_labels(data)
                batch_size_of_3d, _, _, _ = inputs_3d.size()
                batch_nb = batch_size_of_3d // self.batch_size
                for batch_index in range(batch_nb + 1):
                    self.optimizer.zero_grad()
                    batch_start = self.batch_size * batch_index
                    batch_end = (
                        self.batch_size * (batch_index + 1)
                        if batch_index != batch_nb
                        else batch_size_of_3d
                    )
                    if batch_start == batch_end:
                        continue
                    inputs = (inputs_3d[batch_start:batch_end]).to(self.device)
                    labels = (labels_3d[batch_start:batch_end]).to(self.device)

                    if self.mixup:
                        mixup_inputs, targets_a, targets_b, lam = mixup_data(
                            inputs, labels, self.alpha
                        )
                        outputs = self.net(mixup_inputs)
                    else:
                        outputs = self.net(inputs)
                    loss = 0
                    if self.mixup:
                        for class_index in range(len(self.n_classes_list)):
                            loss += mixup_criterion(
                                self.criterion,
                                outputs[class_index],
                                targets_a[:, class_index],
                                targets_b[:, class_index],
                                lam,
                            )
                    else:
                        for class_index in range(len(self.n_classes_list)):
                            loss += self.criterion(
                                outputs[class_index], labels[:, class_index]
                            )
                    loss.backward()
                    self.optimizer.step()

                    with torch.no_grad():
                        if self.mixup:
                            outputs = self.net(inputs)
                        running_loss += loss.item() * labels.size()[0]
                        total += labels.size(0)
                        predicted_list = [
                            torch.max(output.data, 1)[1] for output in outputs
                        ]
                        for class_index, predict_batch in enumerate(predicted_list):
                            labels_batch = labels[:, class_index]
                            for label, predict in zip(labels_batch, predict_batch):
                                confusion_matrix_list[class_index][label, predict] += 1

            print("Train samples: {}".format(confusion_matrix_list[0].sum(axis=1)))
            for index, confusion_matrix_matrix in enumerate(confusion_matrix_list):
                confusion_matrix = (
                    confusion_matrix_matrix
                    / confusion_matrix_matrix.sum(axis=1)[:, None]
                )
                print(
                    "Train confusion matrix of {}th prediction: \n {}".format(
                        index, confusion_matrix
                    )
                )
                writer.add_scalar(
                    "/fold{}/train/accuracy/class{}/TPR".format(self.fold + 1, index),
                    confusion_matrix[1, 1],
                    epoch,
                )
                writer.add_scalar(
                    "/fold{}/train/accuracy/class{}/TNR".format(self.fold + 1, index),
                    confusion_matrix[0, 0],
                    epoch,
                )
            print("Train Loss: {}".format(running_loss / total))
            writer.add_scalar(
                "/fold{}/train/loss".format(self.fold + 1), running_loss / total, epoch
            )

            gt_predict_matrix_list, result_dict_in_this_fold = test(
                self.net,
                testloader,
                self.criterion,
                self.device,
                self.n_classes_list,
                self.fold,
                epoch,
                result_dict_in_this_fold,
                writer,
            )
            del inputs, labels, data, outputs, loss, correct, total
            del trainloader
            gc.collect()
            if (epoch + 1) % 100 == 0:
                torch.save(
                    self.net.state_dict(),
                    "{}/{}_{}_{}.pth".format(
                        self.log_dir, self.args.net, self.fold, epoch + 1
                    ),
                )
        for index, gt_predict_matrix in enumerate(gt_predict_matrix_list):
            gt_predict_matrix_ave_list[index] += gt_predict_matrix


class CrossValidator:
    def __init__(self, args):
        self.args = args
        self.n_classes_list = [2]
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
        ) = construct_dataset(self.args)
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
        self.gt_predict_matrix_ave_list = [
            np.zeros((n_classes, n_classes)) for n_classes in self.n_classes_list
        ]

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
                self.gt_predict_matrix_ave_list,
                self.writer,
            )
            self.result_dict["N_fold_train_results"].append(result_dict_in_this_fold)
            with open(self.save_json_path, "w") as f:
                json.dump(self.result_dict, f)

        print("Cross validation was finished")
        print("--------------------------------------")
        print("--------------[[summary]]--------------")
        print("--------------------------------------")
        for index, gt_predict_matrix_ave in enumerate(self.gt_predict_matrix_ave_list):
            confusion_matrix = (
                gt_predict_matrix_ave / gt_predict_matrix_ave.sum(axis=1)[:, None]
            )
            print(
                "Test confusion matrix in all fold of {}th class: \n {}".format(
                    index, confusion_matrix
                )
            )
        with open(self.save_json_path, "w") as f:
            json.dump(self.result_dict, f)


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
