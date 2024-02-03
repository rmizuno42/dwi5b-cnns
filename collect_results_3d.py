import os
import csv
import json
import argparse
import numpy as np
import torch


class TumorInfo:
    def __init__(self, outputs, label, tumor_id):
        self.outputs = outputs
        self.label = label
        self.tumor_id = tumor_id


class Analyzer:
    def __init__(self):
        self.outputs = []
        self.labels = []
        self.paths = []
        self.tumor_infos = []

    def append_results(self, outputs, labels, paths):
        self.outputs += outputs
        self.labels += labels
        self.paths += paths

    def collect_each_tumor_info(self):
        self.outputs = torch.nn.functional.softmax(torch.tensor(self.outputs), dim=1)
        self.labels = torch.tensor(self.labels)
        self.slices = [int(path.split("/")[-1].split(".")[0]) for path in self.paths]
        self.tumor_ids = np.array([path.split("/")[1] for path in self.paths])
        for tumor_id in list(set(self.tumor_ids)):
            indexes = self.tumor_ids == tumor_id
            assert self.test_indexes(np.where(indexes)[0]), np.where(indexes)[0]
            outputs = self.outputs[indexes]
            labels = self.labels[indexes]
            assert (labels == labels[0]).all()
            self.tumor_infos.append(TumorInfo(outputs, labels[0], tumor_id))

    def test_indexes(self, indexes):
        start = indexes[0]
        flag = True
        for i in range(len(indexes)):
            if indexes[i] == start + i:
                pass
            else:
                flag = False
        return flag


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", nargs="*", default=None, required=True)
    parser.add_argument("--target_type", choices=["validation", "eval"], required=True)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--output_dir", default="csvs_3d")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    target_log_list = args.target
    if args.target_type == "validation":
        target_epoch = args.epoch
    elif args.target_type == "eval":
        target_epoch = 1
    results = {}
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for target_dir in target_log_list:
        for logdir in os.listdir(target_dir):
            target_log = os.path.join(target_dir, logdir, "cm.json")
            if not os.path.exists(target_log):
                continue
            train_setting = json.load(
                open(os.path.join(target_dir, logdir, "train_setting.json"))
            )
            print("---------------------")
            print(target_log)
            print("---------------------")
            with open(target_log, "r") as f:
                json_file = json.load(f)

            nb_folds = len(json_file["N_fold_train_results"])
            analyzer = Analyzer()
            for folds in range(nb_folds):
                outputs = json_file["N_fold_train_results"][folds]["cms"][
                    target_epoch - 1
                ]["outputs"]
                labels = json_file["N_fold_train_results"][folds]["cms"][
                    target_epoch - 1
                ]["labels"]
                paths = json_file["N_fold_train_results"][folds]["cms"][
                    target_epoch - 1
                ]["paths"]

                analyzer.append_results(outputs, labels, paths)
            meta_name = (
                train_setting["lr"],
                train_setting["net"],
                train_setting["affine_noise"],
                train_setting["mixup"],
                train_setting["adc"],
                train_setting["remove_ncf"],
                False,
                train_setting["elastic"],
            )
            analyzer.collect_each_tumor_info()
            if meta_name in results.keys():
                results[meta_name].append(analyzer)
            else:
                results[meta_name] = [analyzer]
    for key in results.keys():
        csv_file_name = (
            "per_tumor_net3d{}_affine+noise{}_mixup{}_elastic{}_adc{}.csv".format(
                key[1], key[2], key[3], key[7], key[4]
            )
        )
        if os.path.exists(csv_file_name):
            print(csv_file_name, " is exists")
            assert 1 == 0
        with open(os.path.join(output_dir, csv_file_name), "w") as f:
            writer = csv.writer(f)
            summary_dict = {}
            for analyzer in results[key]:
                for path_index in range(len(analyzer.paths)):
                    path = analyzer.paths[path_index]
                    label = analyzer.labels[path_index]
                    output = analyzer.outputs[path_index]
                    if path in summary_dict.keys():
                        assert summary_dict[path]["label"] == label
                        summary_dict[path]["outputs"].append(output)
                    else:
                        summary_dict[path] = {}
                        summary_dict[path]["label"] = label
                        summary_dict[path]["outputs"] = [output]
            nb_samples = [
                len(summary_dict[path]["outputs"]) for path in summary_dict.keys()
            ]
            assert max(nb_samples) == min(nb_samples)
            head_samples = []
            for train_nb in range(1, max(nb_samples) + 1):
                for class_name in ["B", "M"]:
                    head_samples += [
                        "train {}:predict score {}".format(train_nb, class_name)
                    ]
            head = ["ID", "label"] + head_samples
            writer.writerow(head)
            for path in summary_dict.keys():
                contents = [path, summary_dict[path]["label"].item()]
                for i in range(max(nb_samples)):
                    for j in range(2):
                        contents += [summary_dict[path]["outputs"][i][j].item()]
                writer.writerow(contents)
