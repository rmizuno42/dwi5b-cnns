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
        self.results_each_slice = np.array([[0, 0], [0, 0]])
        self.results_mean = np.array([[0, 0], [0, 0]])
        self.results_vote = np.array([[0, 0], [0, 0]])
        self.results_weighted_by_location = np.array([[0, 0], [0, 0]])
        self.results_each_slice_auc = 0
        self.results_mean_auc = 0
        self.results_vote_auc = 0
        self.results_weighted_by_location_auc = 0

    def append_results(self, outputs, labels, paths):
        for path in paths:
            assert path not in self.paths
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

    def argmax_from(self, predict, baseline, dim=None):
        assert predict.size() == baseline.size()
        if dim is None:
            return torch.argmax(predict - baseline)
        return torch.argmax(predict - baseline, dim=dim)

    def predict_test(self, path, baseline=torch.tensor([0.5, 0.5]), debug=False):
        for tumor_info in self.tumor_infos:
            if path != tumor_info.tumor_id:
                continue
            if debug:
                print("tumor_info.outputs", tumor_info.outputs)
            if len(tumor_info.outputs) >= 3:
                outputs = tumor_info.outputs[1:-1]
            else:
                outputs = tumor_info.outputs
            if debug:
                print("tumor_info.outputs 2", outputs)
            predict = self.argmax_from(outputs, baseline.repeat(len(outputs), 1), dim=1)
            if debug:
                print("predict", predict)
            vote = torch.sum(predict)
            if debug:
                print("vote", vote)
                print("len(outputs) / 2 < vote", len(outputs) / 2 < vote)
                print("len(outputs) / 2 > vote", len(outputs) / 2 > vote)
            if len(outputs) / 2 < vote:
                return 1
            elif len(outputs) / 2 > vote:
                return 0
            else:
                mean = torch.sum(outputs, dim=0) / torch.sum(outputs)
                if debug:
                    print("mean", mean)
                predict = self.argmax_from(mean, baseline)
                return predict

    def test_indexes(self, indexes):
        start = indexes[0]
        flag = True
        for i in range(len(indexes)):
            if indexes[i] == start + i:
                pass
            else:
                flag = False
        return flag


def slicescore2tumorscore(outputs):
    if len(outputs) >= 3:
        outputs = outputs[1:-1]
    outputs = sorted(outputs, key=lambda x: x[0])
    if len(outputs) % 2 == 1:
        return outputs[len(outputs) // 2]
    outputs = torch.tensor([i.tolist() for i in outputs])
    mean = torch.sum(outputs, dim=0) / torch.sum(outputs)
    outputs = [outputs[len(outputs) // 2 - 1], mean, outputs[len(outputs) // 2]]
    outputs = sorted(outputs, key=lambda x: x[0])
    return outputs[len(outputs) // 2]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", nargs="*", default=None, required=True)
    parser.add_argument("--target_type", choices=["validation", "eval"], required=True)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--output_dir", default="csvs_2d")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    target_log_list = args.target
    assert len(target_log_list) == len(set(target_log_list))
    if args.target_type == "validation":
        target_epoch = args.epoch
    elif args.target_type == "eval":
        target_epoch = 1
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    results = {}
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
                ]["outputs"]["0"]
                labels = (
                    np.array(
                        json_file["N_fold_train_results"][folds]["cms"][
                            target_epoch - 1
                        ]["labels"]
                    )[:, 0]
                ).tolist()
                paths = json_file["N_fold_train_results"][folds]["cms"][
                    target_epoch - 1
                ]["paths"]
                assert len(outputs) == len(labels) == len(paths)
                analyzer.append_results(outputs, labels, paths)
            analyzer.collect_each_tumor_info()
            meta_name = (
                train_setting["lr"],
                train_setting["net"],
                train_setting["affine_noise"],
                train_setting["mixup"],
                train_setting["adc"],
                train_setting["remove_ncf"],
                False,
                False,
                train_setting["elastic"],
                train_setting["finetune"],
            )
            if meta_name in results.keys():
                results[meta_name].append(analyzer)
            else:
                results[meta_name] = [analyzer]
    for key in results.keys():
        csv_file_name = "per_slice_net{}_affine+noise{}_mixup{}_elastic{}_adc{}_lr{}_finetune{}.csv".format(
            key[1], key[2], key[3], key[8], key[4], key[0], key[9]
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
        csv_file_name = "per_tumor_net{}_affine+noise{}_mixup{}_elastic{}_adc{}_lr{}_finetune{}.csv".format(
            key[1], key[2], key[3], key[8], key[4], key[0], key[9]
        )
        if os.path.exists(csv_file_name):
            print(csv_file_name, " is exists")
            assert 1 == 0

        with open(os.path.join(output_dir, csv_file_name), "w") as f:
            writer = csv.writer(f)
            summary_dict = {}
            for analyzer in results[key]:
                for tumor_info in analyzer.tumor_infos:
                    outputs = tumor_info.outputs
                    label = tumor_info.label
                    path = tumor_info.tumor_id
                    if path in summary_dict.keys():
                        assert summary_dict[path]["label"] == label
                        summary_dict[path]["outputs"].append(outputs)
                    else:
                        summary_dict[path] = {}
                        summary_dict[path]["label"] = label
                        summary_dict[path]["outputs"] = [outputs]
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
                asd = summary_dict[path]["outputs"].copy()

                summary_dict[path]["outputs"] = [
                    slicescore2tumorscore(outputs_all_slice)
                    for outputs_all_slice in summary_dict[path]["outputs"]
                ]
                for i in range(max(nb_samples)):
                    analyzer_predict = results[key][i].predict_test(path)
                    csv_predict = np.argmax(summary_dict[path]["outputs"][i])
                    if analyzer_predict != csv_predict:
                        print(path)
                        print(i)
                        print("slice outputs", asd[i])
                        print("tumor predict", summary_dict[path]["outputs"][i])
                        print("analyzer_predict", analyzer_predict)
                        analyzer_predict = results[key][i].predict_test(
                            path, debug=True
                        )
                        assert 1 == 0
                contents = [path, summary_dict[path]["label"].item()]
                for i in range(max(nb_samples)):
                    for j in range(2):
                        contents += [summary_dict[path]["outputs"][i][j].item()]
                writer.writerow(contents)
