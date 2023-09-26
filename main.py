#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [
            base_lr * (self.last_epoch + 1) / self.total_epoch
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description="Spatial Temporal Graph Convolution Network"
    )
    parser.add_argument(
        "--work-dir",
        default="./work_dir/temp",
        help="the work folder for storing results",
    )
    parser.add_argument("-model_saved_name", default="")
    parser.add_argument(
        "--config",
        default="./config/nturgbd-cross-view/agcn/train_joint_agcn.yaml",
        help="path to the configuration file",
    )

    parser.add_argument("--results_file_name", default="results.txt")

    # processor
    parser.add_argument("--phase", default="train", help="must be train or test")
    parser.add_argument(
        "--save-score",
        type=str2bool,
        default=False,
        help="if ture, the classification score will be stored",
    )
    # visulize and debug
    parser.add_argument("--seed", type=int, default=1, help="random seed for pytorch")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="the interval for printing messages (#iteration)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=2,
        help="the interval for storing models (#iteration)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="the interval for evaluating models (#iteration)",
    )
    parser.add_argument(
        "--print-log", type=str2bool, default=True, help="print logging or not"
    )
    parser.add_argument(
        "--show-topk",
        type=int,
        default=[1, 5],
        nargs="+",
        help="which Top K accuracy will be shown",
    )
    # feeder
    parser.add_argument(
        "--feeder", default="feeder.feeder", help="data loader will be used"
    )
    parser.add_argument(
        "--num-worker",
        type=int,
        default=32,
        help="the number of worker for data loader",
    )
    parser.add_argument(
        "--train-feeder-args",
        default=dict(),
        help="the arguments of data loader for training",
    )
    parser.add_argument(
        "--test-feeder-args",
        default=dict(),
        help="the arguments of data loader for test",
    )
    # model
    parser.add_argument("--model", default=None, help="the model will be used")
    parser.add_argument(
        "--model_name", default="stgcn", help="the name of the model will be used"
    )
    parser.add_argument(
        "--model-args", type=dict, default=dict(), help="the arguments of model"
    )
    parser.add_argument(
        "--weights", default=None, help="the weights for network initialization"
    )
    parser.add_argument(
        "--old_model_path",
        default=None,
        help="the weights for the old model for network initialization",
    )
    parser.add_argument(
        "--ignore-weights",
        type=str,
        default=[],
        nargs="+",
        help="the name of weights which will be ignored in the initialization",
    )
    parser.add_argument(
        "--blocksize", default=20, help="the size of the cout for each block"
    )
    parser.add_argument(
        "--numblocks", default=10, help="the maximum number of blocks in each layer"
    )
    parser.add_argument("--numlayers", default=10, help="the maximum number of layers")
    parser.add_argument("--topology", type=list, default=[], help="model topology")
    parser.add_argument(
        "--layer_threshold", default=1e-4, help="the threshold to stop adding layers"
    )
    parser.add_argument(
        "--block_threshold", default=1e-4, help="the threshold to stop adding blocks"
    )
    # optim
    parser.add_argument(
        "--base-lr", type=float, default=0.01, help="initial learning rate"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=[20, 40, 60],
        nargs="+",
        help="the epoch where optimizer reduce the learning rate",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        nargs="+",
        help="the indexes of GPUs for training or testing",
    )
    parser.add_argument("--optimizer", default="SGD", help="type of optimizer")
    parser.add_argument(
        "--nesterov", type=str2bool, default=False, help="use nesterov or not"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="training batch size"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=256, help="test batch size"
    )
    parser.add_argument(
        "--start-epoch", type=int, default=0, help="start training from which epoch"
    )
    parser.add_argument(
        "--num-epoch", type=int, default=50, help="stop training in which epoch"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0005, help="weight decay for optimizer"
    )
    parser.add_argument("--only_train_part", default=False)
    parser.add_argument("--only_train_epoch", default=0)
    parser.add_argument("--warm_up_epoch", default=0)
    parser.add_argument("--init_vnn_from", default="", type=str)
    parser.add_argument("--continue_global_step", default=False)
    parser.add_argument("--test_samples", default=None)
    parser.add_argument("--test_batch_sizes", default=None)
    parser.add_argument("--end_test", default=True)
    parser.add_argument("--DATASET_NAME", default=None)
    parser.add_argument("--SPLIT_NAME", default=None)
    parser.add_argument("--STREAMS_NAME", default=None)
    parser.add_argument("--MODEL_NAME", default=None)
    return parser


class Processor:
    """
    Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == "train":
            if not arg.train_feeder_args["debug"]:
                # if os.path.isdir(arg.model_saved_name):
                #     print("log_dir: ", arg.model_saved_name, "already exist")
                #     answer = input("delete it? y/[n]:")
                #     if answer == "y":
                #         shutil.rmtree(arg.model_saved_name)
                #         print("Dir removed: ", arg.model_saved_name)
                #     else:
                #         print("Dir not removed: ", arg.model_saved_name)
                self.train_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "train"), "train"
                )
                self.val_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "val"), "val"
                )
            else:
                self.train_writer = self.val_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "test"), "test"
                )
        self.global_step = 0
        if arg.model_name.lower() in ["stgcn", "agcn", "tagcn", "stbln"]:
            self.load_model()
            self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == "train":
            self.data_loader["train"] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed,
            )
        self.data_loader["test"] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed,
        )

    def load_model(self):
        output_device = (
            self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        )
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        # print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        if not self.arg.multiple:
            print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        if self.arg.weights:
            self.global_step = (
                0
                if (".latest." in self.arg.weights or ".best." in self.arg.weights)
                else int(self.arg.weights[:-3].split("-")[-1])
            )
            self.print_log("Load weights from {}.".format(self.arg.weights))
            if ".pkl" in self.arg.weights:
                with open(self.arg.weights, "r") as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)
            weights = OrderedDict(
                [
                    [k.split("module.")[-1], v.cuda(output_device)]
                    for k, v in weights.items()
                ]
            )
            keys = list(weights.keys())
            # print(keys)
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log(
                                "Sucessfully Remove Weights: {}.".format(key)
                            )
                        else:
                            self.print_log("Can Not Remove Weights: {}.".format(key))
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print("Can not find these weights:")
                for d in diff:
                    print("  " + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if self.arg.init_vnn_from:
            if self.arg.continue_global_step:
                self.global_step = int(arg.init_vnn_from[:-3].split("-")[-1])

            self.print_log("Init vnn weights from {}.".format(self.arg.init_vnn_from))
            if ".pkl" in self.arg.init_vnn_from:
                with open(self.arg.init_vnn_from, "r") as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.init_vnn_from)
            weights = OrderedDict(
                [
                    [k.split("module.")[-1], v.cuda(output_device)]
                    for k, v in weights.items()
                ]
            )

            def pair_parameter(name):
                if "tcn.means.0.weight" in name:
                    return (
                        name,
                        name.replace("tcn.means.0.weight", "tcn.t_conv.weight"),
                    )
                elif "tcn.means.1" in name:
                    return (name, name.replace("tcn.means.1", "tcn.bn"))
                if "residual.means.0.weight" in name:
                    return (
                        name,
                        name.replace(
                            "residual.means.0.weight", "residual.t_conv.weight"
                        ),
                    )
                elif "residual.means.1" in name:
                    return (name, name.replace("residual.means.1", "residual.bn"))
                else:
                    return (name, name.replace("means.0.", ""))

            paired_parameters = [
                pair_parameter(a)
                for a in self.model.state_dict().keys()
                if "means" in a
            ]
            unpaired_parameters = [
                a
                for a in self.model.state_dict().keys()
                if ("means" not in a) and ("stds" not in a)
            ]

            final_params = {}

            for a, b in paired_parameters:
                final_params[a] = weights[b]

            for a in unpaired_parameters:
                final_params[a] = weights[a]

            self.model.load_state_dict(final_params, strict=False)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model, device_ids=self.arg.device, output_device=output_device
                )

    def load_optimizer(self):
        if self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay,
            )
        elif self.arg.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
            )
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1
        )

        self.lr_scheduler = GradualWarmupScheduler(
            self.optimizer,
            total_epoch=self.arg.warm_up_epoch,
            after_scheduler=lr_scheduler_pre,
        )
        self.print_log("using warm up, epoch: {}".format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open("{}/config.yaml".format(self.arg.work_dir), "w") as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == "SGD" or self.arg.optimizer == "Adam":
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step))
                )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + " ] " + str
        print(str)
        if self.arg.print_log:
            with open("{}/log.txt".format(self.arg.work_dir), "a") as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_model(self, arg, epoch, is_best):

        if is_best:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k, v.cpu()] for k, v in state_dict.items()])
            if arg.model_name.lower() in ["pstgcn", "pstbln"]:
                torch.save(
                    weights,
                    self.arg.model_saved_name
                    + "-"
                    + str(len(self.arg.model_args["topology"]))
                    + "-"
                    + str(self.arg.model_args["topology"][-1])
                    + ".pt",
                )
                self.arg.best_model_path = (
                    self.arg.model_saved_name
                    + "-"
                    + str(len(self.arg.model_args["topology"]))
                    + "-"
                    + str(self.arg.model_args["topology"][-1])
                    + ".pt"
                )
            elif arg.model_name.lower() in ["stgcn", "agcn", "tagcn", "stbln"]:
                # torch.save(
                #     weights,
                #     self.arg.model_saved_name
                #     + "-"
                #     + str(epoch)
                #     + "-"
                #     + str(int(self.global_step))
                #     + ".pt",
                # )
                torch.save(
                    weights,
                    self.arg.model_saved_name
                    + ".best.pt",
                )
                self.arg.best_model_path = (
                    self.arg.model_saved_name
                    + ".best.pt"
                )
        else:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k, v.cpu()] for k, v in state_dict.items()])
            if arg.model_name.lower() in ["pstgcn", "pstbln"]:
                torch.save(
                    weights,
                    self.arg.model_saved_name
                    + "-"
                    + str(len(self.arg.model_args["topology"]))
                    + "-"
                    + str(self.arg.model_args["topology"][-1])
                    + ".latest.pt",
                )
                self.arg.trained_model_path = (
                    self.arg.model_saved_name
                    + "-"
                    + str(len(self.arg.model_args["topology"]))
                    + "-"
                    + str(self.arg.model_args["topology"][-1])
                    + ".latest.pt"
                )
            elif arg.model_name.lower() in ["stgcn", "agcn", "tagcn", "stbln"]:
                torch.save(weights, self.arg.model_saved_name + ".latest.pt")
                with open(self.arg.model_saved_name + ".latest.params", "w") as f:
                    print(
                        self.arg.model_saved_name
                        + "-"
                        + str(epoch)
                        + "-"
                        + str(int(self.global_step))
                        + ".pt",
                        file=f,
                    )
                    self.arg.trained_model_path = self.arg.model_saved_name + ".latest.pt"

    def train(self, epoch):
        self.model.train()
        self.print_log("Training epoch: {}".format(epoch + 1))
        loader = self.data_loader["train"]
        self.adjust_learning_rate(epoch)
        loss_value = []
        self.train_writer.add_scalar("epoch", epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print("only train part, require grad")
                for key, value in self.model.named_parameters():
                    if "PA" in key:
                        value.requires_grad = True
            else:
                print("only train part, do not require grad")
                for key, value in self.model.named_parameters():
                    if "PA" in key:
                        value.requires_grad = False
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer["dataloader"] += self.split_time()

            # forward
            output = self.model(data)
            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0
            loss = self.loss(output, label) + l1

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer["model"] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            self.train_writer.add_scalar("acc", acc, self.global_step)
            self.train_writer.add_scalar("loss", loss.data.item(), self.global_step)
            self.train_writer.add_scalar("loss_l1", l1, self.global_step)
            # statistics
            self.lr = self.optimizer.param_groups[0]["lr"]
            self.train_writer.add_scalar("lr", self.lr, self.global_step)
            timer["statistics"] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: "{:02d}%".format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log("\tMean training loss: {:.4f}.".format(np.mean(loss_value)))
        self.print_log(
            "\tTime consumption: [Data]{dataloader}, [Network]{model}".format(
                **proportion
            )
        )

        self.save_model(arg, epoch, False)

        return loss, acc

    def eval(
        self,
        epoch,
        save_score=False,
        loader_name=["test"],
        wrong_file=None,
        result_file=None,
    ):
        if wrong_file is not None:
            f_w = open(wrong_file, "w")
        if result_file is not None:
            f_r = open(result_file, "w")
        self.model.eval()
        self.print_log("Eval epoch: {}".format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            lbls = []
            preds = []
            outs = []
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = Variable(
                        data.float().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True,
                    )
                    label = Variable(
                        label.long().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True,
                    )

                    output = self.model(data)

                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1
                    lbls.append(label.data.cpu().numpy())
                    preds.append(predict_label.data.cpu().numpy())
                    outs.append(output.data.cpu().numpy())

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + "," + str(true[i]) + "\n")
                        if x != true[i] and wrong_file is not None:
                            f_w.write(
                                str(index[i]) + "," + str(x) + "," + str(true[i]) + "\n"
                            )
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            preds_val = np.concatenate(preds)
            lbls_val = np.concatenate(lbls)
            accuracy = np.mean((preds_val == lbls_val))
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.should_save = True
            # self.lr_scheduler.step(loss)

            result = ""

            print("Accuracy: ", accuracy, " model: ", self.arg.model_saved_name)
            if self.arg.phase == "train":
                self.val_writer.add_scalar("loss", loss, self.global_step)
                self.val_writer.add_scalar("loss_l1", l1, self.global_step)
                self.val_writer.add_scalar("acc", accuracy, self.global_step)

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log(
                "\tMean {} loss of {} batches: {}.".format(
                    ln, len(self.data_loader[ln]), np.mean(loss_value)
                )
            )
            for k in self.arg.show_topk:
                self.print_log(
                    "\tTop{}: {:.2f}%".format(
                        k, 100 * self.data_loader[ln].dataset.top_k(score, k)
                    )
                )

                result += "Top{}: {:.2f}% ".format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)
                )

            if save_score:
                with open(
                    "{}/epoch{}_{}_score.pkl".format(self.arg.work_dir, epoch + 1, ln),
                    "wb",
                ) as f:
                    pickle.dump(score_dict, f)

            return result

    def prog_init(self, block_iter):
        if block_iter == 0:
            weights = torch.load(
                self.arg.model_saved_name
                + "-"
                + str(len(self.arg.model_args["topology"]) - 1)
                + "-"
                + str(self.arg.model_args["topology"][-2])
                + ".pt"
            )
        else:
            weights = torch.load(
                self.arg.model_saved_name
                + "-"
                + str(len(self.arg.model_args["topology"]))
                + "-"
                + str(self.arg.model_args["topology"][-1] - 1)
                + ".pt"
            )
        weights = OrderedDict(
            [[k, v.cuda(self.output_device)] for k, v in weights.items()]
        )
        old_keys = list(weights.keys())
        for current_key in self.model.state_dict():
            if ("graph_attn" or "rand_graph") in current_key:
                if current_key in old_keys:
                    new_state_dict = OrderedDict({current_key: weights[current_key]})
                    self.model.load_state_dict(new_state_dict, strict=False)
            if (
                "g_conv"
                or "gcn_residual"
                or "bln_residual"
                or "tcn.t_conv.bias"
                or "residual"
                or "bn.weight"
                or "bn.bias"
                or "bn.running_mean"
                or "bn.running_var"
            ) in current_key:
                if current_key in old_keys:
                    A = self.model.state_dict()[current_key]
                    old_sh = weights[current_key].shape
                    print("old_sh", old_sh)
                    A[: old_sh[0]] = weights[current_key]
                    new_state_dict = OrderedDict({current_key: A})
                    self.model.load_state_dict(new_state_dict, strict=False)
            if "tcn.t_conv.weight" in current_key:
                if current_key in old_keys:
                    A = self.model.state_dict()[current_key]
                    old_sh = weights[current_key].shape
                    A[: old_sh[0], : old_sh[1]] = weights[current_key]
                    new_state_dict = OrderedDict({current_key: A})
                    self.model.load_state_dict(new_state_dict, strict=False)
            if ("fc.weight" in current_key) and (block_iter > 0):
                if current_key in old_keys:
                    A = self.model.state_dict()[current_key]
                    old_sh = weights[current_key].shape
                    A[: old_sh[0], : old_sh[1]] = weights[current_key]
                    new_state_dict = OrderedDict({current_key: A})
                    self.model.load_state_dict(new_state_dict, strict=False)

    def start(self):
        if self.arg.phase == "train":
            if (
                arg.model_name.lower() in ["stgcn", "agcn", "tagcn", "stbln"]
                or self.arg.model_args["topology"] != []
            ):
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    # if self.lr < 1e-3:
                    #     break
                    self.train(epoch)
                    self.should_save = False
                    self.eval(
                        epoch, save_score=self.arg.save_score, loader_name=["test"]
                    )

                    if self.should_save:
                        self.save_model(self.arg, epoch, True)

                print(
                    "best accuracy: ",
                    self.best_acc,
                    " model_name: ",
                    self.arg.model_saved_name,
                )
            elif (
                arg.model_name.lower() in ["pstgcn", "pstbln"]
                and self.arg.model_args["topology"] == []
            ):
                acc_layer_old = acc_block_old = acc_layer_new = acc_block_new = 1e-10
                loss_layer_old = loss_block_old = loss_layer_new = loss_block_new = 1e10
                for layer_iter in range(self.arg.numlayers):
                    self.arg.model_args["topology"].append(0)
                    # add one layer
                    for block_iter in range(self.arg.numblocks):
                        print(
                            "######################################################################\n"
                        )
                        print("layer." + str(layer_iter) + "_block." + str(block_iter))
                        print(
                            "\n######################################################################\n"
                        )
                        # add one block
                        self.arg.model_args["topology"][layer_iter] = (
                            self.arg.model_args["topology"][layer_iter] + 1
                        )
                        self.load_model()
                        self.load_optimizer()
                        self.lr = self.arg.base_lr
                        self.best_acc = 0
                        self.global_step = (
                            self.arg.start_epoch
                            * len(self.data_loader["train"])
                            / self.arg.batch_size
                        )
                        if layer_iter > 0 or block_iter > 0:
                            self.prog_init(block_iter)
                        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                            if self.lr < 1e-3:
                                break
                            save_model = epoch + 1 == self.arg.num_epoch
                            train_loss, train_acc = self.train(
                                epoch
                            )
                            # if train_acc > self.best_train_acc:
                            #   self.best_train_acc = train_acc
                            self.eval(
                                epoch,
                                save_score=self.arg.save_score,
                                loader_name=["test"],
                            )
                            acc_block_new = train_acc
                            loss_block_new = train_loss
                        # training is finished in N epochs

                        print(
                            "best accuracy: ",
                            self.best_acc,
                            " model_name: ",
                            self.arg.model_saved_name,
                        )
                        with open(arg.results_file_name, "a") as fid:
                            fid.write("layer %.2f \t" % (layer_iter))
                            fid.write("block %.2f \n" % (block_iter))
                            fid.write(
                                "Network Topology: %s \n"
                                % (self.arg.model_args["topology"])
                            )
                            fid.write("Finish training with following performance: \n")
                            fid.write(
                                "best test Acc: %.4f, block_size: %.2f \n"
                                % (self.best_acc, self.arg.model_args["blocksize"])
                            )
                            fid.write("train loss: %.4f \n" % (loss_block_new))
                        if block_iter > 0:
                            loss_b = (
                                -1 * (loss_block_new - loss_block_old) / loss_block_old
                            )
                            acc_b = (acc_block_new - acc_block_old) / acc_block_old
                            if loss_b <= self.arg.block_threshold:
                                self.arg.model_args["topology"][layer_iter] = (
                                    self.arg.model_args["topology"][layer_iter] - 1
                                )
                                print(
                                    "block"
                                    + str(block_iter)
                                    + "of layer"
                                    + str(layer_iter)
                                    + "is removed \n"
                                )
                                print(
                                    "block progression is stopped in layer"
                                    + str(layer_iter)
                                )
                                break
                        acc_block_old = acc_block_new
                        acc_layer_new = acc_block_new
                        loss_block_old = loss_block_new
                        loss_layer_new = loss_block_new
                    if layer_iter > 0:
                        loss_l = -1 * (loss_layer_new - loss_layer_old) / loss_layer_old
                        acc_l = (acc_layer_new - acc_layer_old) / acc_layer_old
                        if loss_l <= self.arg.layer_threshold:
                            self.arg.model_args[
                                "topology"
                            ].pop()  # remove the last layer
                            print("layer" + str(layer_iter) + "is removed \n")
                            print("layer progression is stopped")
                            print(acc_layer_old)
                            break
                    acc_layer_old = acc_layer_new
                    loss_layer_old = loss_layer_new

        elif self.arg.phase == "test":
            if not self.arg.test_feeder_args["debug"]:
                wf = self.arg.model_saved_name + "_wrong.txt"
                rf = self.arg.model_saved_name + "_right.txt"
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError("Please appoint --weights.")
            self.arg.print_log = False

            if not self.arg.multiple:
                self.print_log("Model:   {}.".format(self.arg.model))
                self.print_log("Weights: {}.".format(self.arg.weights))

            result = self.eval(
                epoch=0,
                save_score=self.arg.save_score,
                loader_name=["test"],
                wrong_file=wf,
                result_file=rf,
            )
            self.arg.all_results[-1]["result"] = result

            if not self.arg.multiple:
                self.print_log("Done.\n")


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def parse_config_file(config):
    with open(config, "r") as f:
        args = yaml.load(f, Loader=yaml.Loader)

    def decode_config_path(new_config):
        if new_config == "base":
            return os.path.join(os.path.dirname(config), "..", "base.yaml")
        elif new_config[0] == "/":
            return os.path.join("./config", new_config[1:] + ".yaml")

        return new_config

    def merge_configs(parent, child):
        result = {}

        def merge_objects(local_parent: dict, local_child: dict, local_result: dict):
            for k, v in local_parent.items():
                local_result[k] = v

            for k, v in local_child.items():
                if k in local_result:
                    if isinstance(local_result[k], dict):
                        merge_objects(local_parent[k], local_child[k], local_result[k])
                    else:
                        local_result[k] = v
                else:
                    local_result[k] = v

        merge_objects(parent, child, result)

        return result

    if "include" in args:
        all_include_args = [
            parse_config_file(decode_config_path(a)) for a in args["include"]
        ]

        for include_args in all_include_args:
            args = merge_configs(include_args, args)

        del args["include"]

    return args


def parse_config_parametrized_values(args):

    if args["device"] == -1:
        args["device"] = [*range(torch.cuda.device_count())] # All CUDA_VISIBLE_DEVICES

    def replace_strings(name):
        if "DATASET_NAME" in args:
            args[name] = args[name].replace("$DATASET", args["DATASET_NAME"])

        if "SPLIT_NAME" in args:
            args[name] = args[name].replace("$SPLIT", args["SPLIT_NAME"])

        if "STREAMS_NAME" in args:
            args[name] = args[name].replace("$STREAMS", args["STREAMS_NAME"])

        if "MODEL_NAME" in args:
            args[name] = args[name].replace("$MODEL", args["MODEL_NAME"])

        if "model_args" in args and "samples" in args["model_args"]:
            args[name] = args[name].replace("$SAMPLES", str(args["model_args"]["samples"]))

        args[name] = args[name].replace("$NUM_EPOCH", str(args["num_epoch"]))
        args[name] = args[name].replace("$BATCH_SIZE", str(args["batch_size"]))

    replace_strings("work_dir")

    if "init_vnn_from" in args:
        replace_strings("init_vnn_from")

    if "weights" in args:
        replace_strings("weights")

    if "model_saved_name" not in args:
        args["model_saved_name"] = args["work_dir"]

    return args


if __name__ == "__main__":
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        default_arg = parse_config_file(p.config)
        default_arg = parse_config_parametrized_values(default_arg)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG: {}".format(k))
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    arg.multiple = False

    if arg.phase == "train":
        init_seed(0)
        processor = Processor(arg)
        processor.start()
    
    if arg.phase == "test" or arg.end_test:
        arg.phase = "test"
        if "best_model_path" in arg:
            arg.weights = arg.best_model_path

        arg.all_results = []
        
        if "test_samples" not in arg or arg.test_samples is None:
            arg.all_results.append(
                {
                    "result": "",
                }
            )
            init_seed(0)
            processor = Processor(arg)
            processor.start()
            
            with open(f"{arg.model_saved_name}.test.result", "w") as f:
                for a in arg.all_results:
                    print(a, file=f)
        else:
            print(":::Multiple tests:::")

            samples = arg.test_samples
            batch_sizes = arg.test_batch_sizes
            arg.multiple = True

            for sample, batch_size in zip(samples, batch_sizes):
                arg.samples = None
                arg.model_args["test_samples"] = sample
                arg.test_batch_size = batch_size

                print(f":::samples={sample} batch={batch_size}:::")

                arg.all_results.append(
                    {
                        "samples": sample,
                        "batch": batch_size,
                        "result": "",
                    }
                )

                init_seed(0)
                processor = Processor(arg)
                processor.start()

                with open(f"{arg.model_saved_name}.test.result", "w") as f:
                    for a in arg.all_results:
                        print(a, file=f)

            for a in arg.all_results:
                print(a)
