# Based on https://github.com/deeplearning-wisc/dice

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_msp_score(logits, method_args):
    # Numerically stable softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    scores = np.max(softmax, axis=1)
    return scores

def get_msp_var_score(logits, logit_vars, method_args):
    # Numerically stable softmax
    logits -= logit_vars
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    scores = np.max(softmax, axis=1)
    return scores

def get_msp_var2_score(logits, logit_vars, method_args):
    # Numerically stable softmax
    logits
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    exp_logit_vars = np.exp(logit_vars - np.max(logit_vars, axis=1, keepdims=True))
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    softmax_logit_vars = exp_logit_vars / np.sum(exp_logit_vars, axis=1, keepdims=True)
    scores = np.max(softmax, axis=1) - np.max(softmax_logit_vars, axis=1)
    return scores


def get_sofl_score(logits, method_args):
    num_classes = method_args["num_classes"]
    logits_np = logits.detach().cpu().numpy()
    exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    scores = -softmax[:, num_classes:].sum(axis=1)
    return scores


def get_rowl_score(logits, method_args, raw_score=False):
    num_classes = method_args["num_classes"]
    logits_np = logits.detach().cpu().numpy()
    if raw_score:
        exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        scores = -1.0 * softmax[:, num_classes]
    else:
        scores = -1.0 * (logits_np.argmax(axis=1) == num_classes).astype(np.float32)
    return scores


def get_atom_score(logits, method_args):
    logits_np = logits.detach().cpu().numpy()
    exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    scores = -1.0 * softmax[:, -1]
    return scores


def get_energy_score(logits, method_args):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input

    # temper = method_args['temperature']

    # logits = logits / temper
    nnOutputs = logits
    scores = np.log(np.sum(np.exp(nnOutputs), axis=1))

    return scores

    
def get_energy_var_score(logits, logits_var, method_args):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input

    # temper = method_args['temperature']

    # logits = logits / temper
    nnOutputs = logits
    scores = np.log(np.sum(np.exp(nnOutputs), axis=1))

    return scores



def get_score(logits, method, method_args={}, logit_vars=None, raw_score=False):
    if method == "msp":
        scores = get_msp_score(logits, method_args)
    if method == "msp_var":
        scores = get_msp_var_score(logits, logit_vars, method_args)
    if method == "msp_var2":
        scores = get_msp_var2_score(logits, logit_vars, method_args)
    elif method == "energy":
        scores = get_energy_score(logits, method_args)
    elif method == "energy_var":
        scores = get_energy_var_score(logits, logit_vars, method_args)
    elif method == "sofl":
        scores = get_sofl_score(logits, method_args)
    elif method == "rowl":
        scores = get_rowl_score(logits, method_args, raw_score)
    elif method == "atom":
        scores = get_atom_score(logits, method_args)
    return scores
