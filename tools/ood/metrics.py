# Based on https://github.com/deeplearning-wisc/dice

import numpy as np


def get_curve(known, novel, method=None, false_negatives=0.05):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known), np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == "row":
        threshold = -0.5
    else:
        threshold = known[round(false_negatives * num_k)]

    tp = -np.ones([num_k + num_n + 1], dtype=int)
    fp = -np.ones([num_k + num_n + 1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k + num_n):
        if k == num_k:
            tp[l + 1 :] = tp[l]
            fp[l + 1 :] = np.arange(fp[l] - 1, -1, -1)
            break
        elif n == num_n:
            tp[l + 1 :] = np.arange(tp[l] - 1, -1, -1)
            fp[l + 1 :] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l + 1] = tp[l]
                fp[l + 1] = fp[l] - 1
            else:
                k += 1
                tp[l + 1] = tp[l] - 1
                fp[l + 1] = fp[l]

    j = num_k + num_n - 1
    for l in range(num_k + num_n - 1):
        if all[j] == all[j - 1]:
            tp[j] = tp[j + 1]
            fp[j] = fp[j + 1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


def cal_metric(known, novel, method=None):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    _, _, fpr_at_tpr80 = get_curve(known, novel, method, false_negatives=0.20)
    results = dict()
    mtypes = ["FPR", "AUROC", "DTERR", "AUIN", "AUOUT"]

    results = dict()

    # FPR
    mtype = "FPR"
    results[mtype] = fpr_at_tpr95

    mtype = "FPR-80"
    results[mtype] = fpr_at_tpr80

    # AUROC
    mtype = "AUROC"
    tpr = np.concatenate([[1.0], tp / tp[0], [0.0]])
    fpr = np.concatenate([[1.0], fp / fp[0], [0.0]])
    results[mtype] = -np.trapz(1.0 - fpr, tpr)

    # DTERR
    mtype = "DTERR"
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = "AUIN"
    denom = tp + fp
    denom[denom == 0.0] = -1.0
    pin_ind = np.concatenate([[True], denom > 0.0, [True]])
    pin = np.concatenate([[0.5], tp / denom, [0.0]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = "AUOUT"
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.0] = -1.0
    pout_ind = np.concatenate([[True], denom > 0.0, [True]])
    pout = np.concatenate([[0.0], (fp[0] - fp) / denom, [0.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.0 - fpr[pout_ind])

    return results


def print_all_results(result, method, file=None, dataset="NTU-120"):
    mtypes = ["FPR", "FPR-80", "AUROC", "AUIN"]
    print(" OOD detection method: " + method, file=file)
    print("             ", end="", file=file)
    for mtype in mtypes:
        print(" {mtype:6s}".format(mtype=mtype), end="", file=file)
    print("\n{dataset:12s}".format(dataset=dataset), end="", file=file)
    print(" {val:6.2f}".format(val=100.0 * result["FPR"]), end="", file=file)
    print(" {val:6.2f}".format(val=100.0 * result["FPR-80"]), end="", file=file)
    print(" {val:6.2f}".format(val=100.0 * result["AUROC"]), end="", file=file)
    print(" {val:6.2f}".format(val=100.0 * result["AUIN"]), end="", file=file)
    print("", file=file)


def compute_traditional_ood(in_distribution_scores, test_scores, method):

    known = in_distribution_scores
    novel = test_scores
    result = cal_metric(known, novel, method)

    print_all_results(result, method)

    return result
