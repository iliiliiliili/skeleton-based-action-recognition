import os
import pickle

from fire import Fire
import numpy as np
from tabulate import tabulate, SEPARATING_LINE

from metrics import compute_traditional_ood, cal_metric
from ood_detection_scores import get_score
from dirichlet_estimator import (
    dirichlet_max_confidence,
    estimate_dirichlet_params,
    estimate_mle_dirichlet,
    estimate_mle_dirichlet,
    plot_dirichlet_list,
    estimate_mle_dirichlet_batched,
    dirichlet_entropy,
    is_ood_dirichlet,
)

def detect_traditional_ood(ind_logits, ood_logits, table=[["Method", "FPR-95", "FPR-80", "AUROC"]]):

    results = {}

    for method in ["energy", "msp"]:

        ind_scores = get_score(ind_logits, method)
        ood_scores = get_score(ood_logits, method)

        results[method] = compute_traditional_ood(ind_scores, ood_scores, method)
        table.append([method, f"{100.0 * results[method]['FPR']:.2f}", f"{100.0 * results[method]['FPR-80']:.2f}", f"{100.0 * results[method]['AUROC']:.2f}"])


    method = "Energy + MSP"

    ind_scores1 = get_score(ind_logits, "energy")
    ood_scores1 = get_score(ood_logits, "energy")

    ind_scores2 = get_score(ind_logits, "msp")
    ood_scores2 = get_score(ood_logits, "msp")

    ind_scores = ind_scores1 + ind_scores2
    ood_scores = ood_scores1 + ood_scores2

    results[method] = compute_traditional_ood(ind_scores, ood_scores, method)
    table.append([method, f"{100.0 * results[method]['FPR']:.2f}", f"{100.0 * results[method]['FPR-80']:.2f}", f"{100.0 * results[method]['AUROC']:.2f}"])

    return results, table


def detect_traditional_var_ood(ind_logits, ood_logits, ind_logits_vars, ood_logits_vars, table=[["Method", "FPR-95", "FPR-80", "AUROC"]]):

    results = {}

    for method in ["msp_var", "msp_var2", "energy_var"]:

        ind_scores = get_score(ind_logits, method, logit_vars=ind_logits_vars)
        ood_scores = get_score(ood_logits, method, logit_vars=ood_logits_vars)

        results[method] = compute_traditional_ood(ind_scores, ood_scores, method)
        table.append([method, f"{100.0 * results[method]['FPR']:.2f}", f"{100.0 * results[method]['FPR-80']:.2f}", f"{100.0 * results[method]['AUROC']:.2f}"])

    return results, table


# Used to create OOD table (1 in the paper)
def detect_ood(
    ind_variances,
    ood_variances,
    dirichlets_ind,
    dirichlets_ood,
    dirichlets_moments_ind,
    dirichlets_moments_ood,
    entropies_ind,
    entropies_ood,
    entropies_moments_ind,
    entropies_moments_ood,
    ind_max_variances,
    ood_max_variances,
    ind_logits,
    ood_logits,
    ind_logits_vars,
    ood_logits_vars,
    thresholds={},
    table=[["Method", "FPR-95", "FPR-80", "AUROC"]]
):

    print(
        "OOD variance: ", ood_variances.mean(), " IND variance: ", ind_variances.mean()
    )
    print("OOD variance / IND variance: ", ood_variances.mean() / ind_variances.mean())

    
    print("-------------------------------")
    print("Variance detection: logits mean of all categories vs mean")

    best_thresholds = {}

    best_f1 = None
    best_threshold = None

    for mean_variance_threshold in (
        thresholds["variance:logits-mean-vs-mean"]
        if "variance:logits-mean-vs-mean" in thresholds
        else [1, 2, 3, 4, 5, 10, 20, 50, 100]
    ):

        ind_scores = -ind_logits_vars.mean(axis=-1)
        ood_scores = -ood_logits_vars.mean(axis=-1)

        fraction_of_ind_as_ood = (
            np.sum(
                ind_logits_vars.mean(axis=-1)
                > ind_logits_vars.mean() * mean_variance_threshold
            )
            / ind_logits_vars.shape[0]
        )
        fraction_of_ood_as_ood = (
            np.sum(
                ood_logits_vars.mean(axis=-1)
                > ind_logits_vars.mean() * mean_variance_threshold
            )
            / ood_logits_vars.shape[0]
        )

        f1_score = (
            2
            * (fraction_of_ood_as_ood * (1 - fraction_of_ind_as_ood))
            / (fraction_of_ood_as_ood + (1 - fraction_of_ind_as_ood) + 1e-8)
        )

        if best_f1 is None or f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = mean_variance_threshold

        print(
            f"Threshold: {mean_variance_threshold}x mean IND variance - Fraction of IND classified as OOD: {fraction_of_ind_as_ood:.3f}, Fraction of OOD classified as OOD: {fraction_of_ood_as_ood:.3f}, F1 Score: {f1_score:.3f}"
        )
    
    metrics = cal_metric(ind_scores, ood_scores)
    print(f"FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Variance: Logits mean vs mean", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])

    best_thresholds["variance:logits-mean-vs-mean"] = [best_threshold]


    print("-------------------------------")
    print("Variance detection: logits max of all categories vs mean")

    best_thresholds = {}

    best_f1 = None
    best_threshold = None

    for mean_variance_threshold in (
        thresholds["variance:max-vs-mean"]
        if "variance:max-vs-mean" in thresholds
        else [1, 2, 3, 4, 5, 10, 20, 50, 100]
    ):

        ind_scores = -ind_logits_vars.max(axis=-1)
        ood_scores = -ood_logits_vars.max(axis=-1)

        fraction_of_ind_as_ood = (
            np.sum(
                ind_logits_vars.max(axis=-1)
                > ind_logits_vars.mean() * mean_variance_threshold
            )
            / ind_logits_vars.shape[0]
        )
        fraction_of_ood_as_ood = (
            np.sum(
                ood_logits_vars.max(axis=-1)
                > ind_logits_vars.mean() * mean_variance_threshold
            )
            / ood_logits_vars.shape[0]
        )

        f1_score = (
            2
            * (fraction_of_ood_as_ood * (1 - fraction_of_ind_as_ood))
            / (fraction_of_ood_as_ood + (1 - fraction_of_ind_as_ood) + 1e-8)
        )

        if best_f1 is None or f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = mean_variance_threshold

        print(
            f"Threshold: {mean_variance_threshold}x mean IND variance - Fraction of IND classified as OOD: {fraction_of_ind_as_ood:.3f}, Fraction of OOD classified as OOD: {fraction_of_ood_as_ood:.3f}, F1 Score: {f1_score:.3f}"
        )
    
    metrics = cal_metric(ind_scores, ood_scores)
    print(f"FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Variance: Logits max vs mean", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])


    print("-------------------------------")
    print("Variance detection: mean of all categories vs mean")

    best_thresholds = {}

    best_f1 = None
    best_threshold = None

    for mean_variance_threshold in (
        thresholds["variance:mean-vs-mean"]
        if "variance:mean-vs-mean" in thresholds
        else [1, 2, 3, 4, 5, 10, 20, 50, 100]
    ):

        ind_scores = -ind_variances.mean(axis=-1)
        ood_scores = -ood_variances.mean(axis=-1)

        fraction_of_ind_as_ood = (
            np.sum(
                ind_variances.mean(axis=-1)
                > ind_variances.mean() * mean_variance_threshold
            )
            / ind_variances.shape[0]
        )
        fraction_of_ood_as_ood = (
            np.sum(
                ood_variances.mean(axis=-1)
                > ind_variances.mean() * mean_variance_threshold
            )
            / ood_variances.shape[0]
        )

        f1_score = (
            2
            * (fraction_of_ood_as_ood * (1 - fraction_of_ind_as_ood))
            / (fraction_of_ood_as_ood + (1 - fraction_of_ind_as_ood) + 1e-8)
        )

        if best_f1 is None or f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = mean_variance_threshold

        print(
            f"Threshold: {mean_variance_threshold}x mean IND variance - Fraction of IND classified as OOD: {fraction_of_ind_as_ood:.3f}, Fraction of OOD classified as OOD: {fraction_of_ood_as_ood:.3f}, F1 Score: {f1_score:.3f}"
        )
    
    metrics = cal_metric(ind_scores, ood_scores)
    print(f"FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Variance: Mean vs mean", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])

    best_thresholds["variance:mean-vs-mean"] = [best_threshold]


    print("-------------------------------")
    print("Variance detection: max of all categories vs mean")

    best_thresholds = {}

    best_f1 = None
    best_threshold = None

    for mean_variance_threshold in (
        thresholds["variance:max-vs-mean"]
        if "variance:max-vs-mean" in thresholds
        else [1, 2, 3, 4, 5, 10, 20, 50, 100]
    ):

        ind_scores = -ind_variances.max(axis=-1)
        ood_scores = -ood_variances.max(axis=-1)

        fraction_of_ind_as_ood = (
            np.sum(
                ind_variances.max(axis=-1)
                > ind_variances.mean() * mean_variance_threshold
            )
            / ind_variances.shape[0]
        )
        fraction_of_ood_as_ood = (
            np.sum(
                ood_variances.max(axis=-1)
                > ind_variances.mean() * mean_variance_threshold
            )
            / ood_variances.shape[0]
        )

        f1_score = (
            2
            * (fraction_of_ood_as_ood * (1 - fraction_of_ind_as_ood))
            / (fraction_of_ood_as_ood + (1 - fraction_of_ind_as_ood) + 1e-8)
        )

        if best_f1 is None or f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = mean_variance_threshold

        print(
            f"Threshold: {mean_variance_threshold}x mean IND variance - Fraction of IND classified as OOD: {fraction_of_ind_as_ood:.3f}, Fraction of OOD classified as OOD: {fraction_of_ood_as_ood:.3f}, F1 Score: {f1_score:.3f}"
        )
    
    metrics = cal_metric(ind_scores, ood_scores)
    print(f"FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Variance: Max vs mean", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])

    best_thresholds["variance:max-vs-mean"] = [best_threshold]

    print("-------------------------------")
    print(
        "Variance detection: variance of max category vs mean variance of max category"
    )

    best_f1 = None
    best_threshold = None

    for mean_variance_threshold in (
        thresholds["variance:max-category-vs-mean"]
        if "variance:max-category-vs-mean" in thresholds
        else [1, 2, 3, 4, 5, 10, 20, 50, 100]
    ):

        ind_scores = -ind_max_variances
        ood_scores = -ood_max_variances

        fraction_of_ind_as_ood = (
            np.sum(
                ind_max_variances > ind_max_variances.mean() * mean_variance_threshold
            )
            / ind_variances.shape[0]
        )
        fraction_of_ood_as_ood = (
            np.sum(
                ood_max_variances > ind_max_variances.mean() * mean_variance_threshold
            )
            / ood_variances.shape[0]
        )

        f1_score = (
            2
            * (fraction_of_ood_as_ood * (1 - fraction_of_ind_as_ood))
            / (fraction_of_ood_as_ood + (1 - fraction_of_ind_as_ood) + 1e-8)
        )

        if best_f1 is None or f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = mean_variance_threshold

        print(
            f"Threshold: {mean_variance_threshold}x mean IND variance - Fraction of IND classified as OOD: {fraction_of_ind_as_ood:.3f}, Fraction of OOD classified as OOD: {fraction_of_ood_as_ood:.3f}, F1 Score: {f1_score:.3f}"
        )


    metrics = cal_metric(ind_scores, ood_scores)
    print(f"FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Variance: Max category vs mean", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])

    best_thresholds["variance:max-category-vs-mean"] = [best_threshold]

    print("-------------------------------")
    print(
        "Variance detection: variance of max category vs max variance of max category"
    )

    best_f1 = None
    best_threshold = None

    for max_variance_threshold in (
        thresholds["variance:max-category-vs-max"]
        if "variance:max-category-vs-max" in thresholds
        else [0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 2]
    ):
        fraction_of_ind_as_ood = (
            np.sum(ind_max_variances > ind_max_variances.max() * max_variance_threshold)
            / ind_variances.shape[0]
        )
        fraction_of_ood_as_ood = (
            np.sum(ood_max_variances > ind_max_variances.max() * max_variance_threshold)
            / ood_variances.shape[0]
        )

        f1_score = (
            2
            * (fraction_of_ood_as_ood * (1 - fraction_of_ind_as_ood))
            / (fraction_of_ood_as_ood + (1 - fraction_of_ind_as_ood) + 1e-8)
        )

        if best_f1 is None or f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = max_variance_threshold

        print(
            f"Threshold: {max_variance_threshold}x mean IND variance - Fraction of IND classified as OOD: {fraction_of_ind_as_ood:.3f}, Fraction of OOD classified as OOD: {fraction_of_ood_as_ood:.3f}, F1 Score: {f1_score:.3f}"
        )

    best_thresholds["variance:max-category-vs-max"] = [best_threshold]

    print("-------------------------------")
    print("Maximum confidence detection MLE")

    best_f1 = None
    best_threshold = None

    for threshold in (
        thresholds["max-confidence:MLE"]
        if "max-confidence:MLE" in thresholds
        else [0.5, 0.7, 0.8, 0.9, 0.95]
    ):

        ind_scores = np.array([dirichlet_max_confidence(dirichlets_ind[i]) for i in range(len(dirichlets_ind))])
        ood_scores = np.array([dirichlet_max_confidence(dirichlets_ood[i]) for i in range(len(dirichlets_ood))])

        fraction_of_ind_as_ood = np.sum(
            [
                is_ood_dirichlet(dirichlets_ind[i], threshold)
                for i in range(len(dirichlets_ind))
            ]
        ) / len(dirichlets_ind)
        fraction_of_ood_as_ood = np.sum(
            [
                is_ood_dirichlet(dirichlets_ood[i], threshold)
                for i in range(len(dirichlets_ood))
            ]
        ) / len(dirichlets_ood)

        f1_score = (
            2
            * (fraction_of_ood_as_ood * (1 - fraction_of_ind_as_ood))
            / (fraction_of_ood_as_ood + (1 - fraction_of_ind_as_ood) + 1e-8)
        )

        if best_f1 is None or f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold

        print(
            f"Threshold: {threshold} - Fraction of IND classified as OOD: {fraction_of_ind_as_ood:.3f}, Fraction of OOD classified as OOD: {fraction_of_ood_as_ood:.3f}, F1 Score: {f1_score:.3f}"
        )
        
    metrics = cal_metric(ind_scores, ood_scores)
    print(f"FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Max confidence: MLE", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])

    best_thresholds["max-confidence:MLE"] = [best_threshold]

    print("-------------------------------")
    print("Maximum confidence detection Moments")

    best_f1 = None
    best_threshold = None

    for threshold in (
        thresholds["max-confidence:Moments"]
        if "max-confidence:Moments" in thresholds
        else [0.01, 0.02, 0.03, 0.035, 0.04, 0.0425, 0.045, 0.05, 0.16]
    ):

        ind_scores = np.array([dirichlet_max_confidence(dirichlets_moments_ind[i]) for i in range(len(dirichlets_moments_ind))])
        ood_scores = np.array([dirichlet_max_confidence(dirichlets_moments_ood[i]) for i in range(len(dirichlets_moments_ood))])

        fraction_of_ind_as_ood = np.sum(
            [
                is_ood_dirichlet(dirichlets_moments_ind[i], threshold)
                for i in range(len(dirichlets_moments_ind))
            ]
        ) / len(dirichlets_moments_ind)
        fraction_of_ood_as_ood = np.sum(
            [
                is_ood_dirichlet(dirichlets_moments_ood[i], threshold)
                for i in range(len(dirichlets_moments_ood))
            ]
        ) / len(dirichlets_moments_ood)

        f1_score = (
            2
            * (fraction_of_ood_as_ood * (1 - fraction_of_ind_as_ood))
            / (fraction_of_ood_as_ood + (1 - fraction_of_ind_as_ood) + 1e-8)
        )

        if best_f1 is None or f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold

        print(
            f"Threshold: {threshold} - Fraction of IND classified as OOD: {fraction_of_ind_as_ood:.3f}, Fraction of OOD classified as OOD: {fraction_of_ood_as_ood:.3f}, F1 Score: {f1_score:.3f}"
        )
        
    metrics = cal_metric(ind_scores, ood_scores)
    print(f"Scores FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Max confidence: Moments", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])

    msp_ind_scores = get_score(ind_logits, "msp")
    msp_ood_scores = get_score(ood_logits, "msp")

    energy_ind_scores = get_score(ind_logits, "energy")
    energy_ood_scores = get_score(ood_logits, "energy")

    metrics = cal_metric(ind_scores + msp_ind_scores, ood_scores + msp_ood_scores)
    print(f"Scores+MSP FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Max confidence: Moments + MSP", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])

    metrics = cal_metric(ind_scores + energy_ind_scores, ood_scores + energy_ood_scores)
    print(f"Scores+Energy FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Max confidence: Moments + Energy", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])

    metrics = cal_metric(ind_scores + energy_ind_scores + msp_ind_scores, ood_scores + energy_ood_scores + msp_ood_scores)
    print(f"Scores+Energy+FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Max confidence: Moments + Energy + MSP", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])

    best_thresholds["max-confidence:Moments"] = [best_threshold]

    print("-------------------------------")
    print("Entropy_detection MLE")

    best_f1 = None
    best_threshold = None

    for threshold in (
        thresholds["entropy:MLE"]
        if "entropy:MLE" in thresholds
        else [0.5, 1, 2, 5, 10, 20]
    ):
        
        ind_scores = -entropies_ind
        ood_scores = -entropies_ood

        entropy_threshold = (
            np.mean(entropies_ind[~np.isnan(entropies_ind)]) * threshold
        )  # Scale threshold by max entropy
        fraction_of_ind_as_ood = np.sum(
            np.array(entropies_ind) > entropy_threshold
        ) / len(entropies_ind)
        fraction_of_ood_as_ood = np.sum(
            np.array(entropies_ood) > entropy_threshold
        ) / len(entropies_ood)

        f1_score = (
            2
            * (fraction_of_ood_as_ood * (1 - fraction_of_ind_as_ood))
            / (fraction_of_ood_as_ood + (1 - fraction_of_ind_as_ood) + 1e-8)
        )

        if best_f1 is None or f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold

        print(
            f"Threshold: {threshold}x mean IND entropy - Fraction of IND classified as OOD: {fraction_of_ind_as_ood:.3f}, Fraction of OOD classified as OOD: {fraction_of_ood_as_ood:.3f}, F1 Score: {f1_score:.3f}"
        )

    metrics = cal_metric(ind_scores, ood_scores)
    print(f"Negative Entropy FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Entropy: MLE", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])
    
    metrics = cal_metric(-ind_scores, -ood_scores)
    print(f"Positive Entropy FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Entropy: MLE (positive)", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])

    best_thresholds["entropy:MLE"] = [best_threshold]

    print("-------------------------------")
    print("Entropy_detection Moments")

    best_f1 = None
    best_threshold = None

    for threshold in (
        thresholds["entropy:Moments"]
        if "entropy:Moments" in thresholds
        else [0.5, 1, 2, 5, 10, 20]
    ):
        
        ind_scores = -entropies_moments_ind
        ood_scores = -entropies_moments_ood

        entropy_threshold = (
            np.mean(entropies_moments_ind[~np.isnan(entropies_moments_ind)]) * threshold
        )  # Scale threshold by max entropy
        fraction_of_ind_as_ood = np.sum(
            np.array(entropies_moments_ind) > entropy_threshold
        ) / len(entropies_moments_ind)
        fraction_of_ood_as_ood = np.sum(
            np.array(entropies_moments_ood) > entropy_threshold
        ) / len(entropies_moments_ood)

        f1_score = (
            2
            * (fraction_of_ood_as_ood * (1 - fraction_of_ind_as_ood))
            / (fraction_of_ood_as_ood + (1 - fraction_of_ind_as_ood) + 1e-8)
        )

        if best_f1 is None or f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold

        print(
            f"Threshold: {threshold}x mean IND entropy - Fraction of IND classified as OOD: {fraction_of_ind_as_ood:.3f}, Fraction of OOD classified as OOD: {fraction_of_ood_as_ood:.3f}, F1 Score: {f1_score:.3f}"
        )

    metrics = cal_metric(ind_scores, ood_scores)
    print(f"Negative Entropy FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Entropy: Moments", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])
    metrics = cal_metric(-ind_scores, -ood_scores)
    print(f"Positive Entropy FPR-95: {100.0 * metrics['FPR']:.2f}, FPR-80: {100.0 * metrics['FPR-80']:.2f}, AUROC: {100.0 * metrics['AUROC']:.2f}, AUIN: {100.0 * metrics['AUIN']:.2f}")
    table.append(["Entropy: Moments (positive)", f"{100.0 * metrics['FPR']:.2f}", f"{100.0 * metrics['FPR-80']:.2f}", f"{100.0 * metrics['AUROC']:.2f}"])

    best_thresholds["entropy:Moments"] = [best_threshold]
    print("-------------------------------")

    return best_thresholds, table


def ood_ntu60to120(
    eval_state_path="./ood_experiments/eval_state.pkl",
    load_dirichlets=True,
    calibration_dataset_size=0.1,
    is_variational_model=True,
):

    with open(eval_state_path, "rb") as f:
        eval_state = pickle.load(f)

    categorical_mean = eval_state["categorical_mean"]
    categorical_var = eval_state["categorical_var"]
    categorical_all = eval_state["categorical_all"]
    lbls_val = eval_state["lbls_val"]
    logits = eval_state["score"]
    logit_sets = eval_state["logits_sets"]

    if is_variational_model:
        logit_vars = np.var(logit_sets, axis=1)


    if is_variational_model:

        dirichelts_file_name = f"./ood_experiments/dirichlets.pkl"

        if os.path.exists(dirichelts_file_name) and load_dirichlets:
            with open(dirichelts_file_name, "rb") as f:
                dirichlets_save_object = pickle.load(f)
                dirichlets_moments = dirichlets_save_object["dirichlets_moments"]
                dirichlets_mle = dirichlets_save_object["dirichlets_mle"]
                print("Loaded estimated Dirichlet parameters from file")
        else:
            dirichlets_moments = [
                estimate_dirichlet_params(categorical_mean[i], categorical_var[i])
                for i in range(len(categorical_mean))
            ]
            dirichlets_mle = estimate_mle_dirichlet_batched(categorical_all)

            print("Estimated Dirichlet parameters")

            dirichlets_save_object = {
                "dirichlets_moments": dirichlets_moments,
                "dirichlets_mle": dirichlets_mle,
            }

        with open(dirichelts_file_name, "wb") as f:
            pickle.dump(dirichlets_save_object, f)

    ind_mask = lbls_val < 60

    if is_variational_model:

        dirichlets_ind = np.array(
            [dirichlets_mle[i] for i in range(len(categorical_mean)) if ind_mask[i]]
        )
        dirichlets_ood = np.array(
            [dirichlets_mle[i] for i in range(len(categorical_mean)) if not ind_mask[i]]
        )

        dirichlets_moments_ind = np.array(
            [dirichlets_moments[i] for i in range(len(categorical_mean)) if ind_mask[i]]
        )
        dirichlets_moments_ood = np.array(
            [dirichlets_moments[i] for i in range(len(categorical_mean)) if not ind_mask[i]]
        )

        entropies_ind = np.array([dirichlet_entropy(a) for a in dirichlets_ind])
        entropies_ood = np.array([dirichlet_entropy(a) for a in dirichlets_ood])

        entropies_moments_ind = np.array(
            [dirichlet_entropy(a) for a in dirichlets_moments_ind if not np.isnan(a).any()]
        )
        entropies_moments_ood = np.array(
            [dirichlet_entropy(a) for a in dirichlets_moments_ood if not np.isnan(a).any()]
        )

        ind_variances = categorical_var[ind_mask]
        ood_variances = categorical_var[~ind_mask]

        max_categories = np.argmax(categorical_mean, axis=-1)
        ind_max_variances = categorical_var[ind_mask, max_categories[ind_mask]]
        ood_max_variances = categorical_var[~ind_mask, max_categories[~ind_mask]]
        
        calibration_set_mask_ind = np.array(
            [
                i < len(dirichlets_ind) * calibration_dataset_size
                for i in range(len(dirichlets_ind))
            ]
        )
        calibration_set_mask_ood = np.array(
            [
                i < len(dirichlets_ood) * calibration_dataset_size
                for i in range(len(dirichlets_ood))
            ]
        )
        
        ind_logits_vars = logit_vars[ind_mask]
        ood_logits_vars = logit_vars[~ind_mask]


    ind_logits = logits[ind_mask]
    ood_logits = logits[~ind_mask]


    _, table = detect_traditional_ood(
        ind_logits,
        ood_logits,
    )

    if is_variational_model:

        _, table = detect_traditional_var_ood(
            ind_logits,
            ood_logits,
            ind_logits_vars,
            ood_logits_vars,
            table=table,
        )

        _, table = detect_ood(
            ind_variances,
            ood_variances,
            dirichlets_ind,
            dirichlets_ood,
            dirichlets_moments_ind,
            dirichlets_moments_ood,
            entropies_ind,
            entropies_ood,
            entropies_moments_ind,
            entropies_moments_ood,
            ind_max_variances,
            ood_max_variances,
            ind_logits,
            ood_logits,
            ind_logits_vars,
            ood_logits_vars,
            table=table,
        )


        # best_thresholds = detect_ood(
        #     ind_variances[calibration_set_mask_ind],
        #     ood_variances[calibration_set_mask_ood],
        #     dirichlets_ind[calibration_set_mask_ind],
        #     dirichlets_ood[calibration_set_mask_ood],
        #     dirichlets_moments_ind[calibration_set_mask_ind],
        #     dirichlets_moments_ood[calibration_set_mask_ood],
        #     entropies_ind[calibration_set_mask_ind],
        #     entropies_ood[calibration_set_mask_ood],
        #     entropies_moments_ind[calibration_set_mask_ind],
        #     entropies_moments_ood[calibration_set_mask_ood],
        #     ind_max_variances[calibration_set_mask_ind],
        #     ood_max_variances[calibration_set_mask_ood],
        #     ind_logits_vars[calibration_set_mask_ind],
        #     ood_logits_vars[calibration_set_mask_ood],
        # )

        # print()
        # print()
        # print("Evaluating on test set with best thresholds...")

        # detect_ood(
        #     ind_variances[calibration_set_mask_ind],
        #     ood_variances[~calibration_set_mask_ood],
        #     dirichlets_ind[calibration_set_mask_ind],
        #     dirichlets_ood[~calibration_set_mask_ood],
        #     dirichlets_moments_ind[calibration_set_mask_ind],
        #     dirichlets_moments_ood[~calibration_set_mask_ood],
        #     entropies_ind[calibration_set_mask_ind],
        #     entropies_ood[~calibration_set_mask_ood],
        #     entropies_moments_ind[calibration_set_mask_ind],
        #     entropies_moments_ood[~calibration_set_mask_ood],
        #     ind_max_variances[calibration_set_mask_ind],
        #     ood_max_variances[~calibration_set_mask_ood],
        #     ind_logits_vars[calibration_set_mask_ind],
        #     ood_logits_vars[~calibration_set_mask_ood],
        #     thresholds=best_thresholds,
        # )

        # print("Calibrating on the full ind dataset and partial ood dataset...")

        # calibration_set_mask_ind = np.array(
        #     [i < len(dirichlets_ind) * 1 for i in range(len(dirichlets_ind))]
        # )
        # calibration_set_mask_ood = np.array(
        #     [
        #         i < len(dirichlets_ood) * calibration_dataset_size
        #         for i in range(len(dirichlets_ood))
        #     ]
        # )

        # best_thresholds = detect_ood(
        #     ind_variances[calibration_set_mask_ind],
        #     ood_variances[calibration_set_mask_ood],
        #     dirichlets_ind[calibration_set_mask_ind],
        #     dirichlets_ood[calibration_set_mask_ood],
        #     dirichlets_moments_ind[calibration_set_mask_ind],
        #     dirichlets_moments_ood[calibration_set_mask_ood],
        #     entropies_ind[calibration_set_mask_ind],
        #     entropies_ood[calibration_set_mask_ood],
        #     entropies_moments_ind[calibration_set_mask_ind],
        #     entropies_moments_ood[calibration_set_mask_ood],
        #     ind_max_variances[calibration_set_mask_ind],
        #     ood_max_variances[calibration_set_mask_ood],
        #     ind_logits_vars[calibration_set_mask_ind],
        #     ood_logits_vars[calibration_set_mask_ood],
        # )

        # print()
        # print()
        # print("Evaluating on test set with best thresholds...")

        # detect_ood(
        #     ind_variances[calibration_set_mask_ind],
        #     ood_variances[~calibration_set_mask_ood],
        #     dirichlets_ind[calibration_set_mask_ind],
        #     dirichlets_ood[~calibration_set_mask_ood],
        #     dirichlets_moments_ind[calibration_set_mask_ind],
        #     dirichlets_moments_ood[~calibration_set_mask_ood],
        #     entropies_ind[calibration_set_mask_ind],
        #     entropies_ood[~calibration_set_mask_ood],
        #     entropies_moments_ind[calibration_set_mask_ind],
        #     entropies_moments_ood[~calibration_set_mask_ood],
        #     ind_max_variances[calibration_set_mask_ind],
        #     ood_max_variances[~calibration_set_mask_ood],
        #     ind_logits_vars[calibration_set_mask_ind],
        #     ood_logits_vars[~calibration_set_mask_ood],
        #     thresholds=best_thresholds,
        # )

    tab = tabulate(table)
    print(tab)

    with open(f"./ood_experiments/metrics_table{'' if is_variational_model else '_baseline'}.txt", "w") as f:
        f.write(tab)

    if is_variational_model:

        print("Plotting Dirichlet parameters...")

        plot_dirichlet_list(dirichlets_ind[:6], name="dirichlets_ind")
        plot_dirichlet_list(dirichlets_ood[:6], name="dirichlets_ood")
        plot_dirichlet_list(dirichlets_moments_ind[:6], name="dirichlets_moments_ind")
        plot_dirichlet_list(dirichlets_moments_ood[:6], name="dirichlets_moments_ood")

        print()


if __name__ == "__main__":
    Fire(ood_ntu60to120)
