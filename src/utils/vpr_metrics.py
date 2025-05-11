# Heavily adapted from: https://github.com/stschubert/VPR_Tutorial/blob/main/evaluation/metrics.py

import numpy as np
from sklearn.metrics import auc


def createPR(S_in, GThard, n_thresh=100):
    assert S_in.shape == GThard.shape, "S_in and GThard must have the same shape"
    assert S_in.ndim == 2, "S_in, GThard and GTsoft must be two-dimensional"
    assert n_thresh > 1, "n_thresh must be >1"

    # ensure logical datatype in GT and GTsoft
    GT = GThard.astype("bool")

    # copy S and set elements that are only true in GTsoft to min(S) to ignore them during evaluation
    S = S_in.copy()

    # count the number of ground-truth positives (GTP)
    GTP = np.count_nonzero(GT.any(0))

    if GTP == 0:
        return [1], [0], 0.0

    # GT-values for best match per query (i.e., per column)
    GT = GT[np.argmax(S, axis=0), np.arange(GT.shape[1])]

    # similarities for best match per query (i.e., per column)
    S = np.max(S, axis=0)

    # init precision and recall vectors
    R = [
        0,
    ]
    P = [
        1,
    ]

    # select start and end treshold
    startV = S.max()  # start-value for treshold
    endV = S.min()  # end-value for treshold

    # iterate over different thresholds
    for i in np.linspace(startV, endV, n_thresh):
        B = S >= i  # apply threshold

        TP = np.count_nonzero(GT & B)  # true positives
        FP = np.count_nonzero((~GT) & B)  # false positives

        total_predicted_positives = TP + FP
        if total_predicted_positives == 0:
            P.append(0.0)
        else:
            P.append(TP / (TP + FP))  # precision

        R.append(TP / GTP)  # recall

    # calculate AUPRC
    AUPRC = np.trapz(P, R)
    return P, R, AUPRC


def recallAt100precision(S_in, GThard, n_thresh=100):
    assert S_in.shape == GThard.shape, "S_in and GThard must have the same shape"

    assert S_in.ndim == 2, "S_in, GThard and GTsoft must be two-dimensional"

    assert n_thresh > 1, "n_thresh must be >1"

    # get precision-recall curve
    P, R, _ = createPR(S_in, GThard, n_thresh=n_thresh)
    P = np.array(P)
    R = np.array(R)

    # recall values at 100% precision
    R = R[P == 1]

    # maximum recall at 100% precision
    R = R.max()

    return R


def recallAtK(S, GT, K=1):
    assert S.shape == GT.shape, "S and GT must have the same shape"
    assert S.ndim == 2, "S and GT must be two-dimensional"
    assert K >= 1, "K must be >=1"

    # ensure logical datatype in GT
    GT = GT.astype("bool")

    # discard all query images without an actually matching database image
    j = GT.sum(0) > 0  # columns with matches
    S = S[:, j]  # select columns with a match
    GT = GT[:, j]  # select columns with a match

    # select K highest similarities
    i = S.argsort(0)[-K:, :]
    j = np.tile(np.arange(i.shape[1]), [K, 1])
    GT = GT[i, j]

    # recall@K
    RatK = np.sum(GT.sum(0) > 0) / GT.shape[1]

    return RatK


def meanReciprocalRank(S, GT):
    assert S.shape == GT.shape, "S and GT must have the same shape"
    GT = GT.astype("bool")

    ranks = np.argsort(-S, axis=0)
    mrr_values = []

    for col in range(GT.shape[1]):
        # Find first match position
        match_indices = np.where(GT[ranks[:, col], col])[0]
        if match_indices.size > 0:
            mrr_values.append(1.0 / (match_indices[0] + 1))

    return np.mean(mrr_values) if mrr_values else 0.0
