import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import precision_score, recall_score
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.categorical import Categorical 
# BEGIN CONFORMAL RISK CONTROL
'''
For each sample, returns ratio of number of false negatives to the number of expected true labels for that sample.
Inputs:
    - prediction_set_arr:   a boolean array representing the coverage sets 
                                predicted for each image.
                            dim: (num_examples, num_classes)
                            element at (example_i, class_j) represents 
                                whether class_j is in the prediction set of
                                example_i.
    - true_class_arr:       a boolean array representing the true classes
                                for each image.
                            dim: (num_examples, num_classes)
                            element at (example_i, class_j) represents
                                whether class_j is in the prediction set of
                                example_i.
Output:
    an array of length (true_class_arr), containing the loss for each sample
'''
def samplewise_loss(conformal_set_arr: np.array, true_class_arr: np.array):
    return 1 - np.sum(np.logical_and(conformal_set_arr, true_class_arr), axis=1) / np.sum(true_class_arr, axis=1)

'''
Set the Loss Function for Conformal Risk Control here.
'''
LOSS_FUNCTION = lambda pred_arr, true_arr: np.nanmean(samplewise_loss(pred_arr, true_arr))

'''
Compute the optimal threshold for classification such that the conformal risk control guarantee would still be satisfied:
    E(LOSS using threshold) <= alpha
    threshold := inf{E(LOSS using threshold) <= (n+1)/n*alpha - 1/(n+1)}
Inputs:
    - alpha:                the confidence used to compute the risk control
                                threshold (e.g. 0.05)
    - probability_arr:      an array representing the predictions per class
                                for each sample.
                            dim: (num_examples, num_classes)
                            element at (example_i, class_j) represents
                                P(class_j | example_i)
    - true_class_arr:       a boolean array representing the true classes
                                for each image.
                            dim: (num_examples, num_classes)
                            element at (example_i, class_j) represents
                                whether class_j is in the prediction set of
                                example_i.
Ouptut:
    the optimal (i.e. lowest) threshold that satisfies the conformal risk control
        constraints.
Cite: https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/multilabel-classification-mscoco.ipynb
    '''
def compute_threshold(alpha, probability_arr: np.array, true_class_arr: np.array):
    exp_loss = lambda threshold: LOSS_FUNCTION(probability_arr >= threshold, true_class_arr)
    N = np.shape(probability_arr)[0]
    return brentq(lambda trial_thresh: exp_loss(trial_thresh) - ((N+1.)/N*alpha - 1/(N+1)), 0, 1)

'''
Compute the conformal prediction sets for each sample, given their scores and a threshold.
Inputs: 
    - threshold:            the threshold above which classification scores
                                are considered positive, and below which 
                                scores are considered negative.
    - probability_arr:      an array representing the predictions per class
                                for each sample.
                            dim: (num_examples, num_classes)
                            element at (example_i, class_j) represents
                                P(class_j | example_i)
Ouptut:
    a boolean array representing the coverage sets predicted for each image.
        dim: (num_examples, num_classes
        element at (example_i, class_j) represents 
            whether class_j is in the prediction set of
            example_i.
'''
def compute_prediction_sets_threshold(probability_arr, threshold):
    return probability_arr >= threshold

'''
Compute the conformal prediction sets for each sample, given their scores and a fixed target set size.
Inputs: 
    - threshold:            the threshold above which classification scores
                                are considered positive, and below which 
                                scores are considered negative.
    - probability_arr:      an array representing the predictions per class
                                for each sample.
                            dim: (num_examples, num_classes)
                            element at (example_i, class_j) represents
                                P(class_j | example_i)
Ouptut:
    a boolean array representing the coverage sets predicted for each image.
        dim: (num_examples, num_classes
        element at (example_i, class_j) represents 
            whether class_j is in the prediction set of
            example_i.
'''
# END CONFORMAL RISK CONTROL

# BEGIN FIXED SET SIZE EXPERIMENTS
def compute_prediction_sets_fixed_size(probability_arr, size):
    indices = np.argsort(probability_arr, axis=1)[:, ::-1][:, 0:size]
    res = np.zeros(probability_arr.shape)
    np.put_along_axis(res, indices, 1, axis=1)
    return res
# END FIXED SET SIZE EXPERIMENTS

# BEGIN SET SIZE CALIBRATION / INFERENCING
def set_size_inference_prec_recall(conformal_set_arr: np.array, true_class_arr: np.array, method='global'):
    pred_set_sizes = np.sum(conformal_set_arr, axis=1, keepdims=True)
    all_set_sizes = np.unique(pred_set_sizes)

    prec_list = []
    recall_list = []
    for set_size in all_set_sizes:
        this_pred_arr = conformal_set_arr * (pred_set_sizes >= set_size)
        
        prec_list.append(precision_score(true_class_arr, this_pred_arr, average=('binary' if method=='global' else 'samples')))
        recall_list.append(recall_score(true_class_arr, this_pred_arr, average=('binary' if method=='global' else 'samples')))
    
    return np.array(prec_list), np.array(recall_list)

def set_size_inference_tpr_fpr(conformal_set_arr: np.array, true_class_arr: np.array, method='global'):
    pred_set_sizes = np.sum(conformal_set_arr, axis=1, keepdims=True)
    all_set_sizes = np.unique(pred_set_sizes)

    tpr_list = []
    fpr_list = []
    for set_size in all_set_sizes:
        this_pred_arr = conformal_set_arr * (pred_set_sizes >= set_size)

        agree = this_pred_arr == true_class_arr
        disagree = this_pred_arr != true_class_arr

        tp = agree[this_pred_arr == 1]
        tn = agree[this_pred_arr == 0]
        fp = disagree[this_pred_arr == 1]
        fn = disagree[this_pred_arr == 0]
        
        tp_sum = np.sum(tp, axis=(None if method=='global' else 1))
        tn_sum = np.sum(tn, axis=(None if method=='global' else 1))
        fp_sum = np.sum(fp, axis=(None if method=='global' else 1))
        fn_sum = np.sum(fn, axis=(None if method=='global' else 1))

        tpr_list.append(np.mean(tp_sum / (tp_sum + fn_sum)))
        fpr_list.append(np.mean(fp_sum / (fp_sum + tn_sum)))

    return np.array(tpr_list), np.array(fpr_list)
# END SET SIZE CALIBRATION / INFERENCING

# AMBIGUOUS CONFORMAL PREDICTION
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.categorical import Categorical 

def monte_carlo_cp_deprecated(predictions: torch.Tensor, plausibilities: torch.Tensor, alpha: float, sample_num: int):
    assert predictions.size(dim=0) == predictions.size(dim=0) and predictions.size(dim=0) == plausibilities.size(dim=0)

    assert predictions.size(dim=1) == predictions.size(dim=1) and predictions.size(dim=1) == plausibilities.size(dim=1) - 1

    distro = OneHotCategorical(probs=plausibilities)

    results = distro.sample(sample_shape=torch.Size([sample_num]))

    cp_actual = []
    cp_pred = []
    for i in range(sample_num):
        for j in range(predictions.size(dim=0)):
            if (results[i][j][plausibilities.size(dim=1) - 1] == 1):
                continue

            cp_actual.append(results[i][j][0:plausibilities.size(dim=1) - 1])
            cp_pred.append(predictions[j])

    #this is from https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-smallest-sets.ipynb
    cal_scores = 1 - torch.sum(torch.stack(cp_actual, 0) * torch.stack(cp_pred, 0), axis=1)

    q_level = np.ceil((predictions.size(dim=0)+1) * (1-alpha))/predictions.size(dim=0)
    qhat = np.quantile(cal_scores.numpy(), q_level, interpolation='higher')
    return 1-qhat

def monte_carlo_cp(predictions: torch.Tensor, plausibilities: torch.Tensor, alpha: float, sample_num: int):
    assert predictions.size(dim=0) == plausibilities.size(dim=0)

    assert predictions.size(dim=1) == plausibilities.size(dim=1) - 1

    distro = OneHotCategorical(probs=plausibilities)

    results = distro.sample(sample_shape=torch.Size([sample_num]))

    cp_actual = []
    cp_pred = []
    for i in range(sample_num):
        this_actual = []
        this_pred = []
        for j in range(predictions.size(dim=0)):
            if (results[i][j][plausibilities.size(dim=1) - 1] == 1):
                continue

            this_actual.append(results[i][j][0:plausibilities.size(dim=1) - 1])
            this_pred.append(predictions[j])
        cp_actual.append(torch.stack(this_actual, dim=0) if len(this_actual) > 0 else torch.tensor([]))
        cp_pred.append(torch.stack(this_pred, dim=0) if len(this_pred) > 0 else torch.tensor([]))
    return compute_threshold_amb(cp_actual, cp_pred, alpha)
def compute_threshold_amb(actual: list, pred: list, alpha):
    assert len(actual) == len(pred)
    m = len(actual)

    #this is from https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-smallest-sets.ipynb
    cal_scores = [(1 - torch.sum(this_actual * this_pred, axis=1) if this_actual.dim() > 1 else torch.tensor([])) for this_actual, this_pred in zip(actual, pred)]

    exp_loss = lambda threshold: 1/m * sum([(torch.sum(this_cal_score <= threshold) + 1) / (this_cal_score.size(dim=0) + 1) for this_cal_score in cal_scores])
    
    N = sum([this_cal_score.size(dim=0) for this_cal_score in cal_scores])
    return  1 - brentq(lambda trial_thresh: exp_loss(trial_thresh) - ((N+1.)/N*(1-alpha) - 1/(N+1)), 0, 1)
