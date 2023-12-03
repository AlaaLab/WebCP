import numpy as np
from conformal_prediction_methods import LOSS_FUNCTION
'''
Coverage and Efficiency Metrics:
Note that there are two universal arguments:
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
'''

'''
Computes the overall coverage (total true positive / total number of expected labels).
Inputs:
    - prediction_set_arr:   see above
    - true_class_arr:       see above
Output:
    the overall coverage across all classes and samples, as a proportion in [0, 1].
'''
def overall_coverage(conformal_set_arr: np.array, true_class_arr: np.array):
    return np.sum(np.logical_and(conformal_set_arr, true_class_arr)) / np.sum(true_class_arr)


'''
Computes the coverage for each class.
Inputs:
    - prediction_set_arr:   see above
    - true_class_arr:       see above
Output:
    an array of length (true_class_arr), containing the coverage score for
    each class.
'''
def class_stratified_coverage(conformal_set_arr: np.array, true_class_arr: np.array):
    return np.sum(np.logical_and(conformal_set_arr, true_class_arr), axis=0) / np.sum(true_class_arr, axis=0)


'''
Computes the coverage, stratified across the size of the **expected set** of true labels.
Inputs:
    - prediction_set_arr:   see above
    - true_class_arr:       see above
Output: a variable number of bins of the form (set_size, num_samples_in_bin, mean_coverage),
            represented as 3 arrays of the same variable length:
    - an array containing in increasing order the size of the true label sets represented 
        by each bin.
    - an array containing the number of samples in each bin.
    - an array containing the mean sample-wise coverage in each bin.
'''
def size_stratified_coverage(conformal_set_arr: np.array, true_class_arr: np.array):
    size_arr = np.sum(true_class_arr, axis=1)

    unique, unique_counts = np.unique(size_arr, return_counts=True)
    idx_arr = np.argsort(size_arr)
    sorted_conformal_set_arr = conformal_set_arr[idx_arr]
    sorted_true_class_arr = true_class_arr[idx_arr]

    split_idx = np.cumsum(unique_counts)[:-1]

    split_conformal_set_list = np.split(sorted_conformal_set_arr, split_idx)
    split_true_class_list = np.split(sorted_true_class_arr, split_idx)

    final_losses = np.array([1-LOSS_FUNCTION(this_conf_set, this_true_set) for (this_conf_set, this_true_set) in zip(split_conformal_set_list, split_true_class_list)])

    return unique, final_losses, unique_counts

'''
For each sample, returns the number of extraneous classes in the prediction set relative to the expected set of labels.
Inputs:
    - prediction_set_arr:   see above
    - true_class_arr:       see above
Output:
    an array of length (true_class_arr), containing the coverage score for
    each class.
'''
def samplewise_efficiency(conformal_set_arr: np.array, true_class_arr: np.array):
    return np.sum(conformal_set_arr, axis=1) - np.sum(np.logical_and(conformal_set_arr, true_class_arr), axis=1)

def false_positive_rate(conformal_set_arr: np.array, true_class_arr: np.array):
    false_positives = np.sum(conformal_set_arr - true_class_arr == 1, axis = 1)
    #return np.mean(np.nan_to_num(false_positives/np.sum(conformal_set_arr, axis = 1)))
    return np.mean(np.nan_to_num(false_positives/np.sum(true_class_arr == 0, axis = 1)))

def true_positive_rate(conformal_set_arr: np.array, true_class_arr: np.array):
    true_positives = np.sum(np.logical_and(conformal_set_arr, true_class_arr), axis = 1)
    return np.mean(np.nan_to_num(true_positives/np.sum(true_class_arr, axis = 1)))

def size_stratified_efficiency(conformal_set_arr: np.array, true_class_arr: np.array):
    size_arr = np.sum(true_class_arr, axis=1)

    unique, unique_counts = np.unique(size_arr, return_counts=True)
    idx_arr = np.argsort(size_arr)
    sorted_conformal_set_arr = conformal_set_arr[idx_arr]
    sorted_true_class_arr = true_class_arr[idx_arr]

    split_idx = np.cumsum(unique_counts)[:-1]

    split_conformal_set_list = np.split(sorted_conformal_set_arr, split_idx)
    split_true_class_list = np.split(sorted_true_class_arr, split_idx)

    final_efficiencies = np.array([np.mean(samplewise_efficiency(this_conf_set, this_true_set)) for (this_conf_set, this_true_set) in zip(split_conformal_set_list, split_true_class_list)])

    return unique, final_efficiencies, unique_counts
'''
Returns a tuple of arrays, .
Inputs:
    - prediction_set_arr:   see above
Output:
    a tuple of two arrays of the same length, each of which respectively
    contain the set size and the number of samples with a conformal set
    having the said size.
'''
def prediction_set_size(conformal_set_arr: np.array):
    set_size_arr = np.sum(conformal_set_arr, axis=1)
    return np.unique(set_size_arr, return_counts=True)