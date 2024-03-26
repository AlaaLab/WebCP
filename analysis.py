import sys
import os

from pathlib import Path
import pandas as pd
import scipy as sp
import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
import pickle
import json
import argparse

script_path = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
base_path = script_path.parent.absolute()
sys.path.append(base_path / 'cp')
sys.path.append(base_path / 'utils')
from cp.conformal_prediction_methods import *
from cp.metrics import *

# Methods
def performance_report(threshold, calib_sim_score_arr, test_sim_score_arr, calib_true_class_arr, test_true_class_arr):
    # Get prediction sets
    calib_prediction_set = compute_prediction_sets_threshold(calib_sim_score_arr, threshold)
    test_prediction_set = compute_prediction_sets_threshold(test_sim_score_arr, threshold)
    # Compute performance metrics
    calib_coverage = overall_coverage(calib_prediction_set, calib_true_class_arr)
    test_coverage = overall_coverage(test_prediction_set, test_true_class_arr)
    calib_samplewise_efficiency = samplewise_efficiency(calib_prediction_set, calib_true_class_arr)
    test_samplewise_efficiency = samplewise_efficiency(test_prediction_set, test_true_class_arr)
    # Output Performance Metrics
    print(f"OVERALL COVERAGE (proportion of true labels covered):")
    print(f"Calibration Set: {calib_coverage}")
    print(f"Test Set: {test_coverage}")
    print(f'OVERALL EFFICIENCY (mean num of extraneous classes per sample): ')
    print(f"Calibration Set: {np.mean(calib_samplewise_efficiency)}")
    print(f"Test Set: {np.mean(test_samplewise_efficiency)}")
    return (calib_coverage, np.mean(calib_samplewise_efficiency), test_coverage, np.mean(test_samplewise_efficiency))

#Parse Arguments
'''parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, help='Experiment in experiment_configs to run')
parser.add_argument('--out', type=str, help='Where to output charts')
args = parser.parse_args()

# Parameters
reader = open(base_path + "\\experiment_configs\\"  + args.exp)
config = json.load(reader)'''

dataset = 'caltech256'
source = 'google'
version = '2'

if True:
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-pets_flickr_1")
if False:
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-pets_01-06-24_1")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-pets_01-06-24_owlvit")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-pets_01-06-24_flava")
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-pets_01-06-24_clipa")
if False:
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-fitz17k_01-06-24_1")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-fitz17k_01-06-24_owlvit")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-fitz17k_01-06-24_flava")
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-fitz17k_01-06-24_clipa")
if False:
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-medmnist_01-14-24_1")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-medmnist_01-06-24_owlvit")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-medmnist_01-06-24_flava")
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-medmnist_01-06-24_clipa")
if False:
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-imagenet_02-20-24_1")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-imagenet_01-15-24_owlvit")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-imagenet_01-15-24_flava")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-imagenet_01-15-24_clipa")
if False:
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-caltech256_01-17-24_1")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-caltech256_01-17-24_owlvit")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experimenpythts\\google-caltech256_01-17-24_flava")
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-caltech256_01-17-24_clipa")

folder_name = source + "_" + dataset + "_" + version
RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\" + folder_name)
OUTPUT_RESULT_DIR = RESULTS_DIRECTORY

CALIB_SIZE_CURVE = False
ALPHA_CURVE = False
UNCERTAIN_HIST = False
PLAUSIBILITY_HISTOGRAM = False
ORACLE = True
ALPHA = 0.5
NUM_SAMPLES = 1000
LOGIT_SCALE = 100.0

# Load Files
calib_plausibility_score_arr = torch.load(RESULTS_DIRECTORY / "calib_plausibility_score_arr")
calib_sim_score_arr = torch.load(RESULTS_DIRECTORY / "calib_sim_score_arr")
calib_true_class_arr = torch.load(RESULTS_DIRECTORY / "calib_true_class_arr")
test_sim_score_arr = torch.load(RESULTS_DIRECTORY / "test_sim_score_arr")
test_true_class_arr = torch.load(RESULTS_DIRECTORY / "test_true_class_arr")
n_calib = calib_sim_score_arr.shape[0]
n_test = test_sim_score_arr.shape[0]
m = test_true_class_arr.shape[1]

# Amb CP on data-mined vs. Normal CP on original dataset
if ORACLE:
    ratio = 0.5
    print("Begin Alpha Curve")
    # Initialize metrics lists and set size list
    oracle_metrics = []
    norm_metrics = []
    amb_metrics = []
    alpha_values = [0.05*i for i in range(1, 5)] + [0.1*i for i in range(3, 10)]# + [0.95]
    print(alpha_values)
    # Generate numpy matrices
    calib_sim_score_arr_np = calib_sim_score_arr.detach().cpu().numpy()
    calib_true_class_arr_np = calib_true_class_arr.detach().cpu().numpy()
    test_sim_score_arr_np = test_sim_score_arr.detach().cpu().numpy()
    test_true_class_arr_np = test_true_class_arr.detach().cpu().numpy()
    # Shuffle values
    random_order = np.random.permutation(len(test_sim_score_arr_np))
    test_sim_score_arr_np = test_sim_score_arr_np[random_order]
    test_true_class_arr_np = test_true_class_arr_np[random_order]
    random_order2 = np.random.permutation(len(calib_sim_score_arr_np))
    calib_sim_score_arr_np = calib_sim_score_arr_np[random_order2]
    calib_sim_score_arr = calib_sim_score_arr[random_order2]
    calib_true_class_arr_np = calib_true_class_arr_np[random_order2]
    calib_plausibility_score_arr = calib_plausibility_score_arr[random_order2]
    calib_length = min(int(ratio * len(test_sim_score_arr_np)), len(calib_sim_score_arr))
    # Loop through possible per class set sizes
    for alpha in alpha_values:
        #Perform Conformal Prediction
        print("Performing Conformal Prediction")
        threshold_amb = monte_carlo_cp_eff(calib_sim_score_arr[:calib_length], calib_plausibility_score_arr[:calib_length], alpha, NUM_SAMPLES)
        threshold_oracle = compute_threshold(alpha, test_sim_score_arr_np[:calib_length], test_true_class_arr_np[:calib_length])
        threshold_norm = compute_threshold(alpha, calib_sim_score_arr_np[:calib_length], calib_true_class_arr_np[:calib_length])
        print(threshold_oracle)
        print(threshold_norm)
        print(threshold_amb)
        #Output Metrics
        print("\nAlpha Value: {alpha}".format(alpha = alpha))
        print("Normal CP:")
        norm_metrics.append(performance_report(threshold_norm, calib_sim_score_arr_np[:calib_length], test_sim_score_arr_np[calib_length:], \
                                               calib_true_class_arr_np[:calib_length], test_true_class_arr_np[calib_length:]))
        print("Oracle Normal CP:")
        oracle_metrics.append(performance_report(threshold_oracle, calib_sim_score_arr_np[:calib_length], test_sim_score_arr_np[calib_length:], \
                                               calib_true_class_arr_np[:calib_length], test_true_class_arr_np[calib_length:]))
        print("Ambiguous CP:")
        amb_metrics.append(performance_report(threshold_amb, calib_sim_score_arr_np[:calib_length], test_sim_score_arr_np[calib_length:], \
                                              calib_true_class_arr_np[:calib_length], test_true_class_arr_np[calib_length:]))
    # Generate deltas
    delta_norm = [norm_metrics[i][2]+alpha_values[i]-1 for i in range(0, len(norm_metrics))]
    delta_amb = [amb_metrics[i][2]+alpha_values[i]-1 for i in range(0, len(amb_metrics))]
    delta_oracle = [oracle_metrics[i][2]+alpha_values[i]-1 for i in range(0, len(oracle_metrics))]
    norm = [norm_metrics[i][2] for i in range(0, len(norm_metrics))]
    amb = [amb_metrics[i][2] for i in range(0, len(amb_metrics))]
    oracle = [oracle_metrics[i][2] for i in range(0, len(oracle_metrics))]
    target = [1-alpha for alpha in alpha_values]
    eff_norm = [norm_metrics[i][3] for i in range(0, len(norm_metrics))]
    eff_amb = [amb_metrics[i][3] for i in range(0, len(amb_metrics))]
    eff_oracle = [oracle_metrics[i][3] for i in range(0, len(oracle_metrics))]
    raw_data = {"alpha_values": alpha_values, "delta_norm": delta_norm, "delta_amb": delta_amb, "delta_oracle": delta_oracle, "eff_norm": eff_norm, "eff_amb": eff_amb, "eff_oracle": eff_oracle}
    #with open(OUTPUT_RESULT_DIR / "Method_Comparison.pkl", 'wb') as f: pickle.dump(raw_data, f)
    print("Oracle Results")
    print([oracle_metrics[i][2] for i in range(0, len(oracle_metrics))])
    print([oracle_metrics[i][3] for i in range(0, len(oracle_metrics))])
    print("Normal Results")
    print([norm_metrics[i][2] for i in range(0, len(norm_metrics))])
    print([norm_metrics[i][3] for i in range(0, len(norm_metrics))])
    print("WebCP Results")
    print([amb_metrics[i][2] for i in range(0, len(amb_metrics))])
    print([amb_metrics[i][3] for i in range(0, len(amb_metrics))])
    print(delta_oracle)
    # Generate Plots
    #plt.plot(alpha_values, delta_norm, color='blue', label='normal')
    #plt.plot(alpha_values, delta_amb, color='red', label = 'ambiguous')
    #plt.plot(alpha_values, delta_oracle, color='green', label = 'oracle')
    alpha_values = [1-val for val in alpha_values]
    plt.rcParams.update({'font.size': 14})
    plt.plot(alpha_values, norm, color='blue', label='Normal CP')
    plt.plot(alpha_values, amb, color='red', label = 'WebCP')
    plt.plot(alpha_values, oracle, color='purple', label = 'Oracle CP')
    plt.plot(alpha_values, target, color='green', label = 'Target Coverage')
    plt.axhline(y = 0.0, color = 'grey', linestyle = '-')
    plt.title(u'Target v. Test Coverage')
    plt.xlabel(u'Target Coverage')
    #plt.xticks(alpha_values)
    #plt.ylabel(u'Target (1-α) v. Test Coverage Δ')
    plt.ylabel(u'Test Coverage')
    plt.legend()
    plt.savefig(OUTPUT_RESULT_DIR / "Coverage_Alpha.png")
    plt.show()
    plt.plot(alpha_values, eff_norm, color='blue', label='Normal CP')
    plt.plot(alpha_values, eff_amb, color='red', label = 'WebCP')
    plt.plot(alpha_values, eff_oracle, color='green', label = 'Oracle CP')
    plt.title(u'Target v. Test Efficiency')
    plt.xlabel(u'Target Coverage')
    plt.ylabel(u'Efficiency (extraneous classes per sample)')
    plt.legend()
    plt.savefig(OUTPUT_RESULT_DIR / "Efficiency_Alpha.png")
    plt.show()

# CP Metrics vs. Calibration Set Size
if CALIB_SIZE_CURVE:
    print("Begin Calibration Set Size Curve")
    # Initialize metrics lists and set size list
    norm_metrics = []
    amb_metrics = []
    nums_per_class = [i for i in range(5, 40)]
    # Loop through possible per class set sizes
    for pruned_size in nums_per_class:
        # Initialize pruned lists and num instances
        covered = [0 for i in range(0, test_true_class_arr.shape[1])]
        pruned_calib_sim_score = []
        pruned_calib_true_class = []
        pruned_plausibility_score = []
        for i in range(0, n_calib):
            # Find instance true class and check if num instances pruned
            true_class = torch.argmax(calib_true_class_arr[i])
            if covered[true_class] >= pruned_size: continue
            if true_class > 1000: continue
            # Update pruned list and num instances
            covered[true_class] += 1
            pruned_calib_sim_score.append(calib_sim_score_arr[i])
            pruned_calib_true_class.append(calib_true_class_arr[i])
            pruned_plausibility_score.append(calib_plausibility_score_arr[i])
        # Convert lists into matrices
        pruned_calib_true_class = torch.vstack(pruned_calib_true_class)
        pruned_calib_sim_score = torch.vstack(pruned_calib_sim_score)
        pruned_plausibility_score = torch.vstack(pruned_plausibility_score)
        # Perform Conformal Prediction
        threshold_amb = monte_carlo_cp(pruned_calib_sim_score, pruned_plausibility_score, ALPHA, NUM_SAMPLES)
        pruned_calib_sim_score_np = pruned_calib_sim_score.detach().cpu().numpy()
        pruned_calib_true_class_np = pruned_calib_true_class.detach().cpu().numpy()
        test_sim_score_arr_np = test_sim_score_arr.detach().cpu().numpy()
        test_true_class_arr_np = test_true_class_arr.detach().cpu().numpy()
        threshold_norm = compute_threshold(ALPHA, pruned_calib_sim_score_np, pruned_calib_true_class_np)
        # Calculate Metrics
        print("\nCalib Set Size: {Size}".format(Size = pruned_size))
        print("Normal CP:")
        norm_metric = performance_report(threshold_norm, pruned_calib_sim_score_np, test_sim_score_arr_np, pruned_calib_true_class_np, test_true_class_arr_np)
        norm_metrics.append(norm_metric)
        print("Ambiguous CP:")
        amb_metric = performance_report(threshold_amb, pruned_calib_sim_score_np, test_sim_score_arr_np, pruned_calib_true_class_np, test_true_class_arr_np)
        amb_metrics.append(amb_metric)

    plt.plot(nums_per_class, [norm_metric[2] for norm_metric in norm_metrics], color='blue', label='normal')
    plt.plot(nums_per_class, [amb_metric[2] for amb_metric in amb_metrics], color='red', label = 'ambiguous')
    plt.axhline(y = 1-ALPHA, color = 'green', linestyle = '-')
    plt.title("Calib Size Per Class v. Test Coverage Curve: {alpha}".format(alpha=ALPHA))
    plt.xlabel("# Calib Instances Per Class")
    plt.ylabel("Coverage (proportion of true labels covered)")
    plt.legend()
    plt.savefig(OUTPUT_RESULT_DIR / "Coverage_Size-{alpha}.png".format(alpha=int(ALPHA*100)))
    plt.show()

    plt.plot(nums_per_class, [norm_metric[3] for norm_metric in norm_metrics], color='blue', label='normal')
    plt.plot(nums_per_class, [amb_metric[3] for amb_metric in amb_metrics], color='red', label = 'ambiguous')
    plt.title("Calib Size Per Class v. Test Efficiency Curve: {alpha}".format(alpha=ALPHA))
    plt.xlabel("# Calib Instances Per Class")
    plt.ylabel("Efficiency (mean num of extraneous classes per sample)")
    plt.legend()
    plt.savefig(OUTPUT_RESULT_DIR / "Efficiency_Size-{alpha}.png".format(alpha=int(ALPHA*100)))
    plt.show()

# CP Metrics vs. Alpha value
# Delta between target and actual coverage value
if ALPHA_CURVE:
    print("Begin Alpha Curve")
    # Initialize metrics lists and set size list
    norm_metrics = []
    amb_metrics = []
    alpha_values = [0.1*i for i in range(1, 6)]
    #alpha_values = [0.025*i for i in range(1, 21)]
    # Generate numpy matrices
    calib_sim_score_arr_np = calib_sim_score_arr.detach().cpu().numpy()
    calib_true_class_arr_np = calib_true_class_arr.detach().cpu().numpy()
    test_sim_score_arr_np = test_sim_score_arr.detach().cpu().numpy()
    test_true_class_arr_np = test_true_class_arr.detach().cpu().numpy()
    # Loop through possible per class set sizes
    for alpha in alpha_values:
        #Perform Conformal Prediction
        print("Performing Conformal Prediction")
        threshold_amb = monte_carlo_cp(calib_sim_score_arr, calib_plausibility_score_arr, alpha, NUM_SAMPLES)
        threshold_norm = compute_threshold(alpha, calib_sim_score_arr_np, calib_true_class_arr_np)
        #Output Metrics
        print("\nAlpha Value: {alpha}".format(alpha = alpha))
        print("Normal CP:")
        norm_metrics.append(performance_report(threshold_norm, calib_sim_score_arr_np, test_sim_score_arr_np, calib_true_class_arr_np, test_true_class_arr_np))
        print("Ambiguous CP:")
        amb_metrics.append(performance_report(threshold_amb, calib_sim_score_arr_np, test_sim_score_arr_np, calib_true_class_arr_np, test_true_class_arr_np))
    # Generate deltas
    delta_norm = [norm_metrics[i][2]+alpha_values[i]-1 for i in range(0, len(norm_metrics))]
    delta_amb = [amb_metrics[i][2]+alpha_values[i]-1 for i in range(0, len(norm_metrics))]
    print(delta_norm)
    print(delta_amb)
    # Generate Plot
    plt.plot(alpha_values, delta_norm, color='blue', label='normal')
    plt.plot(alpha_values, delta_amb, color='red', label = 'ambiguous')
    plt.axhline(y = 0.0, color = 'green', linestyle = '-')
    plt.title(u'Alpha Value v. Target-Test Coverage Δ')
    plt.xlabel(u'Alpha Value (α)')
    plt.ylabel(u'Target (1-α) v. Test Coverage Δ')
    plt.legend()
    plt.savefig(OUTPUT_RESULT_DIR / "Delta-Coverage_Alpha.png")
    plt.show()

# Histogram of uncertainty scores 
# 1-np.dot(true_class_arr, sim_score) v. 1-np.dot(plausibility_arr, sim_score) v. 1 - plausibility_arr[argmax(plausibility_arr)]*sim_score[argmax(plausibility_arr)]
# May want to take out junk score and renormalize
# Reverse softmax
if UNCERTAIN_HIST:
    print("Begin Uncertainty Histogram")
    # Initialize score lists
    true_class_calib_scores = []
    true_class_test_scores = []
    plausibility_expected_scores = []
    plausibility_max_scores = []
    # Generate numpy matrices
    calib_sim_score_arr_np = calib_sim_score_arr.detach().cpu().numpy()
    calib_true_class_arr_np = calib_true_class_arr.detach().cpu().numpy()
    test_sim_score_arr_np = test_sim_score_arr.detach().cpu().numpy()
    test_true_class_arr_np = test_true_class_arr.detach().cpu().numpy()
    calib_plausibility_score_arr_np = calib_plausibility_score_arr.detach().cpu().numpy()
    # Invert Softmax
    calib_sim_score_arr_np = (np.log(calib_sim_score_arr_np) + LOGIT_SCALE)/LOGIT_SCALE
    test_sim_score_arr_np = (np.log(test_sim_score_arr_np) + LOGIT_SCALE)/LOGIT_SCALE
    for i in range(0, n_calib):
        # Append scores
        true_class_calib_scores.append(np.dot(calib_sim_score_arr_np[i], calib_true_class_arr_np[i]))
        true_class_test_scores.append(np.dot(test_sim_score_arr_np[i], test_true_class_arr_np[i]))
        # Calculate plausibility scores
        plausibility_point = calib_plausibility_score_arr_np[i][:-1]
        plausibility_point = plausibility_point/np.sum(plausibility_point)
        max_at = np.argmax(plausibility_point)
        plausibility_expected_scores.append(np.dot(calib_sim_score_arr_np[i], plausibility_point))
        plausibility_max_scores.append((calib_sim_score_arr_np[i][max_at] * plausibility_point[max_at]))

    raw_data = {"true_class_test_scores": true_class_test_scores, "plausibility_expected_scores": plausibility_expected_scores, "true_class_calib_scores": true_class_calib_scores, "plausibility_max_scores": plausibility_max_scores}
    with open(OUTPUT_RESULT_DIR / "Histogram_Comparison.pkl", 'wb') as f: pickle.dump(raw_data, f)

    # Generate Histogram
    bins = [0.01*i for i in range(0, 100)]
    plt.hist(plausibility_expected_scores, bins=bins, density=True, alpha=0.5, label=u'Expected Scores $E(x, λ)$', color = 'blue')
    plt.hist(true_class_test_scores, bins=bins, density=True, alpha=0.5, label=u'Test True Scores $E(x, y)$', color = 'green')
    plt.xlabel('Conformity Score')
    plt.ylabel('Frequency')
    plt.title(u'Expected Scores $E(x, λ)$ Histogram')
    plt.legend()
    plt.savefig(OUTPUT_RESULT_DIR / "Conformity_Histogram_Expected-Plausibility.png")
    plt.show()

    plt.hist(true_class_calib_scores, bins=bins, density=True, alpha=0.5, label=u'Calib True Scores $E(x, y)$', color = 'red')
    plt.hist(true_class_test_scores, bins=bins, density=True, alpha=0.5, label=u'Test True Scores $E(x, y)$', color = 'green')
    plt.xlabel('Conformity Score')
    plt.ylabel('Frequency')
    plt.title(u'Calib True Scores $E(x, y)$ Histogram')
    plt.legend()
    plt.savefig(OUTPUT_RESULT_DIR / "Conformity_Histogram_Calib-True.png")
    plt.show()

    plt.hist(plausibility_max_scores, bins=bins, density=True, alpha=0.5, label=u'Top-1 Expected Scores $E(x, argmax_{k}λ_{k})$', color = 'purple')
    plt.hist(true_class_test_scores, bins=bins, density=True, alpha=0.5, label=u'Test True Scores $E(x, y)$', color = 'green')
    plt.xlabel('Conformity Score')
    plt.ylabel('Frequency')
    plt.title(u'Top-1 Expected Scores $E(x, argmax_{k}λ_{k})$ Histogram')
    plt.legend()
    plt.savefig(OUTPUT_RESULT_DIR / "Conformity_Histogram_Argmax-Plausibility.png")
    plt.show()

# Confusion Matrix of Plausibilities
if PLAUSIBILITY_HISTOGRAM:
    print("Begin Plausibility Histogram")
    # Initialize Score Matrices
    true_plausibilities = []
    other_plausibilities = []
    junk_plausibilities = []
    # Generate numpy matrices
    calib_sim_score_arr_np = calib_sim_score_arr.detach().cpu().numpy()
    calib_true_class_arr_np = calib_true_class_arr.detach().cpu().numpy()
    test_sim_score_arr_np = test_sim_score_arr.detach().cpu().numpy()
    test_true_class_arr_np = test_true_class_arr.detach().cpu().numpy()
    calib_plausibility_score_arr_np = calib_plausibility_score_arr.detach().cpu().numpy()
    for i in range(0, n_calib):
        # Calculate scores
        true_class = torch.argmax(calib_true_class_arr[i])
        true = calib_plausibility_score_arr_np[i][true_class]
        junk = calib_plausibility_score_arr_np[i][-1]
        other = np.sum(calib_plausibility_score_arr_np[i])-true-junk
        # Add to score lists
        junk_plausibilities.append(junk)
        true_plausibilities.append(true)
        other_plausibilities.append(other)
    # Generate Histogram
    bins = [0.01*i for i in range(0, 101)]
    plt.hist(true_plausibilities, bins=bins, density=True, alpha=0.5, label=u'$λ_{y}$', color = 'green')
    plt.xlabel(u'Plausibility Score (λ)')
    plt.ylabel('Frequency')
    plt.title(u'Intended Class Plausibility Distribution')
    plt.legend()
    plt.savefig(OUTPUT_RESULT_DIR / "True_Class_Histogram-Plausibility.png")
    plt.show()

    plt.hist(junk_plausibilities, bins=bins, density=True, alpha=0.5, label=u'$λ_{j}$', color = 'red')
    plt.xlabel(u'Plausibility Score (λ)')
    plt.ylabel('Frequency')
    plt.title(u'Junk Plausibility Distribution')
    plt.legend()
    plt.savefig(OUTPUT_RESULT_DIR / "Junk_Class_Histogram-Plausibility.png")
    plt.show()

    plt.hist(other_plausibilities, bins=bins, density=True, alpha=0.5, label=u'$\sum_{i\\neq y \\neq j}^{}\lambda_{i}$', color = 'purple')
    plt.xlabel('Plausibility Score (λ)')
    plt.ylabel('Frequency')
    plt.title(u'Other Class Plausibility Distribution')
    plt.legend()
    plt.savefig(OUTPUT_RESULT_DIR / "Other_Class_Histogram-Plausibility.png")
    plt.show()
