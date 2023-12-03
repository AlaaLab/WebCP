import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
import open_clip
import argparse
from conformal_prediction_methods import *
from metrics import *
import yaml

# Model Methods


def openclip_image_preprocess(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_logits = model.encode_image(image)
        image_logits /= image_logits.norm(dim=-1, keepdim=True)
    return image_logits.to("cpu")


def openclip_text_preprocess(text):
    text = tokenizer(text).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_logits = model.encode_text(text)
        text_logits /= text_logits.norm(dim=-1, keepdim=True)
    return text_logits.to("cpu")


def openclip_process(image_logits, text_logits):
    image_logits, text_logits = image_logits.type(
        torch.float32), text_logits.type(torch.float32)
    return (LOGIT_SCALE * image_logits @ text_logits.T).softmax(dim=-1)[0]


def performance_report(threshold):
    # Get prediction sets
    calib_prediction_set = compute_prediction_sets_threshold(
        calib_sim_score_arr, threshold)
    test_prediction_set = compute_prediction_sets_threshold(
        test_sim_score_arr, threshold)
    # Compute performance metrics
    calib_coverage = overall_coverage(
        calib_prediction_set, calib_true_class_arr)
    test_coverage = overall_coverage(test_prediction_set, test_true_class_arr)
    calib_samplewise_efficiency = samplewise_efficiency(
        calib_prediction_set, calib_true_class_arr)
    test_samplewise_efficiency = samplewise_efficiency(
        test_prediction_set, test_true_class_arr)
    # Output Performance Metrics
    print(f"OVERALL COVERAGE (proportion of true labels covered):")
    print(f"Calibration Set: {calib_coverage}")
    print(f"Test Set: {test_coverage}")
    print(f'OVERALL EFFICIENCY (mean num of extraneous classes per sample): ')
    print(f"Calibration Set: {np.mean(calib_samplewise_efficiency)}")
    print(f"Test Set: {np.mean(test_samplewise_efficiency)}")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", help="the path to the yaml config file.", type=str)
    args = parser.parse_args()
    config = {}
    with open(args.config, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    for k, v in config.items():
        if (k in ['calib_image_dir', 'test_image_dir', 'intermediate_data_dir', 'results_store_dir', 'classification_checkpoint', 'context_dir', 'char_output_dir']):
            config[k] = Path(v)

    config['results_store_dir'].mkdir(exist_ok=False)

    # Parameters
    TEST_IMAGE_DIRECTORY = config["test_image_dir"]
    IMAGE_PLAUSIBILITIES = config["intermediate_data_dir"]
    RESULTS_DIRECTORY = config["results_store_dir"]
    CLASSIFICATION_CHECKPOINT = config["classification_checkpoint"]
    CALIB_IMAGE_DIRECTORY = config['calibration_image_dir']

    class_dict = {}
    class_df = pd.read_csv(config['class_list_csv'])

    for i in range(len(class_df)):
        row = class_df.iloc[i, :]
        class_dict[row["Class Index"]] = row["Class"]

    ALPHA = 0.05
    NUM_SAMPLES = 1000
    USE_SOFTMAX = True
    LOGIT_SCALE = 100.0 if USE_SOFTMAX else 1.0
    MODEL_ID = CLASSIFICATION_CHECKPOINT

    # Initialize Model
    # -----------------------------------------------------------------------------------
    print("Initializing Models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CUDA ENABLED: {}".format(str(torch.cuda.is_available())))
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    model.to(device)

    # Generate Label Logits
    # -----------------------------------------------------------------------------------
    print("Preprocess Text Prompts")
    def PROMPT_GENERATOR(cls): return f"{cls}."
    prompt_list = [PROMPT_GENERATOR(cls) for id, cls in class_dict.items()]
    label_logits = openclip_text_preprocess(prompt_list)

    # Generate Calibration Matrices
    # -----------------------------------------------------------------------------------
    print("Generating Calibration Matrices")
    calib_true_class_arr = []
    calib_sim_score_arr = []
    calib_plausibility_score_arr = []
    # Loop through image
    for label in os.listdir(CALIB_IMAGE_DIRECTORY):
        print("Beginning Calibration Embedding Generation: {label}".format(
            label=label))
        avg = torch.zeros(len(class_dict)+1)
        num_images = 0
        for img in os.listdir(CALIB_IMAGE_DIRECTORY / label):
            # Retrieve plausibilities array
            try:
                plausibility_arr = torch.load(
                    IMAGE_PLAUSIBILITIES / label / (img.split('.')[0]+"_plausibilities"))
            except:
                print("Error")
                continue
            # Build label array
            class_onehot = torch.zeros(len(class_dict))
            class_onehot[int(label)] = 1
            # Build similarity array
            image = Image.open(CALIB_IMAGE_DIRECTORY / label / img)
            image_logit = openclip_image_preprocess(image)
            label_probs = openclip_process(image_logit, label_logits)
            # Append to matrices
            calib_true_class_arr.append(class_onehot)
            calib_sim_score_arr.append(label_probs)
            calib_plausibility_score_arr.append(plausibility_arr)
            # Update metrics
            avg = avg + plausibility_arr
            num_images += 1
            if num_images >= NUM_SAMPLES:
                break
        avg = avg/len(os.listdir(CALIB_IMAGE_DIRECTORY / label))
        print(avg[int(label)])
    # Append Matrices
    calib_true_class_arr = torch.vstack(calib_true_class_arr)
    calib_sim_score_arr = torch.vstack(calib_sim_score_arr)
    calib_plausibility_score_arr = torch.vstack(calib_plausibility_score_arr)

    # Generate Test Matrices
    # -----------------------------------------------------------------------------------
    print("Generating Test Matrices")
    test_true_class_arr = []
    test_sim_score_arr = []
    # Loop through image
    for label in os.listdir(TEST_IMAGE_DIRECTORY):
        print("Beginning Test Embedding Generation: {label}".format(
            label=label))
        for img in os.listdir(TEST_IMAGE_DIRECTORY / label):
            # Build label array
            class_onehot = torch.zeros(len(class_dict))
            class_onehot[int(label)] = 1
            test_true_class_arr.append(class_onehot)
            # Build similarity array
            image = Image.open(TEST_IMAGE_DIRECTORY / label / img)
            image_logit = openclip_image_preprocess(image)
            label_probs = openclip_process(image_logit, label_logits)
            test_sim_score_arr.append(label_probs)
    # Append Matrices
    test_true_class_arr = torch.vstack(test_true_class_arr)
    test_sim_score_arr = torch.vstack(test_sim_score_arr)

    # Perform Conformal Prediction
    # -----------------------------------------------------------------------------------
    # Save Data Arrays
    torch.save(calib_plausibility_score_arr, RESULTS_DIRECTORY /
               "calib_plausibility_score_arr")
    torch.save(calib_sim_score_arr, RESULTS_DIRECTORY / "calib_sim_score_arr")
    torch.save(calib_true_class_arr, RESULTS_DIRECTORY /
               "calib_true_class_arr")
    torch.save(test_sim_score_arr, RESULTS_DIRECTORY / "test_sim_score_arr")
    torch.save(test_true_class_arr, RESULTS_DIRECTORY / "test_true_class_arr")
    # Perform Conformal Prediction
    print("Performing Conformal Prediction")
    threshold_amb = monte_carlo_cp(
        calib_sim_score_arr, calib_plausibility_score_arr, ALPHA, NUM_SAMPLES)
    calib_sim_score_arr = calib_sim_score_arr.detach().cpu().numpy()
    calib_true_class_arr = calib_true_class_arr.detach().cpu().numpy()
    test_sim_score_arr = test_sim_score_arr.detach().cpu().numpy()
    test_true_class_arr = test_true_class_arr.detach().cpu().numpy()
    threshold_norm = compute_threshold(
        ALPHA, calib_sim_score_arr, calib_true_class_arr)
    # Output Metrics
    print("Normal CP:")
    performance_report(threshold_norm)
    print("\nAmbiguous CP:")
    performance_report(threshold_amb)
