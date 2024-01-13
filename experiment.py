import sys
import os

from pathlib import Path
import pandas as pd
import scipy as sp
import numpy as np
from PIL import Image
import torch
import json
import open_clip
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPModel, CLIPProcessor
import argparse

script_path = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
base_path = script_path.parent.absolute()
sys.path.append(base_path / 'cp')
sys.path.append(base_path / 'utils')
from utils.pets_classes import PETS_CLASSES
from utils.fitz17k_classes import FITZ17K_CLASSES
from utils.medmnist_classes import MEDMNIST_CLASSES
from cp.conformal_prediction_methods import *
from cp.metrics import *

#Parse Arguments
#-----------------------------------------------------------------------------------
'''parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, help='Experiment in experiment_configs to run')
args = parser.parse_args()'''

#Parameters
#-----------------------------------------------------------------------------------
'''reader = open(base_path + "\\experiment_configs\\"  + args.exp)
config = json.load(reader)
TEST_IMAGE_DIRECTORY = config["test_image_directory"]
IMAGE_PLAUSIBILITIES = config["intermediate_data_directory"]
RESULTS_DIRECTORY = config["results_data_directory"]
CLASSIFICATION_CHECKPOINT = config["classification_checkpoint"]'''

ALPHA = 0.1
NUM_SAMPLES = 1000
USE_SOFTMAX = True
LOGIT_SCALE = 100.0 if USE_SOFTMAX else 1.0

MODEL_ID = "hf-hub:laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg" #"hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

if False:
    TEST_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets\\google-pets\\oxford-pets")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\oxford-pets\\web_scraping_0105_selenium_reverse-image-selenium_oxford-pets_plausibilities")
    CALIB_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\oxford-pets\\web_scraping_0105_selenium_reverse-image-selenium_oxford-pets")
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-pets_01-06-24_1")
    dataset = 'OxfordPets'
if False:
    TEST_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets\\google-fitz17k\\fitzpatrick-17k")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\fitzpatrick17k\\web_scraping_0105_selenium_reverse-image-selenium_fitz-17k_plausibilities")
    CALIB_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\fitzpatrick17k\\web_scraping_0105_selenium_reverse-image-selenium_fitz-17k")
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-fitz17k_01-06-24_1")
    dataset = 'FitzPatrick17k'
if True:
    TEST_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets\\selenium-medmnist\\medmnist_microscopy")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\medmnist\\web_scraping_1225_reverse-image-selenium_medmnist_plausibilities")
    CALIB_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\medmnist\\web_scraping_1225_reverse-image-selenium_medmnist")
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-medmnist_01-03-24_1")
    dataset = 'MedMNIST'


if dataset == 'MedMNIST':
    LABELS = MEDMNIST_CLASSES
elif dataset == 'FitzPatrick17k':
    LABELS = FITZ17K_CLASSES
elif dataset == 'OxfordPets':
    LABELS = PETS_CLASSES
else:
    LABELS = None

#Model Methods
#-----------------------------------------------------------------------------------
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
    image_logits, text_logits = image_logits.type(torch.float32), text_logits.type(torch.float32)
    return (LOGIT_SCALE * image_logits @ text_logits.T).softmax(dim=-1)[0]

def performance_report(threshold):
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

#Initialize Model
#-----------------------------------------------------------------------------------
print("Initializing Models")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA ENABLED: {}".format(str(torch.cuda.is_available())))
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
tokenizer = open_clip.get_tokenizer(MODEL_ID)
model.to(device)

#Generate Label Logits
#-----------------------------------------------------------------------------------
print("Preprocess Text Prompts")
PROMPT_GENERATOR = lambda cls : f"{cls}."
prompt_list = [PROMPT_GENERATOR(cls) for id, cls in LABELS.items()]
label_logits = openclip_text_preprocess(prompt_list)

#Generate Calibration Matrices
#-----------------------------------------------------------------------------------
print("Generating Calibration Matrices")
calib_true_class_arr = []
calib_sim_score_arr = []
calib_plausibility_score_arr = []
#Loop through image
for label in os.listdir(IMAGE_PLAUSIBILITIES):
    print("Beginning Calibration Embedding Generation: {label}".format(label=label))
    avg = torch.zeros(len(LABELS.items())+1)
    num_images = 0
    for plaus in os.listdir(IMAGE_PLAUSIBILITIES / label):
        # Retrieve plausibilities array
        try: 
            plausibility_arr = torch.load(IMAGE_PLAUSIBILITIES / label / plaus)
            if not torch.all(torch.isnan(plausibility_arr)==False):
                print('ERROR')
            image = Image.open(CALIB_IMAGE_DIRECTORY / label / (plaus+'.jpeg'))
        except:
            print("Error")
            continue
        # Build label array
        class_onehot = torch.zeros(len(LABELS.items()))
        class_onehot[int(label)] = 1
        # Build similarity array
        image_logit = openclip_image_preprocess(image)
        label_probs = openclip_process(image_logit, label_logits)
        # Append to matrices
        calib_true_class_arr.append(class_onehot)
        calib_sim_score_arr.append(label_probs)
        calib_plausibility_score_arr.append(plausibility_arr)
        # Update metrics
        avg = avg + plausibility_arr
        num_images += 1
        if num_images >= NUM_SAMPLES: break
    avg = avg/len(os.listdir(CALIB_IMAGE_DIRECTORY / label))
#Append Matrices
calib_true_class_arr = torch.vstack(calib_true_class_arr)
calib_sim_score_arr = torch.vstack(calib_sim_score_arr)
calib_plausibility_score_arr = torch.vstack(calib_plausibility_score_arr)
print(torch.all(torch.isnan(calib_plausibility_score_arr)==False))
print(torch.min(calib_plausibility_score_arr))
#Generate Test Matrices
#-----------------------------------------------------------------------------------
print("Generating Test Matrices")
test_true_class_arr = []
test_sim_score_arr = []
#Loop through image
for label in os.listdir(TEST_IMAGE_DIRECTORY):
    print("Beginning Test Embedding Generation: {label}".format(label=label))
    for img in os.listdir(TEST_IMAGE_DIRECTORY / label):
        # Build label array
        class_onehot = torch.zeros(len(LABELS.items()))
        class_onehot[int(label)] = 1
        test_true_class_arr.append(class_onehot)
        # Build similarity array
        image = Image.open(TEST_IMAGE_DIRECTORY / label / img)
        image_logit = openclip_image_preprocess(image)
        label_probs = openclip_process(image_logit, label_logits)
        test_sim_score_arr.append(label_probs)
#Append Matrices
test_true_class_arr = torch.vstack(test_true_class_arr)
test_sim_score_arr = torch.vstack(test_sim_score_arr)

#Perform Conformal Prediction
#-----------------------------------------------------------------------------------
#Save Data Arrays
torch.save(calib_plausibility_score_arr, RESULTS_DIRECTORY / "calib_plausibility_score_arr")
torch.save(calib_sim_score_arr, RESULTS_DIRECTORY / "calib_sim_score_arr")
torch.save(calib_true_class_arr, RESULTS_DIRECTORY / "calib_true_class_arr")
torch.save(test_sim_score_arr, RESULTS_DIRECTORY / "test_sim_score_arr")
torch.save(test_true_class_arr, RESULTS_DIRECTORY / "test_true_class_arr")
#Perform Conformal Prediction
print("Performing Conformal Prediction")
threshold_amb = monte_carlo_cp(calib_sim_score_arr, calib_plausibility_score_arr, ALPHA, NUM_SAMPLES)
calib_sim_score_arr = calib_sim_score_arr.detach().cpu().numpy()
calib_true_class_arr = calib_true_class_arr.detach().cpu().numpy()
test_sim_score_arr = test_sim_score_arr.detach().cpu().numpy()
test_true_class_arr = test_true_class_arr.detach().cpu().numpy()
threshold_norm = compute_threshold(ALPHA, calib_sim_score_arr, calib_true_class_arr)
#Output Metrics
print("Normal CP:")
performance_report(threshold_norm)
print("\nAmbiguous CP:")
performance_report(threshold_amb)
