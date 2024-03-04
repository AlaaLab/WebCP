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
from transformers import AutoProcessor, OwlViTModel, GroupViTModel
from transformers import FlavaModel, BertTokenizer, FlavaFeatureExtractor
import argparse
import random
script_path = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
base_path = script_path.parent.absolute()
sys.path.append(base_path / 'cp')
sys.path.append(base_path / 'utils')
from utils.pets_classes import PETS_CLASSES
from utils.fitz17k_classes import FITZ17K_CLASSES
from utils.medmnist_classes import MEDMNIST_CLASSES
from utils.imagenet_classes import IMAGENET_CLASSES
from utils.caltech256_classes import CALTECH256_CLASSES
from cp.conformal_prediction_methods import *
from cp.metrics import *
import easyocr

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
NUM_SAMPLES = 30
USE_SOFTMAX = True

TEST_RELOAD = False
CALIB_RELOAD =  True

if False:
    TEST_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets\\google-pets\\oxford-pets")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\oxford-pets\\web_scraping_0105_selenium_reverse-image-selenium_oxford-pets_plausibilities")
    CALIB_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\oxford-pets\\web_scraping_0105_selenium_reverse-image-selenium_oxford-pets")
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-pets_01-06-24_1")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-pets_01-06-24_owlvit")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-pets_01-06-24_flava")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-pets_01-06-24_clipa")
    dataset = 'OxfordPets'
if False:
    TEST_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets\\google-fitz17k\\fitzpatrick-17k")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\fitzpatrick17k\\web_scraping_0105_selenium_reverse-image-selenium_fitz-17k_plausibilities")
    CALIB_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\fitzpatrick17k\\web_scraping_0105_selenium_reverse-image-selenium_fitz-17k")
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-fitz17k_01-06-24_1")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-fitz17k_01-06-24_owlvit")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-fitz17k_01-06-24_flava")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-fitz17k_01-06-24_clipa")
    dataset = 'FitzPatrick17k'
if False:
    TEST_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets\\selenium-medmnist\\medmnist_microscopy")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\medmnist\\web_scraping_0114_reverse-image-selenium_medmnist_new-plausibilities")
    CALIB_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\medmnist\\web_scraping_0114_reverse-image-selenium_medmnist_new")
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-medmnist_01-14-24_1")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-medmnist_01-06-24_owlvit")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-medmnist_01-06-24_flava")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-medmnist_01-06-24_clipa")
    dataset = 'MedMNIST'
if True:
    TEST_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\imagenet\\imagenet_2012")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\imagenet\\web_scraping_0220_selenium_reverse-image-selenium_imagenet_plausibilities")
    CALIB_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\imagenet\\web_scraping_0103_selenium_reverse-image-selenium_imagenet")
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-imagenet_02-20-24_1")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-imagenet_01-15-24_1")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-imagenet_01-15-24_owlvit")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-imagenet_01-15-24_flava")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-imagenet_01-15-24_clipa")
    dataset =  'ImageNet'
if False:
    TEST_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\caltech256\\256_ObjectCategories")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\caltech256\\web_scraping_0114_selenium_reverse-search-selenium_caltech-256_plausibilities")
    CALIB_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\caltech256\\web_scraping_0114_selenium_reverse-search-selenium_caltech-256")
    RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-caltech256_01-17-24_1")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-caltech256_01-17-24_owlvit")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-caltech256_01-17-24_flava")
    #RESULTS_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\analysis\\ambiguous_experiments\\google-caltech256_01-17-24_clipa")
    dataset =  'Caltech256'

if dataset == 'MedMNIST':
    LABELS = MEDMNIST_CLASSES
elif dataset == 'FitzPatrick17k':
    LABELS = FITZ17K_CLASSES
elif dataset == 'OxfordPets':
    LABELS = PETS_CLASSES
elif dataset == 'ImageNet':
    LABELS = IMAGENET_CLASSES
elif dataset == 'Caltech256':
    LABELS = CALTECH256_CLASSES
else:
    LABELS = None


if GENERATE_DEBUG_CSV: 
    CALIB_DEBUG_CSV_PATH = Path("~/reesearch/debug.csv")
    calib_debug_df = pd.read_csv(CALIB_DEBUG_CSV_PATH).set_index("index")
    calib_debug_df['clip_score_label'] = -1.0
    for i in range(len(LABELS)):
        calib_debug_df[f'clip_score_{i}'] = -1.0

    CALIB_OUTPUT_CSV_PATH = Path("~/reesearch/debug_clip.csv")

    TEST_DEBUG_CSV_PATH = Path("~/reesearch/debug_test.csv")
    test_debug_keys = ["index", "label", "filename", "clip_score_label"] + [f"clip_score_{i}" for i in range(len(LABELS))]
    test_debug_dict = {k: [] for k in test_debug_keys}
    # test_debug_dict = pd.read_csv(TEST_DEBUG_CSV_PATH)    
#Model Methods
#-----------------------------------------------------------------------------------
def owlvit_image_preprocess(image):
    with torch.no_grad(), torch.cuda.amp.autocast():
        inputs = owlvit_processor(text=['Dummy'], images=image, return_tensors="pt", padding=True, truncation=True).to(device)
        image_logits = owlvit_model(**inputs).image_embeds
        image_logits /= image_logits.norm(dim=-1, keepdim=True)
    return image_logits.to("cpu")

def owlvit_text_preprocess(text):
    with torch.no_grad(), torch.cuda.amp.autocast():
        inputs = owlvit_processor(text=text, images=Image.new('RGB', (100, 100)), return_tensors="pt", padding=True, truncation=True).to(device)
        text_logits = owlvit_model(**inputs).text_embeds
        text_logits /= text_logits.norm(dim=-1, keepdim=True)
    return text_logits.to("cpu")

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

def flava_image_preprocess(image):
    image_input = fe(image, return_tensors="pt")
    for k, v in image_input.items():
        image_input[k] = v.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        return flava_model.get_image_features(**image_input)[:, 0, :].to("cpu")
    
def flava_text_preprocess(text):
    text_inputs = flava_tokenizer(text, return_tensors="pt", padding="max_length")
    for k, v in text_inputs.items():
        text_inputs[k] = v.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        return flava_model.get_text_features(**text_inputs)[:, 0, :].to("cpu")

def score_process(image_logits, text_logits):
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
if True:
    MODEL_ID = "hf-hub:laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg" #"hf-hub:UCSC-VLAA/ViT-L-14-CLIPA-datacomp1B" "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    model.to(device)
    text_preprocess = openclip_text_preprocess
    image_preprocess = openclip_image_preprocess
    LOGIT_SCALE = 100.0
if False:
    #owlvit_model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
    #owlvit_processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32", padding=True)
    owlvit_model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
    owlvit_processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc", padding=True)
    owlvit_model.to(device)
    text_preprocess = owlvit_text_preprocess
    image_preprocess = owlvit_image_preprocess
    LOGIT_SCALE = 10.0
if False:
    flava_model = FlavaModel.from_pretrained("facebook/flava-full")
    fe = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
    flava_tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")
    flava_model.eval()
    flava_model.to(device)
    text_preprocess = flava_text_preprocess
    image_preprocess = flava_image_preprocess
    LOGIT_SCALE = 1.0
reader = easyocr.Reader(['en'])

#Generate Label Logits
#-----------------------------------------------------------------------------------
print("Preprocess Text Prompts")
PROMPT_GENERATOR = lambda cls : f"{cls}."
prompt_list = [PROMPT_GENERATOR(cls) for id, cls in LABELS.items()]
#label_logits = openclip_text_preprocess(prompt_list)
#label_logits = owlvit_text_preprocess(prompt_list)
label_logits = text_preprocess(prompt_list)
#Generate Calibration Matrices
#-----------------------------------------------------------------------------------
if not CALIB_RELOAD:
    print("Skipping Generating Calibration Matrices")
else:
    avg2 = [0, 0]
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
            if not 'plausibilities' in plaus: continue
            try: 
                plausibility_arr = torch.load(IMAGE_PLAUSIBILITIES / label / plaus)
                if not torch.all(torch.isnan(plausibility_arr)==False):
                    print('ERROR')
                path = CALIB_IMAGE_DIRECTORY / label / (plaus.split('_')[0]+'.jpeg')
                image = Image.open(path).convert('RGB')      
                extract_info = reader.readtext(str(path))
            except:
                print("Error")
                continue
            # Build label array
            class_onehot = torch.zeros(len(LABELS.items()))
            class_onehot[int(label)] = 1
            # Build similarity array
            image_logit = image_preprocess(image)
            label_probs = score_process(image_logit, label_logits)
            # Check if diagram
            len_text = 0
            for info in extract_info:
                if info[2] > 0.3:
                    len_text += len(info[1])
            if len_text > 10:
                print('Diagram Detected')
                plausibility_arr = plausibility_arr * 0.0
                plausibility_arr[len(plausibility_arr)-1] = 1.0
            # Append to matrices
            calib_true_class_arr.append(class_onehot)
            calib_sim_score_arr.append(label_probs)
            calib_plausibility_score_arr.append(plausibility_arr)
            # Update metrics
            avg = avg + plausibility_arr
            # Check if break
            num_images += 1
            if num_images >= NUM_SAMPLES: break
        avg = avg/NUM_SAMPLES
        print(avg[int(label)])
        print(avg[len(avg)-1])
    #Append Matrices
    calib_true_class_arr = torch.vstack(calib_true_class_arr)
    calib_sim_score_arr = torch.vstack(calib_sim_score_arr)
    calib_plausibility_score_arr = torch.vstack(calib_plausibility_score_arr)
    #Save Data Arrays
    torch.save(calib_plausibility_score_arr, RESULTS_DIRECTORY / "calib_plausibility_score_arr")
    torch.save(calib_sim_score_arr, RESULTS_DIRECTORY / "calib_sim_score_arr")
    torch.save(calib_true_class_arr, RESULTS_DIRECTORY / "calib_true_class_arr")
    if GENERATE_DEBUG_CSV:
        calib_debug_df.to_csv(CALIB_OUTPUT_CSV_PATH)
#Generate Test Matrices
#-----------------------------------------------------------------------------------
if not TEST_RELOAD:
    print('Skipping Generating Test Matrices')
else:
    print("Generating Test Matrices")
    test_true_class_arr = []
    test_sim_score_arr = []
    #Loop through image
    for label in os.listdir(TEST_IMAGE_DIRECTORY):
        print("Beginning Test Embedding Generation: {label}".format(label=label))
        num_images = 0
        test_image_list = list(os.listdir(TEST_IMAGE_DIRECTORY / label))
        random.shuffle(test_image_list)
        for img in test_image_list:
            # Build label array
            class_onehot = torch.zeros(len(LABELS.items()))
            if '.' in label:
                label_int = str(int(label.split('.')[0])-1)
            else:
                label_int = label
            class_onehot[int(label_int)] = 1
            # Build similarity array
            try:
                image = Image.open(TEST_IMAGE_DIRECTORY / label / img).convert('RGB')
            except:
                print("IMG load error:", img)
                continue
            image_logit = image_preprocess(image)
            label_probs = score_process(image_logit, label_logits)
            test_true_class_arr.append(class_onehot)
            test_sim_score_arr.append(label_probs)

            if GENERATE_DEBUG_CSV:
                test_debug_dict["filename"].append(str(img))
                test_debug_dict["label"].append(str(label_int))
                test_debug_dict["index"].append(f"{str(label_int)},{str(img)}")
                test_debug_dict["clip_score_label"].append(label_probs[int(label_int)].numpy())
                for i in range(len(LABELS)):
                    test_debug_dict[f"clip_score_{i}"].append(label_probs[i].numpy())

            num_images += 1
            if num_images >= NUM_SAMPLES: break
    #Append Matrices
    test_true_class_arr = torch.vstack(test_true_class_arr)
    test_sim_score_arr = torch.vstack(test_sim_score_arr)
    #Save Data Arrays
    torch.save(test_sim_score_arr, RESULTS_DIRECTORY / "test_sim_score_arr")
    torch.save(test_true_class_arr, RESULTS_DIRECTORY / "test_true_class_arr")

    if GENERATE_DEBUG_CSV:
        pd.DataFrame(test_debug_dict).set_index("index").sort_values(["label", "index"]).to_csv(TEST_DEBUG_CSV_PATH)

#Perform Conformal Prediction
#-----------------------------------------------------------------------------------
calib_plausibility_score_arr =  torch.load(RESULTS_DIRECTORY / "calib_plausibility_score_arr")
calib_sim_score_arr = torch.load(RESULTS_DIRECTORY / "calib_sim_score_arr")
calib_true_class_arr = torch.load(RESULTS_DIRECTORY / "calib_true_class_arr")
test_sim_score_arr = torch.load(RESULTS_DIRECTORY / "test_sim_score_arr")
test_true_class_arr = torch.load(RESULTS_DIRECTORY / "test_true_class_arr")
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
