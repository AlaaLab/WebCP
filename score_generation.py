import sys
import os
from pathlib import Path
import pandas as pd
import scipy.special as sp
import numpy as np
from PIL import Image
import torch
import open_clip
import json
import argparse
import pickle
import traceback
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

script_path = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
base_path = script_path.parent.absolute()
sys.path.append(base_path / 'cp')
sys.path.append(base_path / 'utils')
from utils.pets_classes import PETS_CLASSES, PETS_GENERIC_CLASSES
from utils.fitz17k_classes import FITZ17K_CLASSES, FITZ17K_GENERIC_CLASSES
from utils.medmnist_classes import MEDMNIST_CLASSES, MEDMNIST_GENERIC_CLASSES
from utils.imagenet_classes import IMAGENET_CLASSES, IMAGENET_GENERIC_CLASSES
from utils.caltech256_classes import CALTECH256_CLASSES, CALTECH256_GENERIC_CLASSES

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA ENABLED: {}".format(str(torch.cuda.is_available())))

def scores_converter(scores, labels):
    dict_scores = {}
    n = len(scores['labels'])
    for i in range(0, n):
        label = scores['labels'][i]
        dict_scores[label] = scores['scores'][i]
    score_vals = []
    for label in labels:
        score_vals.append(dict_scores[label])
    return score_vals


# Parameters
if False:
    CONTEXT_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\oxford-pets\\web_scraping_0105_selenium_reverse-image-selenium_oxford-pets_caption-results")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\oxford-pets\\web_scraping_0105_selenium_reverse-image-selenium_oxford-pets_plausibilities")
    CALIB_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\oxford-pets\\web_scraping_0105_selenium_reverse-image-selenium_oxford-pets")
    DATASET = 'OxfordPets'
if False:
    CONTEXT_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\fitzpatrick17k\\web_scraping_0105_selenium_reverse-image-selenium_fitz-17k_caption-results")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\fitzpatrick17k\\web_scraping_0105_selenium_reverse-image-selenium_fitz-17k_plausibilities")
    CALIB_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\fitzpatrick17k\\web_scraping_0105_selenium_reverse-image-selenium_fitz-17k")
    DATASET = 'FitzPatrick17k'
if True:
    CONTEXT_DIRECTORY = Path("/home/hwei/reesearch/datasets/web_scraping_0114_selenium_reverse-image-search-selenium_NEW-CAPTION-METHOD_medmnist_25size_caption-results")
    IMAGE_PLAUSIBILITIES = Path("/home/hwei/reesearch/datasets/web_scraping_0114_selenium_reverse-image-search-selenium_NEW-CAPTION-METHOD_medmnist_25size_plausibilities")
    CALIB_IMAGE_DIRECTORY = Path("/home/hwei/reesearch/datasets/web_scraping_0114_selenium_reverse-image-search-selenium_NEW-CAPTION-METHOD_medmnist_25size")
    DATASET = 'MedMNIST'
if True:
    CONTEXT_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\imagenet\\web_scraping_0114_selenium_reverse-image-selenium_imagenet_new-caption-results")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\imagenet\\web_scraping_0220_selenium_reverse-image-selenium_imagenet_plausibilities")
    CALIB_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\imagenet\\web_scraping_0114_selenium_reverse-image-search-selenium_imagenet_new")
    #CONTEXT_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\imagenet\\web_scraping_0114_selenium_reverse-image-selenium_imagenet_new-caption-results")
    #IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\imagenet\\web_scraping_0114_selenium_reverse-image-selenium_imagenet_plausibilities")
    #CALIB_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\imagenet\\web_scraping_0114_selenium_reverse-image-search-selenium_imagenet_new")
    DATASET =  'ImageNet'
if False:
    CONTEXT_DIRECTORY = Path("C:\\Users\\Robert Wei\\reesearch\\datasets\\web_scraping_0114_selenium_reverse-image-search-selenium_caltech-256_25size_caption-results")
    IMAGE_PLAUSIBILITIES = Path("C:\\Users\\Robert Wei\\reesearch\\datasets\\web_scraping_0114_selenium_reverse-search-selenium_caltech-256_plausibilities")
    CALIB_IMAGE_DIRECTORY = Path("C:\\Users\\Robert Wei\\reesearch\\datasets\\web_scraping_0114_selenium_reverse-image-search-selenium_caltech-256_25size")
    DATASET =  'Caltech256'

if DATASET == 'MedMNIST':
    LABELS = MEDMNIST_CLASSES
    PSEUDO_LABELS = MEDMNIST_GENERIC_CLASSES
elif DATASET == 'FitzPatrick17k':
    LABELS = FITZ17K_CLASSES
    PSEUDO_LABELS = FITZ17K_GENERIC_CLASSES
elif DATASET == 'OxfordPets':
    LABELS = PETS_CLASSES
    PSEUDO_LABELS = PETS_GENERIC_CLASSES
elif DATASET == 'ImageNet':
    LABELS = IMAGENET_CLASSES
    PSEUDO_LABELS = IMAGENET_GENERIC_CLASSES
elif DATASET == "Caltech256":
    LABELS = CALTECH256_CLASSES
    PSEUDO_LABELS = CALTECH256_GENERIC_CLASSES
else:
    LABELS = None

# Model Initialization
#model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') #'sentence-transformers/all-mpnet-base-v2') 'sentence-transformers/all-MiniLM-L6-v2') 'sentence-transformers/msmarco-bert-base-dot-v5')
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1", device=0) #"valhalla/distilbart-mnli-12-1" "facebook/bart-large-mnli"
# Encode Labels
#label_embed = model.encode([label for label in LABELS.values()])
labels = [label.split(',')[0] for label in LABELS.values()]
print(labels)
#pseudo_embed = model.encode([label for label in PSEUDO_LABELS.values()])
# Loop through caption folders
for label in os.listdir(CONTEXT_DIRECTORY):
    if label.endswith("events.log"): 
        continue
    print("Beginning Score Generation: {label}".format(label=label))
    if label > "110": continue
    os.makedirs(IMAGE_PLAUSIBILITIES / label, exist_ok=True)
    n = 0
    for file in os.listdir(CONTEXT_DIRECTORY / label):
        # Load captions 
        if file.endswith("_debug.pkl") or file.endswith("events.log"): continue
        try:
            with open(CALIB_IMAGE_DIRECTORY / label / (file.split('.')[0]+'.caption'), 'r') as read:
                title = "\n".join([line.rstrip() for line in read])
        except:
            print('ERROR PKL LOAD')
            print(traceback.format_exc())
            continue
        captions = pickle.load(open(CONTEXT_DIRECTORY / label / file, 'rb'))
        if len(captions) <= 1: 
            print('ERROR # CAPTIONS:' + str(captions))
            continue
        # Main Score 
        main_score = classifier(title, list(LABELS.values()))
        main_score = scores_converter(main_score, list(LABELS.values()))
        main_score = torch.tensor(main_score)
        # Second Score
        second_score = []
        second_search = captions[0:min(10, len(captions))]
        label_set = list(set(PSEUDO_LABELS.values()))
        for caption in second_search:
            label_set = label_set + [labels[int(label)]]
            score_dict = classifier(caption, label_set, multi_label=True)
            score = scores_converter(score_dict, list(PSEUDO_LABELS.values()))
            second_score.append(score)
        second_score = [torch.tensor(score) for score in second_score]
        second_score = torch.stack(second_score)
        torch.save(main_score, IMAGE_PLAUSIBILITIES / label / (file.split(".")[0] + '_main'))
        torch.save(second_score, IMAGE_PLAUSIBILITIES / label / (file.split(".")[0] + '_second'))
        print('a')
        n += 1
        if n >= 10: break