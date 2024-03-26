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

dataset = 'caltech256'
source = 'google'
version = '2'

folder_name = source + "_" + dataset + "_" + version
CONTEXT_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\" + dataset + "\\" + folder_name + "_caption-results")
IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\" + dataset + "\\" + folder_name + "_plausibilities")
CALIB_IMAGE_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\" + dataset + "\\" + folder_name)
os.makedirs(IMAGE_PLAUSIBILITIES, exist_ok=True)
if dataset == 'medmnist':
    LABELS = MEDMNIST_CLASSES
    PSEUDO_LABELS = MEDMNIST_GENERIC_CLASSES
elif dataset == 'fitzpatrick17k':
    LABELS = FITZ17K_CLASSES
    PSEUDO_LABELS = FITZ17K_GENERIC_CLASSES
elif dataset == 'oxford-pets':
    LABELS = PETS_CLASSES
    PSEUDO_LABELS = PETS_GENERIC_CLASSES
elif dataset == 'imagenet':
    LABELS = IMAGENET_CLASSES
    PSEUDO_LABELS = IMAGENET_GENERIC_CLASSES
elif dataset == "caltech256":
    LABELS = CALTECH256_CLASSES
    PSEUDO_LABELS = CALTECH256_GENERIC_CLASSES
else:
    LABELS = None

# Model Initialization
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') #'sentence-transformers/all-mpnet-base-v2') 'sentence-transformers/all-MiniLM-L6-v2') 'sentence-transformers/msmarco-bert-base-dot-v5')
#classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0) #"valhalla/distilbart-mnli-12-1" "facebook/bart-large-mnli"
# Encode Labels
#label_embed = model.encode([label for label in LABELS.values()])
labels = [label.split(',')[0] for label in LABELS.values()]
label_embeddings = torch.tensor(model.encode(labels))
print(labels)
os.makedirs(IMAGE_PLAUSIBILITIES, exist_ok=True)
#pseudo_embed = model.encode([label for label in PSEUDO_LABELS.values()])
# Loop through caption folders
for label in os.listdir(CONTEXT_DIRECTORY):
    if label.endswith("events.log"): 
        continue
    print("Beginning Score Generation: {label}".format(label=label))
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
        #print(str(CONTEXT_DIRECTORY / label / file))
        captions = pickle.load(open(CONTEXT_DIRECTORY / label / (file.split('.')[0]+'.pkl'), 'rb'))
        if len(captions) <= 1: 
            print('ERROR # CAPTIONS:' + str(captions))
            continue
        # Main Score 
        #main_score = classifier(title, list(LABELS.values()))
        #main_score = scores_converter(main_score, list(LABELS.values()))
        #main_score = torch.tensor(main_score)
        main_embeddings = torch.tensor(model.encode(title))
        main_score = main_embeddings @ label_embeddings.T
        # Second Score
        second_score = []
        second_search = captions[0:min(10, len(captions))]
        label_set = list(set(PSEUDO_LABELS.values()))
        second_embeddings = torch.tensor(model.encode(second_search))
        second_score = second_embeddings @ label_embeddings.T
        #for caption in second_search:
        #    score_dict = classifier(caption, label_set, multi_label=True)
        #    score = scores_converter(score_dict, list(PSEUDO_LABELS.values()))
        #    second_score.append(score)
        #second_score = [torch.tensor(score) for score in second_score]
        #second_score = torch.stack(second_score)
        torch.save(main_score, IMAGE_PLAUSIBILITIES / label / (file.split(".")[0] + '_main'))
        torch.save(second_score, IMAGE_PLAUSIBILITIES / label / (file.split(".")[0] + '_second'))
        n += 1
        if n >= 10: break