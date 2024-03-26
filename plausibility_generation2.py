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
IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\" + dataset + "\\" + folder_name + "_plausibilities")
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

GENERATE_DEBUG_CSV = True
if GENERATE_DEBUG_CSV:
    DEBUG_CSV_PATH = Path("~/reesearch/debug.csv")
    debug_header_list = ["index", "label", "filename", "main_caption", "secondary_captions", "score_junk", "score_label", "score_nonlabels"] + [f"score_{i}" for i in range(len(LABELS))] + [f"first_score_{i}" for i in range(len(LABELS))] + [f"second_score_{i}" for i in range(len(LABELS))]# + [f"clip_score_label"] + [f"clip_score_{i}" for i in range(len(LABELS))]
    debug_dict = {k: [] for k in debug_header_list}
torch.set_grad_enabled(False)
# Model Initialization
#model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') #'sentence-transformers/all-mpnet-base-v2') 'sentence-transformers/all-MiniLM-L6-v2') 'sentence-transformers/msmarco-bert-base-dot-v5')
#classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1", device=0) #"valhalla/distilbart-mnli-12-1" "facebook/bart-large-mnli"
# Encode Labels
#label_embed = model.encode([label for label in LABELS.values()])
#pseudo_embed = model.encode([label for label in PSEUDO_LABELS.values()])
# Loop through caption folders
for label in os.listdir(IMAGE_PLAUSIBILITIES):

    print("Beginning Plausibility Generation: {label}".format(label=label))
    os.makedirs(IMAGE_PLAUSIBILITIES / label, exist_ok=True)
    # Loop through image captions
    avg = 0.0
    avg2 = 0.0
    avgn = 0
    for file in os.listdir(IMAGE_PLAUSIBILITIES / label):
        # Load captions 
        if not file.endswith("_main"): continue
        name = file.split("_")[0]
        main_score = torch.load(IMAGE_PLAUSIBILITIES / label / (name+'_main'))
        second_score = torch.load(IMAGE_PLAUSIBILITIES / label / (name+'_second'))
        # Calculate embedding of main caption
        #main_embed = model.encode(captions[0])
        # Calculate softmax dot product between embedding and list of labels
        if True:
            main_score = main_score
            #print(main_score[0])
            main_score = torch.nn.functional.softmax(main_score*100.0)
        if False:
            main_score = torch.from_numpy(label_embed @ main_embed)
            main_score = torch.nn.functional.softmax(main_score*200.0)
        if False:
            main_score = torch.from_numpy(label_embed @ main_embed)
            #print(main_score)
            leftover = 1 - main_score[int(label)]
            leftover_total = torch.sum(main_score[:int(label)]) + torch.sum(main_score[int(label)+1:])
            main_score[:int(label)] *= (leftover/leftover_total)
            main_score[int(label)+1:] *= (leftover/leftover_total)
            main_score = torch.nn.functional.softmax(main_score*10.0)
        # Calculate embeddings of secondary captions
        #second_embed = model.encode(captions[1:])
        # Calculate softmax average dot product between embedding and list of pseudo-labels
        if True:
            #second_score = second_score[1:]
            #maxitem = torch.argmax(second_score, dim=1)
            #second_score=second_score*0.0
            #for i in range(len(maxitem)): second_score[i][maxitem[i]]=1.0
            second_score = torch.mean(second_score, dim=0)
            #topx = max(1, int(0.2*second_score.shape[0]))
            #second_score, _ = torch.sort(second_score, dim=0)
            #second_score = torch.mean(second_score[-1*topx:], dim=0)
        if False:
            second_score = torch.from_numpy(pseudo_embed @ second_embed.T)
            #second_score, _ = torch.sort(second_score, dim=1)
            #topx = max(1, int(0.3*second_score.shape[1]))
            #second_score = torch.mean(second_score[:,-1*topx:], dim=1)
            second_score = torch.mean(second_score, dim=1)
            #second_score /= 0.5
            second_score[second_score > 0.4] = 1.0
            avg += second_score[int(label)]
        if False:
            second_main_score = torch.from_numpy(second_embed @ label_embed[int(label)])
            second_score = torch.from_numpy(pseudo_embed @ second_embed.T)
            print(second_score[int(label)])
            #second_score[int(label)] = torch.maximum(second_score[int(label)], second_main_score)
            #second_score[int(label)] = 1 - ((1-second_score[int(label)]) * (1-second_main_score))
            topx = max(1, int(1.0*second_score.shape[1]))
            second_score, _ = torch.sort(second_score, dim=1)
            second_main_score, _ = torch.sort(second_main_score, dim=0)
            second_main_score = torch.mean(second_main_score[-1*topx:])
            second_score = torch.mean(second_score[:,-1*topx:], dim=1)
            second_score[int(label)] = 1 - ((1-second_score[int(label)]) * (1-second_main_score))
            avg += second_score[int(label)]
            print(second_score)
            print('------------------------------------')
            #topx = max(1, int(0.3*second_score.shape[1]))
            #second_score = torch.mean(second_score, dim=1)
            #second_score /= 0.5
            #second_score[second_score > 1] = 1.0
        # Calculate final plausibility and store
        scores = torch.mul(main_score, second_score)
        junk_prob = 1.0 - torch.sum(scores)
        avg2 += main_score[int(label)]
        avg += scores[int(label)]
        scores = torch.cat([scores, torch.tensor([junk_prob])])
        scores[scores < 0.0] = 0.0
        #if junk_prob > 0.95:
        #    scores = scores * 0.0
        #    scores[len(scores)-1] = 1.0
        #avg += scores[int(label)]
        #print(scores[int(label)], second_score[int(label)], torch.argmax(scores))
        '''print(LABELS["0"])
        print(title)
        print(main_score[0])
        print(captions[1:])
        print(scores)
        print('------------------------------')'''
        torch.save(scores, IMAGE_PLAUSIBILITIES / label / (file.split(".")[0]+'_plausibilities'))
        avgn += 1
        if avgn >= 10: break
    print(avg/avgn)
    print(avg2/avgn)