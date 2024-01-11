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
from utils.imagenet_classes import IMAGENET2012_CLASSES

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA ENABLED: {}".format(str(torch.cuda.is_available())))

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def sliding_window(tokens):
    tokens = tokens.unfold(dimension=1, size=512, step=256)
    out = model(**tokens)
    embed = mean_pooling(out, tokens['attention_mask'])

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
    DATASET = 'OxfordPets'
if True:
    CONTEXT_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\fitzpatrick17k\\web_scraping_0105_selenium_reverse-image-selenium_fitz-17k_caption-results")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\fitzpatrick17k\\web_scraping_0105_selenium_reverse-image-selenium_fitz-17k_plausibilities")
    DATASET = 'FitzPatrick17k'
if False:
    CONTEXT_DIRECTORY = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\medmnist\\web_scraping_1225_reverse-image-selenium_medmnist_caption-results")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\medmnist\\web_scraping_1225_reverse-image-selenium_medmnist_plausibilities")
    DATASET = 'MedMNIST'

if DATASET == 'MedMNIST':
    LABELS = MEDMNIST_CLASSES
    PSEUDO_LABELS = MEDMNIST_GENERIC_CLASSES
elif DATASET == 'FitzPatrick17k':
    LABELS = FITZ17K_CLASSES
    PSEUDO_LABELS = FITZ17K_GENERIC_CLASSES
elif DATASET == 'OxfordPets':
    LABELS = PETS_CLASSES
    PSEUDO_LABELS = PETS_GENERIC_CLASSES
else:
    LABELS = None

# Model Initialization
#model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') #'sentence-transformers/all-mpnet-base-v2') 'sentence-transformers/all-MiniLM-L6-v2') 'sentence-transformers/msmarco-bert-base-dot-v5')
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1", device=0) #"valhalla/distilbart-mnli-12-1" "facebook/bart-large-mnli"
# Encode Labels
#label_embed = model.encode([label for label in LABELS.values()])
#pseudo_embed = model.encode([label for label in PSEUDO_LABELS.values()])
# Loop through caption folders
for label in os.listdir(CONTEXT_DIRECTORY):
    print("Beginning Plausibility Generation: {label}".format(label=label))
    os.makedirs(IMAGE_PLAUSIBILITIES / label, exist_ok=True)
    # Loop through image captions
    avg = 0.0
    avg2 = 0.0
    avgn = 0
    for file in os.listdir(CONTEXT_DIRECTORY / label):
        # Load captions 
        print('a')
        if file.endswith("_debug.pkl"): continue
        captions = pickle.load(open(CONTEXT_DIRECTORY / label / file, 'rb'))
        if len(captions) <= 1: continue
        # Calculate embedding of main caption
        #main_embed = model.encode(captions[0])
        # Calculate softmax dot product between embedding and list of labels
        if True:
            main_score = classifier(captions[0], list(LABELS.values()))
            main_score = scores_converter(main_score, list(LABELS.values()))
            main_score = torch.tensor(main_score)
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
            second_score = []
            second_search = captions[1:]
            for caption in second_search:
                score = classifier(caption, list(set(PSEUDO_LABELS.values())), multi_label=True)
                score = scores_converter(score, list(PSEUDO_LABELS.values()))
                second_score.append(score)
            second_score = [torch.tensor(score) for score in second_score]
            second_score = torch.stack(second_score)
            second_score = torch.mean(second_score, dim=0)
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
        avg += scores[int(label)]
        avg2 += junk_prob
        scores = torch.cat([scores, torch.tensor([junk_prob])])
        scores[scores < 0.0] = 0.0
        #avg += scores[int(label)]
        avgn += 1
        if avgn >= 10: break
        '''print(LABELS["0"])
        print(captions[0])
        print(main_score)
        print(captions[1:])
        print(second_score)
        print(scores)
        exit()'''
        torch.save(scores, IMAGE_PLAUSIBILITIES / label / (str(int(file.split(".")[0]))))
    print(avg/avgn)
    print(avg2/avgn)