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
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\oxford-pets\\web_scraping_0105_selenium_reverse-image-selenium_oxford-pets_plausibilities")
    DATASET = 'OxfordPets'
if False:
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\fitzpatrick17k\\web_scraping_0105_selenium_reverse-image-selenium_fitz-17k_plausibilities")
    DATASET = 'FitzPatrick17k'
if False:
    #IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\medmnist\\web_scraping_1225_reverse-image-selenium_medmnist_plausibilities")
    IMAGE_PLAUSIBILITIES = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\medmnist\\web_scraping_0114_reverse-image-selenium_medmnist_new-plausibilities")
    DATASET = 'MedMNIST'
if True:
    IMAGE_PLAUSIBILITIES = Path("/home/hwei/reesearch/datasets/web_scraping_0114_selenium_reverse-image-search-selenium_NEW-CAPTION-METHOD_imagenet_25size_plausibilities")
    CAPTION_DIR = Path("/home/hwei/reesearch/datasets/web_scraping_0114_selenium_reverse-image-search-selenium_NEW-CAPTION-METHOD_imagenet_25size_caption-results")
    IMAGE_DIR = Path("/home/hwei/reesearch/datasets/web_scraping_0114_selenium_reverse-image-search-selenium_NEW-CAPTION-METHOD_imagenet_25size")
    DATASET =  'ImageNet'
if False:
    IMAGE_PLAUSIBILITIES = Path("/home/hwei/reesearch/datasets/web_scraping_0114_selenium_reverse-search-selenium_caltech-256_plausibilities")
    CAPTION_DIR = Path("/home/hwei/reesearch/datasets/web_scraping_0114_selenium_reverse-image-search-selenium_caltech-256_25size_caption-results")
    IMAGE_DIR = Path("/home/hwei/reesearch/datasets/web_scraping_0114_selenium_reverse-image-search-selenium_caltech-256_25size")
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
elif DATASET == 'Caltech256':
    LABELS = CALTECH256_CLASSES
    PSEUDO_LABELS = CALTECH256_GENERIC_CLASSES
else:
    LABELS = None

GENERATE_DEBUG_CSV = True
if GENERATE_DEBUG_CSV:
    DEBUG_CSV_PATH = Path("~/reesearch/debug_imagenet.csv")
    debug_header_list = ["index", "label", "filename", "main_caption", "secondary_captions", "score_junk", "score_label", "score_nonlabels", "first_score_label", "second_score_label"] + [f"score_{i}" for i in range(len(LABELS))] + [f"first_score_{i}" for i in range(len(LABELS))] + [f"second_score_{i}" for i in range(len(LABELS))]# + [f"clip_score_label"] + [f"clip_score_{i}" for i in range(len(LABELS))]
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
    if str(label) < "550": 
        continue
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
            main_score = torch.nn.functional.softmax(main_score*500.0)
            #print(main_score)
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
        if GENERATE_DEBUG_CSV:
            debug_dict["index"].append(f"{int(str(label))},{int(str(file)[0:str(file).rindex('_')])}")
            debug_dict["label"].append(int(str(label)))
            debug_dict["filename"].append(str(int(str(file)[0:str(file).rindex('_')])))
            debug_dict["score_junk"].append(junk_prob.numpy())
            debug_dict["score_label"].append(scores[int(str(label))].numpy())
            debug_dict["score_nonlabels"].append(torch.sum(scores[:-1]).numpy() - scores[int(str(label))].numpy())
            debug_dict["first_score_label"].append(main_score[int(str(label))].numpy())
            debug_dict["second_score_label"].append(second_score[int(str(label))].numpy())
            try:
                with open(IMAGE_DIR / label / f"{int(str(file)[0:str(file).rindex('_')])}.caption", "r") as main_caption_file:
                    debug_dict["main_caption"].append("\n".join([line.rstrip() for line in main_caption_file]))
            except Exception as e:
                debug_dict["main_caption"].append("ERROR")
            
            try:
                with open(CAPTION_DIR / label / f"{int(str(file)[0:str(file).rindex('_')])}.pkl", 'rb') as sec_caption_file:
                    sec_caption_list = (pickle.load(sec_caption_file))
                    debug_dict["secondary_captions"].append("\n".join(sec_caption_list))
            except Exception as e:
                debug_dict["secondary_captions"].append("ERROR")
            for i in range(len(LABELS)):
                debug_dict[f"score_{i}"].append(scores[i].numpy())
                debug_dict[f"first_score_{i}"].append(main_score[i].numpy())
                debug_dict[f"second_score_{i}"].append(second_score[i].numpy())


        torch.save(scores, IMAGE_PLAUSIBILITIES / label / (file.split(".")[0]+'_plausibilities'))
        avgn += 1
        if avgn >= 10: break
    print(avg/avgn)
    print(avg2/avgn)

if GENERATE_DEBUG_CSV:
    pd.DataFrame(debug_dict).set_index("index").sort_values(by=['label', 'filename']).to_csv(DEBUG_CSV_PATH)