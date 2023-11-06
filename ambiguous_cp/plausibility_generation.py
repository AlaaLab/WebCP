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

script_path = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
base_path = script_path.parent.absolute()
sys.path.append(base_path + '\\cp')
sys.path.append(base_path + '\\utils')
from pets_classes import PETS_CLASSES, PETS_GENERIC_CLASSES
from fitz17k_classes import FITZ17K_CLASSES, FITZ17K_GENERIC_CLASSES
from medmnist_classes import MEDMNIST_CLASSES, MEDMNIST_GENERIC_CLASSES

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA ENABLED: {}".format(str(torch.cuda.is_available())))

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, help='Experiment in experiment_configs to run')
args = parser.parse_args()

# Parameters
reader = open(base_path + "\\experiment_configs\\"  + args.exp)
config = json.load(reader)
CONTEXT_DIRECTORY = config["context_directory"]
IMAGE_PLAUSIBILITIES = config["intermediate_data_directory"]
if config["dataset"] == 'MedMNIST':
    LABELS = MEDMNIST_GENERIC_CLASSES
elif config["dataset"] == 'FitzPatrick17k':
    LABELS = FITZ17K_GENERIC_CLASSES
else:
    LABELS = None

# Prompt Engineering
junk_labels = [
    "an image",
    "an image with a lot of text",
    "an image of a diagram",
    "an image of a graph",
    "an image of a drawing"
]
content_labels = ["an image primarily of {label}".format(label=label) for label in GENERIC_CLASSES.values()]

#Methods
def openclip_text_preprocess(text):
    text = tokenizer(text).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_logits = model.encode_text(text)
        text_logits /= text_logits.norm(dim=-1, keepdim=True)
    return text_logits.to("cpu")

def openclip_process(image_logits, text_logits, temp = 100.0):
    image_logits, text_logits = image_logits.type(torch.float32), text_logits.type(torch.float32)
    logits = (temp * image_logits @ text_logits.T).softmax(dim=-1)[0]
    return logits

#Model Initialization
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg')
model.to(device)

# Calculate Plausibilities
print("Preprocess Text Prompts")
junk_logits = openclip_text_preprocess(junk_labels)
labels_logits = openclip_text_preprocess(content_labels)
print("Begin Plausibility Generation")
avg2 = 0.0
for label in os.listdir(CONTEXT_DIRECTORY):
    print("Beginning Plausibility Generation: {label}".format(label=label))
    # Context Probs
    context_csv = pd.read_csv(CONTEXT_DIRECTORY / str(label) / "scores.csv")
    context_list = context_csv.to_numpy()
    n = context_list.shape[0]
    for i in range(0, n):
        # Preparation
        context_entry = context_list[i]
        id, context_vals = context_entry[1], context_entry[2:]
        context_probs = sp.softmax(context_vals)
        image_logit = torch.load(IMAGE_PLAUSIBILITIES / str(label) / (str(int(id))+"_encoding"))
        # Type Alignment
        junk_probs = openclip_process(image_logit, junk_logits, 1000.0)
        context_probs = junk_probs[0]*context_probs
        # Label Alignment
        for j in range(0, labels_logits.shape[0]):
            binary_prob = openclip_process(image_logit, torch.vstack((labels_logits[j], junk_logits[0])))
            context_probs[j] *= binary_prob[0]
        junk_prob = 1.0 - torch.sum(context_probs)
        # Content Probs
        context_probs = torch.cat([context_probs, torch.tensor([junk_prob])])
        torch.save(context_probs, IMAGE_PLAUSIBILITIES / label / (str(int(id))+"_plausibilities"))














