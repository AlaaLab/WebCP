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
import argparse

# Scan environment
script_path = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
base_path = script_path.parent.absolute()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA ENABLED: {}".format(str(torch.cuda.is_available())))

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, help='Experiment in experiment_configs to run')
args = parser.parse_args()

# Load Config
reader = open(base_path + "\\experiment_configs\\"  + args.exp)
config = json.load(reader)
CALIB_IMAGE_DIRECTORY = config["calib_image_directory"]
IMAGE_LOGITS = config["intermediate_data_directory"]
PLAUSIBILITY_CHECKPOINT = config["plausibility_checkpoint"]

def openclip_image_preprocess(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_logits = model.encode_image(image)
        image_logits /= image_logits.norm(dim=-1, keepdim=True)
    return image_logits.to("cpu")

model, _, preprocess = open_clip.create_model_and_transforms(PLAUSIBILITY_CHECKPOINT)
tokenizer = open_clip.get_tokenizer(PLAUSIBILITY_CHECKPOINT)
model.to(device)

print("Begin Image Encoding")
for label in os.listdir(CALIB_IMAGE_DIRECTORY):
    print("Beginning Encoding for Label: {label}".format(label=label))
    os.makedirs(IMAGE_LOGITS / label, exist_ok=True)
    for img in os.listdir(CALIB_IMAGE_DIRECTORY / label):
        image = Image.open(CALIB_IMAGE_DIRECTORY / label / img)
        image_logit = openclip_image_preprocess(image)
        torch.save(image_logit, IMAGE_LOGITS / label / (img.split('.')[0]+"_encoding"))