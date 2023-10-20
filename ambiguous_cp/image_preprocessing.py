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

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA ENABLED: {}".format(str(torch.cuda.is_available())))
# Load Config
reader = open("C:\\Documents\\Alaa Lab\\CP-CLIP\\WebCP\\experiment_configs\\google-hybrid_medmnist_09-01-2023.json")
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