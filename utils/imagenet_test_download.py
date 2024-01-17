import sys

import json
import os
from pathlib import Path
import pickle
from PIL import Image

from datasets import load_dataset
from huggingface_hub import login

DESTINATION_DATASET_DIR = Path("C:\\Documents\\Alaa Lab\\CP-CLIP\\datasets2\\imagenet\\imagenet_2012")
SIZE = 25000
RANDOM_SEED = 420

# Access Dataset
login(token="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")
data_files = {"val": "https://huggingface.co/datasets/imagenet-1k/resolve/1500f8c59b214ce459c0a593fa1c87993aeb7700/data/val_images.tar.gz"}
dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
dataset = dataset.shuffle(RANDOM_SEED, buffer_size=2500)

# Iterate Through Dataset
print("Extracting Image Info")
at = 0
minimum = 100000
for testCase in dataset:
    label = int(testCase['label'])
    minimum = min(minimum, label)
    os.makedirs(DESTINATION_DATASET_DIR / str(label), exist_ok=True)
    testCase['image'].save(DESTINATION_DATASET_DIR / str(label) / (str(at)+'.jpg'))
    at+=1
    if at%100==0: print("Extraction at Instance: ", str(at))
    if at==SIZE: break
print(minimum)
