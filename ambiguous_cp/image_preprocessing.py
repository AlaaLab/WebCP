import os
from pathlib import Path
from PIL import Image
import torch
import open_clip
import argparse
import yaml


def openclip_image_preprocess(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_logits = model.encode_image(image)
        image_logits /= image_logits.norm(dim=-1, keepdim=True)
    return image_logits.to("cpu")


if __name__ == "__main__":
    # Scan environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CUDA ENABLED: {}".format(str(torch.cuda.is_available())))

    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str,
                        help='Experiment in experiment_configs to run')
    args = parser.parse_args()

    # Load Config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", help="the path to the yaml config file.", type=str)
    args = parser.parse_args()
    config = {}
    with open(args.config, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    for k, v in config.items():
        if (k in ['calib_image_dir', 'test_image_dir', 'intermediate_data_dir', 'results_store_dir', 'classification_checkpoint', 'context_dir', 'char_output_dir']):
            config[k] = Path(v)

    CALIB_IMAGE_DIRECTORY = config["calib_image_dir"]
    IMAGE_LOGITS = config["intermediate_data_dir"]
    IMAGE_LOGITS.mkdir(exist_ok=False)
    PLAUSIBILITY_CHECKPOINT = config["plausibility_checkpoint"]

    model, _, preprocess = open_clip.create_model_and_transforms(
        PLAUSIBILITY_CHECKPOINT)
    tokenizer = open_clip.get_tokenizer(PLAUSIBILITY_CHECKPOINT)
    model.to(device)

    print("Begin Image Encoding")
    for label in os.listdir(CALIB_IMAGE_DIRECTORY):
        print("Beginning Encoding for Label: {label}".format(label=label))
        os.makedirs(IMAGE_LOGITS / label, exist_ok=False)
        for img in os.listdir(CALIB_IMAGE_DIRECTORY / label):
            image = Image.open(CALIB_IMAGE_DIRECTORY / label / img)
            image_logit = openclip_image_preprocess(image)
            torch.save(image_logit, IMAGE_LOGITS / label /
                       (img.split('.')[0]+"_encoding"))
