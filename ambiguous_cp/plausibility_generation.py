import os
from pathlib import Path
import pandas as pd
import scipy.special as sp
import torch
import open_clip
import argparse
import yaml

# Methods


def openclip_text_preprocess(text):
    text = tokenizer(text).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_logits = model.encode_text(text)
        text_logits /= text_logits.norm(dim=-1, keepdim=True)
    return text_logits.to("cpu")


def openclip_process(image_logits, text_logits, temp=100.0):
    image_logits, text_logits = image_logits.type(
        torch.float32), text_logits.type(torch.float32)
    logits = (temp * image_logits @ text_logits.T).softmax(dim=-1)[0]
    return logits


if __name__ == "__main__":
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

    CONTEXT_DIRECTORY = config["context_dir"]
    IMAGE_PLAUSIBILITIES = config["intermediate_data_dir"]

    generic_class_dict = {}
    class_df = pd.read_csv(config['class_list_csv'])

    for i in range(len(class_df)):
        row = class_df.iloc[i, :]
        generic_class_dict[row["Class Index"]] = row["Generic Class"]

    # Prompt Engineering
    junk_labels = config['junk_labels']
    content_labels = ["an image primarily of {label}".format(
        label=label) for label in generic_class_dict.values()]

    # Model Initialization
    model, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg')
    tokenizer = open_clip.get_tokenizer(
        'hf-hub:laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg')
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
        context_csv = pd.read_csv(
            CONTEXT_DIRECTORY / str(label) / "scores.csv")
        context_list = context_csv.to_numpy()
        n = context_list.shape[0]
        for i in range(0, n):
            # Preparation
            context_entry = context_list[i]
            id, context_vals = context_entry[1], context_entry[2:]
            context_probs = sp.softmax(context_vals)
            image_logit = torch.load(
                IMAGE_PLAUSIBILITIES / str(label) / (str(int(id))+"_encoding"))
            # Type Alignment
            junk_probs = openclip_process(image_logit, junk_logits, 1000.0)
            context_probs = junk_probs[0]*context_probs
            # Label Alignment
            for j in range(0, labels_logits.shape[0]):
                binary_prob = openclip_process(
                    image_logit, torch.vstack((labels_logits[j], junk_logits[0])))
                context_probs[j] *= binary_prob[0]
            junk_prob = 1.0 - torch.sum(context_probs)
            # Content Probs
            context_probs = torch.cat(
                [context_probs, torch.tensor([junk_prob])])
            torch.save(context_probs, IMAGE_PLAUSIBILITIES /
                       label / (str(int(id))+"_plausibilities"))
