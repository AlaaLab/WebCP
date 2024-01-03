import argparse
import yaml
import pandas as pd
import numpy as np 
from pathlib import Path
import logging
from google.cloud import vision
import pickle
import concurrent.futures
import traceback
def annotate(args:tuple) -> vision.WebDetection:
    """Returns web annotations given the path to an image.

    Args:
        path: path to the input image.

    Returns:
        An WebDetection object with relevant information of the
        image from the internet (i.e., the annotations).
    """
    try: 
        this_class_store, file_idx, path, logger = args

        client = vision.ImageAnnotatorClient()

        if path.startswith("http") or path.startswith("gs:"):
            image = vision.Image()
            image.source.image_uri = path

        else:
            with open(path, "rb") as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

        web_detection = client.web_detection(image=image).web_detection

        with open(this_class_store / f"{file_idx}.web_detection", "wb") as this_res_file:
            pickle.dump(web_detection, this_res_file)
    except Exception as e:
        logger.error(traceback.format_exc())

def get_fileidx_list(dataset_subfolder):
    idxSet = set()
    for filePath in dataset_subfolder.iterdir():
        first = str(filePath).rindex("/")+1
        last = str(filePath).rindex(".")
        fileIdx = int(str(str(filePath)[first:last]))
        idxSet.add(fileIdx)

    res = list(idxSet)
    res.sort()
    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", help="the path to the yaml config file.", type=str)
    args = parser.parse_args()

    config = {}
    with open(args.config, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    for k, v in config.items():
        if (k[-4:] == '_dir'):
            config[k] = Path(v)

    config['web_detection_store_dir'].mkdir(exist_ok=False)

    class_df = pd.read_csv(config['class_list_csv'])


    logging.basicConfig(filename=config['web_detection_store_dir']/f"events.log",
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode="w")
    logger = logging.getLogger("my-logger")
    logger.setLevel(logging.INFO)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info("start process")
    class_dict = {}
    for i in range(len(class_df)):
        row = class_df.iloc[i, :]
        class_dict[row["Class Index"]] = row["Class"]

    request_list = []
    for class_idx, class_name in class_dict.items():
        # print("CLASS " + str(class_idx))
        this_class_store = (config['web_detection_store_dir'] / f"{class_idx}")
        this_class_store.mkdir(exist_ok=True)

        this_dataset_store = (
            config['scraping_store_dir'] / f"{class_idx}")
        assert this_dataset_store.exists()

        for file_idx in get_fileidx_list(this_dataset_store):
            # print(file_idx)
            url = open(this_dataset_store /
                           f"{file_idx}.image-url", "r").read()

            request_list.append((this_class_store, file_idx, url, logger))

    with concurrent.futures.ThreadPoolExecutor(max_workers=config['num_threads']) as executor:
        executor.map(annotate, request_list)