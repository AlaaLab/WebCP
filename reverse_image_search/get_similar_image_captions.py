from scraper import *

import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
import yaml
import pickle
import argparse
from multiprocessing import Pool

def get_fileidx_list(dataset_subfolder):
    idxSet = set()
    for filePath in dataset_subfolder.iterdir():
        if (filePath.is_dir()):
            continue
        first = str(filePath).rindex("\\")+1
        last = str(filePath).rindex(".")
        fileIdx = int(str(str(filePath)[first:last]))
        idxSet.add(fileIdx)

    res = list(idxSet)
    res.sort()
    return res


def selenium_process(ls):
    class_id, image_id, logger, config = ls

    this_res_dir = config['reverse_image_store_dir'] / f"{class_id}"
    this_res_dir.mkdir(exist_ok=True)
    try:
        scraper = Google_Reverse_Image_Search(
            engine="google",  # or "google", "yahoo", "flickr"
            num_images=config['num_similar_captions'],
            headless=True,
            driver="~/Downloads/chromedriver",
            expand=False,
            # k=3
        )

        res_dict = scraper.reverse_image_search(config['scraping_store_dir'] / f"{class_id}" / f"{image_id}.jpeg")

        res_list = [elem['caption'] for _, elem in res_dict.items()]

        with open(this_res_dir / f"{image_id}.pkl", "wb") as this_res_file:
            pickle.dump(res_list, this_res_file)

        # with open(this_res_dir / f"{image_id}_debug.pkl", "wb") as this_res_file:
        #     pickle.dump(res_dict, this_res_file)
        scraper.close()
    except Exception as e:
        logger.error(traceback.format_exc())


def main():
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

    config['reverse_image_store_dir'].mkdir(exist_ok=False)

    logging.basicConfig(filename=config['reverse_image_store_dir']/f"events.log",
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode="w")
    logger = logging.getLogger("my-logger")
    logger.setLevel(logging.INFO)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    class_dict = {}
    class_df = pd.read_csv(config['class_list_csv'])

    for i in range(len(class_df)):
        row = class_df.iloc[i, :]
        class_dict[row["Class Index"]] = row["Class"]

    request_list = []
    for class_idx, _ in class_dict.items():
        # print("CLASS " + str(class_idx))
        this_class_store = (config['reverse_image_store_dir'] / f"{class_idx}")
        this_class_store.mkdir(exist_ok=True)

        this_dataset_store = (
            config['scraping_store_dir'] / f"{class_idx}")
        assert this_dataset_store.exists()

        for file_idx in get_fileidx_list(this_dataset_store):
            request_list.append((class_idx, file_idx, logger, config))

    with Pool(processes=config['num_selenium_threads']) as executor:
        executor.map(selenium_process, request_list)

if __name__ == "__main__":
    main()
