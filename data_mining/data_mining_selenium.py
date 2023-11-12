from math import ceil
import os
import pandas as pd
import numpy as np
from pathlib import Path
from time import sleep
from googleapiclient.errors import HttpError
import json
import shutil
import logging
import sys
import requests
import os
import pandas as pd
import numpy as np
from pathlib import Path
from time import sleep
import json
import shutil
import logging
import sys
import threading
from PIL import Image
import requests
from io import BytesIO
import traceback
from requests.exceptions import ConnectTimeout
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from urllib.parse import urlparse
from googleapiclient.discovery import build
import pickle
from image_caption_scraper import Image_Caption_Scraper
from data_mining_utils import process_image
import yaml
import argparse 
from multiprocessing import Pool
import threading

def selenium_process(ls):
    class_id, class_name, config = ls

    this_res_dir = config['results_store_dir'] / f"{class_id}"
    this_res_dir.mkdir(exist_ok=True)
    logging.basicConfig(filename=this_res_dir/f"events.log", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode="w")
    logger = logging.getLogger("my-logger")
    logger.setLevel(logging.INFO)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info(f"Class: {class_id}, {class_name}")

    try:
        scraper = Image_Caption_Scraper(
                engine="google", # or "google", "yahoo", "flickr"
                num_images=config['set_size'],
                query=class_name,
                out_dir=str(this_res_dir),
                headless=False,
                driver="~/Downloads/chromedriver",
                expand=False,
                # k=3
            )

        scraper.scrape(save_images=True)
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
        if (k in ['results_store_dir', 'calibration_dataset_dir']):
            config[k] = Path(v)

    READY = np.array([0], dtype='i')
    TERMINATE = np.array([-1], dtype='i')

    config['results_store_dir'].mkdir(exist_ok=True)

    logging.basicConfig(filename=config['results_store_dir']/f"events.log", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode="w")
    logger = logging.getLogger("my-logger")
    logger.setLevel(logging.INFO)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    API_KEY = os.environ["GOOGLE_API_KEY"]
    PROJECT_CX_KEY = os.environ["GOOGLE_CX_ID"]

    class_dict = {}
    class_df = pd.read_csv(config['class_list_csv'])
    
    for i in range(len(class_df)):
        row = class_df.iloc[i, :]
        class_dict[row["Class Index"]] = row["Class"]


    request_list = [(class_id, class_name, config) for class_id, class_name in class_dict.items() if (class_id not in config['class_id_exclude_list'] and (True if config['class_id_start'] is None else config['class_id_start'] <= class_id) and (True if config['class_id_end'] is None else class_id < config['class_id_end']))]
    print(request_list)

    scraper = Image_Caption_Scraper(
        engine="google", # or "google", "yahoo", "flickr"
        num_images=config['set_size'],
        query=request_list[0][1],
        out_dir=str(request_list[0][1]),
        headless=False,
        driver="~/Downloads/chromedriver",
        expand=False,
        # k=3
    )


    scraper.scrape(save_images=True)
    scraper.close()

    with Pool(processes=config['num_threads']) as executor:
        executor.map(selenium_process, request_list)
        
if __name__ == "__main__":
    main()