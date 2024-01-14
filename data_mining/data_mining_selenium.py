import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
from image_caption_scraper.scraper import Image_Caption_Scraper
import yaml
import argparse
from multiprocessing import Pool


def selenium_process(ls):
    class_id, class_name, config, logger = ls

    this_res_dir = config['scraping_store_dir'] / f"{class_id}"
    this_res_dir.mkdir(exist_ok=True)

    logger.info(f"Class: {class_id}, {class_name}")

    try:
        scraper = Image_Caption_Scraper(
            engine="google",  # or "google", "yahoo", "flickr"
            num_images=config['set_size'],
            query=class_name,
            out_dir=str(this_res_dir),
            headless=True,
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
        if (k[-4:] == '_dir'):
            config[k] = Path(v)

    config['scraping_store_dir'].mkdir(exist_ok=False)

    logging.basicConfig(filename=config['scraping_store_dir']/f"events.log",
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

    request_list = [(class_id, class_name, config, logger) for class_id, class_name in class_dict.items() if (class_id not in config['class_id_exclude_list'] and (
        True if config['class_id_start'] is None else config['class_id_start'] <= class_id) and (True if config['class_id_end'] is None else class_id < config['class_id_end']))]

    with Pool(processes=config['num_selenium_threads']) as executor:
        executor.map(selenium_process, request_list)


if __name__ == "__main__":
    main()
