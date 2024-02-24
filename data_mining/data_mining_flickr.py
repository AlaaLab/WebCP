# # define search params
# # option for commonly used search param are shown below for easy reference.
# # For param marked with '##':
# #   - Multiselect is currently not feasible. Choose ONE option only
# #   - This param can also be omitted from _search_params if you do not wish to define any value
# _search_params = {
#     'q': '...',
#     'num': 10,
#     'fileType': 'jpg|gif|png',
#     'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived',
#     'safe': 'active|high|medium|off|safeUndefined', ##
#     'imgType': 'clipart|face|lineart|stock|photo|animated|imgTypeUndefined', ##
#     'imgSize': 'huge|icon|large|medium|small|xlarge|xxlarge|imgSizeUndefined', ##
#     'imgDominantColor': 'black|blue|brown|gray|green|orange|pink|purple|red|teal|white|yellow|imgDominantColorUndefined', ##
#     'imgColorType': 'color|gray|mono|trans|imgColorTypeUndefined' ##
# }

import os
import pandas as pd
import numpy as np
from pathlib import Path
from time import sleep
import json
import shutil
import logging
import sys
from mpi4py import MPI
import flickrapi
import threading
from PIL import Image
import requests
from io import BytesIO
import concurrent.futures
import traceback
from data_mining_utils import process_image_caption
import argparse
import yaml

if __name__ == "__main__":

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/27.0.1453.94 '
        'Safari/537.36'
    }

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

    class_dict = {}
    class_df = pd.read_csv(config['class_list_csv'])

    for i in range(len(class_df)):
        row = class_df.iloc[i, :]
        class_dict[row["Class Index"]] = row["Class"]

    API_KEY = os.environ["FLICKR_API_KEY"]
    API_SECRET = os.environ["FLICKR_API_SECRET"]

    flickr = flickrapi.FlickrAPI(API_KEY, API_SECRET, format="parsed-json")

    READY = np.array([0], dtype='i')
    TERMINATE = np.array([-1], dtype='i')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        config['scraping_store_dir'].mkdir(exist_ok=False)
    else:
        sleep(10)

    logging.basicConfig(filename=config['scraping_store_dir']/f"events_{rank}.log", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode="w")
    logger = logging.getLogger("my-logger")
    logger.setLevel(logging.INFO)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    if rank == 0:
        logger.info(f"MAIN: using {size} processes.")

        request_list = [(class_id, class_name) for class_id, class_name in class_dict.items() if (class_id not in config['class_id_exclude_list'] and (
            True if config['class_id_start'] is None else config['class_id_start'] <= class_id) and (True if config['class_id_end'] is None else class_id < config['class_id_end']))]

        cnt = 0
        while cnt < len(request_list):
            class_id, class_name = request_list[cnt]

            data = np.empty(1, dtype='i')
            comm.Recv([data, MPI.INT], source=MPI.ANY_SOURCE, tag=0)

            dest_rank = data[0]
            comm.send((class_id, class_name), dest=dest_rank, tag=0)

            logger.info(f"MAIN: assigned ({class_id}, {class_name}) to process {dest_rank}")        
            cnt += 1
        
        for i in range(size-1):
            data = np.empty(1, dtype='i')
            comm.Recv([data, MPI.INT], source=MPI.ANY_SOURCE, tag=0)

            dest_rank = data[0]
            comm.send((-1, "NULL"), dest=dest_rank, tag=0)

    else:
        while True:
            comm.Send([np.array([rank], dtype='i'), MPI.INT], dest=0, tag=0)

            data = comm.recv(source=0, tag=0)
            class_id, class_name = data

            if (class_id == -1):
                MPI.Finalize()
                break

            logger.info(f"PROCESS {rank}: Class: {class_id}, {class_name}")
            res = flickr.photos.search(text=class_name, sort="relevance", safe_search=2, content_type=1, media="photos", per_page=int(1.25 * config['set_size']), extras="url_c, url_o")
            # logger.info(f"PROCESS {rank}: Class: {class_id}, {class_name}, {res}")
            try:
                this_img_dir = config['scraping_store_dir'] / f"{class_id}"
                name_url_list = [  [str(this_img_dir / f"{idx}.jpeg"), 
                                    str(this_img_dir / f"{idx}.caption"), 
                                    photo['url_c'] if 'url_c' in photo else (photo['url_o'] if 'url_o' in photo else None), 
                                    photo['title']]
                                        for idx, photo in enumerate(res['photos']['photo'])
                                ]
                this_img_dir.mkdir(exist_ok=True)

                with concurrent.futures.ThreadPoolExecutor(max_workers=config['num_threads']) as executor:
                    executor.map(process_image_caption, name_url_list, [config for i in range(len(name_url_list))], [headers for i in range(len(name_url_list))], [logger for i in range(len(name_url_list))])

            except Exception as e:
                logger.critical(f"Error when processing {class_id} {class_name}: {traceback.format_exc()}")
            
            finally:
                continue