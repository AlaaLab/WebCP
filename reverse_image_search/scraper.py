import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from loguru import logger
import traceback
from datetime import datetime
import re
import json
import os
from pathlib import Path
import uuid
import json
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException

# print(uuid.uuid4())
def get_public_ip_address():
    """Read the public IP address of the host"""
    response = requests.get('https://api.ipify.org')
    return response.text
class parse_args():
    def __init__(self,engine,num_images,headless,driver,expand,k):
        self.engine = engine
        self.num_images = num_images
        self.headless = headless
        self.driver = driver
        self.expand = expand
        self.k = k

class Google_Reverse_Image_Search():
    public_ip = get_public_ip_address()
    def __init__(self,engine="all",num_images=100,headless=True,driver="chromedriver",expand=False,k=3):
        """Initialization is only starting the web driver and getting the public IP address"""
        logger.info("Initializing scraper")
        
        self.public_ip = Google_Reverse_Image_Search.public_ip
        self.google_start_index = 0

        self.cfg = parse_args(engine,num_images,headless,driver,expand,k)
        self.start_web_driver()
    
    def close(self):
        self.wd.close()

    def start_web_driver(self):
        """Create the webdriver and point it to the specific search engine"""
        logger.info("Starting the engine")
        chrome_options = Options()

        if self.cfg.headless:
            chrome_options.add_argument("--headless")

        self.wd = webdriver.Chrome(options=chrome_options) # service=Service(executable_path=self.cfg.driver)


    def scrape_matching_pages(self, local_image_path):
        """Main function to scrape"""
        matching_pages = {}

        logger.info(f"Scraping for similar images of {local_image_path}")
        matching_pages = self.crawl(local_image_path)

        self.save_matching_pages(matching_pages)


    def make_query(self, local_image_path):
        """Given the target engine and query, build the target url"""
        self.wd.get("https://images.google.com/")
        time.sleep(1)

        search_button = self.wd.find_elements(By.XPATH, "//div[@aria-label='Search by image' and @role='button']")[0]
        self.wd.execute_script("arguments[0].click();", search_button)
        time.sleep(1)

        upload_button = self.wd.find_elements(By.XPATH, "//*[contains(text(), 'upload a file')]")[0]
        self.wd.execute_script("arguments[0].click();", upload_button)

        # Find image input
        upload_btn = self.wd.find_element(By.NAME, 'encoded_image')
        upload_btn.send_keys(str(local_image_path))
        
        time.sleep(2)

    def scroll_to_end(self):
        """Function for Google Images to scroll to new images after finishing all existing images"""
        logger.info("Loading new images")
        self.wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

    def goto_image_source_page(self):
        source_button = self.wd.find_elements(By.XPATH, "//div[@aria-label=\"Find image source\" and @role=\"button\"]")[0]
        self.wd.execute_script("arguments[0].click();", source_button)

    def reverse_image_search(self, local_image_path, start=0):
        """Retrieve urls for images and captions from Google Images search engine"""
        logger.info("Scraping google images")
        self.make_query(local_image_path)

        # time.sleep(2)
        # try:
        #     button = self.wd.find_element(By.XPATH, "//button[contains(@class, 'VfPpkd-LgbsSe') and @jsname='b3VHJd']")
        #     button.click()
        #     time.sleep(2)
        # except (NoSuchElementException, ElementClickInterceptedException):
        #     # Handle the exception or just pass
        #     pass

        img_data = {}

        start = 0
        prevLength = 0
        while(len(img_data)<self.cfg.num_images):
            self.scroll_to_end();i=0

            thumbnail_results = self.wd.find_elements(By.CSS_SELECTOR, "div[class^='Vd9M6'] > a")

            if(len(thumbnail_results)==prevLength):
                logger.info("Loaded all images for Google")
                break

            prevLength = len(thumbnail_results)
            logger.info(f"There are {len(thumbnail_results)} images")

            for i,content in enumerate(thumbnail_results[start:len(thumbnail_results)]):
                try:

                    url = content.get_attribute("href")
                    caption = content.get_attribute("aria-label")
                    
                    now = datetime.now().astimezone()
                    now = now.strftime("%m-%d-%Y %H:%M:%S %z %Z")

                    name = uuid.uuid4() # len(img_data)
                    img_data[f'{name}.jpg']={
                        'local_image': local_image_path,
                        'url':url,
                        'caption':caption,
                        'datetime': now,
                        'source': 'google',
                        'public_ip': self.public_ip
                    }
                    logger.info(f"Finished {len(img_data)}/{self.cfg.num_images} images for Google.")
                except:
                    logger.debug("Couldn't load image and caption for Google")
                    logger.debug(traceback.format_exc())
                
                if(len(img_data)>self.cfg.num_images-1): 
                    logger.info(f"Finished scraping {self.cfg.num_images} for Google!")
                    # logger.info("Loaded all the images and captions!")
                    break
            
            start = len(thumbnail_results)

        if (len(img_data) < self.cfg.num_images):
            logger.warning(f"NUMBER OF IMAGES LESS THAN DESIRED.")
        return img_data

    def get_matching_pages(self, local_image_path):
        """Retrieve urls for images and captions from Google Images search engine"""
        logger.info("Scraping google images")
        self.make_query(local_image_path)
        self.goto_image_source_page()

        img_data = {}

        thumbnail_results = self.wd.find_elements(By.CSS_SELECTOR, "li[class^='anSuc'] > a")

        # logger.info(f"There are {len(thumbnail_results)} images")

        for i,content in enumerate(thumbnail_results[start:min(5,len(thumbnail_results))]):
            try:
                page_url = content.get_attribute("href")
                caption = content.get_attribute("aria-label")

                now = datetime.now().astimezone()
                now = now.strftime("%m-%d-%Y %H:%M:%S %z %Z")

                name = uuid.uuid4() # len(img_data)
                img_data[f'{name}.jpg']={
                    'local_image': local_image_path,
                    'url':page_url,
                    'caption':caption,
                    'datetime': now,
                    'source': 'google',
                    'public_ip': self.public_ip
                }
                logger.info(f"Finished {len(img_data)}/{self.cfg.num_images} images for Google.")
            except:
                logger.debug("Couldn't load image and caption for Google")
            
            if(len(img_data)>self.cfg.num_images-1): 
                logger.info(f"Finished scraping {self.cfg.num_images} for Google!")
                # logger.info("Loaded all the images and captions!")
                break
        
        start = len(thumbnail_results)
        return img_data

    # def save_images_and_captions(self,img_data):
    #     """Retrieve the images and save them in directory with the captions"""
    #     query = '_'.join(self.cfg.query.lower().split())
        
    #     out_dir = self.cfg.out_dir
    #     Path(out_dir).mkdir(parents=True, exist_ok=True)
    #     os.chdir(out_dir)

    #     target_folder = os.path.join(f'{self.cfg.engine}', query)
    #     Path(target_folder).mkdir(parents=True, exist_ok=True)

    #     result_items = img_data.copy()

    #     for i,(key,val) in enumerate(img_data.items()):
    #         try:
    #             url = val['url']
    #             caption = val['caption']

    #             if(url.startswith('http')):
    #                 read_http(url,self.cfg.engine,query,caption,i)

    #             elif(url.startswith('data')):
    #                 read_base64(url,self.cfg.engine,query,caption,i)

    #             else:
    #                 del result_items[key]
    #                 logger.debug(f"Couldn't save image {i}: not http nor base64 encoded.")
    #         except:
    #             del result_items[key]
    #             logger.debug(f"Couldn't save image {i}")

    #     file_path = f'{self.cfg.engine}/{query}/{query}.json'
    #     with open(file_path, 'w+') as fp:
    #         json.dump(result_items, fp)
    #     logger.info(f"Saved urls file at: {os.path.join(os.getcwd(),file_path)}")

    # def save_images_data(self,img_data):
    #     """Save only the meta data without the images"""
    #     query = '_'.join(self.cfg.query.lower().split())
    #     out_dir = self.cfg.out_dir
    #     Path(out_dir).mkdir(parents=True, exist_ok=True)
    #     os.chdir(out_dir)

    #     file_path = f'{self.cfg.engine}/{query}'
    #     Path(file_path).mkdir(parents=True, exist_ok=True)
    #     file_path += f'/{query}.json'
    #     with open(file_path, 'w+') as fp:
    #         json.dump(img_data, fp)
    #     logger.info(f"Saved json data file at: {os.path.join(os.getcwd(),file_path)}")
