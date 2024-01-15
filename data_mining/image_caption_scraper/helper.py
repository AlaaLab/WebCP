import requests
import io
from PIL import Image
import os
from loguru import logger
import base64
import traceback

def read_http(url, engine, query, caption, i):
    # logger.info("Image is http")
    img_content = requests.get(url).content
    img_file = io.BytesIO(img_content)
    img = Image.open(img_file)
    try:
        img = img.convert('RGB')
    except:
        pass

    try:
        image_file_path = os.path.join(f'{i}.jpeg')
        with open(image_file_path, 'wb') as f:
            img.save(f, "JPEG", quality=95)
        
        caption_file_path = os.path.join(f'{i}.caption')
        with open(caption_file_path, "w", errors="replace") as write_file:
            write_file.write(caption)
    except Exception as e:
        logger.info(traceback.format_exc())
        try:
            os.remove(os.path.join(f'{i}.jpeg'))
        except OSError:
            pass

        try:
            os.remove(os.path.join(f'{i}.caption'))
        except OSError:
            pass

        raise e
    # image_url_file_path = os.path.join(f'{i}.image-url')
    # with open(image_url_file_path, "w") as image_url_file:
    #     image_url_file.write(url)
 
    logger.info(f"Saved image {i}")
def read_base64(url, engine, query, caption, i):
    # logger.info("Image is base64")
    base64_img = url.split(',')[1]
    img = Image.open(io.BytesIO(base64.b64decode(base64_img)))
    img = img.convert("RGB")
    try:
        image_file_path = os.path.join(f'{i}.jpeg')
        with open(image_file_path, "wb") as f:
            img.save(f, "JPEG", quality=95)
    
        caption_file_path = os.path.join(f'{i}.caption') 
        with open(caption_file_path, "w", errors="replace") as write_file:
            write_file.write(caption)
    except Exception as e:
        logger.info(traceback.format_exc())
        try:
            os.remove(os.path.join(f'{i}.jpeg'))
        except OSError:
            pass

        try:
            os.remove(os.path.join(f'{i}.caption'))
        except OSError:
            pass

        raise e
    # image_url_file_path = os.path.join(f'{i}.image-url')
    # with open(image_url_file_path, "w") as image_url_file:
    #     image_url_file.write(url)

    logger.info(f"Saved image {i}")

class parse_args():
    def __init__(self,engine,num_images,query,out_dir,headless,driver,expand,k):
        self.engine = engine
        self.num_images = num_images
        self.query = query
        self.out_dir = out_dir
        self.headless = headless
        self.driver = driver
        self.expand = expand
        self.k = k
