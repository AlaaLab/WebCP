from PIL import Image
import requests
from io import BytesIO
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
import pickle
from pathlib import Path
import traceback
def process_image_caption(ls, config, headers, logger):
    image_file_dir, caption_file_dir, image_url, caption = ls

    if image_url is None:
        logger.warn(f"image url is none; image {image_file_dir}, caption {caption}")
        return
    try:
        response = requests.get(image_url, headers=headers, timeout=(5, 5), stream=True)
        img = Image.open(BytesIO(response.content))

        old_width, old_height = img.size

        ratio = max(old_width, old_height) / config['max_dim_size']
        new_width, new_height = int(old_width / ratio), int(old_height / ratio)

        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        img = img.resize((new_width, new_height))
        img.save(image_file_dir, "jpeg", quality=95)

        with open(caption_file_dir, "w") as caption_file:
            caption_file.write(str(caption))
    except Exception as e:
        logger.warn(traceback.format_exc())
        image_file_dir.unlink(missing_ok=True)
        caption_file_dir.unlink(missing_ok=True)

def process_image(ls, config, headers):
    file_dir, image_url_dir, url = ls
    # try:
    response = requests.get(url, headers=headers, timeout=(5, 5), stream=True)
    img = Image.open(BytesIO(response.content))

    old_width, old_height = img.size

    ratio = max(old_width, old_height) / config['max_dim_size']
    new_width, new_height = int(old_width / ratio), int(old_height / ratio)

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    img = img.resize((new_width, new_height))
    img.save(file_dir, "jpeg", quality=95)

    with open(image_url_dir, "w") as image_url_file:
        image_url_file.write(str(url))

    # except Exception as e:
    # logger.critical(f"Error when processing {file_dir}, url {url}: {traceback.format_exc()}; Request Content is {response.content}")


def trimArgs(path_url):
    path_url = str(path_url)
    index = -1
    if ".jpg" in path_url:
        index = max(index, path_url.rindex(".jpg") + 4)
    if ".jpeg" in path_url:
        index = max(index, path_url.rindex(".jpeg") + 5)
    if ".png" in path_url:
        index = max(index, path_url.rindex(".png") + 4)
    path_url = path_url[0:(index if index >= 0 else len(path_url))]

    return path_url


def getBestURLMatch(bs, verify_path):
    bestTag = None
    bestScore = 0.0
    for tag in bs.select(f"img[alt]"):
        if (tag['alt'] == ""):
            continue
        if (urlparse(tag['alt']).scheme in ['file', 'http', 'https']):
            continue
        if ("src" not in tag.attrs and "data-src" not in tag.attrs):
            continue

        chosenScore = 0
        for type in ['src', 'data-src']:
            if (type not in tag.attrs):
                continue

            type_content_url = os.path.basename(urlparse(tag[type]).path)
            type_content_url = trimArgs(type_content_url)
            type_score = SequenceMatcher(
                None, verify_path, type_content_url).ratio()

            if (chosenScore < type_score):
                chosenScore = type_score

        if (chosenScore > bestScore):
            bestScore = chosenScore
            bestTag = tag

    return bestTag, bestScore


def process_webpage(ls, headers, logger):
    file_dir, url_dir, url, verifyurl = ls
    # try:
    response = requests.get(url, headers=headers, timeout=(5, 5))
    content = response.content.decode("utf-8")

    imgname = os.path.basename(urlparse(verifyurl).path)
    imgname = trimArgs(imgname)

    bs = BeautifulSoup(content, 'html.parser')
    # kill all script and style elements
    for script in bs(["script", "style"]):
        script.extract()    # rip it out

    bestTag, bestScore = getBestURLMatch(bs, imgname)

    if (bestTag is None):
        raise Exception(
            f"file image not found: image {verifyurl} in url {url}")
    if (bestScore < 0.8):
        raise Exception(
            f"no good matches: image {verifyurl} in url {url} (image name {imgname}, best match {bestScore}, tag {bestTag})")

    logger.info(
        f"image {verifyurl}, url {url}, best match is score {bestScore}, tag {bestTag}")

    with open(file_dir, "w") as text_file:
        text_file.write(content)
        text_file.close()

    di = {"src": bestTag.attrs.get("src", None), "data-src": bestTag.attrs.get(
        "data-src", None), "alt": bestTag.attrs.get("alt", None)}
    with open(url_dir, "wb") as url_file:
        pickle.dump(di, url_file)
        url_file.close()
    # except Exception as e:
    #     logger.critical(f"Error when processing {file_dir}, url {url}: {traceback.format_exc()}; Request Content is {response.content}")


def process_both(imgLs, webLs, config, headers, logger):
    try:
        process_image(imgLs, config, headers)
        if config['use_webpage_context']:
            process_webpage(webLs, headers, logger)
    except Exception as e:
        Path(imgLs[0]).unlink(missing_ok=True)
        Path(imgLs[1]).unlink(missing_ok=True)
        if config['use_webpage_context']:
            Path(webLs[0]).unlink(missing_ok=True)
            Path(webLs[1]).unlink(missing_ok=True)
        logger.critical(
            f"Error when processing {imgLs}, web {webLs}: {traceback.format_exc()}")


def makeQuery(startIdx, query, cse):
    res = cse.list(c2coff="1", cx=os.environ['GOOGLE_CX_ID'], fileType="jpg,png", filter="1",
                   num="10", q=query, safe="off", searchType="image", start=str(startIdx)).execute()
    return [item['image']['contextLink'] for item in res['items']], [item['link'] for item in res['items']]
