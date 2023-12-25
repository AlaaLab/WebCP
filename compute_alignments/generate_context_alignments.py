"""
Script to generate context alignment scores using textual retriever models, for non-selenium experiments.
"""
import torch
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
import pickle
import nltk.data
import yaml
import argparse


def getText(content, bestTagDict, config):
    nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def processText(text):

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip()
                  for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        split_arr = text.split('\n')
        res = []
        for text in split_arr:
            res.extend(nltk_tokenizer.tokenize(text))

        return res

    bs = BeautifulSoup(content, "html.parser")

    # kill all script and style elements
    for script in bs(["script", "style"]):
        script.extract()    # rip it out

    for tag in bs.select(f"img"):
        if tag.attrs.get("src", None) == bestTagDict['src'] and tag.attrs.get("data-src", None) == bestTagDict['data-src'] and tag.attrs.get("alt", None) == bestTagDict['alt']:
            break
    assert tag.attrs.get("src", None) == bestTagDict['src'] and tag.attrs.get(
        "data-src", None) == bestTagDict['data-src'] and tag.attrs.get("alt", None) == bestTagDict['alt'], f"tag {tag}, best {bestTagDict}"

    textBeforeList, textAfterList = [], []

    for parent in list([tag]) + list(tag.parents):
        for prev_sibling in parent.previous_siblings:
            textBeforeList.extend(processText(prev_sibling.get_text()))
        for next_sibling in parent.next_siblings:
            textAfterList.extend(processText(next_sibling.get_text()))

    captionText = []

    if "alt" in tag.attrs and bool(BeautifulSoup(tag["alt"], "html.parser").find()):
        captionBS = BeautifulSoup(tag["alt"], "html.parser")

        # kill all script and style elements
        for script in captionBS(["script", "style"]):
            script.extract()    # rip it out

        captionText = processText(captionBS.get_text())
    elif "alt" in tag.attrs:
        captionText = processText(tag['alt'])

    textBeforeList.reverse()

    shortenedBeforeList, shortenedAfterList = [], []

    tokenCnt = 0
    idx = len(textBeforeList) - 1
    while (len(textBeforeList) - 1 - idx) < config['sentence_window_length'] and idx >= 0 and tokenCnt <= config['token_window_length']:
        nextTokens = tokenizer([textBeforeList[idx]], padding=True, truncation=True,
                               max_length=config['max_tokens'], return_tensors='pt')['input_ids']
        if (tokenCnt + nextTokens.size(dim=1) > tokenCnt):
            break
        shortenedBeforeList.append(textBeforeList[idx])
        tokenCnt += nextTokens.size(dim=1)
        idx -= 1

    tokenCnt = 0
    idx = 0
    while idx < min(config['sentence_window_length'], len(textAfterList)) and tokenCnt <= config['token_window_length']:
        nextTokens = tokenizer([textAfterList[idx]], padding=True, truncation=True,
                               max_length=config['max_tokens'], return_tensors='pt')['input_ids']
        if (tokenCnt + nextTokens.size(dim=1) > tokenCnt):
            break
        shortenedAfterList.append(textAfterList[idx])
        tokenCnt += nextTokens.size(dim=1)
        idx += 1

    return shortenedBeforeList, captionText, shortenedAfterList


def compute_simscores(context_encoder, query_embedding_dict: dict, tokenized_context):
    scores = {}
    with torch.no_grad():
        ctx_emb = context_encoder(
            **tokenized_context).last_hidden_state[:, 0, :]
    for cls, query_emb in query_embedding_dict.items():
        this_score_arr = query_emb @ ctx_emb.T
        scores[cls] = torch.max(this_score_arr).item()

    return scores


def get_fileidx_list(dataset_subfolder):
    idxSet = set()
    for filePath in dataset_subfolder.iterdir():
        first = str(filePath).rindex("_")+1
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
    with open(args["config"], "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    for k, v in config.items():
        if (k[-4:] == '_dir'):
            config[k] = Path(v)

    config['context_alignment_store_dir'].mkdir(exist_ok=False)

    class_df = pd.read_csv(config['class_list_csv'])

    tokenizer = AutoTokenizer.from_pretrained(
        config['tokenizer']['path'], **config['tokenizer']['kwargs'])
    query_encoder = AutoModel.from_pretrained(
        config['query_encoder']['path'], **config['query_encoder']['kwargs'])
    context_encoder = AutoModel.from_pretrained(
        config['context_encoder']['path'], **config['context_encoder']['kwargs'])

    query_encoder.eval()
    context_encoder.eval()

    class_dict = {}
    for i in range(len(class_df)):
        row = class_df.iloc[i, :]
        class_dict[row["Class Index"]] = row["Class"]

    query_encoders = {cls_name: tokenizer(
        cls_name, return_tensors='pt') for cls_name in class_dict.values()}
    for query_encoding in query_encoders.values():
        for arg in ['input_ids', 'token_type_ids', 'attention_mask']:
            assert query_encoding[arg].size(
                dim=1) <= config['max_tokens'], f"{query_encoding[arg].size(dim=1)}"
    with torch.no_grad():
        query_embeddings = {cls_name: query_encoder(
            **query_encoding).last_hidden_state[:, 0, :] for cls_name, query_encoding in query_encoders.items()}

    for class_idx, class_name in class_dict.items():
        print("CLASS " + str(class_idx))
        this_result_store = (config['context_alignment_store_dir'] / f"{class_idx}")
        this_result_store.mkdir(exist_ok=True)

        this_dataset_store = (
            config['calibration_dataset_dir'] / f"{class_idx}")
        assert this_dataset_store.exists()

        result_dict = {cls_name: []
                       for cls_name in ['Index'] + list(class_dict.values())}
        text_dict = {field: []
                     for field in ['Index', 'text_before', 'caption', 'text_after']}
        for file_idx in get_fileidx_list(this_dataset_store):
            print(file_idx)
            context = open(this_dataset_store /
                           f"{class_name}_{file_idx}.context", "r").read()

            bestTagDict = {}
            with open(this_dataset_store / f"{class_name}_{file_idx}.url", "rb") as urlFile:
                bestTagDict = pickle.load(urlFile)

            try:
                textBeforeList, captionList, textAfterList = getText(
                    context, bestTagDict)
            except Exception as e:
                print(f"skipping {file_idx} because of error.")
                continue
            text_dict['Index'].append(file_idx)
            text_dict['text_before'].append(str(textBeforeList))
            text_dict['text_after'].append(str(textAfterList))
            text_dict['caption'].append(str(captionList))

            tokenizedContext = tokenizer(textBeforeList + captionList + textAfterList, padding=True,
                                         truncation=True, max_length=config['max_tokens'], return_tensors='pt')
            for token in tokenizedContext.values():
                assert token.size(dim=1) <= config['max_tokens']

            scoreDict = compute_simscores(
                context_encoder, query_embeddings, tokenizedContext)
            result_dict['Index'].append(file_idx)
            for cls_name in class_dict.values():
                result_dict[cls_name].append(scoreDict[cls_name])

        this_df = pd.DataFrame(result_dict)
        this_df.to_csv(this_result_store / f"scores.csv")

        this_text_df = pd.DataFrame(text_dict)
        this_text_df.to_csv(this_result_store / f"text_contents.csv")
