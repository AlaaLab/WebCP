# WebCP
Shiladitya Dutta, Hongbo Wei, Ahmed Alaa, and Lars van der Laan

This repository contains the corresponding implementations for the experiments in our [work on web-based conformal prediction for zero-shot models](https://arxiv.org/abs/2310.09926).

## Note: as of 10/14/2023 this repository is still a WIP 
We are doing some refactoring of our original experimental code, and will remove this message once changes are completed.

## Repository Workflow (and overview along the way)
We describe our repository, ordering based on the workflow for plausbility generation as described in our paper.

We must start off with the goal of our procedure in mind. We are given a classification task that we seek to apply our zero-shot model to, and we are seeking to apply ambiguous conformal prediction to generate guarantees on the uncertainty of the model on this application. In our case, we assume that the zero-shot model is a variant of CLIP. To do this, we do the following:
### 0. (Input) specify what classes are the targets of the classification task we define.
For example, let's say we are trying to apply CLIP to the classification task of distinguishing dog and cat species (e.g. similar to [OxfordPets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)). We must provide a specification file to highlight these classes that are part of our classification task. 

In our repo, we require that these classes are enumerated in a .csv file, with these columns: 
- **Class Index**: just for numbering/enumerating over the classes. These have to be unique for each class. 
- **Class**: the classes in the classification task. to avoid ambiguity (e.g. just specifying "newfoundland" as a class name is unclear because it can refer to "newfoundland island" or "newfoundland dog breed" which can be confusing for search engines), adding a descriptor can help the search engine return more relevant results. 
- **Generic Class**: these are actually used later on in the process (plausibility score generation based on the image's features). These are used so that when we use CLIP to determine if a calibration image actually resembles its corresponding labels, CLIP can have a better time doing so. For example, if a class in our classification task is, say, "yorkshire terrier", then a generic class corresponding to this could be "dog"; as CLIP has better performance on more generic classes, we can test whether the image actually seems to roughly resemble a "yorkshire terrier" by checking if it resembles a "dog". 

Then, we can initiate our procedure by...
### 1. Mining from the open web. 
To generate final calibration sets, we need raw images and their metadata from the internet. We currently have two scripts for this: 
- **data_mining/data_mining.py**: using the csv file with our list of classes, this script calls our custom Google Image Search Engine (defined using [Google Custom Search API](https://developers.google.com/custom-search/v1/overview)) on each of our classes and returning a list of entries, where each entry has the URL of the image and a URL to the webpage the image is linked on. The script then tries to GET the image from the image URL, and find the location where the image is embedded in the HTML page associated with the webpage HTML (by iterating through HTML ``<img>`` elements and finding which ones have a ``URL`` source resembling the image URL). If it fails to find this HTML element (which occurs quite frequently), then the entry is discarded; it if is found, then the image, the image's HTML webpage, and the identifier of the matching ``<img>`` element are saved. **NOTE**: this script uses both multiprocessing AND multithreading at the same time, i.e. it spawns multiple concurrent worker processes (+ a coordinator process), each of which spawn multiple concurrent worker threads. To run this script, it is necessary to use ``mpiexec``; i.e. ``mpiexec -n <num of processes> python data_mining/data_mining.py``, where the number of TOTAL processes (including the coordinator) is specified by the ``-n`` argument. To specify the number of worker threads spawned in each process, use the ``num_threads`` argument in the .yaml file. 
- **data_mining/data_mining_selenium.py**: here, instead of calling a search engine API, we use a selenium web scraper that essentially loads the image search results page in Selenium, and scrapes each of the images and their corresponding captions from that page. However, the captions here are derived directly from the search engine's search result page, **not** the webpage the image is embedded in. Hence, in this method we are unable to obtain a window of text surrounding the image's embedding location in the webpage; we just take the image's caption, and save it along with the image. **NOTE**: unlike data_mining/data_mining.py, we do not use mpiexec here, and just use "multithreading". It is sufficient to just use thread pools (i.e. no mpi multiprocessing) because when each thread calls the scraping instance, a fork call is done that automatically creates a separate process for each selenium scraper. In fact, using MPI here would result in some errors due to trying to call fork within a MPI process. Thus, this script can be called using normal python arguments. 

In both scripts, a directory is generated that for each class ID (specified in the csv file in step 0), contains a subdirectory with the following files: ``./{class_id}/{image_id}.caption`` and ``./{class_id}/{image_id}.image`` which respectively contain (search engine's caption, if using the selenium script, or the webpage, if not using the selenium script) and the actual image scraped from the script.

### 2. Generating Context Alignments
Now that we've scraped the raw images and their corresponding captions + surrounding text, we can test the image's textual context to see if it is relevant to each of the class included in the task. This computed metric is called a context alignment, which for right now is left in an unnormalized form (i.e. softmax is not taken during this step).

Depending on which script was used in the previous step, the following scripts should be used accordingly (i.e. use context_alignment/generate_context_alignments.py if data_mining/data_mining.py was used in the last step, and context_alignment/generate_context_alignments_selenium.py if data_mining/data_mining_selenium.py was used in the last step):
- **compute_alignments/generate_context_alignments.py**: this takes in the csv file of class listings from step 0, as well as the webpage plaintext + ``<img>`` element in data_mining/data_mining.py to compute a context alignment for each image. It first uses BeautifulSoup to parse the HTML page and retrieve the ``alt`` text of the matching ``<img>`` element (ideally the image caption), as well as all plaintext residing in a window around that element that is presumed to perhaps describe something about the image (we specify our window size to be a minimum of 256 tokens and 10 sentences on each side of the ``<img>`` element). Then, for each class in the task, it uses a retriever model (in our case, BERT) to compute a similarity score between each sentence in the window and the class name, and takes the maximum score over all sentences as the score for the class. 
- **compute_alignments/generate_context_alignments_selenium.py**: the only difference between this file and compute_alignments/generate_context_alignments.py is that because we only used the search engine's caption (rather than the entire webpage), we just compute similarity scores using this caption rather than an entire window.

In both cases, for each class ID a csv file under ``{class_id}/scores.csv`` is generated that contains the generated unnormalized content alignment scores. 

### 3. Generating ConteNt Alignments, and normalize to get plausibilities.
Now, we use a version of CLIP (chosen to be distinct from the CLIP variant we are running inferencing on) to compute content alignments (as compared to context alignments, which use the textual metadata of an image in the form of captions + surrounding context) that analyze an image's features to see how closely they match to a "relaxed" version of each class' names. 
#### 3.1 Some data processing preliminaires. 
The chosen CLIP version requires preprocessing to be done on all scraped calibration images. The script **ambiguous_cp/image_preprocessing.py** does this preprocessing on the scraped images from step 1, specific for the chosen CLIP model. 

#### 3.2 Generating content alignments, and final plausibilities. 
The list of classes from step 0, generated context alignments from step 2, and normalized images from step 3.1 are utilized to generate the content alignment as mentioned above by inputting the generic classes and normalized images into CLIP; after this, softmax normalization is done and final plausibility scores are generated for each calibration image using the normalized content and context alignments. This entire process is done in **ambiguous_cp/plausibility_generation.py**.

### 4. Running Ambiguous CP with the Generated Plausibilities + Images. 
Now that we have our calibration sets with plausibilities, we can run ambiguous CP to generate an uncertainty threshold we desire (see the "Monte Carlo CP" algorithm in [Stutz et al., 2023](https://arxiv.org/pdf/2307.09302.pdf) for more background). This is done in **ambiguous_cp/experiment.py**, which also supports testing the performance of this generated threshold (e.g. efficiency and coverage) on a held-out test set. 

Following this, additional analyses can be done using **ambiguous_cp/analysis.py** to see the behaviour of the threshold and plausibility scores at varying confidence levels. 



## Setup
TODO: MENTION GOOGLE
### Required Packages:
NOTE: The requirements may not be comprehensive, is work in progress.
```
cd WebCP
pip install -r requirements.txt
```

## Experiments

There are 2 included experiments, one for MedMNIST and one for FitzPatrick17k. Their configs are in the experiment_configs folder.

### Running MedMNIST:

Download Google-MedMNIST data folder and unzip: https://drive.google.com/file/d/1QrIAPwQUtEIKX--E8CDlq4SkXo5DmBEn/view?usp=sharing  

Modify experiment_configs/google-hybrid_medmnist_09-01-2023.json with correct directories. 1, 2, 3 are all inside .zip file that was downloaded.
1. "test_image_directory": modify to where you downloaded google-medmnist/medmnist_microscopy
2. "calib_image_directory": modify to where you downloaded google-medmnist/web_scraping_0922_google_medmnist
3. "context_directory": modify to where you downloaded google-medmnist/msmarco-bert_0922_google_medmnist
4. "intermediate_data_directory": Where you want intermediate data (i.e. plausibility distributions, image encodings) stored
5. "results_data_directory": Where you want conformal prediction results stored

Then run these commands.
```
cd ambiguous_cp
python image_preprocessing.py --exp "google-hybrid_medmnist_09-01-2023.json"
python plausibility_generation.py --exp "google-hybrid_medmnist_09-01-2023.json"
python experiment.py --exp "google-hybrid_medmnist_09-01-2023.json"
python analysis.py --exp "google-hybrid_medmnist_09-01-2023.json" --out [DIRECTORY WHERE YOU WANT PLOTS SAVED]
```

### Running FitzPatrick17k:

Download Google-Fitz17k data folder and unzip: https://drive.google.com/file/d/1KVjTUXRMrxSl9rhrAe4T0ExtYdrKgiZU/view?usp=sharing 

Modify experiment_configs/google-hybrid_fitzpatrick17_09-01-2023.json with correct directories. 1, 2, 3 are all inside .zip file that was downloaded.
1. "test_image_directory": modify to where you downloaded google-fitz17k2/fitzpatrick-17k
2. "calib_image_directory": modify to where you downloaded google-fitz17k2/web-scraping_0906_google-hybrid_fitzpatrick_50size
3. "context_directory": modify to where you downloaded google-fitz17k2/bert-base_0906_google-hybrid_fitzpatrick17k_50size
4. "intermediate_data_directory": Where you want intermediate data (i.e. plausibility distributions, image encodings) stored
5. "results_data_directory": Where you want conformal prediction results stored

Then run these commands.
```
cd ambiguous_cp
python image_preprocessing.py --exp "google-hybrid_fitzpatrick17_09-01-2023.json"
python plausibility_generation.py --exp "google-hybrid_fitzpatrick17_09-01-2023.json"
python experiment.py --exp "google-hybrid_fitzpatrick17_09-01-2023.json"
python analysis.py --exp "google-hybrid_fitzpatrick17_09-01-2023.json" --out [DIRECTORY WHERE YOU WANT PLOTS SAVED]
```

## Known Bugs

1. May need to modify code for cpu if your machine doesn't support GPU acceleration (cuda). In particular, may need to download specific torch version from website (i.e. 1.13.1+cu117)

## Acknowledgements 
1. The selenium-based data mining procedure in our library (under ./image_caption_scraper) were forked from [alishibli97/iamge-caption-scraper](https://github.com/alishibli97/image-caption-scraper) and subsequently modified.
2. Our implementations of ambiguous conformal prediction (under ./ambiguous_cp) were based upon the framework established in [Stutz et al., 2023](https://arxiv.org/pdf/2307.09302.pdf).
3. Our implementations of the conformal prediction methods under ./ambiguous_cp were derived from Anastasios Angelopoulos's highly accessible literature review on conformal prediction ([Angelopoulos and Bates (2021)](https://arxiv.org/abs/2107.07511)), and their corresponding well-maintained repository ([aangelopoulos/conformal-prediction](https://github.com/aangelopoulos/conformal-prediction))

## Citation

If you find our work useful in your research, please cite:

```BiBTeX
@misc{dutta2023estimating,
      title={Estimating Uncertainty in Multimodal Foundation Models using Public Internet Data}, 
      author={Shiladitya Dutta and Hongbo Wei and Lars van der Laan and Ahmed M. Alaa},
      year={2023},
      eprint={2310.09926},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```