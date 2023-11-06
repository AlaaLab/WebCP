# WebCP
Conformal prediction in zero-shot settings

## Note: as of 10/14/2023 this repository is still a WIP 
We are currently in the process of migrating from the /main and /dutta branch in our original experimental repository. Code for performing data-mining, context extraction, and context plausibility generation currently in the process of being ported to this repo.

## Setup

### Required Packages:
1. transformers
2. open_clip
3. torch
4. matplotlib
5. pandas
6. scipy
7. numpy
8. sklearn
9. pillow

NOTE: List may not be comprehensive, is work in progress.

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
python image_preprocessing.py --exp "google-hybrid_fitzpatrick17_09-01-2023.json"
python plausibility_generation.py --exp "google-hybrid_fitzpatrick17_09-01-2023.json"
python experiment.py --exp "google-hybrid_fitzpatrick17_09-01-2023.json"
python analysis.py --exp "google-hybrid_fitzpatrick17_09-01-2023.json" --out [DIRECTORY WHERE YOU WANT PLOTS SAVED]
```

## Known Bugs

1. May need to modify code for cpu if your machine doesn't support GPU acceleration (cuda)
