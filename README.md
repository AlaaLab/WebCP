# WebCP
Conformal prediction in zero-shot settings

## Setup

### Required Packages:
NOTE: Need to download different torch version from website instead of 1.13.1+cu117 based on if you have CUDA or if you have a different CUDA version
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

1. May need to modify code for cpu if your machine doesn't support GPU acceleration via CUDA

## Credits

1. The selenium-based data mining procedure in our library (all files in ./image_caption_scraper) were forked from the following repository and subsequently modified: https://github.com/alishibli97/image-caption-scraper. We thank the author for creating such a convenient open-source data mining library.
2. All models including CLIP and BERT variants are sourced from HuggingFace model hub via their transformers library.
