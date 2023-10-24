# WebCP
Conformal prediction in zero-shot settings

## as of 10/14/2023 -- THIS REPOSITORY IS A WORK IN PROGRESS! 
We are currently in the process of migrating from the /main and /dutta branch in our original experimental repository: https://github.com/AlaaLab/zero-shot-conformal-prediction 


Required Packages:
transformers
open_clip
torch
matplotlib
pandas
scipy
numpy
sklearn
pillow
NOTE: List may not be comprehensive, is work in progress

Guide to formatting input folders
NOTE: Every Dataset has .py file in the util directory. Each .py file contains two OrderedDicts. One is for the specific label names and other is for the generic label names. The key for each label entry is the label id. These keys are sequential integers (i.e. "1" for 1st Label, "2" for 2nd Label, "3" for 3rd Label)

Google Drive of Data Folders (Test Set and Data-mined Calibration Set + Context Preprocessing):
Google-MedMNIST: https://drive.google.com/file/d/1QrIAPwQUtEIKX--E8CDlq4SkXo5DmBEn/view?usp=sharing 
Google-Fitz17k: https://drive.google.com/file/d/1KVjTUXRMrxSl9rhrAe4T0ExtYdrKgiZU/view?usp=sharing 
Note: Code for performing data-mining, context extraction, and context plausibility generation are still in the zero-shot-conformal-prediction folder. Currently in the process of porting it to this repo.

All data folders (image directory, context directory, intermediate data, etc.) are formatted as such:
1. Primary containing folder
2. Subfolders named as sequential numbers (1, 2, 3, 4, ...). 
3. Each subfolder contains the details for a given class. The sequential number are the label IDs
4. IMPORTANT NOTE: the folder label IDs MUST match the IDs in the dataset OrderedDict

Guide To Running Ambiguous CP
1. Change base_path in analysis.py, experiment.py, and plausibility_generation.py to point to directory containing WebCP project
2. Create an experiments .json file containing paths to relevant file and checkpoints
    a. "test_image_directory": Directory containing test set images
    b. "calib_image_directory": Directory containing data-mined images
    c. "context_directory": Directory containing data-mined context
    d. "intermediate_data_directory": Directory holding intermediate data such as plausibilities and encodings
    e. "results_data_directory": Directory holding the results data
    f. "plausibility_checkpoint": The model checkpoint of the CLIP variant used for plausibility generation
    g. "classification_checkpoint": The model checkpoint of the CLIP variant used for classification
3. Change the parameters in all files in /ambiguous_cp
    a. Change argument in reader = open() to point towards experiment .json
    b. In experiment.py and plausibility_generation.py, change LABELS and GENERIC_LABELS to be the correct labels OrderedDict
    c. Change OUTPUT_RESULT_DIR to point to the path you want the charts outputted to
4. Run 4 Files in Order
    a. image_preprocessing.py: uses plausibility CLIP to generate calib image encoding files
    b. plausibility_generation.py: generates plausibilities using encoded calib images and context filees
    c. experiment.py: uses plausibility and classification CLIP to generate adjusted sim scores and carry out CP
    d. analysis.py: uses results to output various charts on results
