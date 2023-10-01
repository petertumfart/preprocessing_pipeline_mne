# preprocessing_pipeline_mne
This repository serves as a resource to perform preprocessing steps using the mne-python library.

## You will learn how to..
- ..load and convert EEG and Marker streams from xdf files.
- ..handle multiple runs and blocks for multiple subjects.
- ..apply filters, re-reference EEG, clean data (statistically and visually).
- ..further use the pre-processed data.

## Installation
1. Clone the repository
2. ```pip install -r requirements.txt```
3. ```pip install "mne>=1.0" matplotlib mne-qt-browser```
4. Download the data from here: ***tbd***

## How to use
- Run ```jupyter-lab``` or ```jupyter notebook```

- Open [01_preprocessing_pipeline.ipynb](01_preprocessing_pipeline.ipynb) and modify the DATA_PATH to your local path.

- Have fun playing around and investigating the data.

## Additional information

- This code was implemented and tested with python 3.10

