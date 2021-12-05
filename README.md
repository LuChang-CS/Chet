# Context-Aware-Healthcare

Codes for AAAI 2022 paper: *Context-aware Health Event Prediction via Transition Functions on Dynamic Disease Graphs*

## Download the MIMIC-III and MIMIC-IV datasets
Go to [https://mimic.physionet.org/](https://mimic.physionet.org/gettingstarted/access/) for access. Once you have the authority for the dataset, download the dataset and extract the csv files to `data/mimic3/raw/` and `data/mimic4/raw/` in this project.

## Preprocess
```bash
python run_preprocess.py
```

## Train model
```bash
python train.py
```

## Configuration
Please see `train.py` for detailed configurations.
