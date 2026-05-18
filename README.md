# DB-Kinetic
A deep learning framework for predicting enzyme kinetic parameters (kcat, Km, kcat/Km) with physical constraint and knowledge fusion.

## Overview
DB-Kinetic integrates cross-modal enzyme-substrate interaction and physical constraint knowledge fusion to improve prediction generalization under strict sequence similarity constraints (40% identity cutoff).

## Requirements & Environment
We provide `environment.yml` for Conda environment setup.

```bash
conda env create -f environment.yml
conda activate DB-Kinetic
```
In addition, CataPro also relies on additional pre-trained models, including prot_t5_xl_uniref50 and molt5-base-smiles2caption. These two models are used for extracting features from enzymes and substrates, respectively. You need to place the weights for these two pre-trained models in the models directory.
## Model Training
All training scripts are located in the `training/` folder.

### Train kcat model
```bash
python training/kcat/train_model_freeze_1.py
```

### Train kcat/Km model
```bash
python training/kcat_km/train_model_kcat_km_1_3.py
```

### Train Km model
```bash
python training/km/train_model2_1.py
```

## Citation
If you use this work, please cite our paper.
