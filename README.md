# DyCODER: Dynamically Interleaving Continuous and Discrete Reasoning

This repo contains the code for the MATH-700 project submission by Mete Ismayilzada, Angelina Frolova and Carolyn Crichton. It has been forked from the original codebase for the [COCONUT project](git@github.com:facebookresearch/coconut.git). For the details of the COCONUT project, check out the [COCONUT docs](./coconut_README.md).

## Setup
```sh
conda create --name dycoder python=3.12
conda activate dycoder
pip install -r requirements.txt
```

## Repo structure
- `dataset.py`: Code for data related utils both for COCONUT and DyCODER.
- `utils.py`: Generic utilities.
- `coconut.py`: Code for the COCONUT module.
- `run_coconut.py`: Script for training COCONUT models including the baselines included in the COCONUT paper such as no-CoT, CoT etc.
- `run_coconut.sh`: Script containing commands to reproduce our experiments with COCONUT models.
- `dycoder.py`: Code for our DyCODER module.
- `dycoder_with_kv_cache.py`: Code for our DyCODER module that incorporates kv caching similar to COCONUT, however, in our experiments we ended not using as the original version turned out to be faster than this.
- `run_dycoder.py`: Script for training DyCODER models.
- `run_dycoder.sh`: Script containing commands to reproduce our experiments with DyCODER models.
- `configs`: Directory containing training and evaluation configurations for both COCONUT and DYCODER models.
- `data`: Directory containing the training and evaluation data for GSM, ProsQA and MATH datasets.
- `docker`: Directory containing docker deployment scripts.
- `figures`: Directory containing various figures used in the report.
- `preprocessing`: Directory containing data preprocessing scripts. `math_annotation.py` and `math.py` particularly contains our preprocessing steps for annotation with DeepSeek and final dataset preparation.
- `analysis.ipynb`: Notebook containing analysis code for MATH dataset.