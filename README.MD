
![repo version](https://img.shields.io/badge/Version-v.%201.0-green)
![python version](https://img.shields.io/badge/python-3.9_|_3.10_|_3.11-blue)
![license](https://img.shields.io/badge/license-MIT-orange)
[![Static Badge](https://img.shields.io/badge/ChemRxiv-10.26434/chemrxiv--2025--qj4k3-8A2BE2)](https://doi.org/10.26434/chemrxiv-2025-qj4k3)

<h2 id="Title">Molecular deep learning at the edge of chemical space</h2>

**Derek van Tilborg**, **Luke Rossen**, **Francesca Grisoni**<sup>*</sup>\
<sup>*</sup>Corresponding author: f.grisoni@tue.nl

**Abstract**\
Molecular machine learning models often fail to generalize beyond the chemical space of their training data, limiting their ability to reliably perform predictions on structurally novel bioactive molecules. To advance the ability of machine learning to go beyond the “edge” of their training chemical space, we introduce a joint modeling approach that combines molecular property prediction with molecular reconstruction. This approach allows the introduction of unfamiliarity, a novel reconstruction-based metric that enables the estimation of model generalizability. Via a systematic analysis spanning more than 30 bioactivity datasets, we demonstrate that unfamiliarity not only effectively identifies out-of-distribution molecules but also serves as a reliable predictor of classifier performance. Even when faced with the presence of strong distribution shifts on large-scale molecular libraries, unfamiliarity yields robust and meaningful molecular insights that go unnoticed by traditional methods. Finally, we experimentally validate unfamiliarity-based molecule screening in the wet lab for two clinically relevant kinases, discovering seven compounds with low micromolar potency, and limited similarity to training molecules. This demonstrates that unfamiliarity can extend the reach of machine learning beyond the edge of the charted chemical space, advancing the discovery of diverse and structurally novel molecules.

![Figure 1](img/fig1.png?raw=true "Figure1")
**Figure 1. The architecture of the Joint Molecular Model (JMM) estimates how ‘unfamiliar’ a molecule is to the model through its reconstruction loss.**

<!-- Prerequisites-->
<h2 id="Prerequisites">Prerequisites</h2>

The following Python packages are required to run this codebase. Tested on macOS 15.1.1https://raw.githubusercontent.com/molML/JointMolecularModel/refs/heads/main/README.MD#:~:text=The%20following%20Python%20packages%20are,learn.org%2F%29%20%281.2.1:
- [PyTorch](https://pytorch.org/) (1.12.1)
- [Pandas](https://pandas.pydata.org/) (1.5.3)
- [Numpy](https://numpy.org/) (1.23.5)
- [XGBoost](https://xgboost.readthedocs.io/) (1.7.3)
- [Scikit‑learn](https://scikit-learn.org/) (1.2.1)

<h2 id="Installation">Installation</h2>
Install dependencies from the provided `env.yaml` file. This typically takes a couple of minutes:

```bash
conda env create -f env.yaml
```

<h2 id="content">Content</h2>

This repository is structured in the following way:

- **data**: contains all data  
- **cheminformatics**: the starting data set  
- **experiments**: all Python scripts required to replicate the study  
- **jcm**: all deep learning code  
- **results**: collection of results  
- **plots**: all scripts required to plot the figures in the paper  

<!-- How to cite-->
<h2 id="How-to-cite">How to cite</h2>

You can currently cite our [pre‑print](https://doi.org/10.26434/chemrxiv-2025-qj4k3):

van Tilborg *et al.* (2025). Molecular deep learning at the edge of chemical space. *ChemRxiv*.

<!-- License-->
<h2 id="License">License</h2>

This codebase is under MIT license. For use of specific models, please refer to the model licenses found in the original packages.

---

## System requirements

The Joint Molecular Model is research code rather than a general‑purpose software package.  It has been tested on macOS 15.1.1https://raw.githubusercontent.com/molML/JointMolecularModel/refs/heads/main/README.MD#:~:text=The%20following%20Python%20packages%20are,learn.org%2F%29%20%281.2.1 and Linux machines and requires a recent Python installation (3.9–3.11).  The provided `env.yml` specifies a conda environment for Python 3.12.4 and lists all dependencieshttps://raw.githubusercontent.com/molML/JointMolecularModel/refs/heads/main/env.yml#:~:text=,zipp%3D%3D3.18.1.

### Software and operating systems

| component | tested/required versions (from `env.yml`) | notes |
| --- | --- | --- |
| **Python** | 3.9–3.11 tested; conda environment uses 3.12.4https://raw.githubusercontent.com/molML/JointMolecularModel/refs/heads/main/README.MD#:~:text=The%20following%20Python%20packages%20are,learn.org%2F%29%20%281.2.1https://raw.githubusercontent.com/molML/JointMolecularModel/refs/heads/main/env.yml#:~:text=,numpy%3D1.26.4 | use the supplied conda environment for reproducibility |
| **PyTorch** | 2.3.0https://raw.githubusercontent.com/molML/JointMolecularModel/refs/heads/main/env.yml#:~:text=,zipp%3D%3D3.18.1 | GPU support via CUDA 11.3https://raw.githubusercontent.com/molML/JointMolecularModel/refs/heads/main/env.yml#:~:text=,pip |
| **RDKit** | 2024.3.3https://raw.githubusercontent.com/molML/JointMolecularModel/refs/heads/main/env.yml#:~:text=,learn%3D%3D1.5.1 | required for SMILES parsing and molecular descriptors |
| **Scikit‑learn** | 1.5.1https://raw.githubusercontent.com/molML/JointMolecularModel/refs/heads/main/env.yml#:~:text=,2 | classical ML models (RF/MLP baselines) |
| **XGBoost** | 2.1.0https://raw.githubusercontent.com/molML/JointMolecularModel/refs/heads/main/env.yml#:~:text=,1 | used for tree‑based baselines |
| **Pandas** | 2.2.2https://raw.githubusercontent.com/molML/JointMolecularModel/refs/heads/main/env.yml#:~:text= | tabular data handling |
| **CUDA toolkit** | 11.3https://raw.githubusercontent.com/molML/JointMolecularModel/refs/heads/main/env.yml#:~:text=,pip | needed for GPU training |

### Non‑standard hardware

Training the auto‑encoder and joint model on the full ChEMBL dataset is compute‑intensive.  A **CUDA‑enabled GPU** (e.g., an NVIDIA RTX‑class card with ≥8 GB VRAM) is recommended.  Training on CPU is possible but significantly slower.  For reproducing the paper’s experiments, access to multiple GPUs and at least 32 GB of RAM is advised.

## Using this repository

This repository was used to generate the results in the accompanying pre‑print.  It does **not** provide a turnkey command‑line interface or API.  The scripts in the `experiments/` folder are pipeline components that expect input files in specific locations.  To reproduce the study:

1. **Data preparation** – run `experiments/0_clean_data.py`, `experiments/1_filter_chembl.py` and `experiments/2.0_split_data.py` sequentially.  These scripts clean the raw SMILES, filter ChEMBL and create train/test/out‑of‑distribution splits for the datasets used in the paper.  They operate on the files in `data/` and do not take custom dataset arguments.

2. **Pre‑training and baseline models** – run `experiments/3.1_ae_pretraining.py` to pre‑train the SMILES auto‑encoder on ChEMBL, followed by `experiments/4.2_ecfp_mlp.py` and `experiments/4.3_smiles_mlp.py` to train baseline classifiers.  Hyper‑parameters are defined in YAML files under `experiments/hyperparams/` and can be modified there if necessary.  The scripts should be executed as provided; they read data paths from the configuration files.

3. **Joint model training** – run `experiments/4.4_jmm.py` to train the Joint Molecular Model by combining the pre‑trained auto‑encoder and MLP.  This script automatically locates the necessary data and models and writes results to `results/`.

4. **Inference and unfamiliarity scoring** – run `experiments/5.2_inference_jmm.py` to generate predictions on the train/test/OOD splits and compute the unfamiliarity metric.  The script writes a CSV file containing SMILES strings, reconstructed SMILES, edit distances, predicted labels, uncertainty metrics and unfamiliarity scores.

Each script writes logs, model checkpoints and output files under `results/`.  You can inspect these files and use the plotting scripts in `plots/` to recreate the figures shown in the paper.

### Running on your own data

This codebase is tightly coupled to the datasets and experimental protocol described in the paper.  There is no simplified interface for arbitrary datasets.  If you wish to experiment with your own data, you will need to:

1. Prepare a CSV file with at least a `smiles` column and an activity label `y`.
2. Modify the data‑cleaning and splitting scripts (`experiments/0_clean_data.py`, `experiments/1_filter_chembl.py`, `experiments/2.0_split_data.py`) to read your file and write cleaned/split versions to `data/`.
3. Adapt the configuration files in `experiments/hyperparams/` so that the pre‑training and training scripts point to your new splits.
4. Run the pipeline scripts in the same order as above.  There is no official API for custom datasets, so you are expected to modify the code yourself.  Providing command‑line examples here would be misleading.

## Reproduction instructions

To reproduce the experiments from the pre‑print:

1. **Install dependencies** using the provided `env.yml` file.
2. **Run the data preparation scripts** (`0_clean_data.py`, `1_filter_chembl.py`, `2.0_split_data.py`) on the included datasets (Lit‑PCBA, MoleculeACE, Ames mutagenicity and ChEMBL).  These scripts will clean the SMILES and create train/test/OOD splits.  Ensure you have sufficient disk space for the ChEMBL dataset.
3. **Execute the hyper‑parameter optimisation and training scripts** in sequence: pre‑train the auto‑encoder (`3.1_ae_pretraining.py`), train the baseline models (`3.0_rnn_pretraining.py`, `4.2_ecfp_mlp.py`, `4.3_smiles_mlp.py`), and then train the joint model (`4.4_jmm.py`).  These scripts use internal configuration files and are not designed to accept dataset arguments.
4. **Evaluate models** by running the inference scripts (`5.0_inference_rnn.py`, `5.1_inference_ae.py`, `5.2_inference_jmm.py`).  These scripts load the trained models, produce predictions for the train/test/OOD splits and compute unfamiliarity scores.
5. **Analyse results**.  Use the scripts and notebooks in `results/` and `plots/` to aggregate predictions, compute performance metrics and generate figures.  Following the same random seeds and hyper‑parameters as in the YAML files should allow you to reproduce the paper’s main findings: unfamiliarity correlates with classifier performance and helps identify structurally novel but active moleculesemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/688bc2c223be8e43d622933a/original/molecular-deep-learning-at-the-edge-of-chemical-space.pdf#:~:text=systematic%20analysis%20spanning%20more%20than,scale.
