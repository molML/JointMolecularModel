
![repo version](https://img.shields.io/badge/Version-v.%201.0-green)
![python version](https://img.shields.io/badge/python-3.9_|_3.10_|_3.11-blue)
![license](https://img.shields.io/badge/license-MIT-orange)
[![Static Badge](https://img.shields.io/badge/ChemRxiv-10.26434/chemrxiv--2025--qj4k3-8A2BE2)](https://doi.org/10.26434/chemrxiv-2025-qj4k3)
[![Static Badge](https://img.shields.io/badge/Zenodo_(data)-10.5281/zenodo.14865513-54af7d)](https://doi.org/10.5281/zenodo.14865513)

<h2 id="Title">Molecular deep learning at the edge of chemical space</h2>

**Derek van Tilborg**, **Luke Rossen**, **Francesca Grisoni**<sup>*</sup>\
<sup>*</sup>Corresponding author: f.grisoni@tue.nl

**Abstract**\
Molecular machine learning models often fail to generalize beyond the chemical space of their training data, limiting their ability to reliably perform predictions on structurally novel bioactive molecules.  To advance the ability of machine learning to go beyond the “edge” of their training chemical space, we introduce a joint modelling approach that combines molecular property prediction with molecular reconstruction.  This approach allows the introduction of **unfamiliarity**, a novel reconstruction‑based metric that enables the estimation of model generalizability.  Via a systematic analysis spanning more than 30 bioactivity datasets, we demonstrate that unfamiliarity not only effectively identifies out‑of‑distribution molecules but also serves as a reliable predictor of classifier performance.  Even when faced with strong distribution shifts on large‑scale molecular libraries, unfamiliarity yields robust and meaningful molecular insights that go unnoticed by traditional methods.  Finally, we experimentally validate unfamiliarity‑based molecule screening in the wet lab for two clinically relevant kinases, discovering seven compounds with low micromolar potency and limited similarity to training molecules.  This demonstrates that unfamiliarity can extend the reach of machine learning beyond the edge of the charted chemical space, advancing the discovery of diverse and structurally novel molecules.

![Figure 1](img/fig1.png?raw=true "Figure 1")
**Figure 1.** The architecture of the Joint Molecular Model (JMM) estimates how “unfamiliar” a molecule is to the model through its reconstruction loss.

<!-- Prerequisites-->
<h2 id="Prerequisites">Prerequisites</h2>

The JMM codebase has been tested on macOS 15.1.1 and recent Linux distributions.  It requires Python and a handful of scientific libraries.  The easiest way to install the correct versions is via the supplied `env.yml` conda environment.  The core packages used in our experiments are listed below:

- **Python** 3.9–3.11 (the conda environment uses Python 3.12.4)
- **PyTorch** 2.3.0 (with optional CUDA 11.3 for GPU support)
- **RDKit** 2024.3.3 (for SMILES parsing and descriptor generation)
- **Scikit‑learn** 1.5.1 (classical ML baselines)
- **XGBoost** 2.1.0 (tree‑based baselines)
- **Pandas** 2.2.2 and **Numpy** 1.26.4 (data handling and numerics)

<!-- Installation-->
<h2 id="Installation">Installation</h2>

1. **Clone the repository**:

   ```bash
   git clone https://github.com/molML/JointMolecularModel.git
   cd JointMolecularModel
   ```

2. **Create and activate the conda environment**.  The `env.yml` file specifies all dependencies.  Creating the environment typically takes 5–10 minutes on a normal desktop machine.

   ```bash
   conda env create -f env.yml
   conda activate jointmolecularmodel
   ```

3. **(Optional) Verify your installation** by importing the main libraries:

   ```bash
   python -c "import torch, pandas, rdkit; print(torch.__version__, pandas.__version__)"
   ```
   If you intend to train on a GPU, ensure that `torch.cuda.is_available()` returns `True`.

<!-- Content-->
<h2 id="content">Content</h2>

This repository is organised as follows:

- **data** – raw and processed datasets used in the study
- **cheminformatics** – scripts for generating the starting dataset
- **experiments** – Python scripts that implement each stage of the pipeline
- **jcm** – deep‑learning model definitions and utilities
- **results** – folders where trained models and logs are written
- **plots** – scripts to reproduce figures from the paper

<!-- How to cite-->
<h2 id="How-to-cite">How to cite</h2>

You can cite our pre‑print as follows:

> van Tilborg *et al.* (2025). *Molecular deep learning at the edge of chemical space*. ChemRxiv. DOI: 10.26434/chemrxiv-2025-qj4k3

<!-- License-->
<h2 id="License">License</h2>

This codebase is released under the MIT license.  When using pre‑trained models or third‑party libraries, please adhere to their respective licences.

---

## System requirements

The Joint Molecular Model is research code rather than a general‑purpose software package.  It has been tested on macOS 15.1.1 and recent Linux distributions, and requires Python 3.9 or newer.  A CUDA‑enabled GPU with at least 8 GB of memory is recommended for training the deep‑learning models.  CPU‑only training is possible but considerably slower.

### Software and operating systems

| Component    | Tested/required versions | Notes                                                |
|--------------|--------------------------|------------------------------------------------------|
| **Python**   | 3.9–3.11 (conda env uses 3.12.4) | Use the supplied `env.yml` for reproducibility      |
| **PyTorch**  | 2.3.0                    | GPU support via CUDA 11.3                            |
| **RDKit**    | 2024.3.3                 | Required for SMILES parsing and molecular descriptors |
| **Scikit‑learn** | 1.5.1                | For classical ML models (RF/MLP baselines)           |
| **XGBoost**  | 2.1.0                    | Used for tree‑based baselines                       |
| **Pandas**   | 2.2.2                    | Tabular data handling                               |
| **Operating system** | macOS 15.1.1; Linux | The code has been tested on macOS and Linux          |

### Non‑standard hardware

For full‑scale training on the ChEMBL dataset, we recommend a CUDA‑enabled GPU (e.g., NVIDIA RTX series with ≥8 GB memory) and at least 32 GB of system RAM.  Smaller datasets can be processed on a normal desktop, but training will be slower.

## Installation guide

The typical workflow for setting up this codebase is as follows:

1. **Clone the repository** and create the conda environment as described in the [Installation](#installation) section above.
2. **Prepare data**.  A separate Zenodo repository (https://doi.org/10.5281/zenodo.14865513) includes cleaned and split datasets under the `data/` folder.  There is no automated interface for loading custom datasets; instead, the pipeline reads fixed file names from `data/` and assumes specific folder structures.  If you intend to reproduce the paper’s results, use the data from the Zenodo repo. No additional data preparation is required.
3. **Run the pipeline scripts** in sequence.  Each script in the `experiments/` directory implements a step of the workflow: data cleaning, splitting, model pre‑training, baseline training, joint training and inference.  See [Demo](#demo) below for an overview.
4. **(Optional) Verify your environment** by running a short Python script or by training a small model.  Installation typically takes only a few minutes; most of the computation time is spent during training.

## Demo

This repository does **not** provide a single command‑line entry point or a user‑friendly demo for arbitrary datasets.  Instead, the scripts in the `experiments/` directory were written to replicate the experiments in our pre‑print.  To reproduce those results, run the scripts **in order** using the included configuration files. Since many experiments are written for a SLURM-based computer cluster as 'jobs', they might need some adjustments for your specific system.
IMPORTANT: we do not redistribute external data sources (original links can be found in the paper), but instead provide processed data separately in a Zenodo repo (https://doi.org/10.6084/m9.figshare.30665201.v1). 

The key steps to replicate this study are:

1. **Data preparation** – run `experiments/0_clean_data.py`, `experiments/1_filter_chembl.py` and `experiments/2.0_split_data.py` to clean raw data, filter ChEMBL and generate train/test/OOD splits. These steps can be skipped when using the pre-processed data from Zenodo.
2. **Model pre‑training and baselines** – run `experiments/3.1_ae_pretraining.py` to pre‑train the SMILES auto‑encoder, then run `experiments/4.2_ecfp_mlp.py` and `experiments/4.3_smiles_mlp.py` to train baseline classifiers.  These scripts read hyper‑parameters from YAML files under `experiments/hyperparams/`.
3. **Joint model training** – run `experiments/4.4_jmm.py` to train the joint model using the pre‑trained components.
4. **Inference** – run `experiments/5.2_inference_jmm.py` to compute predictions and unfamiliarity scores on the train/test/OOD splits.  The outputs (CSV files and logs) will be written to the `results/` directory.

These scripts are designed for our specific datasets and do **not** accept arbitrary command‑line arguments.  Attempting to run them on your own data without modifying the code will likely result in errors.  Running the full pipeline on ChEMBL can take several hours on a single GPU; smaller datasets finish more quickly.  The final outputs are CSV files containing the original SMILES, reconstructed SMILES and their edit distances, predicted labels, uncertainty measures and the unfamiliarity metric.

## Source data for the main figures

We provide the source data for the main figures at figshare (https://doi.org/10.6084/m9.figshare.30665201.v1) and the R scripts to plot them. In these same R scripts, you can find the code for most of the statistical tests we perform for the paper. 
Because we are not allowed to redistribute chemical structures from commercial screening libraries, the column of SMILES strings is removed from the source data of Fig. 3.

## Instructions for use

Because this codebase was developed for a single research project, there is no general API for applying the JMM to arbitrary datasets.  All scripts assume that data live in specific folders and have specific file names.  If you wish to experiment with other datasets or change hyper‑parameters, you will need to delve into the source code and configuration files.  There are no ready‑made command‑line options for specifying a dataset, so such changes require code modifications.

### Reproduction instructions

To reproduce the experiments reported in the pre‑print:

1. **Install dependencies** using the provided `env.yml` file.
2. **Run the data preparation scripts** (`0_clean_data.py`, `1_filter_chembl.py`, `2.0_split_data.py`) on the included datasets (Lit‑PCBA, MoleculeACE, Ames mutagenicity and ChEMBL).  These scripts will clean the SMILES and create the train/test/OOD splits used in our study.
3. **Run the training scripts** (`3.1_ae_pretraining.py`, `4.2_ecfp_mlp.py`, `4.3_smiles_mlp.py`, `4.4_jmm.py`) in sequence.  Each script reads hyper‑parameters from YAML files under `experiments/hyperparams/` and writes its outputs to `results/`.
4. **Evaluate models** using the inference scripts (`5.2_inference_jmm.py` and related files).  These will generate predictions and unfamiliarity scores and save them as CSV files.
5. **Analyse results**.  Use the notebooks and plotting scripts under `results/` and `plots/` to reproduce the figures and analyses in the paper.

By following this workflow and using the same random seeds and hyper‑parameters reported in the configuration files, you should be able to replicate the main findings of our work: unfamiliarity correlates with classifier performance across datasets and helps prioritise structurally novel yet active molecules.
