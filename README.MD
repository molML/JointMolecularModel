![repo version](https://img.shields.io/badge/Version-v.%201.0-green)
![python version](https://img.shields.io/badge/python-3.9_|_3.10_|_3.11-blue)
![license](https://img.shields.io/badge/license-MIT-orange)
[![Static Badge](https://img.shields.io/badge/ChemRxiv-...)](https://doi.org/...)
[![Static Badge](https://img.shields.io/badge/Data-Zenodo:_10.5281/8289605-54af7d)](https:///zenodo.org/records/...)


<h2 id="Title">Molecular deep learning at the edge of chemical space</h2>

**Derek van Tilborg**, **Luke Rossen**, **Francesca Grisoni**<sup>*</sup>\
<sup>*</sup>Corresponding author: f.grisoni@tue.nl

**Abstract**\
Bla bla

![Figure 1](img/fig1.png?raw=true "Figure1")
**Figure 1. The architecture of the Joint Molecular Model (JMM) estimates how ‘unfamiliar’ a molecule is to the model through its reconstruction loss.**

<!-- Prerequisites-->
<h2 id="Prerequisites">Prerequisites</h2>

The following Python packages are required to run this codebase. Tested on macOS 15.1.1
- [PyTorch](https://pytorch.org/) (1.12.1)
- [Pandas](https://pandas.pydata.org/) (1.5.3)
- [Numpy](https://numpy.org/) (1.23.5)
- [XGBoost](https://xgboost.readthedocs.io/) (1.7.3)
- [Scikit-learn](https://scikit-learn.org/) (1.2.1)


<h2 id="Installation">Installation</h2>
Install dependencies from the provided env.yaml file. This typically takes a couple of minutes.

```conda env create -f env.yaml```

<!-- Content-->
<h2 id="content">Content</h2>

This repository is structured in the following way:

- data: contains all data 
- cheminformatics: the starting data set
- experiments: all Python scripts required to replicate the study
- jcm: all deep learning code 
- results: collection of results
- plots: all scripts required to plot the figures in the paper


<!-- How to cite-->
<h2 id="How-to-cite">How to cite</h2>

You can currently cite our [pre-print](https://doi.org/):

van Tilborg *et al.* (2025). Molecular deep learning at the edge of chemical space. ChemRxiv.   


<!-- License-->
<h2 id="License">License</h2>

This codebase is under MIT license. For use of specific models, please refer to the model licenses found in the original 
packages.