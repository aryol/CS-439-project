# Mnist training

This repository aims to include mnist training with frequency analysis
Current projects files include:
- Jupyter notebook having the train, validation and test of frequency analysis in `optimization_mnist.py`
- SCRN Optimizer implemented by us in pytorch (`SCRNOptimizer.py`) from the paper: https://proceedings.neurips.cc/paper/2018/file/db1915052d15f7815c8b88e879465a1e-Paper.pdf


- A customized dataset for Mnist loading in `custome_dataset.py`

- A utility file for seed setting in `determinestic.py`

# Structure of the files related to classification of MNIST digits and their reproducibility

## Structure

Here is the structure of this folder:
* [determinestic.py](determinestic.py): a utility file which ensures deterministic execution needed for reproducibility.
* [custom_dataset.py](custome_dataset.py): a utility file for loading MNIST images. 
* [mnist.ipynb](mnist.ipynb): a notebook which can be used to reproduce results related to MNIST experiment. Note that this notebook was originally executed in the Google colab environment. 
* [SCRNOptimizer.py](SCRNOptimizer.py): the same implementation of SCRN optimizer. 

## Reproducibility

To reproduce results of this section one can run [SCRNOptimizer.py](SCRNOptimizer.py). As previously stated, this notebook was originally made to be executed on Google colab environment to benefit from free GPU resources. 
