# Structure of the files related to classification of MNIST digits and their reproducibility

## Structure

Here is the structure of this folder:
* [determinestic.py](determinestic.py): a utility file which ensures deterministic execution needed for reproducibility.
* [custom_dataset.py](custome_dataset.py): a utility file for loading MNIST images. 
* [mnist.ipynb](mnist.ipynb): a notebook which can be used to reproduce results related to MNIST experiment. Note that this notebook was originally executed in the Google colab environment. 
* [SCRNOptimizer.py](SCRNOptimizer.py): the same implementation of SCRN optimizer. 

## Reproducibility

To reproduce results of this section one can run [SCRNOptimizer.py](SCRNOptimizer.py). As previously stated, this notebook was originally made to be executed on Google colab environment to benefit from free GPU resources. 
