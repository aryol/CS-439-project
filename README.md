# CS-439 project: Generalization of DNNs Trained with Second-order Optimizers Through the Lens of Frequency Analysis

**Summary.**
This repository includes the course project for [CS-439](https://github.com/epfml/OptML_course), Optimization for Machine Learning, at EPFL - Spring 2022. 
In this project, we empirically analyze the spectral bias induced by second-order optimizers (AdaHessian and SCRN) and 
compare it with spectral bias of SGD and Adam which are commonly used in deep learning. To this end, we consider 
both synthetic and real-world tasks such as image classification and segmentation. We discover that second-order methods 
manifest different behavior in frequency domain; particularly, they learn high frequencies faster and they are more stable. 
Furthermore, we relate our observations in the frequency domain with the generalization of different optimization methods. 
All of the implementation has been done in Python and PyTorch. 


# Experiments & Repository Structure

## Experiments
Our project includes three main independent experiments. More precisely, we do frequency analysis for three tasks:
* **Synthetic task.** In this task we consider learning a function which is sum of sinusoid signals.
* **Image classification.** In this task we study the performance of different optimizers on classification of MNIST digits.
* **Image segmentation.** In this task we study the image segmentation task.

## Repository Structure
Given having the aforementioned tasks, we have structured our repository as follows:
* [synthetic_task](synthetic_task): contains the code for our synthetic tasks. More particularly, it is consisted of the following files: 
* [image_classification](image_classification): contains code related to our experiments for MNIST digit classification
* [image_segmentation](image_segmentation): contains code for our experiment on image segmentation
* [optimizers](optimizers): this folder includes the code used for second-order optimizers
  * [SCRNOptimizer.py](optimizers/SCRNOptimizer.py): this file includes an implementation for the SCRN optimizer which has been implemented by our group.
  * [adahessian.py](optimizers/adahessian.py): this code is the official code of AdaHessian optimizer. 
  

# Reproducibility
For all experiments, note that we have set the seeds (along with some other options) in all of our experiments to ensure reproducibility. 
Once more note that our project consists of three independent experiments. Therefore, running each of them is independent of the others and is explained separately in the readme file of their respective folder.

Finally, note that we have listed required packages for our experiments (running and plotting) in [requirements.txt](requirements.txt).







