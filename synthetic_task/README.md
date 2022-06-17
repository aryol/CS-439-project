# Structure of the files related to synthetic task and their reproducibility

## Structure

Here is the structure of this folder:
* [synthetic.py](synthetic.py): the main code that runs the synthetic experiment.
* [train_script.sh](train_script.sh): a bash script that trains our model with different optimizers and seed.
* [synthetic-visualization](synthetic-visualization.ipynb): a notebook which can be used to reproduce the plots reported in the paper. 

## Reproducibility

Main implementation of the synthetic task is done in [synthetic.py](synthetic.py) which is well documented. 
There is a bash training script, [train_scipt.py](train_script.sh) 
which does the training needed for our results. More precisely, for each optimizer, 
it trains the model, 10 times using 10 different seeds. Finally, one can plot the results (and see other results) through 
running our [notebook](synthetic-visualization.ipynb). 
Note that the bash script generates some additional files. Moreover, 
please execute everything from this directory. So the training script can be run by
```shell
cd synthetic_task
./train_scipt.sh
```

Moreover, it is possible to try other hyperparameters and training settings for this task. To do this, you can try commands such as below. 
```shell
python3 synthetic.py -seed 1234 -epochs 10000 -optim sgd -lr 0.001 -momentum 0.9
```
