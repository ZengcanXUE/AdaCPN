# Exploring Contextual and Pairwise Semantic-enhanced Embedding with Adaptive Fusion for Knowledge Graph Completion
This is an implementation of AdaCPN from the paper "Exploring Contextual and Pairwise Semantic-enhanced Embedding with Adaptive Fusion for Knowledge Graph Completion".

## Requirements

Python is running at version <u>3.9.16</u>. Other Python package versions can be found in **requirements.txt**

It is recommended to create a virtual environment with the above version of Python using **conda**, and install the python packages in requirements.txt using **pip** in the virtual environment.



## Running a model

Parameters are configured in `configs`, all the hyperparameters in the configuration file come from the paper.

Start training command:
```
$ python main.py -c configs/FB15k237.json
```

## Citation
