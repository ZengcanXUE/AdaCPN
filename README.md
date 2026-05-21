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

### Citation:
Please cite the following paper if you use this code in your work.

```bibtex
@article{AdaCPN,
title = {Exploring contextual and pairwise semantic-enhanced embedding with adaptive fusion for knowledge graph completion},
journal = {Information Processing & Management},
volume = {63},
number = {7, Part B},
pages = {104849},
year = {2026},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2026.104849},
url = {https://www.sciencedirect.com/science/article/pii/S0306457326002402},
author = {Zengcan Xue and Jiarui Chen and Xiaoyong Hu and Zirou Lin},
keywords = {Knowledge graph completion, Adaptive convolution, Hybrid-granularity semantic, Link prediction},
}
```
