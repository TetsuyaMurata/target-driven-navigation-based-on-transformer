# Target Driven Navigation Based On Transformer
This is repo is created based on [jkulhanek work](https://github.com/jkulhanek/visual-navigation-agent-pytorch) and [norips work (which is already disappeared)](https://github.com/norips/visual-navigation-agent-pytorch)  

## Introduction

This repository provides a PyTorch implementation of the paper "[Object Memory Transformer for Object Goal Navigation](https://arxiv.org/abs/2203.14708)"

## Setup and run

Please install Git LFS before cloning 

Clone the repo and download submodules:

    git clone --recurse-submodules -b rap git@github.com:TetsuyaMurata/target-driven-navigation-based-on-transformer.git

(Before install following pakages, please make sure you have Python 3.7.13 installed (try python3 -V))

Install requirements using pip:

    cd target-driven-navigation-based-on-transformer
    python3 -m venv .vna
    source .vna/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Next you need to construct the dataset. To do so run the command:
    
    python create_dataset.py

The dataset is composed of one hdf5 file per scene.
Each file contains:
- **resnet_feature** 2048-d ResNet-50 feature extracted from the observations
- **observation** 300x400x3 RGB image (agent's first-person view)
- **location** (x,y) coordinates of the sampled scene locations on a discrete grid with 0.5-meter offset
- **rotation** (x,y,z) rortation of the orientation of the agent for each location.
- **bbox** dict : `("ObjectName|3D_position": [x1,y1,x2,y2])` bounding box of object present in the view of the agent 
- **yolo_bbox** Same as abouve but using Yolov3 to extract bounding box.
- **object_visibility** visible object present in the current view
- **semantic_obs** 300x400x3 RGB image Semantic segmentation of the current view
- **graph** a state-action transition graph, where `graph[i][j]` is the location id of the destination by taking action `j` in location `i`, and `-1` indicates collision while the agent stays in the same place.
- **networkx_graph** Same graph using networkx lib
- **object_feature** 2048-d ResNet-50 feature extracted from the objects
- **object_vector** 300-d spacy feature extracted using object name
- **object_vector_visualgenome** 300-d spacy feature extracted using object name using weigh trained on visualgenome caption
- **shortest_path_distance** a square matrix of shortest path distance (in number of steps) between pairwise locations, where `-1` means two states are unreachable from each other.  
  
If you want to change the angle at which the agent bends, change `rotation_possible_inplace` on line 77 of `create_dataset.py`. 360 divided by `rotation_possible_inplace` is the number of directions to bend, e.g., it is good for you to change `rotation_possible_ inplace = 4` for 90 degrees, and `rotation_possible_inplace = 8` for 45 degrees.
  
### Training or Evaluation
to train or evaluate your network you need to use a json file as experiment. You can create a experiment file using the script `create_experiment.py`. One experiment file contains training set and evaluation set, reward function and network used. You can set these values using the script (``--help`` to see documentation).

If you would adopt method of 'grid_memory' to reproduce `Object Memory Transformer`, you cau run `create_experiment.py` like below. You will get a `param.json` with the agent's training and evaluation requirements in EXPERIMENT.
 - `python create_experiment.py --method grid_memory`

and e.g. :
    `python create_experiment.py --env grid_memory`

An experiment file which is named `.env` is for specifying processing target can be found under folder. And In ENVIRONMENT, you can find a folder you made for example "grid_memory" and `param.json` in it.  

If you would like to train or eval, execute below.
- Train : `python -u train.py`
- Eval : `python -u eval.py`

`memory_size.py` is used to make memory size, e.g.,`python memory_size.py --memory 32`

### ABCI
Set up your venv on abci:

    ssh abci (assume you heve suted up abci and ssh already)
    mkdir Code
    cd Code 
    git clone --recurse-submodules git@github.com:TetsuyaMurata/target-driven-navigation-based-on-transformer.git
    qrsh -g grpname -l rt_F=1 -l h_rt=00:10:00 (only need to create venv)
    module load gcc/11.2.0 python/3.7/3.7.13
    cd Code/target-driven-navigation-based-on-transformer/
    python3 -m venv .vna
    source .vna/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
Symbolic link datasets:

    cd Code/target-driven-navigation-based-on-transformer/
    mkdir data 
    ln -s /your/data/path ./data/

In your json file (training setting file), please cahnge `"h5_file_path": "./data/{scene}.h5"` to `"h5_file_path": "./data/8action_45deg/{scene}.h5"`
    
If you would use these scripts at `ABCI`(AI Bridging Cloud Infrastructure), they can be executed in the following way.  

- Train : `bash train_transformer_abci.sh`
- Eval : `bash eval_transformer_abci.sh`

### Calculation
You can use `category_score_mean.py` to calculate an average score of an agent by FloorPlan from the log files after evaluation, even if there are a lot of them. The results will be generated as `category_score_mean_eval...csv`. You also are able to use `step_mean_sample_count.py` to know average steps or a number of data by FloorPlan under the circumstances are "Failure on the way", "Failure at max(300steps)", "Success". These scripts need a `pandas` of library.


# Yolo_dataset

You can find in the `yolo_dataset` folder the cfg and weights of the pretrained network. This network was trained on the same dataset as previously. You can use the script `dataset_to_yolo.py` to create this dataset.

# Visual genome

You can find in the `word2vec_visualgenome` folder, the pretrained word2vec model from gensim with visualgenome dataset. You can re-train it by adding `region_descriptions.json.gz` in dataset folder and running the main.py script

# Citation
Please use this BibTex entry to cite our paper.
```
@ARTICLE{
author={Fukushima, Rui and Ota, Kei and Kanezaki, Asako and Sasaki, Yoko and Yoshiyasu, Yusuke},
journal={ICRA},
title={Object Memory Transformer for Object Goal Navigation},
year={2022},
doi={10.48550/ARXIV.2203.14708},}
