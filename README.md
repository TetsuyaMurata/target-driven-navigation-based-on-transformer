# Target Driven Navigation Based On Transformer
This is repo is created based on [jkulhanek work](https://github.com/jkulhanek/visual-navigation-agent-pytorch) and [norips work (which is already disappeared)](https://github.com/norips/visual-navigation-agent-pytorch)  

## Introduction

This repository provides a PyTorch implementation of the paper "[Object Memory Transformer for Object Goal Navigation](https://arxiv.org/abs/2203.14708)"

## Setup and run

Please install Git LFS before cloning 

Clone the repo and download submodules:

    git clone --recurse-submodules git@github.com:TetsuyaMurata/target-driven-navigation-based-on-transformer.git

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
  
If you would run `create_dateset.py`, you need libraries of `requirements_tem.txt` and run `python -m spacy download en_core_web_lg`. It is also needed for you to put `resnet50_places365.pth.tar` to "agent/resnet" and `yolov3_ai2thor_last.weights` to "yolo_dataset/backup". 

If you want to change the angle at which the agent bends, change `rotation_possible_inplace` on line 77 of `create_dataset.py`. 360 divided by the number of directions to bend, e.g. it is good for you to change `rotation_possible_ inplace = 4` for 90 degrees, and `rotation_possible_inplace = 8` for 45 degrees.
  
### Training or Evaluation
to train or evaluate your network you need to use a json file as experiment. You can create a experiment file using the script `create_experiment.py`. One experiment file contains training set and evaluation set, reward function and network used. You can set these values using the script (``--help`` to see documentation).

If you would adopt method of 'grid_memory', you cau run `create_experiment.py` like below. 
 - `python create_experiment.py --method grid_memory`

An experiment file which is named `target_path.txt` can be found under folder.

Set experiment in `target_path.txt` file e.g. :
    `./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/32hist/61`
    
- Train : `bash train_transformer.sh`
- Eval : `bash eval_transformer.sh`

`memory_size.py` is used getting memory size from hist of `target_path.txt`. It is run by executiing `train_transformer.sh` or `eval_transformer.sh`.

### ABCI
If you would use these scripts at `ABCI`(AI Bridging Cloud Infrastructure), they can be executed in the following way.  

- Train : `bash train_transformer_abci.sh`
- Eval : `bash eval_transformer_abci.sh`

### Calculation
You can use `category_score_mean.py` to calculate an average score of an agent by FloorPlan from the log files after evaluation, even if there are a lot of them. You also are able to use `step_mean_sample_count.py` to know average steps or a number of data by FloorPlan under the circumstances are "Failure on the way", "Failure at max(300steps)", "Success".

# Yolo_dataset

You can find in the `yolo_dataset` folder the cfg and weights of the pretrained network. This network was trained on the same dataset as previously. You can use the script `dataset_to_yolo.py` to create this dataset.

# Visual genome

You can find in the `word2vec_visualgenome` folder, the pretrained word2vec model from gensim with visualgenome dataset. You can re-train it by adding `region_descriptions.json.gz` in dataset folder and running the main.py script

# Singularity

On the condition that you use python3 and singularity is installed, if you would like to use these scripts without environment construction, you cau use a image of singularity. You needn't install packages by using `cenotate_transformer.simg`.

# Citation
Please use this BibTex entry to cite our paper.
```
@ARTICLE{
author={Fukushima, Rui and Ota, Kei and Kanezaki, Asako and Sasaki, Yoko and Yoshiyasu, Yusuke},
journal={ICRA},
title={Object Memory Transformer for Object Goal Navigation},
year={2022},
doi={10.48550/ARXIV.2203.14708},}
```

