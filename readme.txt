# target
/home/dl-box/target-driven-navigation-based-on-transformer/model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/4hist/61/concat/checkpoints

# pth
!!!!! Not Read !!!!!! 
./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/forvideo/know/checkpoints/{checkpoint}.pth

!!! Real Read !!! : ./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/forvideo/know/checkpoints/5000026.pth


### Check ###
!!! checkpoint_path !!! : ./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/forvideo/know/checkpoints/{checkpoint}.pth

Restoring from checkpoint 5000026
!!! torch load !!! : ./model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/forvideo/know/checkpoints/5000026.pth

### eval.py ###
If you would like to get a log file of pth you choose, you could need to modify "eval.py". 

### change these scripts for Fukushima's trained pth model ###
./agent/network.py
./agent/training.py
./agent/evaluation.py #customに最新版スクリプトがある
./eval.py
./agent/transformer.py


福島さんpth評価対象path
/home/dl-box/target-driven-navigation-based-on-transformer/model/Transformer_word2vec/80scene/45deg/1layer/grid_memory/50cm/seed/32hist/61/checkpoints

### evaluation.py ###
evaluation.pyにあるtarget_pathをかきかえるとそのフォルダのcheckpointフォルダ直下のpthが読み込まれ、出力logファイルもそのフォルダのパスの場所になる

### 8action_45degree ###
/home/dl-box/mnt/b/visual-navigation-agent-pytorch/8action_45degree/data

### grid_size=0.25 ###
/home/dl-box/mnt/a/visual-navigation-agent-pytorch-data/data
