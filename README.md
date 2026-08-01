# Implementation of [Variational Graph Convolutional Neural Networks](https://arxiv.org/abs/2507.01699) for skeleton-based human action recognition

## Installation

Use `bash -i install.sh` to install with Conda and module. Edit to select different python and cuda versions or to install without module/Conda tools.

# Data Preparation

 - Download the raw data from [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) and [Skeleton-Kinetics](https://github.com/yysijie/st-gcn). Then put them under the data directory:
 
        -data\  
          -kinetics_raw\  
            -kinetics_train\
              ...
            -kinetics_val\
              ...
            -kinetics_train_label.json
            -keintics_val_label.json
          -nturgbd_raw\  
            -nturgb+d_skeletons\
              ...
            -samples_with_missing_skeletons.txt
            

[https://github.com/shahroudy/NTURGB-D]: NTU-RGB+D
[https://github.com/yysijie/st-gcn]: Skeleton-Kinetics

 - Preprocess the data with
  
    `python data_gen/ntu_gendata.py`
    
    `python data_gen/kinetics-gendata.py.`

 - Generate the bone data with: 
    
    `python data_gen/gen_bone_data.py`
     
# Training & Testing

Modify config files based on your experimental setup and run the scripts as follows:


```
python main.py --config=./config/ntu60/xview/joint/vstgcn/train.yaml
```

```
python main.py --config=./config/ntu60/xsub/joint/stgcn/train.yaml
```

