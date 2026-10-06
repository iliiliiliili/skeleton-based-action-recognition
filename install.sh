#!/bin/bash -i

conda create -n vgcn python=3.10
conda activate vgcn
conda env list

module load cuda/11.6

conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
pip install -r tool_requirements.txt