#!/usr/bin/env bash
echo "Preparing influence_boosting conda env..."
conda create -n influence_boosting python=2.7
source activate influence_boosting
pip install numpy==1.14.0 scipy==0.19.1 pandas==0.20.3 scikit-learn==0.19.0 matplotlib==2.0.2 tensorflow==1.6.0rc0 tqdm==4.19.5
