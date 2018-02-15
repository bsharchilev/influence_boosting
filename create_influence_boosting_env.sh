echo "Preparing influence_boosting conda env..."
conda create -n influence_boosting python=2.7
source activate influence_boosting
pip install numpy scipy pandas scikit-learn matplotlib
