# Finding Influential Training Samples for Gradient Boosted Decision Trees
This repository implements the LeafRefit and LeafInfluence methods described in the paper _Finding Influential Training Samples for Gradient Boosted Decision Trees_.

The paper deals with the problem of finding infuential training samples using the Infuence Functions framework from classical statistics recently revisited in the paper ["Understanding Black-box Predictions via Influence Functions"](https://arxiv.org/abs/1703.04730) ([code](https://github.com/kohpangwei/influence-release)). The classical approach, however, is only applicable to smooth parametric models. In our paper, we introduce _LeafRefit_ and _LeafInfuence_, methods for extending the Infuence Functions framework to non-parametric Gradient Boosted Decision Trees ensembles.

# Requirements
We recommend using the [Anaconda](https://www.anaconda.com/download/) Python distribution for easy installation.
## Python packages
The following Python 2.7 packages are required:

_Note: versions of the packages specified below are the versions with which the experiments reported in the paper were tested._
- numpy==1.14.0
- scipy==0.19.1
- pandas==0.20.3
- scikit-learn==0.19.0
- matplotlib==2.0.2
- tensorflow==1.6.0rc0
- tqdm==4.19.5
- ipywidgets (for Jupyter Notebook rendering)

The ``create_influence_boosting_env.sh`` script creates the `influence_boosting` Conda environment with the required packages installed. You can run the script by running the following in the ``influence_boosting`` directory:
```shell
bash create_influence_boosting_env.sh
```

## CatBoost
The code in this repository uses [CatBoost](https://catboost.yandex/) for an implementation of GBDT. We tested our package with CatBoost version 0.6 built from [GitHub](https://github.com/catboost). Installation instructions are available in the [documentation](https://tech.yandex.com/catboost/doc/dg/concepts/python-installation-docpage/).

**_Note: if you are using the ``influence_boosting`` environment described above, make sure to install CatBoost specifically for this environment._**

## ``export_catboost``
Since CatBoost is written in C++, in order to use CatBoost models with our Python package, we also include ``export_catboost``, a binary that exports a saved CatBoost model to a human-readable JSON.

This repository assumes that a program named ``export_catboost`` is available in the shell. To ensure that, you can do the following:
- Select one of the two binaries, ``export_catboost_macosx`` or ``export_catboost_linux``, depending on your OS.
- Assuming you are using ``export_catboost_linux``, copy it to ``export_catboost`` in the root repository directory.
- Add the path to the root repository directory to the ``PATH`` environment variable.

# Example
An example experiment showing the API and a use-case of Influence Functions can be found in the [``influence_for_error_fixing.ipynb``](https://github.com/bsharchilev/influence_boosting/blob/master/scripts/influence_for_error_fixing.ipynb) notebook.
