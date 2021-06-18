# Optimization for Machine Learning - Mini-Project

## Introduction
The mini-project focuses on the implementation of different methods for hyperparameter tuning,
and the analysis of their impact on a machine learning model.

Three hyperparameter tuning methods are considered:
+ Grid search
+ Random search
+ Bayesian optimization


## Requirements
+ PyTorch
+ [GPy](https://github.com/SheffieldML/GPy)


## Running
Reproduce results by running the iPython notebooks:
+ Experiment1.ipynb
+ Experiment2.ipynb

The notebooks run several times our experiments so that randomness involved in random search and Bayesian optimisation is taken into account when plotting results as it can be seen at the end of thes two notebooks.  
Note that these two notebooks also have what we call offline versions of our experiments where they are ran using results that have already been computed during earlier runs. The values obtained are stored in the file results.py. In these two notebooks, you can see complete trajectories of one experiment at a time.

Experiment3.ipynb is included even though time was missing to correctly run it.