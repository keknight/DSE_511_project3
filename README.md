# DSE 511 - Introduction to Data Science and Computing I
# Final Project

Description:
This github repository is for the binary classification of the impact risk of near-earth objects (NEOs) for planet earth and the earth's population. The main research question is whether standard "off-the-shelf" algorithms can produce outstanding classification performance.


Repository Strcuture:
The repository contains several directories. Two directories containing the raw and processed data, one directory with the written reports (proposal and final report), and one directory with source code written in Python. 

Tasks: 
Proposal (Christoph)
Data importing and pre-processing (Anna)
Data analysis + visualization (Anna)
Data modeling and evaluation with/without dimensionality reduction (Katie, Christoph)

Algorithms: 
Naive bayes, baseline model (Christoph)
SVM (Christoph)
Decision tree (Katie)
XGBoost (Katie)

Modeling and evaluation pipeline:
Train with GridSearchCV (cross validation + hyperparameter tuning) -> Use best model to predict test data -> Report evaluation metrics (F1, Precision, Recall) + plots 
- Set seed = 0 for reproducibility
