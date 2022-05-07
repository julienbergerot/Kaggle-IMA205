# Kaggle IMA205 2022

This project is a kaggle challenge taken from [here](https://www.kaggle.com/competitions/ima205-challenge-2022/leaderboard). The purpose of this challenge is to classify skin lesion images into 8 differents categories.

### Files

*Analyse de donnees.ipynb* contains a quick analysis of the data and shed light on class imbalance. It also create weights to deal with that and a correspondance between the class and an id.

*dataset.py* contains class to create a dataset from the database of the challenge.

*Kaggle IMA205.ipynb* is a script for training a model (fine tunning resnet101) on said data.

### Results

After some iterations, I managed to obtain a score of ***0.8***, where the score is computed as a weighted accuracy based on the weights.