# arvato_capstone_project
*Udacity - Machine Learning Engineer Nanodegree Program*

## Project Overview

In this project, we analyze demographic data for customers of a mail-order sales company in Germany, comparing it against demographics 
information for the general population. Exploratory Data Analysis is performed to understand and clean the data. 
Unsupervised learning techniques are used to perform customer segmentation, identifying the parts of the population 
that best describe the core customer base of the company. Then, we'll apply what we've learned on a third dataset with 
demographic information for targets of a marketing campaign for the company, and use a model to predict which 
individuals are most likely to convert into becoming customers for the company.

# Software and Libraries

This project uses Python 3 and is designed to be completed through the Jupyter Notebooks IDE. It is highly recommended 
that you use the Anaconda distribution to install Python, since the distribution includes all necessary Python libraries
as well as Jupyter Notebooks. The following libraries are expected to be used in this project:

* NumPy
* pandas
* Sklearn / scikit-learn
* Matplotlib (for data visualization)
* Seaborn (for data visualization)
* joblib (use to save sklearn models)
* missingno (to analyze missing data)
* pyarrow (to save to parquet format)
* tqdm (to show progress bars)
* imblearn (for resampling imbalanced data)
* xgboost (for the supervised model)
* hyperopt (for hyperparameter tuning using Bayesian optimization)
* utils/functions.py (various utility functions used through the project)
* other some functions from scikit-learn

## How the project is organized

There are 3 Jupyter Notebooks that are supposed to be ran in order

1.Data Cleaning and Transformation.ipynb
2.Customer Segmentation.ipynb
3.Supervised Learning Model.ipynb

The notebooks expect that the following files:
- Udacity_AZDIAS_052018.csv
- Udacity_CUSTOMERS_052018.csv
- Udacity_MAILOUT_052018_TEST.csv
- Udacity_MAILOUT_052018_TRAIN.csv

**The data is the property of Bertelsmann Arvato Analytics and it is not included in the repository.**

Also `AZDIAS_Feature_Summary.csv` is expected to be presented. This file is constructed for this
project and contains only metadata (the file is provided).

# For the Kaggle competition the submit.csv file is included in the notebooks folder
