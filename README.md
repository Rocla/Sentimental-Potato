# Sentimental Potato
Python 3.5 classification of movie reviews (positive or negative)
Author: Romain Claret

# School project
- Data and analysis are in french
- 1000 corpus are given
- Train on 80% of the corpus
- Test on the 20% left
- Use of **data/movies_tagged**
- Use Python
- Use of scikit-learn

## Steps
1. Pre-Treatment (custom to the corpus given)
    - POS = [NOM, VER, ADV, ADJ]
    1. filter (remove the lines that are not in POS)
    2. Extract the canonical form, infinitive of the verbs, etc...
    3. Split the corpus into 80-20 train-test (random)
2. Vectorisation (based on scikit-learn tutorial)
3. Training (based on scikit-learn tutorial)
4. Evaluation (based on scikit-learn tutorial)

## Requirements
- Load the training corups with **sklearn.datasets.load_files**
- Vectorization with **sklearn.feature_extraction.text.CountVectorizer**
- Indexing with **sklearn.feature_extraction.text.TfidfTransforme**
- Creating training models **(naive Bayes, SVM, …)**
- Remplace the following operation with pipeline **sklearn.pipeline.Pipeline**
- Evaluate the classification model with **sklearn.metrics**
- Extract the best model/parameter by using **sklearn.grid_search.GridSearchCV**

## Data structure
- path: **data/movies_raw_data** is the raw data
- path: **data/movies_tagged** is the treated data giving the tags

## What has been done
- Count the number of categories (generic)
- Cloning the raw data into a processing folder (preserve the raw data integrity before filtering)
- Filtering the data of the processing folder
- Load the data into the memory
- Splitting the data into a training set and a testing set
- Testing different analysers: 2 naive_bayes, 2 linear_models, 3 svm
- Sorting the results of the tests
- Bruteforcing the parameters to find the best values
    - We are optimizing svc only because it's the best from the previous tests