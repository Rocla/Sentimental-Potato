# Sentimental Potato
Python 3.5 classification of movie reviews (positive or negative)

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
    1. filter (remove the lines que are not in POS)
    2. Extract the canonical form, infinitive of the verbs, etc...
    3. Split the corpus into 80-20 train-test (random)
2. Vectorisation (based on scikit-learn tutorial)
3. Training (based on scikit-learn tutorial)
4. Evaluation (based on scikit-learn tutorial)

## Requirements
- Load the training **corups with sklearn.datasets.load_files**
- Vectorization with **sklearn.feature_extraction.text.CountVectorizer**
- Indexing with **sklearn.feature_extraction.text.TfidfTransforme**
- Creating training models **(naive Bayes, SVM, …)**
- Remplace the following operation with pipeline **sklearn.pipeline.Pipeline**
- Evaluate the classification model with **sklearn.metrics**
- Extract the best model/parameter by using **sklearn.grid_search.GridSearchCV**

## Data structure
- path: **data/movies_raw_data** is the raw data
- path: **data/movies_tagged** is the treated data giving the tags