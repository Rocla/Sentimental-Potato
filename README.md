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
- AUTO path: **data/movies_processed** is the treated data giving the tags

## What has been done
- Count the number of categories (generic)
- Cloning the raw data into a processing folder (preserve the raw data integrity before filtering)
- Filtering the data of the processing folder
- Load the data into the memory
- Splitting the data into a training set and a testing set
- Testing different analysers: 2 naive_bayes, 2 linear_models, 3 svm
- Sorting the results of the tests
- Bruteforcing the parameters to find the best values
- MANUALLY: Updating the values of the tests for future tests
    - The tests are already optimized in this version (about 10h of brute force) but the values could be better with more bruteforce..
    - Note that it is note optimized for the time... But for the results!
    
## Results
>>>>>>>>>>>
testing: naive_bayes -> MultinomialNB
>>>>>>>>>>>

Result:  0.897755610973

Classification Metrics:
             precision    recall  f1-score   support

        neg       0.98      0.82      0.89       208
        pos       0.83      0.98      0.90       193

avg / total       0.91      0.90      0.90       401


Confusion Matrix:
[170  38]
[  3 190]

<<<<<<<<<<<<
done testing: naive_bayes -> MultinomialNB
<<<<<<<<<<<<


>>>>>>>>>>>
testing: naive_bayes -> BernoulliNB
>>>>>>>>>>>

Result:  0.900249376559

Classification Metrics:
             precision    recall  f1-score   support

        neg       0.95      0.86      0.90       208
        pos       0.86      0.95      0.90       193

avg / total       0.90      0.90      0.90       401


Confusion Matrix:
[178  30]
[ 10 183]

<<<<<<<<<<<<
done testing: naive_bayes -> BernoulliNB
<<<<<<<<<<<<


>>>>>>>>>>>
testing: linear_model -> SGDClassifier
>>>>>>>>>>>

Result:  0.885286783042

Classification Metrics:
             precision    recall  f1-score   support

        neg       0.88      0.90      0.89       208
        pos       0.89      0.87      0.88       193

avg / total       0.89      0.89      0.89       401


Confusion Matrix:
[188  20]
[ 26 167]

<<<<<<<<<<<<
done testing: linear_model -> SGDClassifier
<<<<<<<<<<<<


>>>>>>>>>>>
testing: linear_model -> LogisticRegression
>>>>>>>>>>>

Result:  0.917705735661

Classification Metrics:
             precision    recall  f1-score   support

        neg       0.94      0.89      0.92       208
        pos       0.89      0.94      0.92       193

avg / total       0.92      0.92      0.92       401


Confusion Matrix:
[186  22]
[ 11 182]

<<<<<<<<<<<<
done testing: linear_model -> LogisticRegression
<<<<<<<<<<<<


>>>>>>>>>>>
testing: svm -> SVC
>>>>>>>>>>>

Result:  0.922693266833

Classification Metrics:
             precision    recall  f1-score   support

        neg       0.94      0.90      0.92       208
        pos       0.90      0.94      0.92       193

avg / total       0.92      0.92      0.92       401


Confusion Matrix:
[188  20]
[ 11 182]

<<<<<<<<<<<<
done testing: svm -> SVC
<<<<<<<<<<<<


>>>>>>>>>>>
testing: svm -> LinearSVC
>>>>>>>>>>>

Result:  0.922693266833

Classification Metrics:
             precision    recall  f1-score   support

        neg       0.94      0.90      0.92       208
        pos       0.90      0.94      0.92       193

avg / total       0.92      0.92      0.92       401


Confusion Matrix:
[188  20]
[ 11 182]

<<<<<<<<<<<<
done testing: svm -> LinearSVC
<<<<<<<<<<<<


>>>>>>>>>>>
testing: svm -> NuSVC
>>>>>>>>>>>

Result:  0.912718204489

Classification Metrics:
             precision    recall  f1-score   support

        neg       0.94      0.88      0.91       208
        pos       0.88      0.94      0.91       193

avg / total       0.91      0.91      0.91       401


Confusion Matrix:
[184  24]
[ 11 182]

<<<<<<<<<<<<
done testing: svm -> NuSVC
<<<<<<<<<<<<

Now let's sort them...
[['SVC', 0.92269326683291775], ['LinearSVC', 0.92269326683291775], ['LogisticRegression', 0.9177057356608479], ['NuSVC', 0.91271820448877805], ['BernoulliNB', 0.90024937655860349], ['MultinomialNB', 0.89775561097256862], ['SGDClassifier', 0.88528678304239405]]
