# -*- coding: utf-8 -*-

from os import mkdir, listdir
from os.path import join, isdir, isfile
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.grid_search import GridSearchCV
from shutil import rmtree, copy
from random import randint
from numpy import mean
import re

__author__ = "Romain Claret"
__maintainer__ = "Romain Claret"
__copyright__ = "Copyright 2016, Romain Claret"
__credits__ = ["Romain Claret"]

# Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# Do whatever you want with this but don't make money with it :)
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0"
__version__ = "1.0.0"
__email__ = "contact[at]rocla.ch"
__status__ = "Prototype"  # Prototype, Development, Production
__date__ = "14.04.2016"

"""@package extractor
Global documentation for the use of the sentimental extractor from data.
"""

"""
Sentimental Potato analyse the sentiment of movie reviews
"""


class Bunch(dict):
    """Container object from sklearn.datasets
    Dictionary-like object that exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass


def process_raw_data(raw_path, process_path, data_categories, filter_list):
    if isdir(process_path):
        rmtree(process_path)
    mkdir(process_path)

    for directory in data_categories:
        mkdir(process_path + "/" + directory)

        for file in [files for files in listdir(raw_path + "/" + directory) if
                     isfile(join(raw_path + "/" + directory, files))]:
            if file != ".DS_Store":
                copy(raw_path + "/" + directory + "/" + file, process_path + "/" + directory + "/" + file)
                filter_data(process_path + "/" + directory + "/" + file, filter_list)


def get_categories(data_path):
    return [directory for directory in listdir(data_path) if isdir(join(data_path, directory))]


def load_data(data_path, data_categories):
    return load_files(container_path=data_path,
                      description=None,
                      categories=data_categories,
                      load_content=True,
                      shuffle=True,
                      encoding='latin-1',
                      decode_error='strict',
                      random_state=randint(0, 999999))


def filter_data(file_path, filter_list):
    regex = ".*"
    for type in filter_list:
        regex += type + "|"
    regex = regex[:-1]
    regex += ".*"
    regex = re.compile(regex).search

    with open(file_path, 'r') as f:
        new = [line for line in f.readlines() if regex(line)]
    with open(file_path, 'w') as f:
        f.writelines(new)


def split_data(data, percent):
    # first_part_data = data.data[:int(len(data.data) * percent)]
    # second_part_data = data.data[int(len(data.data) * percent):]
    first_part_filenames = data.filenames[:len(data.filenames) * percent]
    second_part_filenames = data.filenames[len(data.filenames) * percent:]
    first_part_target = data.target[:len(data.target) * percent]
    second_part_target = data.target[len(data.target) * percent:]

    first_part_data = []
    for filename in first_part_filenames:
        with open(filename, 'rb') as f:
            first_part_data.append(f.read())
    first_part_data = [d.decode('latin-1', 'strict') for d in first_part_data]

    second_part_data = []
    for filename in second_part_filenames:
        with open(filename, 'rb') as f:
            second_part_data.append(f.read())
    second_part_data = [d.decode('latin-1', 'strict') for d in second_part_data]

    first_part_data = Bunch(filenames=first_part_filenames,
                            data=first_part_data,
                            target_names=data.target_names,
                            target=first_part_target,
                            DESCR=data.DESCR)
    second_part_data = Bunch(filenames=second_part_filenames,
                             data=second_part_data,
                             target_names=data.target_names,
                             target=second_part_target,
                             DESCR=data.DESCR)

    return first_part_data, second_part_data


def test_multinomialnb(data_training, data_testing, verbose=False):
    print_header("testing: naive_bayes -> MultinomialNB")

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 20))),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf', MultinomialNB())])

    text_clf = text_clf.fit(data_training.data, data_training.target)
    docs_test = data_testing.data
    predicted_target = text_clf.predict(docs_test)

    mean_value = mean(data_testing.target == predicted_target)
    print_result(predicted_target=predicted_target, data_testing_set=data_testing, mean_value=mean_value,
                 verbose=verbose)

    print_footer("done testing: naive_bayes -> MultinomialNB")
    return mean_value


def test_bernoullinb(data_training, data_testing, verbose=False):
    print_header("testing: naive_bayes -> BernoulliNB")

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf', BernoulliNB())])

    text_clf.fit(data_training.data, data_training.target)
    docs_test = data_testing.data
    predicted_target = text_clf.predict(docs_test)

    mean_value = mean(data_testing.target == predicted_target)
    print_result(predicted_target=predicted_target, data_testing_set=data_testing, mean_value=mean_value,
                 verbose=verbose)

    print_footer("done testing: naive_bayes -> BernoulliNB")
    return mean_value


def test_sgdclassifier(data_training, data_testing, verbose=False):
    print_header("testing: linear_model -> SGDClassifier")

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 20))),
                         ('tfidf', TfidfTransformer(use_idf=False)),
                         ('clf', SGDClassifier(
                             loss='hinge',
                             penalty='l2',
                             alpha=1e-3,
                             n_iter=5,
                             random_state=99)
                          )])

    text_clf.fit(data_training.data, data_training.target)
    docs_test = data_testing.data
    predicted_target = text_clf.predict(docs_test)

    mean_value = mean(data_testing.target == predicted_target)
    print_result(predicted_target=predicted_target, data_testing_set=data_testing, mean_value=mean_value,
                 verbose=verbose)

    print_footer("done testing: linear_model -> SGDClassifier")
    return mean_value


def test_logisticregression(data_training, data_testing, verbose=False):
    print_header("testing: linear_model -> LogisticRegression")

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf', LogisticRegression(
                             penalty='l2',
                             random_state=0)
                          )])

    text_clf.fit(data_training.data, data_training.target)
    docs_test = data_testing.data
    predicted_target = text_clf.predict(docs_test)

    mean_value = mean(data_testing.target == predicted_target)
    print_result(predicted_target=predicted_target, data_testing_set=data_testing, mean_value=mean_value,
                 verbose=verbose)

    print_footer("done testing: linear_model -> LogisticRegression")
    return mean_value


def test_svc(data_training, data_testing, verbose=False):
    print_header("testing: svm -> SVC")

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf', SVC(C=15, kernel='linear', random_state=0))])

    text_clf.fit(data_training.data, data_training.target)
    docs_test = data_testing.data
    predicted_target = text_clf.predict(docs_test)

    mean_value = mean(data_testing.target == predicted_target)
    print_result(predicted_target=predicted_target, data_testing_set=data_testing, mean_value=mean_value,
                 verbose=verbose)

    print_footer("done testing: svm -> SVC")
    return mean_value


def test_linearsvc(data_training, data_testing, verbose=False):
    print_header("testing: svm -> LinearSVC")

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf', LinearSVC(C=15, random_state=0))])

    text_clf.fit(data_training.data, data_training.target)
    docs_test = data_testing.data
    predicted_target = text_clf.predict(docs_test)

    mean_value = mean(data_testing.target == predicted_target)
    print_result(predicted_target=predicted_target, data_testing_set=data_testing, mean_value=mean_value,
                 verbose=verbose)

    print_footer("done testing: svm -> LinearSVC")
    return mean_value


def test_nusvc(data_training, data_testing, verbose=False):
    print_header("testing: svm -> NuSVC")

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf', NuSVC(kernel='linear', random_state=0))])

    text_clf.fit(data_training.data, data_training.target)
    docs_test = data_testing.data
    predicted_target = text_clf.predict(docs_test)

    mean_value = mean(data_testing.target == predicted_target)
    print_result(predicted_target=predicted_target, data_testing_set=data_testing, mean_value=mean_value,
                 verbose=verbose)

    print_footer("done testing: svm -> NuSVC")
    return mean_value


def print_bruteforce_results(gs_clf, parameters, name):
    print_header("Bruteforce: " + str(name))

    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])

    print("result: ", score)
    print("for the following parameters:")
    for param_name in sorted(parameters.keys()):
        print("\t" + str(param_name) + ": " + str(best_parameters[param_name]))
    print("")

    print_footer("done Bruteforce: " + str(name))


def bruteforce_naive_bayes_badass_multinomialnb(ngram_min=(1, 1), ngram_max=(1, 5),
                                                jobs=2):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])

    parameters = {
        'vect__ngram_range': [ngram_min, ngram_max],
        'tfidf__use_idf': (True, False)
    }

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=jobs)
    gs_clf = gs_clf.fit(data_training.data, data_training.target)

    print_bruteforce_results(gs_clf, parameters, "Naive Bayes MultinomialNB badass bruteforce")


def bruteforce_naive_bayes_badass_bernoullinb(ngram_min=(1, 1), ngram_max=(1, 5),
                                              jobs=2):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', BernoulliNB())])

    parameters = {
        'vect__ngram_range': [ngram_min, ngram_max],
        'tfidf__use_idf': (True, False)
    }

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=jobs)
    gs_clf = gs_clf.fit(data_training.data, data_training.target)

    print_bruteforce_results(gs_clf, parameters, "Naive Bayes BernoulliNB badass bruteforce")


def bruteforce_linear_model_badass_sgdclassifier(random_min=0, random_max=999,
                                                 ngram_min=(1, 1), ngram_max=(1, 5),
                                                 jobs=2):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier())])

    parameters = {
        'clf__random_state': (random_min, random_max),
        'vect__ngram_range': [ngram_min, ngram_max],
        'tfidf__use_idf': (True, False)
    }

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=jobs)
    gs_clf = gs_clf.fit(data_training.data, data_training.target)

    print_bruteforce_results(gs_clf, parameters, "linear model SGDClassifier badass bruteforce")


def bruteforce_linear_model_badass_logisticregression(random_min=0, random_max=999,
                                                      ngram_min=(1, 1), ngram_max=(1, 5),
                                                      jobs=2):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LogisticRegression())])

    parameters = {
        'clf__random_state': (random_min, random_max),
        'vect__ngram_range': [ngram_min, ngram_max],
        'tfidf__use_idf': (True, False)
    }

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=jobs)
    gs_clf = gs_clf.fit(data_training.data, data_training.target)

    print_bruteforce_results(gs_clf, parameters, "linear model LogisticRegression badass bruteforce")


def bruteforce_svm_svc_badass(random_min=0, random_max=999,
                              ngram_min=(1, 1), ngram_max=(1, 5),
                              c_min=9, c_max=10,
                              jobs=2):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SVC())])

    parameters = {
        'clf__random_state': (random_min, random_max),
        'clf__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
        'vect__ngram_range': [ngram_min, ngram_max],
        'tfidf__use_idf': (True, False),
        'clf__C': [c_min, c_max]
    }

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=jobs)
    gs_clf = gs_clf.fit(data_training.data, data_training.target)

    print_bruteforce_results(gs_clf, parameters, "svm SVC badass bruteforce")


def bruteforce_svm_linearsvc_badass(random_min=0, random_max=999,
                                    ngram_min=(1, 1), ngram_max=(1, 5),
                                    c_min=9, c_max=10,
                                    jobs=2):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC())])

    parameters = {
        'clf__random_state': (random_min, random_max),
        'vect__ngram_range': [ngram_min, ngram_max],
        'tfidf__use_idf': (True, False),
        'clf__C': [c_min, c_max]
    }

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=jobs)
    gs_clf = gs_clf.fit(data_training.data, data_training.target)

    print_bruteforce_results(gs_clf, parameters, "svm LinearSVC badass bruteforce")


def bruteforce_svm_nusvc_badass(random_min=0, random_max=999,
                                ngram_min=(1, 1), ngram_max=(1, 5),
                                jobs=2):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', NuSVC())])

    parameters = {
        'clf__random_state': (random_min, random_max),
        'clf__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
        'vect__ngram_range': [ngram_min, ngram_max],
        'tfidf__use_idf': (True, False)
    }

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=jobs)
    gs_clf = gs_clf.fit(data_training.data, data_training.target)

    print_bruteforce_results(gs_clf, parameters, "svm NuSVC badass bruteforce")


def print_result(predicted_target, data_testing_set, mean_value, verbose=False):
    print("Result: ", mean_value)

    if verbose:
        print("")
        print("Classification Metrics:")
        print(metrics.classification_report(
            data_testing_set.target,
            predicted_target,
            target_names=data_testing_set.target_names))
        print("")
        print("Confusion Matrix:")
        print(metrics.confusion_matrix(data_testing_set.target, predicted_target)[0])
        print(metrics.confusion_matrix(data_testing_set.target, predicted_target)[1])


def print_header(header_name):
    print("")
    print(">>>>>>>>>>>")
    print(header_name)
    print(">>>>>>>>>>>")
    print("")


def print_footer(footer_name):
    print("")
    print("<<<<<<<<<<<<")
    print(footer_name)
    print("<<<<<<<<<<<<")
    print("")


if __name__ == "__main__":
    """
    This function is run if this file is run directly.
    """
    path = "data/movies_tagged"
    process_path = "data/movies_processed"
    percent_training_testing = 0.8
    filter_list = ["NOM", "VER", "ADV", "ADJ"]
    results_list = []
    verbose = True
    parallel_jobs = 6

    print("Bello, I am grumpy, and I hate you")
    print("I will work on the path: " + str(path))
    data_categories = get_categories(path)

    print("Let's process data now to: " + str(process_path))
    process_raw_data(path, process_path, data_categories, filter_list)

    print("I will be loading the data now")
    data = load_data(path, data_categories)

    print("Okay.. Now let's have a " + str(percent_training_testing) +
          " percent training and the rest for the testing sets")
    data_training, data_testing = split_data(data, percent_training_testing)

    print("Voila voila... Let's have some fun now... Tests tests tests tests!!!")
    results_list.append(["MultinomialNB", test_multinomialnb(data_training, data_testing, verbose=verbose)])
    results_list.append(["BernoulliNB", test_bernoullinb(data_training, data_testing, verbose=verbose)])
    results_list.append(["SGDClassifier", test_sgdclassifier(data_training, data_testing, verbose=verbose)])
    results_list.append(["LogisticRegression", test_logisticregression(data_training, data_testing, verbose=verbose)])
    results_list.append(["SVC", test_svc(data_training, data_testing, verbose=verbose)])
    results_list.append(["LinearSVC", test_linearsvc(data_training, data_testing, verbose=verbose)])
    results_list.append(["NuSVC", test_nusvc(data_training, data_testing, verbose=verbose)])

    print("Now let's sort them...")
    print(sorted(results_list, key=lambda x: x[1], reverse=True))

    print("Trying to crash your machine! Just kidding, i am bruteforcing the parameters :)")
    # here ngram_max is important
    bruteforce_naive_bayes_badass_multinomialnb(ngram_min=(1, 1),
                                                ngram_max=(1, 30),
                                                jobs=parallel_jobs)

    bruteforce_naive_bayes_badass_bernoullinb(ngram_min=(1, 1),
                                              ngram_max=(1, 10),
                                              jobs=parallel_jobs)

    # here random & ngram are important
    bruteforce_linear_model_badass_sgdclassifier(random_min=0, random_max=99999,
                                                 ngram_min=(1, 1), ngram_max=(1, 30),
                                                 jobs=parallel_jobs)

    bruteforce_linear_model_badass_logisticregression(random_min=0, random_max=1,
                                                      ngram_min=(1, 1), ngram_max=(1, 10),
                                                      jobs=parallel_jobs)

    bruteforce_svm_svc_badass(random_min=0, random_max=1,
                              ngram_min=(1, 1), ngram_max=(1, 10),
                              c_min=1, c_max=50,
                              jobs=parallel_jobs)

    bruteforce_svm_linearsvc_badass(random_min=0, random_max=1,
                                    ngram_min=(1, 1), ngram_max=(1, 30),
                                    c_min=1, c_max=50,
                                    jobs=parallel_jobs)

    bruteforce_svm_nusvc_badass(random_min=0, random_max=1,
                                ngram_min=(1, 1), ngram_max=(1, 10),
                                jobs=parallel_jobs)
