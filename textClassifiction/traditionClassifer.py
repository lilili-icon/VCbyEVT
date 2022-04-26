# Import libraries
import pandas as pd
import numpy as np
import time
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
from sklearn import metrics



# Load datasets
df_train = pd.read_csv('data/with/train.csv')
df_test = pd.read_csv('data/with/test.csv')
# df_test = pd.read_csv('data/test.csv')
X = df_train.description.values
X_Test = df_test.description.values

# Extract NLP features
# config 1: Bag-of-word without tf-idf


def extract_features(config, start_word_ngram, end_word_ngram):
    if config == 1:
        return TfidfVectorizer(stop_words=['aka'], ngram_range=(1, 1), use_idf=False, min_df=0.001,
                               norm=None, smooth_idf=False, token_pattern=r'\S*[A-Za-z]\S+')

    return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_word_ngram, end_word_ngram), use_idf=True,
                           min_df=0.001, norm='l2', token_pattern=r'\S*[A-Za-z]\S+')


# Build the Classifiers
def build_classifiers(config):
    clfs = {'NB': MultinomialNB(),
            'SVM': OneVsRestClassifier(LinearSVC(random_state=42, C=0.1, max_iter=1000), n_jobs=-1)}

    clfs['LR'] = LogisticRegression(C=0.1, multi_class='ovr', n_jobs=-1, solver='lbfgs', max_iter=1000,
                                    random_state=42)

    clfs['RF'] = RandomForestClassifier(n_estimators=100, max_depth=None, max_leaf_nodes=None, random_state=42,
                                        n_jobs=-1)
    clfs['XGB'] = XGBClassifier(objective='multiclass', max_depth=0, max_leaves=5, grow_policy='lossguide',
                                n_jobs=-1, random_state=42, tree_method='hist')
    clfs['LGBM'] = LGBMClassifier(num_leaves=5, max_depth=-1, objective='multiclass', n_jobs=-1, random_state=42)

    return clfs


# Extract the n-grams and transform n-grams into the feature vectors

def feature_model(X_train, X_test, y_test, config, start_word_ngram, end_word_ngram):
    # Create vectorizer
    vectorizer = extract_features(config=config, start_word_ngram=start_word_ngram, end_word_ngram=end_word_ngram)

    X_train = X_train.astype(str)
    X_test = X_test.astype(str)

    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)

    # Remove rows with all zero values
    test_df = pd.DataFrame(X_test_transformed.todense())
    results = test_df.apply(lambda x: x.value_counts().get(0.0, 0), axis=1)
    non_zero_indices = np.where(results < len(test_df.columns))[0]

    X_train_transformed = X_train_transformed.astype(np.float64)
    X_test_transformed = X_test_transformed.astype(np.float64)

    return X_train_transformed, X_test_transformed[non_zero_indices], y_test[non_zero_indices]


# Evaluate the models with Accuracy, Macro and Weighted F-Scores

def evaluate(clf_name,clf, X_train_transformed, X_test_transformed, y_train, y_test):
    clf.fit(X_train_transformed, y_train)
    print(clf_name+"****************")
    y_pred = clf.predict(X_test_transformed)
    report = metrics.classification_report(y_test,y_pred,digits=4)
    return report
    # return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average=None),\
    #        recall_score(y_test, y_pred,average=None),f1_score(y_test, y_pred, average=None)

def validate_data(clf_name,clf, y,y_test, config, start_word_ngram, end_word_ngram):

    t_start = time.clock()

    #y只是标签，x是描述
    X_train, X_test = X, X_Test
    y_train, y_test = y, y_test

     # word n-gram generation
    X_train_transformed, X_test_transformed, y_test = feature_model(X_train, X_test, y_test, config,
                                                                        start_word_ngram, end_word_ngram)

    # training and evaluation
    results = evaluate(clf_name,clf, X_train_transformed, X_test_transformed, y_train, y_test)
    print(results)

    val_time = time.clock() - t_start

    return results


labels = ['type']

# config 1: Bag-of-word without tf-idf
# config 2: Bag-of-word with tf-idf
# config 3: N-gram without tf-idf
# config 4: N-gram with tf-idf

result_file = 'val_without_time.txt'

configs = [1]

with open(result_file, 'w') as fout:
    for config in configs:

        print("Current config:", config)
        fout.write("Current config:" + str(config) + "\n")

        start_word_ngram = 1
        end_word_ngram = 1

        if config == 1:
            print("Bag-of-word without tf-idf")
            fout.write("Bag-of-word without tf-idf\n")
            start_word_ngram = 1
            end_word_ngram = 1
        elif config == 2:
            print("Bag-of-word with tf-idf")
            fout.write("Bag-of-word with tf-idf\n")
            start_word_ngram = 1
            end_word_ngram = 1
        elif config <= 5:
            print("N-gram without tf-idf")
            fout.write("N-gram without tf-idf\n")
            if config == 3:
                start_word_ngram = 1
                end_word_ngram = 2
            elif config == 4:
                start_word_ngram = 1
                end_word_ngram = 3
            elif config == 5:
                start_word_ngram = 1
                end_word_ngram = 4
        else:
            print("N-gram with tf-idf")
            fout.write("N-gram with tf-idf\n")
            if config == 6:
                start_word_ngram = 1
                end_word_ngram = 2
            elif config == 7:
                start_word_ngram = 1
                end_word_ngram = 3
            elif config == 8:
                start_word_ngram = 1
                end_word_ngram = 4

        clfs = build_classifiers(config=config)

        for label in labels:
            print("Current output:", label, "\n")

            fout.write("Current output:" + label + "\n")

            print(
                "Classifier\t Acc\t p\t r\t f1\t Val Time\n")
            fout.write(
                "Classifier\t Acc\t p\t r\t f1\t Val Time\n\n")

            for clf_name, clf in clfs.items():
                print(clf_name + "\t" + "", end='')

                fout.write(clf_name + "\t")

                y = df_train.label.values
                y_test = df_test.label.values

                val_res = validate_data(clf_name,clf, y,y_test, config, start_word_ngram, end_word_ngram)

                fout.write(val_res + "\n")

            print("------------------------------------------------\n")
            fout.write("------------------------------------------------\n\n")

        print("##############################################\n")
        fout.write("##############################################\n\n")