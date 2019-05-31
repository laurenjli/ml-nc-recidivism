#!/usr/bin/env python
# coding: utf-8

## Importing Packeges
import pandas as pd
import numpy as np
import requests
import math
import sys
import graphviz 
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn import tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier,RandomForestClassifier
from sklearn.model_selection import ParameterGrid 

from sklearn.metrics import accuracy_score as accuracy, confusion_matrix, f1_score, auc, roc_auc_score, precision_score
from sklearn.metrics import roc_curve, precision_recall_curve

import os; 


## Get Data

def get_csv(data_dir, file_name):
    '''
    Description: This function takes a csv file and uploads it into a pandas dataframe
    Input:
        csv: file to upload
    Output:
        pandas data frame
    '''

    df = pd.read_csv(os.path.join(data_dir, file_name))
    return df

## Pre-ProcessData

def remove_outliers(df, attribute_lst, sd_threshold=3):
    '''
    Takes a dataframe and number of standard deviations to be considered
    as outlier and returns a df without the observation that have one or
    or more outliers in the attributes selected.
    input:
    df: pandas data frame
    attributes_lst: list of attributes names
    sd_threshold: standard deviations
    output:
    the new datafrane without outliers
    '''
    zscore = lambda x: (x - x.mean())/x.std(ddof=0)
    for attribute in attribute_lst:
        df[attribute] = df[attribute][(np.abs(zscore(df[attribute])) < sd_threshold)]
    return df


def fill_nan(df, attributes_lst, how='mean'):
    '''
    Fills the nan with the "how" specified.
    input:
        df: pandas data frame
        attributes_lst: list of attributes names
        how: {mean, max, min or median} of that attribute in the sample
    output:
        dataframe with the replaced nan
    '''   
    if how == 'mean':
        for attribute in attributes_lst: 
            df[attribute].fillna(df[attribute].mean(), inplace=True)
    elif how == 'max':
        for attribute in attributes_lst: 
            df[attribute].fillna(df[attribute].max(), inplace=True)
    elif how == 'min':
        for attribute in attributes_lst: 
            df[attribute].fillna(df[attribute].min(), inplace=True)
    elif how == 'median':
        for attribute in attributes_lst: 
            df[attribute].fillna(df[attribute].median(), inplace=True)
    else:
        raise Exception("This function only allows to fill the Nan with\
                        the mean, max, min or median of the observations")

        
def to_int(df, attribute_lst):
    '''
    Converts the data type of a string to an integer if possible or another\
    type of numberic data if not.
    input:
        df: pandas data frame
        attributes_lst: list of attributes names
    output:
        a df with the corresponding numeric variables
    '''
    
    for var in attribute_lst:
        df[var] = pd.to_numeric(df[var], errors='coerce', downcast='integer') 
    return df

       

def to_date(df, attribute_lst, years_range=[1000, 3000]):
    '''
    Converts the data type of a string in the format YYYY-MM-DD to a datetime and\
    replace to None the dates that fall outsithe the range of years specified
    input:
        df: pandas data frame
        attributes_lst: list of attributes names
        year_range: a tupple or list where the first year if the lowest bound \
        and the second is the highest bound
    output:
        a df with the corresponding None
    '''
    
    df = df.apply(out_of_range_to_none, axis=1, args=(years_range, attribute_lst))
    for var in attribute_lst:
        df[var] = df[var].astype('datetime64[s]')#, errors = 'ignore')
    return df


def out_of_range_to_none(row, year_range, attributes_lst): 
    '''
    Takes a row and a lists of columns and checks if the value falls out of the \
    intended range. When it does, it converts it to None.
    input:
        row: a series that represents a row
        year_range: a tupple or list where the first year is the lowest bound \
        and the second is the highest bound of the range. Both years are included,\
        i.e. the bounds are not converted to None.
        attributes_lst: list of attributes names
    output:
        a row with the corresponding None
    '''
    
    for col in attributes_lst:
        year = int(row[col].split("/")[2])
        if year < year_range[0] or year > year_range[1]:
            row[col] = None 
    return row


## Generate Features/ Predictors


def discretize_variable(df, attribute_lst):
    '''
    Converts continuous variables into discrete variables
    input:
        df: pandas data frame
        attributes_lst: list of attributes names
    output:
        dataframe with the new variables
    ''' 

    for var in attribute_lst:
        new_var = var + 'cat'
        df[new_var] = pd.qcut(df[var], 10, duplicates="drop", labels=False)
    return df

def categorical_to_dummy(df, attribute_lst):
    '''
    Converts categorical variables into one variabel dummies for each category. 
    input:
        df: pandas data frame
        attributes_lst: list of attributes names
    output:
        dataframe with the new variables
    ''' 

    for var in attribute_lst:
        df = pd.get_dummies(df, columns=[var], dummy_na=True)
    return df

def flag_to_dummy(df, attribute_lst, rename=True):
    '''
    Converts a flag variable to a dummy with 1 for Yes and 0 for No
    '''
    for var in attribute_lst:
        df[var] = df[var].map({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, 'T': 1, 'F': 0, 't': 1, 'f': 0,
                                     'True': 1, 'False': 0, True: 1, False: 0, 'OPEN': 1, 'CLOSED': 0})
        if rename:
            new_var_name = var[:-5]
            df.rename(index=str, columns={var: new_var_name}, inplace=True)
    return df

def gender_to_dummy(df, gender_var):  
    '''
    Converts a gender indicative variable to a dummy with 1 for female and 0 for male
    '''
    df[gender_var] = df[gender_var].map({'FEMALE': 1, 'MALE': 0, 'F': 1, 'M': 0})
    df.rename(index=str, columns={gender_var: "FEMALE"}, inplace=True)
    return df



## Classification

### Classifiers



classifiers = { 'LR': LogisticRegression(penalty='l1', C=1e5),
                'KNN': KNeighborsClassifier(n_neighbors=3),
                'DT': DecisionTreeClassifier(),
                'SVM': LinearSVC(random_state=0, tol=1e-5),
                'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
                'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
                'BA': BaggingClassifier()} 


### Evaluation Metrics


def plot_precision_recall_n(y_true, y_scores, filename=None):
    '''
    This function creates a graph that represent the precision and recall 
    at different point of the threshold. 
    Input:
        y_true: np.array with the observed Ys 
        y_scores: np.array with the predicted scores 
    Output:
        The graph
    '''
    #precision, recall, tresholds = precision_recall_curve(y_test, pred_scoresp[:,1], pos_label=1)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores, pos_label=1)
    
    #plt.plot(recall, precision, marker='.')
    #plt.show()
    population = [1.*sum(y_scores>threshold)/len(y_scores) for threshold in thresholds]+[0]
    p, = plt.plot(population, precision, color ='b')
    r, =  plt.plot(population, recall, color ='r')
    plt.legend([p,r],['precision', 'recall'])
    plt.show()
    if filename is not None:
        plt.savefig(filename)

    
def f1_at_threshold(y_true, y_predicted):
    '''
    This function calculates the evaluation metric F! for a certain level 
    of possitive labeled observations. 
    Input:
        y_true: np.array with the observed Ys 
        y_predicted: np.array with the predicted Ys 
    Output:
        The F1 score
    '''
    return  f1_score(y_true, y_predicted)


def accuracy_at_threshold(y_true, y_predicted):
    '''
    This function calculates the evaluation metric called "accuracy" for a 
    certain level of possitive labeled observations. 
    Input:
        y_true: np.array with the observed Ys 
        y_predicted: np.array with the predicted Ys
    Output:
        The Accuracy score
    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    return 1.0 * (tp + tn) / (tn + fp + fn + tp )

def recall_at_threshold(y_true, y_predicted):
    '''
    This function calculates the evaluation metric called "recall" for a 
    certain level of possitive labeled observations. 
    Input:
        y_true: np.array with the observed Ys 
        y_predicted: np.array with the predicted Ys
    Output:
        The Recall score
    '''
    _, _, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    return 1.0 * tp / (tp + fn)

def precision_at_threshold(y_true, y_predicted):
    '''
    This function calculates the evaluation metric called "precision" for a 
    certain level of possitive labeled observations. 
    Input:
        y_true: np.array with the observed Ys 
        y_predicted: np.array with the predicted Ys
    Output:
        The Precision score
    '''
    
    _, fp, _, tp = confusion_matrix(y_true, y_predicted).ravel()
    return 1.0 * tp / (tp + fp)

def pred_at_level(y_true, y_scores, level):
    '''
    This function takes the predicted score and converts it into label 1 or 0
    based on the level -percentage of observations- decided to include, e.i. label 1. 
    Input:
        y_true: np.array with the observed Ys 
        y_scores: np.array with the predicted scores 
        level: percentage of the population labeled 1
    Output:
        The predicted label {0, 1}
    '''
    
    idx = np.argsort(np.array(y_scores))[::-1]
    y_scores, y_true = np.array(y_scores)[idx], np.array(y_true)[idx]
    cutoff_index = int(len(y_scores) * (level / 100.0))
    y_preds_at_level = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return y_true, y_preds_at_level


metrics = { 'accuracy':accuracy_at_threshold,
            'precision':precision_at_threshold,
            'recall':recall_at_threshold,
            'f1':f1_at_threshold,
            'auc':roc_auc_score}


### Master Classifier


def classify(train_set, test_set, label, models, eval_metrics, eval_metrics_by_level, custom_grid, attributes_lst):
    '''
    This function fits a set of classifiers and a dataframe with performance measures for each
    Input:
        train_set, test_set: dataframe for training and testing the models
        label: name of the Y variable
        n_samples: number of times that the data will be partitioned in training and testing sets.
        eval_metrics: list of threshold-independent metrics.
        eval_metrics_by_level: tuple containing a list of threshold-dependent metrics as first element and a list of thresholds as second element
        attributes_lst: list containing the names of the features (i.e. X variables) to be used.
    Output:
        Dataframe containing performance measures for each classifier
    '''
    results_columns = ['model','classifiers', 'parameters'] + eval_metrics + [metric + '_' + str(level) for level in eval_metrics_by_level[1] for metric in eval_metrics_by_level[0]]
    results =  pd.DataFrame(columns=results_columns)
    y_train = train_set[label]
    X_train = train_set.loc[:, attributes_lst]
    y_test = test_set[label]
    X_test = test_set.loc[:, attributes_lst]
    for model in models:
        grid = ParameterGrid(custom_grid[model])
        for parameters in grid:
            classifier = classifiers[model]
            print(classifier)
            clfr = classifier.set_params(**parameters)
            clfr.fit(X_train, y_train)
            eval_result = [model, classifier, parameters]
            y_pred_prob = clfr.predict_proba(X_test)[:,1]
            #plot_precision_recall_n(y_test, y_pred_prob, model+'.png')
            if eval_metrics:
                eval_result += [metrics[metric](y_test, y_pred_prob) for metric in eval_metrics]
            if eval_metrics_by_level[0]:
                for level in eval_metrics_by_level[1]:
                   y_test, y_pred = pred_at_level(y_test, y_pred_prob, level)
                   eval_result += [metrics[metric](y_test, y_pred) for metric in eval_metrics_by_level[0]]
             
            results.loc[len(results)] = eval_result
    return results
