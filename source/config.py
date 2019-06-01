'''
Configuration File
'''

## DATA FILE / LABEL
#Define constants
CSVFOLDER="../ncdoc_data/data/preprocessed/"
DATABASE_FILENAME=CSVFOLDER + "inmates.db"
DATA_DIR = CSVFOLDER + "traintest"

RESULTS_DIR = "results"
GRAPH_FOLDER = 'graphs'
RESULTS_FILE = "results.csv"
TRAIN_TEST_COL = 'year'
SEED = 0

# done with race, gender, age, incarceration lens, county, minmaxterm, sentences

VARIABLES = {
             'LABEL' : 'LABEL',
#             'TO_DISCRETIZE' : [{'NUM_SENTENCES': (3, ['low','medium','high'])}],
             'DATES' : ['START_DATE', 'END_DATE'],
             'MISSING' : {'MISSING_CAT': ['INMATE_RACE_CODE', 'INMATE_GENDER_CODE'],
                          'AGE': ['AGE_AT_START_DATE', 'AGE_AT_END_DATE','AGE_FIRST_SENTENCE', 
                                  'AGE_AT_OFFENSE_START', 'AGE_AT_OFFENSE_END'],
                          'IMPUTE_MEAN': ['INCARCERATION_LEN_DAYS','TOTAL_INCARCERATION_ALLPRIOR', 
                                          'AVG_INCARCERATION_ALLPRIOR', 'TOTAL_INCARCERATION_LAST5YR', 
                                          'AVG_INCARCERATION_LAST5YR'],
                          'IMPUTE_ZERO': ['INFRACTIONS', 'INFRACTIONS_UNIQUE', 'INFRACTIONS_GUILTY',
                                          'INFRACTIONS_LAST_INCAR', 'INFRACTIONS_LAST_INCAR_GUILTY']
                           },
             'INDICATOR': {'incorrect': ['INCARCERATION_LEN_DAYS'],
                           'missing': ['AGE_AT_START_DATE', 'AGE_AT_END_DATE','AGE_FIRST_SENTENCE', 
                                       'AGE_AT_OFFENSE_START', 'AGE_AT_OFFENSE_END']
                           },
             'CONTINUOUS_VARS_MINMAX' : ['INCARCERATION_LEN_DAYS',
                                         'TOTAL_INCARCERATION_ALLPRIOR', 'NUM_PREV_INCARCERATION_ALLPRIOR', 
                                         'AVG_INCARCERATION_ALLPRIOR','TOTAL_INCARCERATION_LAST5YR', 
                                         'NUM_PREV_INCARCERATION_LAST5YR', 'AVG_INCARCERATION_LAST5YR',
                                         'NUM_SENTENCES', 'TOTAL_SENT_ALLPRIOR', 'NUM_PREV_SENT_ALLPRIOR', 
                                         'AVG_SENT_ALLPRIOR', 'TOTAL_SENT_LAST5YR', 'NUM_PREV_SENT_LAST5YR', 
                                         'AVG_SENT_LAST5YR'],
             'CATEGORICAL_VARS' : ['MINMAXTERM','INMATE_RACE_CODE', 'INMATE_GENDER_CODE',
                                   'PREFIX'],
             'SPECIAL_DUMMY': ['COUNTY_CONVICTION'],
             'VARS_TO_EXCLUDE' : ['ID', 'START_DATE', 'END_DATE', 'LABEL','SENTENCE_YEAR',
                                  'INMATE_RACE_CODE', 'INMATE_GENDER_CODE', 
                                  'PREFIX'],
             'NO_CLEANING_REQ': ['PREV_INCAR_INDIC', 'LABEL']
             }


## RUNNING THE MODELS
GRIDSIZE = 'test'
MODELS = ['LR']
#MODELS = ['RF', 'ET', 'GB', 'AB', 'BAG', 'DT', 'KNN', 'LR', 'SVM', 'NB']
EVAL_METRICS_BY_LEVEL = (['accuracy', 'precision', 'recall', 'f1'],\
                         [1,2,5,10,20,30,50])
EVAL_METRICS = ['auc']
# plot pr: save or show or None
PLOT_PR = 'save'


def define_clfs_params(grid_size):
    """
    This functions defines parameter grid for all the classifiers
    Inputs:
       grid_size: how big of a grid do you want. it can be test, small, or large
    Returns a set of model and parameters
    Raises KeyError: Raises an exception.
    """

    large_grid = { 
    'RF':   {'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],
             'min_samples_split': [2,5,10,50,100], 'n_jobs': [-1], 'random_state': [SEED]},
    'ET':   {'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100],
             'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10,50,100], 'n_jobs': [-1], 'random_state': [SEED]},
    'AB':   {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000], 'random_state': [SEED]},
    'GB':   {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0],
             'max_depth': [1,5,10,20,50,100], 'random_state': [SEED]},
    'KNN':  {'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'DT':   {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'min_samples_split': [2,5,10,50,100], 'random_state': [SEED]},
    'SVM':  {'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10], 'random_state': [SEED]},
    'LR':   {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10], 'random_state': [SEED]},
    'BAG':  {'n_estimators': [1,10,100,1000,10000], 'n_jobs': [-1], 'random_state': [SEED]},
    'NB':   {'alpha': [0.00001,0.0001,0.001,0.01,0.1,1,10], 'fit_prior': [True, False]}
           }
    
    small_grid = {
    'RF':   {'n_estimators': [10,100,1000], 'max_depth': [1,5,10,20], 'max_features': ['sqrt','log2'],
             'min_samples_split': [2,10,50], 'n_jobs': [-1], 'random_state': [SEED]},
    'ET':   {'n_estimators': [10,100,1000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20],
             'max_features': ['sqrt','log2'],'min_samples_split': [2,10,50], 'n_jobs': [-1], 'random_state': [SEED]},
    'AB':   {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [10,100,1000], 'random_state': [SEED]},
    'GB':   {'n_estimators': [10,100,1000], 'learning_rate' : [0.001,0.01],'subsample' : [0.1,0.5],
             'max_depth': [1,5,10,20], 'random_state': [SEED]},
    'KNN':  {'n_neighbors': [1,10,25,50],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'DT':   {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20], 'min_samples_split': [2,10,50], 'random_state': [SEED]},
    'SVM':  {'C' :[0.01,0.1,1,10], 'random_state': [SEED]},
    'LR':   {'penalty': ['l1','l2'], 'C': [0.01,0.1,1,10], 'random_state': [SEED]},
    'BAG':  {'n_estimators': [10,100,1000], 'n_jobs': [-1], 'random_state': [SEED]},
    'NB':   {'alpha': [0.01,0.1,1,10], 'fit_prior': [True, False]}
            }
       
    test_grid = {
    'RF':   {'n_estimators': [1], 'max_depth': [5], 'max_features': ['sqrt'], 'min_samples_split': [10], 'n_jobs': [-1], 'random_state': [SEED]},
    'ET':   {'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [5],
             'max_features': ['sqrt'],'min_samples_split': [10], 'n_jobs': [-1], 'random_state': [SEED]},
    'AB':   {'algorithm': ['SAMME.R'], 'n_estimators': [5], 'random_state': [SEED]},
    'GB':   {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [5], 'random_state': [SEED]},
    'KNN':  {'n_neighbors': [1],'weights': ['uniform'],'algorithm': ['auto']},
    'DT':   {'criterion': ['gini'], 'max_depth': [5], 'min_samples_split': [10], 'random_state': [SEED]},
    'SVM':  {'C' :[10], 'random_state': [SEED]},
    'LR':   {'penalty': ['l1'], 'C': [10], 'random_state': [SEED]},
    'BAG':  {'n_estimators': [1], 'n_jobs': [-1], 'random_state': [SEED]},
    'NB':   {'alpha': [1], 'fit_prior': [True]}    
            }

    
    if (grid_size == 'large'):
        return large_grid
    elif (grid_size == 'small'):
        return small_grid
    elif (grid_size == 'test'):
        return test_grid
    else:
        return 0, 0