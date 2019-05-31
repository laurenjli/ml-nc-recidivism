'''
Configuration File
'''

## DATA FILE / LABEL
DATAFILE = "data/projects_2012_2013.csv"
OUTCOME = "notfullyfundedin60days"

## CHANGE DATA TYPES
TO_DATE = ["date_posted", "datefullyfunded"]

## TEMPORAL VALIDATION
PREDICTION_PERIOD = 60  # in days
TEST_PERIOD = 180       # in days
DATE_TO_SPLIT = "date_posted"
TRAIN_TEST_DATES = [[(2012, 1, 1), (2012, 7, 1)],
                    [(2012, 1, 1), (2013, 1, 1)],
                    [(2012, 1, 1), (2013, 7, 1)]]
TRAIN_TEST_LABELS = ["jul12", "jan13", "jul13"]

## DATA CLEANING
TO_DISCRETIZE = ['total_price_including_optional_support', 'students_reached']
DISCRETE_LEVELS = [(3, ['low','medium','high']), (3,['low','medium','high']) ]
MISSING = ['school_metro', 'school_district', 'primary_focus_subject', 'primary_focus_area',
           'secondary_focus_subject', 'secondary_focus_area', 'resource_type',
           'grade_level', 'students_reached']
CATEGORICAL = ['school_city', 'school_state', 'school_metro', 'school_district', 
            'school_county', 'school_charter', 'school_magnet', 'teacher_prefix', 
            'primary_focus_subject', 'primary_focus_area', 'secondary_focus_subject', 
            'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level',
            'total_price_including_optional_support', 'students_reached',
            'eligible_double_your_impact_match']
CONTINUOUS = ['school_latitude', 'school_longitude']

## FEATURES
FEATURES = CATEGORICAL + CONTINUOUS

## EVALUATION AND RESULTS
METRICS_THRESHOLD = (['precision', 'recall', 'f1'], [1,2,5,10,20,30,50])
OTHER_METRICS = ['auc']

## RUNNING THE MODELS
GRIDSIZE = 'test'
OUTFILE = "results_"+GRIDSIZE+".csv"
MODELS = ['RF', 'ET', 'GB', 'AB', 'BAG', 'DT', 'KNN', 'LR', 'SVM', 'NB']
SEED = 0
GRAPH_FOLDER = 'graphs/'


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
    'RF':   {'n_estimators': [100], 'max_depth': [5], 'max_features': ['sqrt'], 'min_samples_split': [10], 'n_jobs': [-1], 'random_state': [SEED]},
    'ET':   {'n_estimators': [100], 'criterion' : ['gini'] ,'max_depth': [5],
             'max_features': ['sqrt'],'min_samples_split': [10], 'n_jobs': [-1], 'random_state': [SEED]},
    'AB':   {'algorithm': ['SAMME.R'], 'n_estimators': [5], 'random_state': [SEED]},
    'GB':   {'n_estimators': [5], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [5], 'random_state': [SEED]},
    'KNN':  {'n_neighbors': [1],'weights': ['uniform'],'algorithm': ['auto']},
    'DT':   {'criterion': ['gini'], 'max_depth': [5], 'min_samples_split': [10], 'random_state': [SEED]},
    'SVM':  {'C' :[10], 'random_state': [SEED]},
    'LR':   {'penalty': ['l1'], 'C': [10], 'random_state': [SEED]},
    'BAG':  {'n_estimators': [1], 'n_jobs': [-1], 'random_state': [SEED]},
    'NB':   {'alpha': [1], 'fit_prior': [True, False]}    
            }

    
    if (grid_size == 'large'):
        return large_grid
    elif (grid_size == 'small'):
        return small_grid
    elif (grid_size == 'test'):
        return test_grid
    else:
        return 0, 0