
import pandas as pd
import numpy as np 
from sklearn.model_selection import (
    RandomizedSearchCV, 
    RepeatedStratifiedKFold, 
    train_test_split, 
    cross_validate, 
    KFold
)
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import xgboost as xgb 
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    f1_score,  
    make_scorer, 
    recall_score
)
from src.data_loading.training_data import load_and_remove_politicial_entities, ROOT
from sklearn.model_selection import cross_validate


def random_search_cv(model, space, root=ROOT, random_state=824):
    
    # split training and testing data 
    X, y, _, _ = load_and_remove_politicial_entities(root=root)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # random search over 10-fold cross validation 
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)

    search = RandomizedSearchCV(
        model, space, 
        n_iter=20, 
        scoring='roc_auc', 
        n_jobs=-1, cv=cv, 
        random_state=random_state
    )
    result = search.fit(X=X_train,y=y_train)

    # train and evaluate model with the best parameters 
    model = LogisticRegression(**result.best_params_).fit(X=X_train,y=y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    print('best training auc: %s' % result.best_score_)
    print('best testing auc: %s' % auc)
    return model 


def logistic_regression_cv(random_state=824):
    model = LogisticRegression()
    space = dict()
    space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
    space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
    space['C'] = loguniform(1e-5, 100)
    return random_search_cv(model, space, random_state=random_state)


def train_models(X,y, models=None):

    if models is None:
        models = {
            'random-forest': RandomForestClassifier(verbose=False), 
            'catboost': CatBoostClassifier(verbose=False), 
            'logistic': LogisticRegression(solver='liblinear',penalty='l1'),
            'xgboost': xgb.XGBClassifier(),
            'lightgbm': LGBMClassifier(), 
        }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=824)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=824)

    for name, model in models.items():

        # use validation set for early stopping
        additional_params = {}
        if name in ['xgboost', 'lightgbm']:
            additional_params = dict(eval_set=[(X_val, y_val)], verbose=False)

        model.fit(X=X_train,y=y_train, **additional_params)

    preds = pd.DataFrame({name: model.predict_proba(X_test)[:,1] for name, model in models.items()})
    
    test_results = []
    for col in preds.columns:
        test_results.append({
            'model': col, 
            'accuracy': accuracy_score(y_test, models[col].predict(X_test)), 
            'F1': f1_score(y_test, models[col].predict(X_test)), 
            'AUC': roc_auc_score(y_test, models[col].predict_proba(X_test)[:,1]), 
        })
    return models, preds, test_results


def default_models(random_state=824):
    models = {
        'random-forest': RandomForestClassifier(verbose=False, random_state=random_state), 
        'catboost': CatBoostClassifier(verbose=False, random_state=random_state), 
        'logistic': LogisticRegression(solver='liblinear',penalty='l1'),
        'xgboost': xgb.XGBClassifier(random_state=random_state),
        'lightgbm': LGBMClassifier(random_state=random_state), 
    }
    return models


def model_cross_validate(X,y, models=None, random_state=824, verbose=False):
    """
    Perform cross validation over a dictionary of different models 
    """
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'auc': make_scorer(roc_auc_score, needs_proba=True),
        'recall': make_scorer(recall_score), 
        'f1': make_scorer(f1_score)
    }
    results = {}

    if models is None:
        models = default_models(random_state=random_state) 

    cv = KFold(random_state=random_state, shuffle=True)

    for name, model in models.items():
        scores = cross_validate(model, X, y, scoring=scoring, cv=cv)
        results[name] = scores
        if verbose:
            print(f'model: {name}')
            for key, values in scores.items():
                print(f'{key}:{round(np.mean(values),3)} +- {round(np.std(values),3)}')
            print('-'*40)
    return results 
