
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV, 
    RepeatedStratifiedKFold, 
    train_test_split
)
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from catboost import CatBoostClassifier
import xgboost as xgb 
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from src.data_loading.training_data import get_balanced_tfidf_data


def random_search_cv(model, space, random_state=824):
    
    # split training and testing data 
    X, y, _ = get_balanced_tfidf_data()
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
