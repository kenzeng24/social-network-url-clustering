
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV, 
    RepeatedStratifiedKFold, 
    train_test_split
)
from scipy.stats import loguniform
from src.data_loading.training_data import get_tfidf_dataset


def random_search_cv(model, space, random_state=824):
    
    # split training and testing data 
    X, y, _ = get_tfidf_dataset()
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
