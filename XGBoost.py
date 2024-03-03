import Feature_Extraction
import pandas as pd 

features = Feature_Extraction.extracted_features_timeseries
print(features)

from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder


X = features.drop(column = 'Lables')
y = features['Labels']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify = y, random_state = 8)

from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from XGBoost import XGBClassifier

estimators = [
    ('encoder', TargetEncoder()),
    ('clf', XGBClassifier(random_state=8)) # can customize objective function with the objective paramenters
]
pipe = Pipeline(steps = estimators)

#hyperparameter search spaces
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
search_space = {
    'clf__max_depth': Integer(2,8),
    'clf__learning_rate': Real(0.001,1.0,prior = 'log-uniform'),
    'clf__subsample': Real(0.5,1.0),
    'clf__colsample_bytree': Real(0.5,1.0),
    'clf__colsample_bynode': Real(0.5,1.0),
    'clf__reg_alpha': Real(0.0,10.0),
    'clf__reg_lambda': Real(0.0,10.0),
    'clf__gamma': Real(0.0,10.0)
}

opt = BayesSearchCV(pip,search_space,cv=3,n_iter=10,scoring = 'roc_auc', random_state = 8)
opt.fit(X_train,y_train)

#Evaluate the model and make predictions
opt.best_estimator_
opt.best_score_
opt.score(X_test,y_test)
opt.predict(X_test)
opt.predict_proba(X_test)