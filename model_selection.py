import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# This script loads the transformed training set with reduced dimensions (153 principal components) that was created in
# the previous step and compares different classifiers using a 5-fold cross validation scheme. The parameters of the
# different classifiers are tuned manually inside the script. All accuracy score results are stored in a DataFrame and
# are printed at the end of the script's execution. It's worth noting that the Gradient Boost algorithm at the end of
# the script took extremely long to train. Therefore, the corresponding lines are commented.

seed = np.random.seed = 7

# Loading the  Training Set
X_train = pd.read_csv('Train.csv')
y_train = X_train[X_train.columns[-1]]
X_train = X_train.drop(columns=X_train.columns[-1], axis=1)

# Creating a DataFrame to store the accuracy scores of the different models
models = pd.DataFrame({'Score': [0, 0, 0, 0, 0, 0, 0]}, index=['SVC', 'LogisticRegression', 'KNeighborsClassifier',
                                                               'DecisionTreeClassifier', 'RandomForestClassifier',
                                                               'AdaBoostClassifier', 'GaussianNB', 'GradientBoost'])

# Stratified 5-fold split object that will be used
cv = StratifiedKFold(shuffle=True, random_state=seed)

# KNN
classifier = KNeighborsClassifier(n_neighbors=7)
y_pred = cross_val_predict(classifier, X_train, y_train, cv=cv, verbose=3)
models.loc['KNeighborsClassifier', 'Score'] = accuracy_score(y_train, y_pred)

# Logistic Regression
classifier = LogisticRegression(max_iter=1000)
y_pred = cross_val_predict(classifier, X_train, y_train, cv=cv, verbose=3)
models.loc['LogisticRegression', 'Score'] = accuracy_score(y_train, y_pred)

# SVC
classifier = SVC()
y_pred = cross_val_predict(classifier, X_train, y_train, cv=cv, verbose=3)
models.loc['SVC', 'Score'] = accuracy_score(y_train, y_pred)

# Decision Tree
classifier = DecisionTreeClassifier(max_depth=20, criterion='entropy', random_state=seed)
y_pred = cross_val_predict(classifier, X_train, y_train, cv=cv, verbose=3)
models.loc['DecisionTreeClassifier', 'Score'] = accuracy_score(y_train, y_pred)

# Gaussian Naive Bayes
classifier = GaussianNB()
y_pred = cross_val_predict(classifier, X_train, y_train, cv=cv, verbose=3)
models.loc['GaussianNB', 'Score'] = accuracy_score(y_train, y_pred)

# Random Forest
classifier = RandomForestClassifier(n_estimators=100, random_state=seed)
y_pred = cross_val_predict(classifier, X_train, y_train, cv=cv, verbose=3)
models.loc['RandomForestClassifier', 'Score'] = accuracy_score(y_train, y_pred)

# ADABOOST
classifier = AdaBoostClassifier(random_state=seed, n_estimators=250)
y_pred = cross_val_predict(classifier, X_train, y_train, cv=cv, verbose=3)
models.loc['AdaBoostClassifier', 'Score'] = accuracy_score(y_train, y_pred)

# Gradient Boost
#classifier = GradientBoostingClassifier(random_state=seed, n_estimators=100, learning_rate=1.0,)
#y_pred = cross_val_predict(classifier, X_train, y_train, cv=cv, verbose=3)
#models.loc['GradientBoost', 'Score'] = accuracy_score(y_train, y_pred)

print(models)
