import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import StratifiedKFold

# This script conducts a hyperparameter tuning of SVM for our problem. This happens in gradual steps. Moreover, the
# test set is imported and the final evaluation of the selected model takes place and the results are printed.
seed = np.random.seed = 7

# Loading the  Training Set
X_train = pd.read_csv('Train.csv')
y_train = X_train[X_train.columns[-1]]
X_train = X_train.drop(columns=X_train.columns[-1], axis=1)

# Stratified 5-fold split object that will be used
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# SVC: Decide on the best Kernel
svc = SVC(gamma="scale")
parameters = {'kernel': ('poly', 'sigmoid', 'linear', 'rbf'), 'C': [1, 10]}

clf = GridSearchCV(svc, parameters, cv=cv, n_jobs=2, verbose=3)
clf.fit(X_train, y_train)

results = pd.concat([pd.DataFrame(clf.cv_results_["params"]),
                     pd.DataFrame(clf.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
print('Accuracy Score for Different Kernels')
print(results)

# SVC to find optimal hyperparameters
svc = SVC()
parameters = {'kernel': ['rbf'], 'C': [13, 16], 'gamma': [0.55, 0.58]}
clf = GridSearchCV(svc, parameters, cv=cv, n_jobs=1, pre_dispatch=2, refit=False, verbose=3)
clf.fit(X_train, y_train)

results = pd.concat([pd.DataFrame(clf.cv_results_["params"]),
                     pd.DataFrame(clf.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
print('Grid Search Results')
print(results)

# Loading the  Test Set
X_test = pd.read_csv('Test.csv')
y_test = X_test[X_test.columns[-1]]
X_test = X_test.drop(columns=X_test.columns[-1], axis=1)


# Training the optimal SVC for the final evaluation
classifier = SVC(C=10)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred)
train_precision = precision_score(y_train, y_pred, average='weighted')
train_recall = recall_score(y_train, y_pred, average='weighted')
print('Training Accuracy')
print(train_accuracy)
print('------------------------------------------')
print('Training Recall')
print(train_recall)
print('------------------------------------------')
print('Training Precision')
print(train_precision)
print('------------------------------------------')
print('Training Classification Report')
print(classification_report(y_train, y_pred))
print(confusion_matrix(y_train, y_pred))
print('------------------------------------------')

y_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, average='micro')
test_recall = recall_score(y_test, y_pred, average='micro')
print('Test Accuracy')
print(test_accuracy)
print('------------------------------------------')
print('Test Recall')
print(test_recall)
print('------------------------------------------')
print('Test Precision')
print(test_precision)
print('------------------------------------------')
print('Test Classification Report')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
