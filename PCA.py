import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# This script executes a series of preprocessing steps and finally performs a Principal Component Analysis to reduce the
# size of the data set. Moreover, it splits the data in training and test set and stores them in 2 separate files,
# 'Train.csv' and 'Test.csv'. Finally, it calculates the accuracy score of a KNN algorithm to estimate the loss of
# accuracy that was introduced by the dimensionality reduction

seed = np.random.seed = 7    # Setting the seed to ensure reproducibility
scaler = MinMaxScaler()      # The scaling approach that will be used depends on this scaler parameter
digits = pd.read_csv('digit_recognizer_dataset.csv')
stats = digits.describe().loc[['mean', 'std'], :]

# Discard non - informative digits
white_digits = []   # List to put digits that are always white
useful_digits = []  # List for the rest of the digits

for index, column in enumerate(digits.columns.tolist()):
    if stats.loc['mean', column] == 0 and stats.loc['std', column] == 0:
        white_digits.append(column)
    else:
        useful_digits.append(column)

final_digits = digits[useful_digits]
final_digits = final_digits.drop(labels='label', axis=1)
labels = digits['label']

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(final_digits, labels, test_size=0.2,
                                                    random_state=seed, stratify=labels)
# Scaling of Numeric Features Before PCA
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(.95)   # Keeping the components that explain 95% of the variance
pca.fit(X_train_scaled)
print(pca.n_components_)
X_train_scaled = pca.transform(X_train_scaled)
X_test_scaled = pca.transform(X_test_scaled)

# Scaling of Numeric Features After PCA
scaler.fit(X_train_scaled)
X_train_rescaled = scaler.transform(X_train_scaled)
X_test_rescaled = scaler.transform(X_test_scaled)

# Write the Training Data to a CSV file
df_X_train = pd.DataFrame(data=X_train_rescaled)
y_train = y_train.reset_index(drop=True)
df_y_train = pd.DataFrame(data=y_train)
finalDf_train = pd.concat([df_X_train, df_y_train], axis=1, ignore_index=True)
finalDf_train.to_csv('Train.csv', index=False)

# Write the Test data to a CSV file
df_X_test = pd.DataFrame(data=X_test_rescaled)
y_test = y_test.reset_index(drop=True)
df_y_test = pd.DataFrame(data=y_test)
finalDf_test = pd.concat([df_X_test, df_y_test], axis=1, ignore_index=True)
finalDf_test.to_csv('Test.csv', index=False)


# Making predictions with a KNN to estimate accuracy drop due to PCA
X = finalDf_train
y = X[X.columns[-1]]
X = X.drop(columns=X.columns[-1], axis=1)

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                  random_state=seed, stratify=y)

# "Training" KNN
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)

# Predictions
y_knn_pred = knn_clf.predict(X_train)
print('Training Accuracy')
print(accuracy_score(y_train, y_knn_pred))
print('--------------------------------------')

y_knn_pred = knn_clf.predict(X_val)
print('Validation Accuracy')
print(accuracy_score(y_val, y_knn_pred))
print('--------------------------------------')
