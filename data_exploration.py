import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# This script performs an exploratory data analysis. It plots basic histograms and graphs to get a grip on the data.
# Finally, a simple KNN classifier is trained, in order to get a first estimate of a baseline accuracy value

seed = np.random.seed = 7
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

# Explore the correlations between features
correlation = final_digits.corr()
plt.hist(correlation, bins=[-1, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.show()

# Printing the relative frequency of each different class
print('Relative Frequencies of different Class Labels')
print(labels.groupby(labels).count()/len(labels))

# Printing a Histogram of the data
bins = [i for i in range(0, 286, 26)]
plt.hist(final_digits.values, bins=bins)
plt.show()

# Train Test split
X, X_test, y, y_test = train_test_split(final_digits, labels, test_size=0.2,
                                        random_state=seed, stratify=labels)
# Scaling of Numeric Features Before PCA
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_test_scaled = scaler.transform(X_test)
y = y.reset_index(drop=True)

# Training / Validation Split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2,
                                                  random_state=seed, stratify=y)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

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
