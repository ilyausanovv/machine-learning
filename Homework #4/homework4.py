from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split

class KNeighborsClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = np.sqrt(np.sum((x - self.X_train) ** 2, axis=1))
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        return list(map(lambda x: Counter(x).most_common(1)[0][0], neighbors))

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum(y_pred == y_test) / len(y_test)

def normalize_data(dataset):
    for i in range(len(dataset[0])):
        column_values = [row[i] for row in dataset]
        column_min = np.min(column_values)
        column_max = np.max(column_values)
        for row in dataset:
            row[i] = (row[i] - column_min) / (column_max - column_min)


k = 20  # Прогоны с заданным количеством
test_size = 0.4

def sandbox():

    global current_value
    iris = datasets.load_iris()
    targets = iris.target
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = targets

    x = iris.get('data')
    sns.pairplot(df, hue="Species", height=3)
    plt.show()

    normalize_data(x)

    df.data = x
    sns.pairplot(df, hue="Species", height=3)
    plt.show()

    y = iris.get('target')
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    accuracies = []
    for current_value in range(1, k):
        knn = KNeighborsClassifier(k=current_value)
        knn.fit(X_train, y_train)
        accuracy = knn.evaluate(X_test, y_test)
        accuracies.append(accuracy)

    fig, ax = plt.subplots()
    ax.plot(range(1, k), accuracies)
    ax.set(xlabel="k", ylabel="accuracy")
    plt.show()

    best_accuracy = np.array(accuracies).max()
    print(best_accuracy)

    testSet = [[5.1, 3.5, 1.4, 0.2]]  # - setosa
    test = pd.DataFrame(testSet)
    test = np.asarray(test)
    test_min = np.min(test)
    test_max = np.max(test)
    test_array = (test - test_min) / (test_max - test_min)

    di = {0.0: 'Setosa', 1.0: 'Versicolor', 2.0: 'Virginica'}
    knn = KNeighborsClassifier(k=current_value)
    knn.fit(X_train, y_train)
    predictions = knn.predict(test_array)
    for i in predictions:
        print(di[i])

if __name__ == "__main__":
    sandbox()