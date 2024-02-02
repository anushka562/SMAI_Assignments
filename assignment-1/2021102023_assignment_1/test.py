import sys
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate


class optimised_KNN:
    def __init__(self, k=3, encoder_type='resnets', distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.encoder_type = encoder_type
    
    def fit(self, X, Y):
        self.X_train = X[:, 0]
        if self.encoder_type=='resnets':
            self.X_train = X[:, 0]
        elif self.encoder_type=='vits':
            self.X_train = X[:, 1]
#         print(self.X_train.shape)
        self.Y_train = Y
        
    def calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1-x2)**2))
        if(self.distance_metric == 'manhattan'):
            return np.sum(np.abs(x1-x2))
        if(self.distance_metric == 'cosine'):
            distance = dot(x1, np.transpose(x2))/(norm(x1) * norm(x2))
#             print(distance[0][0])
            return 1-distance[0][0]
        
    def _predict(self, x):
        distances = [self.calculate_distance(x, x_train) for x_train in self.X_train]
        indices = np.argsort(distances)
        k_nearest_indices = indices[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def validate(self, X, Y_actual):
        Y_predictions = self.predict(X)
        f1_weighted = f1_score(Y_actual, Y_predictions, average='weighted', zero_division=1)
        f1_micro = f1_score(Y_actual, Y_predictions, average='micro', zero_division=1)
        f1_macro = f1_score(Y_actual, Y_predictions, average='macro', zero_division=1)
        accuracy = accuracy_score(Y_actual, Y_predictions)
        precision = precision_score(Y_actual, Y_predictions, average='weighted', zero_division=1)
        recall = recall_score(Y_actual, Y_predictions, average='weighted', zero_division=1)
        return recall, precision, accuracy, f1_weighted, f1_micro, f1_macro


if __name__ == "__main__":
    data = np.load("data.npy", allow_pickle = True)

    train_ratio = 0.80
    validation_ratio = 0.20
    split_index1 = int(train_ratio*len(data))

    train_data = data[:split_index1]
    validation_data = data[split_index1: :]

    X_train = train_data[:, 1:3]
    y_train = train_data[:, 3]

    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "<test_data_file>")
        sys.exit(1)

    test_data = np.load(sys.argv[1], allow_pickle= True)

    X_test = test_data[:, 2]
    y_test = test_data[:, 3]

    kNN = optimised_KNN(k=7, encoder_type='vits', distance_metric='manhattan')
    kNN.fit(X_train, y_train)
    # print(X_test.shape)
    recall, precision, accuracy, f1_weighted, f1_micro, f1_macro = kNN.validate(X_test, y_test)
    
    metrics = [[accuracy, precision, recall, f1_weighted, f1_micro, f1_macro]]
    print(tabulate(metrics, headers=["Accuracy", "Precision", "Recall", "F1 score(weighted)", "F1 score(micro)", "F1 score(macro)"]))
    # print("recall:", recall)
    # print("precision:", precision)
    # print("accuracy:", accuracy)
    # print("f1 score(weighted):", f1_weighted)
    # print("f1 score(micro):", f1_micro)
    # print("f1 score(macro):", f1_macro)
    
