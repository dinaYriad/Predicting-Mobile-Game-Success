from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import metrics
from sklearn import decomposition
from sklearn.preprocessing import PolynomialFeatures
from python_files.visualization import Graph
import pandas as pd
from python_files.helper_functions import Helper

class PredictModel:

    def __init__(self, model_id):  #model:1 -> multivariate regression      #model:2 -> polynomial regression
        self.model_id = model_id
        self.learning_model = linear_model.LinearRegression()
        self.poly_degree = 2

    def train(self, X_train, y_train):
        if self.model_id == 2: #Polynomial linear model.
            poly_features = PolynomialFeatures(degree=self.poly_degree) #built in.
            X_train = poly_features.fit_transform(X_train) #Transform to higher degree.

        self.learning_model.fit(X_train, y_train)

        y_train_predicted = self.learning_model.predict(X_train)

        helper = Helper()
        helper.save_model(self.learning_model, "prediction", self.model_id)

        return self.metrics_calculations(y_train, y_train_predicted)

    def test(self, X_test, y_test):
        if self.model_id == 2: #Polynomial linear model.
            poly_features = PolynomialFeatures(degree=self.poly_degree)
            X_test = poly_features.fit_transform(X_test) #Transform to higher degree.

        y_test_predicted = self.learning_model.predict(X_test)

        return self.metrics_calculations(y_test, y_test_predicted)

    def test_for_saved_model(self, X_test, y_test):
        helper = Helper()
        loaded_model = helper.retreive_model('prediction', self.model_id)

        if self.model_id == 2: #Polynomial linear model.
            poly_features = PolynomialFeatures(degree=self.poly_degree)
            X_test = poly_features.fit_transform(X_test) #Transform to higher degree.

        y_test_predicted = loaded_model.predict(X_test)

        print("Model Coefs:\n", loaded_model.coef_)
        return self.metrics_calculations(y_test, y_test_predicted)

    def metrics_calculations(self, actual, predicted):
        test_error = metrics.mean_squared_error(actual, predicted)
        train_r2_score = metrics.r2_score(actual, predicted)

        return test_error, train_r2_score

class ClassifyModel:
    def __init__(self, model_id, pca_mode):
        self.model_id = model_id
        self.pca_mode = pca_mode

        if model_id == 1:
            self.learning_model = linear_model.LogisticRegression(max_iter=1000, solver='sag', C=0.1, multi_class='multinomial')
        elif model_id == 2:
            self.learning_model = svm.SVC(C=1, kernel='poly', degree=5, decision_function_shape='ovo', max_iter=2000)
        elif model_id == 3:
            self.weights = 'distance'
            self.algorithm = 'kd_tree'
            self.learning_model = neighbors.KNeighborsClassifier(weights=self.weights, algorithm=self.algorithm)
        else:
            self.learning_model = tree.DecisionTreeClassifier(max_depth=50, criterion="entropy", splitter="best")

    def train(self, X_train, y_train):
        if self.pca_mode:
            principalComponents = self.build_pca(X_train, y_train)
            X_train = principalComponents

        if self.model_id == 3: #KNN model
            self.learning_model.n_neighbors = self.find_best_k(X_train, y_train, 30)

        self.learning_model.fit(X_train, y_train)

        y_train_predicted = self.learning_model.predict(X_train)

        helper = Helper()
        helper.save_model(self.learning_model, "classification", self.model_id)

        return self.metrics_calculations(y_train, y_train_predicted)

    def test(self, X_test, y_test):
        if self.pca_mode:
            principalComponents = self.pca.transform(X_test)
            X_test = principalComponents

        y_test_predicted = self.learning_model.predict(X_test)

        return self.metrics_calculations(y_test, y_test_predicted)

    def test_for_saved_model(self, X_test, y_test):
        helper = Helper()
        loaded_model = helper.retreive_model('classification', self.model_id)

        if self.pca_mode:
            principalComponents = self.pca.transform(X_test)
            X_test = principalComponents

        y_test_predicted = loaded_model.predict(X_test)

        return self.metrics_calculations(y_test, y_test_predicted)

    def metrics_calculations(self, actual, predicted):
        train_accuracy = metrics.accuracy_score(actual, predicted, normalize=True)
        confusion_matrix = metrics.confusion_matrix(actual, predicted)
        miss_count = metrics.zero_one_loss(actual, predicted, normalize=True)

        return train_accuracy, confusion_matrix, miss_count

    def build_pca(self, X, y):
        n_components = 2
        self.pca = decomposition.PCA(n_components=n_components)
        principalComponents = self.pca.fit_transform(X)

        pca_labels = []
        for i in range(n_components):
            pca_labels.append('PCA ' + str(i))

        pcaDataframe = pd.DataFrame(data=principalComponents, columns=pca_labels)
        pcaDataframe = pd.concat([pcaDataframe, y], axis=1)

        graph = Graph(data=pcaDataframe, label=y.name)
        graph.plot_pca()

        print("PCA Variance Ratio:", self.pca.explained_variance_ratio_)
        return principalComponents

    def find_best_k(self, X, y, max_range):
        accuracy = []
        for i in range(1, max_range):
            knn = neighbors.KNeighborsClassifier(n_neighbors=i, weights=self.weights, algorithm=self.algorithm)
            knn.fit(X, y)

            y_predicted = knn.predict(X)
            accuracy.append(metrics.accuracy_score(y, y_predicted, normalize=True))

        graph = Graph(range(1, max_range), accuracy)
        graph.plot('K Accuracy', 'K Value', 'Accuracy')

        return accuracy.index(max(accuracy)) + 1