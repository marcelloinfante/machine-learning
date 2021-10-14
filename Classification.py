import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class ClassificationModels:
    def __init__(self, dataset):
        self.dataset = dataset

    def preprocessing(self):
        dataset = pd.read_csv(self.dataset)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    def feature_scaling(self):
        sc = StandardScaler()
        X_train = sc.fit_transform(self.X_train)
        X_test = sc.transform(self.X_test)
        return {"X_train": X_train, "X_test": X_test}



    def logistic_regression_train(self):
        feature_scaling = self.feature_scaling()
        X_train = feature_scaling["X_train"]
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, self.y_train)
        return classifier

    def logistic_regression_predict(self, list_of_values):
        classifier = self.logistic_regression_train()
        y_pred = classifier.predict(list_of_values)
        return y_pred

    def logistic_regression_score(self):
        feature_scaling = self.feature_scaling()
        X_test = feature_scaling["X_test"]
        y_pred = self.logistic_regression_predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy



    def kernel_svm_train(self):
        feature_scaling = self.feature_scaling()
        X_train = feature_scaling["X_train"]
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, self.y_train)
        return classifier

    def kernel_svm_predict(self, list_of_values):
        classifier = self.kernel_svm_train()
        y_pred = classifier.predict(list_of_values)
        return y_pred

    def kernel_svm_score(self):
        feature_scaling = self.feature_scaling()
        X_test = feature_scaling["X_test"]
        y_pred = self.kernel_svm_predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy



    def k_nearest_neighbors_train(self):
        feature_scaling = self.feature_scaling()
        X_train = feature_scaling["X_train"]
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, self.y_train)
        return classifier

    def k_nearest_neighbors_predict(self, list_of_values):
        classifier = self.k_nearest_neighbors_train()
        y_pred = classifier.predict(list_of_values)
        return y_pred

    def k_nearest_neighbors_score(self):
        feature_scaling = self.feature_scaling()
        X_test = feature_scaling["X_test"]
        y_pred = self.k_nearest_neighbors_predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy



    def support_vector_machine_train(self):
        feature_scaling = self.feature_scaling()
        X_train = feature_scaling["X_train"]
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, self.y_train)
        return classifier

    def support_vector_machine_predict(self, list_of_values):
        classifier = self.support_vector_machine_train()
        y_pred = classifier.predict(list_of_values)
        return y_pred

    def support_vector_machine_score(self):
        feature_scaling = self.feature_scaling()
        X_test = feature_scaling["X_test"]
        y_pred = self.support_vector_machine_predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy



    def naive_bayes_train(self):
        feature_scaling = self.feature_scaling()
        X_train = feature_scaling["X_train"]
        classifier = GaussianNB()
        classifier.fit(X_train, self.y_train)
        return classifier

    def naive_bayes_predict(self, list_of_values):
        classifier = self.naive_bayes_train()
        y_pred = classifier.predict(list_of_values)
        return y_pred

    def naive_bayes_score(self):
        feature_scaling = self.feature_scaling()
        X_test = feature_scaling["X_test"]
        y_pred = self.naive_bayes_predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy



    def decision_tree_classification_train(self):
        feature_scaling = self.feature_scaling()
        X_train = feature_scaling["X_train"]
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, self.y_train)
        return classifier

    def decision_tree_classification_predict(self, list_of_values):
        classifier = self.decision_tree_classification_train()
        y_pred = classifier.predict(list_of_values)
        return y_pred

    def decision_tree_classification_score(self):
        feature_scaling = self.feature_scaling()
        X_test = feature_scaling["X_test"]
        y_pred = self.decision_tree_classification_predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy



    def random_forest_classification_train(self):
        feature_scaling = self.feature_scaling()
        X_train = feature_scaling["X_train"]
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, self.y_train)
        return classifier

    def random_forest_classification_predict(self, list_of_values):
        classifier = self.random_forest_classification_train()
        y_pred = classifier.predict(list_of_values)
        return y_pred

    def random_forest_classification_score(self):
        feature_scaling = self.feature_scaling()
        X_test = feature_scaling["X_test"]
        y_pred = self.random_forest_classification_predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy


    def select_best_regression_model(self):
        regression_models = {
            'logistic_regression' : self.logistic_regression_score(),
            'kernel_svm' : self.kernel_svm_score(),
            'k_nearest_neighbors' : self.k_nearest_neighbors_score(),
            'support_vector_machine' : self.support_vector_machine_score(),
            'naive_bayes' : self.naive_bayes_score(),
            'decision_tree_classification' : self.decision_tree_classification_score(),
            'random_forest_classification' : self.random_forest_classification_score()
        }
        best_regression_model = max(regression_models, key=regression_models.get)
        return {"best_regression_model": best_regression_model, "accuracy": regression_models[best_regression_model]}


    def predict_with_best_regression_model(self, list_of_values):
        selected_regression_models = self.select_best_regression_model()
        best_regression_model = selected_regression_models["best_regression_model"]
        best_regression_model_accuracy = selected_regression_models["accuracy"]
        if best_regression_model == 'logistic_regression':
            y_pred = self.logistic_regression_predict([list_of_values])
        elif best_regression_model == 'kernel_svm':
            y_pred = self.kernel_svm_predict([list_of_values])
        elif best_regression_model == 'k_nearest_neighbors':
            y_pred = self.k_nearest_neighbors_predict([list_of_values])
        elif best_regression_model == 'support_vector_machine':
            y_pred = self.support_vector_machine_predict([list_of_values])
        elif best_regression_model == 'naive_bayes':
            y_pred = self.naive_bayes_predict([list_of_values])
        elif best_regression_model == 'decision_tree_classification':
            y_pred = self.decision_tree_classification_predict([list_of_values])
        elif best_regression_model == 'random_forest_classification':
            y_pred = self.random_forest_classification_predict([list_of_values])
        return {"prediction": y_pred, "best_regression_model": best_regression_model, "accuracy": best_regression_model_accuracy}


classification = ClassificationModels('./data_examples/Data_classification.csv')
classification.preprocessing()
prediction = classification.predict_with_best_regression_model([1173347, 1, 1, 1, 1, 2, 5, 1, 1, 1])
print("Best Classification Model: " + prediction["best_regression_model"])
print("Accuracy: " + str(prediction["accuracy"]))
print("Prediction: " + str(prediction["prediction"]))
print("---------------------------------------------")
print("Models Accuracy:")
print("logistic_regression: " + str(classification.logistic_regression_score()))
print("kernel_svm: " + str(classification.kernel_svm_score()))
print("k_nearest_neighbors: " + str(classification.k_nearest_neighbors_score()))
print("support_vector_machine: " + str(classification.support_vector_machine_score()))
print("naive_bayes: " + str(classification.naive_bayes_score()))
print("decision_tree_classification: " + str(classification.decision_tree_classification_score()))
print("random_forest_classification: " + str(classification.random_forest_classification_score()))
