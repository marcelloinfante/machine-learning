import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

class RegressionModels:
    def __init__(self, dataset):
        self.dataset = dataset

    def preprocessing(self):
        dataset = pd.read_csv(self.dataset)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



    def multiple_linear_regression_train(self):
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)
        return regressor

    def multiple_linear_regression_predict(self, list_of_values):
        regressor = self.multiple_linear_regression_train()
        y_pred = regressor.predict(list_of_values)
        return y_pred

    def multiple_linear_regression_score(self):
        y_pred = self.multiple_linear_regression_predict(self.X_test)
        return r2_score(self.y_test, y_pred)



    def polynomial_regression_train(self):
        poly_reg = PolynomialFeatures(degree = 8)
        X_poly = poly_reg.fit_transform(self.X_train)
        regressor = LinearRegression()
        regressor.fit(X_poly, self.y_train)
        return {"regressor": regressor, "poly_reg": poly_reg}

    def polynomial_regression_predict(self, list_of_values):
        regressor = self.polynomial_regression_train()["regressor"]
        poly_reg = self.polynomial_regression_train()["poly_reg"]
        y_pred = regressor.predict(poly_reg.transform(list_of_values))
        return y_pred

    def polynomial_regression_score(self):
        y_pred = self.polynomial_regression_predict(self.X_test)
        return r2_score(self.y_test, y_pred)



    def support_vector_regression_train(self):
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train = sc_X.fit_transform(self.X_train)
        y_train = sc_y.fit_transform(self.y_train.reshape(-1, 1))
        regressor = SVR(kernel = 'rbf')
        regressor.fit(X_train, y_train)
        return {"regressor": regressor, "sc_X": sc_X, "sc_y": sc_y}

    def support_vector_regression_predict(self, list_of_values):
        regressor = self.support_vector_regression_train()["regressor"]
        sc_X = self.support_vector_regression_train()["sc_X"]
        sc_y = self.support_vector_regression_train()["sc_y"]
        y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(list_of_values)))
        return y_pred

    def support_vector_regression_score(self):
        y_pred = self.support_vector_regression_predict(self.X_test)
        return r2_score(self.y_test, y_pred)



    def decision_tree_regression_train(self):
        regressor = DecisionTreeRegressor(random_state = 0)
        regressor.fit(self.X_train, self.y_train)
        return regressor

    def decision_tree_regression_predict(self, list_of_values):
        regressor = self.decision_tree_regression_train()
        y_pred = regressor.predict(list_of_values)
        return y_pred

    def decision_tree_regression_score(self):
        y_pred = self.decision_tree_regression_predict(self.X_test)
        return r2_score(self.y_test, y_pred)



    def random_forest_regression_train(self):
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(self.X_train, self.y_train)
        return regressor

    def random_forest_regression_predict(self, list_of_values):
        regressor = self.random_forest_regression_train()
        y_pred = regressor.predict(list_of_values)
        return y_pred

    def random_forest_regression_score(self):
        y_pred = self.random_forest_regression_predict(self.X_test)
        return r2_score(self.y_test, y_pred)



    def select_best_regression_model(self):
        regression_models = {
            'multiple_linear_regression' : self.multiple_linear_regression_score(),
            'polynomial_regression' : self.polynomial_regression_score(),
            'support_vector_regression' : self.support_vector_regression_score(),
            'decision_tree_regression' : self.decision_tree_regression_score(),
            'random_forest_regression' : self.random_forest_regression_score()
        }
        best_regression_model = max(regression_models, key=regression_models.get)
        return {"best_regression_model": best_regression_model, "accuracy": regression_models[best_regression_model]}


    def predict_with_best_regression_model(self, list_of_values):
        selected_regression_models = self.select_best_regression_model()
        best_regression_model = selected_regression_models["best_regression_model"]
        best_regression_model_accuracy = selected_regression_models["accuracy"]
        if best_regression_model == 'multiple_linear_regression':
            y_pred = self.multiple_linear_regression_predict([list_of_values])
        elif best_regression_model == 'polynomial_regression':
            y_pred = self.polynomial_regression_predict([list_of_values])
        elif best_regression_model == 'support_vector_regression':
            y_pred = self.support_vector_regression_predict([list_of_values])
        elif best_regression_model == 'decision_tree_regression':
            y_pred = self.decision_tree_regression_predict([list_of_values])
        elif best_regression_model == 'random_forest_regression':
            y_pred = self.random_forest_regression_predict([list_of_values])
        return {"prediction": y_pred, "best_regression_model": best_regression_model, "accuracy": best_regression_model_accuracy}

regression = RegressionModels('./Data.csv')
regression.preprocessing()
print(regression.predict_with_best_regression_model([28.66, 77.95, 1009.59, 69.07]))
