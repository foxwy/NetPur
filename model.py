# -*- coding: utf-8 -*-
# @Author: WY
# @Date:   2022-03-08 10:04:54
# @Last Modified by:   WY
# @Last Modified time: 2022-03-08 21:21:45

from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, \
                             ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, \
                             StackingRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ExpSineSquared, RationalQuadratic, Matern
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_squared_error, mean_absolute_error

#from algorithm.DNN_regressor import *


def Get_model(m_type='RF'):
    #----model-----
    # SVM
    if m_type == 'svr_rbf':
        model = SVR(kernel="rbf", C=10, gamma=1, epsilon=0.1)
    elif m_type == 'svr_poly':
        model = SVR(kernel="poly", C=1000, gamma="auto", degree=3, epsilon=0.1, coef0=1)

    # linear model
    elif m_type == 'bayes':
        model = linear_model.BayesianRidge()
    elif m_type == 'LR':
        model = linear_model.LinearRegression()
    elif m_type == 'LOR':
        model = linear_model.LogisticRegression()
    elif m_type == 'Huber':
        model = linear_model.HuberRegressor()
    elif m_type == 'SGD':
        model = linear_model.SGDRegressor()
    elif m_type == 'lasso':
        model = linear_model.Lasso()
    elif m_type == 'percetron':
        model = linear_model.Perceptron()
    elif m_type == 'elast':
        model = linear_model.ElasticNet()

    # ridge
    elif m_type == 'ridge':
        model = linear_model.Ridge(alpha=0.5)
    elif m_type == 'Kridge':
        model = KernelRidge(alpha=1.0)

    # naive bayes
    elif m_type == 'GNB':
        model = GaussianNB()

    # KNN
    elif m_type == 'KNN':
        model = KNeighborsRegressor(n_neighbors=5)
    elif m_type == 'RKNN':
        model = RadiusNeighborsRegressor(radius=2)

    # Gaussian
    elif m_type == 'Gauss':
        kernel = WhiteKernel() + Matern(length_scale=1.0, nu=1.5)
        model = GaussianProcessRegressor(kernel=kernel)

    # cross decomposition
    elif m_type == 'PLS':
        model = PLSRegression(n_components=2)

    # tree
    elif m_type == 'DT':
        model = DecisionTreeRegressor(max_depth=10)
    elif m_type == 'RF':
        model = RandomForestRegressor(n_estimators=150)

    # boost
    elif m_type == 'Adaboost':
        model = AdaBoostRegressor()
    elif m_type == 'Bagging':
        model = BaggingRegressor()
    elif m_type == 'extraT':
        model = ExtraTreesRegressor()
    elif m_type == 'GB':
        model = GradientBoostingRegressor()
    elif m_type == 'HistGB':
        model = HistGradientBoostingRegressor()
    elif m_type == 'stack':
        estimators = [
        ('rf', RandomForestRegressor(n_estimators=100)),
        ('knn', KNeighborsRegressor(n_neighbors=5))]

        model = StackingRegressor(estimators=estimators)
    elif m_type == 'voting':
        estimators = [
        ('rf', RandomForestRegressor(n_estimators=100)),
        ('knn', KNeighborsRegressor(n_neighbors=5))]

        model = VotingRegressor(estimators=estimators)

    # NN
    elif m_type == 'MLP':
        model = MLPRegressor(hidden_layer_sizes=(100, 200, 200, 200, 100), early_stopping=True, 
                             activation='relu')

    return model


def Model_metric(dataset, model):
    X_train, X_test, y_train, y_test = dataset[0], dataset[1], dataset[2], dataset[3]

    y_pre = model.predict(X_test)
    print('MSE train:', mean_squared_error(y_train, model.predict(X_train)))
    print('MSE test:', mean_squared_error(y_test, y_pre))

    print('MAE train:', mean_absolute_error(y_train, model.predict(X_train)))
    print('MAE test:', mean_absolute_error(y_test, y_pre))


if __name__ == '__main__':
    model = Get_model('RF')