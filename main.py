# -*- coding: utf-8 -*-
# @Author: WY
# @Date:   2022-03-07 19:03:32
# @Last Modified by:   WY
# @Last Modified time: 2022-03-08 15:29:01

import numpy as np
from dataset import Dataset_state
from model import Get_model, Model_metric

import matplotlib.pyplot as plt

plt.style.use(['science', 'no-latex'])
plt.rcParams["font.family"] = 'Arial'


def Plt_set(ax, xlabel='x', ylabel='y'):
    font_size = 18
    font = {'size': font_size, 'weight': 'normal'}

    ax.set_xlabel(xlabel, font)
    ax.set_ylabel(ylabel, font)
    ax.tick_params(labelsize=font_size)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.tick_params(width=1.5)
    ax.legend(prop=font, loc=1)
    plt.tight_layout()


def Get_figure(dataset, model):
    #-------------------------------------
    X_train, X_test, y_train, y_test = dataset[0], dataset[1], dataset[2], dataset[3]

    #-----figure-----
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].scatter(abs(X_train.dot(np.array([1, 1, 1, -1]))), y_train, label='train_real', linewidth=2.5)
    ax[0].scatter(abs(X_train.dot(np.array([1, 1, 1, -1]))), model.predict(X_train), label='train_pre', linewidth=2.5)

    ax[1].scatter(abs(X_test.dot(np.array([1, 1, 1, -1]))), y_test, label='test_real', linewidth=2.5)
    ax[1].scatter(abs(X_test.dot(np.array([1, 1, 1, -1]))), model.predict(X_test), label='test_pre', linewidth=2.5)

    Plt_set(ax[0], 'S', 'Purity')
    Plt_set(ax[1], 'S', 'Purity')
    plt.savefig('3.jpg', dpi=600)
    plt.show()

    #-------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    '''
    ax.scatter(range(len(y_train)), y_train, label='train_real', linewidth=2.5)
    ax.scatter(range(len(y_train)), model.predict(X_train), label='train_pre', linewidth=2.5)
    ax.scatter(range(len(y_test)), y_test, label='test_real', linewidth=2.5)
    ax.scatter(range(len(y_test)), model.predict(X_test), label='test_pre', linewidth=2.5)'''
    
    ax.scatter(y_train, model.predict(X_train), label='train', linewidth=2.5)
    ax.scatter(y_test, model.predict(X_test), label='test', linewidth=2.5)
    ax.plot(y_train, y_train, label='k=1', linewidth=2.5)

    Plt_set(ax, 'real value', 'predict')
    plt.savefig('4.jpg', dpi=600)
    plt.show()


if __name__ == "__main__":
    #-----dataset-----
    X_train, X_test, y_train, y_test = Dataset_state(1000, 0.4, 'Werner')

    #-----model-----
    model = Get_model('Gauss')  # KNN, RF, Gauss, Bagging

    #-----predict-----
    model.fit(X_train, y_train)
    Model_metric([X_train, X_test, y_train, y_test], model)

    #-----figure-----
    Get_figure([X_train, X_test, y_train, y_test], model)

    #dnn_regressor(X_train, y_train, [10, 10], 100, 8)
