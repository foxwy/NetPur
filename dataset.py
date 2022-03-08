# -*- coding: utf-8 -*-
# @Author: WY
# @Date:   2022-03-07 19:03:11
# @Last Modified by:   WY
# @Last Modified time: 2022-03-08 15:26:03

import numpy as np
from numpy import sin, cos, sqrt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


#----------- definition -----------
# basic state
state0 = np.matrix([[1], [0]])
state1 = np.matrix([[0], [1]])

# basic gate
X = np.matrix([[0.0j, 1.0], [1.0, 0.0j]])
Y = np.matrix([[0.0j, -1.0j], [1.0j, 0.0j]])
Z = np.matrix([[1.0, 0.0j], [0.0j, -1.0]])
I = np.matrix([[1.0, 0.0j], [0.0j, 1.0]])


#----------- function -----------
def Get_rho(v, s_type):
    if s_type == 'Werner':  # Werner state
        psi = (np.kron(state0, state1) - np.kron(state1, state0)) / sqrt(2)
        rho = v * psi.dot(psi.T.conjugate()) + (1 - v) * np.kron(I, I) / 4
        
    elif s_type == 'random':  # random mixed state
        A = np.random.uniform(size=(4, 4))
        B = np.random.uniform(size=(4, 4))
        sig = A + 1j * B
        rho = sig.dot(sig.T.conjugate()) / np.trace(sig.dot(sig.T.conjugate()))

    return rho


def Get_theta_Matrix(rho):
    sigma = [X, Y, Z]

    # T
    T = np.zeros((3, 3))
    for j in range(3):
        for k in range(3):
            T[j, k] = np.trace(rho.dot(np.kron(sigma[j], sigma[k]))).real
    #print('T:', T)

    # a
    a = np.zeros((3, 1))
    for j in range(3):
        a[j, 0] = np.trace(rho.dot(np.kron(sigma[j], I))).real
    #print('a:', a)

    # b
    b = np.zeros((3, 1))
    for j in range(3):
        b[j, 0] = np.trace(rho.dot(np.kron(I, sigma[j]))).real
    #print('b:', b)

    # Theta
    Theta = np.zeros((4, 4))
    Theta[0, 0] = 1
    Theta[0, 1:] = b.T
    Theta[1:, 0] = a.T
    Theta[1:, 1:] = T
    #print('\nTheta:\n', Theta)

    return Theta


def Get_measure(Theta, rho):
    A0 = np.matrix([[0], [1], [0], [0]])
    A1 = np.matrix([[0], [0], [0], [1]])
    B0 = np.matrix([[0], [1/sqrt(2)], [0], [1/sqrt(2)]])
    B1 = np.matrix([[0], [1/sqrt(2)], [0], [-1/sqrt(2)]])

    A0B0 = A0.T.dot(Theta).dot(B0)[0, 0]
    A1B0 = A1.T.dot(Theta).dot(B0)[0, 0]
    A0B1 = A0.T.dot(Theta).dot(B1)[0, 0]
    A1B1 = A1.T.dot(Theta).dot(B1)[0, 0]

    Pur = np.trace(rho**2)

    return [A0B0, A1B0, A0B1, A1B1, Pur]


def Data_processing(X, y):
    y = y.reshape(len(y), 1)
    x_MinMax = preprocessing.MinMaxScaler()
    y_MinMax = preprocessing.MinMaxScaler()

    X = x_MinMax.fit_transform(X)
    y = y_MinMax.fit_transform(y)

    return X, y


def Dataset_state(N, te_size, s_type):
    V = np.random.uniform(0, 1, N)  # uniform random number in [0, 1]
    dataset = []
    for v in V:
        rho = Get_rho(v, s_type)
        Theta_M = Get_theta_Matrix(rho)
        data_state = Get_measure(Theta_M, rho)
        dataset.append(data_state)

    dataset = np.real(np.array(dataset))
    X = dataset[:, :4]
    y = dataset[:, 4]
    #X, y = Data_processing(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=te_size)

    #print('X_train:\n', X_train)
    #print('y_train:\n', y_train)
    #print('X_test:\n', X_test)
    #print('y_test:\n', y_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = Dataset_state(1000, 0.3, 'Werner')
    print('X_train:\n', X_train)
    print('y_train:\n', y_train)
    print('X_test:\n', X_test)
    print('y_test:\n', y_test)
