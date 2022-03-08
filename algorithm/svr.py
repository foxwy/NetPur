'''
@author: foxwy
@time: 2019-09-06
@description: SVM regression
'''

#/
#model
#/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import os

from .source import *

#--------------------------------------------------参数选择--------------------------------------------------
#/
#the train processing of SVM regression
#/
def Svr_train_reKFold(indata, outdata, K, repeated_num):
    #----------ELM Init begin----------
    test_error_aver_min = 1e10  #相对误差最小
    test_size = len(outdata) * len(outdata[0]) / K
    test_error = []  #保存每个参数对应的误差

    #ELM参数
    c_best = 0

    #cross validation
    kf = RepeatedKFold(n_splits=K, n_repeats=repeated_num, random_state=12883823)

    #----------elm train begin----------
    print('\n-----svr parameter selection-----')
    c = 1

    while c <= 2**10:
        test_error_aver = 0
        exception_flag = 0
        i = 0
        for train_index, test_index in kf.split(indata):
            print(i)
            i += 1
            Input_train = indata[train_index]
            Output_train = outdata[train_index]
            Input_test = indata[test_index]
            Output_test = outdata[test_index]

            try:  #求逆异常
                clf = SVR(kernel='rbf', C=c)
                clf = MultiOutputRegressor(clf)
                clf.fit(Input_train, Output_train)
            except:
                exception_flag = 1
                break
            else:
                capacity_test = clf.predict(Input_test)
                test_error_aver += np.linalg.norm(Output_test - capacity_test)  #矩阵F范数

        if exception_flag == 0:  #非异常
            test_error.append([c, (test_error_aver / (K * repeated_num))**2 / test_size])
            if test_error_aver_min > test_error_aver:  #更优参数
                print('+++++++++++++++++++++++++')
                test_error_aver_min = test_error_aver
                c_best = c

        c *= 2
        print('error', (test_error_aver_min / (K * repeated_num))**2 / test_size)
        print('\n-----------------c-----------------')
        print('c:', c)
    #----------elm train end----------
    test_error_aver_min = (test_error_aver_min / (K * repeated_num))**2 / test_size  #预测均方误差

    return c_best, test_error_aver_min, test_error

#--------------------------------------------------文件存储和提取--------------------------------------------------
#/
#存储ELM参数
#/
def Store_svr_data(c_best, test_error_aver, test_error, save_num):
    #创建文件夹
    path = r'save\result'
    if not os.path.exists(path+'\\'+str(save_num)):
        os.mkdir(path+'\\'+str(save_num))
        os.mkdir(path+'\\'+str(save_num)+'\\'+'weights')
        os.mkdir(path+'\\'+str(save_num)+'\\'+'analysis')

    StoreWeights(c_best, path+'\\'+str(save_num)+r'\weights\c_best.txt')
    StoreWeights(test_error_aver, path+'\\'+str(save_num)+r'\weights\test_error_aver.txt')
    StoreWeights(test_error, path+'\\'+str(save_num)+r'\weights\test_error.txt')

def Grab_svr_data(save_num):
    path = r'save\result'

    c_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\c_best.txt')
    test_error_aver = GrabWeights(path+'\\'+str(save_num)+r'\weights\test_error_aver.txt')
    test_error = GrabWeights(path+'\\'+str(save_num)+r'\weights\test_error.txt')

    return c_best, test_error_aver, test_error

#/
#参数图形绘制
#/
def Figure_svr(save_num):
    path = r'save\result'
    test_error = GrabWeights(path+'\\'+str(save_num)+r'\weights\test_error.txt')
    c_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\c_best.txt')
    test_error_aver = GrabWeights(path+'\\'+str(save_num)+r'\weights\test_error_aver.txt')

    print('c_best:', c_best)
    print('test_error_aver:', test_error_aver)

    x = np.array(test_error)[:, 0]
    y = np.array(test_error)[:, 1]

    plt.plot(x, y)
    plt.xlabel('c')
    plt.ylabel('RMSE')

    # 添加坐标轴(顺序是Z, Y, X)
    plt.savefig(path+'\\'+str(save_num)+r'\analysis\svr_figure'+'.png', dpi=1200, format='png')
    plt.show()