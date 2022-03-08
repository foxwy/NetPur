'''
@author: foxwy
@time: 2019-09-06
@description: kernel based extreme learning machine
'''

#/
#model
#/
from sklearn.model_selection import RepeatedKFold
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import choices

from .source import *
from .elmk import *

#/
#极限学习机的训练
#function:['rbf', 'linear', 'poly']
#/
def ELM_kernel_train(X, Y, params):
    elmk = ELMKernel(params)
    elmk._local_train(X, Y, params)

    # 返回计算结果
    return elmk

#/
#极限学习机的预测
#/
def ELM_kernel_predict(X, Y, elmk):
    return elmk._local_test(X, Y, predicting=False)

#--------------------------------------------------elm 参数选择--------------------------------------------------
#/
#the train processing of kernel based extreme learning machine using cross-validation method
#funcition:'rbf', 'linear', 'poly'
#/
def ELM_kernel_train_reKFold(indata, outdata, K, repeated_num, function):
	#----------ELM Init begin----------
	hidden_C = -3
	test_error_aver_min = 1e10  #相对误差最小
	if len(outdata.shape) == 1:
		test_size = len(outdata) / K
	else:
		test_size = len(outdata) * len(outdata[0]) / K
	test_error = []  #保存每个C,L对应的误差

	#ELM参数
	C_best = 0
	params_best = [0, 0]
	elmk_best = 0

	#cross validation
	kf = RepeatedKFold(n_splits=K, n_repeats=repeated_num, random_state=12883823)

	#----------elm train begin----------
	print('\n-----kernel elm parameter selection-----')
	while hidden_C <= 3:
		#--------------------linear--------------------
		if function == 'linear':
			test_error_aver = 0
			exception_flag = 0

			for train_index, test_index in kf.split(indata):
				Input_train = indata[train_index]
				Output_train = outdata[train_index]
				Input_test = indata[test_index]
				Output_test = outdata[test_index]

				try:  #求逆异常
					elmk = ELM_kernel_train(Input_train, Output_train, [function, hidden_C, []])
				except:
					exception_flag = 1
					break
				else:
					capacity_test = ELM_kernel_predict(Input_test, Output_test, elmk)  #elm测试
					test_error_aver += np.linalg.norm(Output_test - capacity_test)  #矩阵F范数

			if exception_flag == 0:  #非异常
				test_error.append([hidden_C, (test_error_aver / (K * repeated_num))**2 / test_size])
				if test_error_aver_min > test_error_aver:  #更优参数
					#print('+++++++++++++++++++++++++')
					test_error_aver_min = test_error_aver
					C_best = hidden_C
					elmk_best = elmk
		elif function == 'rbf':
			params_1 = -10
			while params_1 <= 10:
				test_error_aver = 0
				exception_flag = 0

				for train_index, test_index in kf.split(indata):
					Input_train = indata[train_index]
					Output_train = outdata[train_index]
					Input_test = indata[test_index]
					Output_test = outdata[test_index]

					try:  #求逆异常
						elmk = ELM_kernel_train(Input_train, Output_train, [function, hidden_C, [params_1]])
					except:
						exception_flag = 1
						break
					else:
						capacity_test = ELM_kernel_predict(Input_test, Output_test, elmk)  #elm测试
						test_error_aver += np.linalg.norm(Output_test - capacity_test)  #矩阵F范数

				if exception_flag == 0:  #非异常
					test_error.append([hidden_C, params_1, (test_error_aver / (K * repeated_num))**2 / test_size])
					if test_error_aver_min > test_error_aver:  #更优参数
						#print('+++++++++++++++++++++++++')
						test_error_aver_min = test_error_aver
						C_best = hidden_C
						params_best[0] = params_1
						elmk_best = elmk

					#print(params_1)
				params_1 += 1
		elif function == 'poly':
			params_1 = -5
			while params_1 <= 5:
				params_2 = -5
				while params_2 <= 5:
					test_error_aver = 0
					exception_flag = 0

					for train_index, test_index in kf.split(indata):
						Input_train = indata[train_index]
						Output_train = outdata[train_index]
						Input_test = indata[test_index]
						Output_test = outdata[test_index]

						try:  #求逆异常[function, C, []]
							elmk = ELM_kernel_train(Input_train, Output_train, [function, hidden_C, [params_1, params_2]])
						except:
							exception_flag = 1
							break
						else:
							capacity_test = ELM_kernel_predict(Input_test, Output_test, elmk)  #elm测试
							test_error_aver += np.linalg.norm(Output_test - capacity_test)  #矩阵F范数

					if exception_flag == 0:  #非异常
						test_error.append([hidden_C, params_1, params_2, (test_error_aver / (K * repeated_num))**2 / test_size])
						if test_error_aver_min > test_error_aver:  #更优参数
							#print('+++++++++++++++++++++++++')
							test_error_aver_min = test_error_aver
							C_best = hidden_C
							params_best = [params_1, params_2]
							elmk_best = elmk
						#print('params_2:', params_2)
					params_2 += 1

				#print('params_1:', params_1)
				params_1 += 1

		hidden_C += 1
		print('error', (test_error_aver_min / (K * repeated_num))**2 / test_size)
		#print('\n-----------------hidden_C-----------------')
		#print(hidden_C)
	#----------elm train end----------
	test_error_aver_min = (test_error_aver_min / (K * repeated_num))**2 / test_size  #预测均方误差

	return C_best, params_best, elmk_best, test_error_aver_min, test_error

#/
#the test processing of kernel based extreme learning machine
#/
def ELM_kernel_test(indata, outdata, elmk_best):
	#----------elm test begin----------
	print('\n-----kernel elm test-----')
	if len(outdata.shape) == 1:
		test_size = len(outdata)
	else:
		test_size = len(outdata) * len(outdata[0])

	#--------------------linear--------------------
	capacity_test = ELM_kernel_predict(indata, outdata, elmk_best)  #elm测试
	test_error = np.linalg.norm(outdata - capacity_test)  #矩阵F范数
	test_error_aver = (test_error**2) / test_size

	print('error', test_error_aver)  #预测均方误差
	#----------elm train end----------

	return test_error_aver

#/
#集成学习bagging
#/
def ELM_kernel_train_reKFold_ensember(x_train, y_train, K, repeated_num, ensember_num, function):
	print('bagging ensember train......')
	elmkset = []
	for i in range(ensember_num):
		x_choiced = choices(x_train, k=50)
		y_choiced = []
		for item in x_choiced:
			y_choiced.append(y_train[x_train.index(item)])
		C_best, params_best, elmk_best, test_error_aver, test_error_C_L = ELM_kernel_train_reKFold(np.array(x_choiced), np.array(y_choiced), K, repeated_num, 'rbf')
		elmkset.append(elmk_best)
		print('集成学习预测均方误差:', test_error_aver)

	return elmkset

def ELM_kernel_test_ensember(x_test, y_test, elmkset, ensember_num):
	print('bagging ensember test......')
	capacity_test = 0
	if len(y_test.shape) == 1:
		test_size = len(y_test)
	else:
		test_size = len(y_test) * len(y_test[0])

	for i in range(ensember_num):
		capacity_test += ELM_kernel_predict(x_test, y_test, elmkset[i])  #elm测试
	test_error = np.linalg.norm(y_test - capacity_test / ensember_num)
	test_error_aver = (test_error**2) / test_size
	print('集成学习测试均方误差:', test_error_aver)

	return test_error_aver

#--------------------------------------------------文件存储和提取--------------------------------------------------
#/
#存储kernel ELM参数
#/
def Store_kernel_elmdata(C_best, params_best, elmk_best, test_error_aver, test_error_C_L, save_num):
    #创建文件夹
    path = r'save\result'
    if not os.path.exists(path+'\\'+str(save_num)):
        os.mkdir(path+'\\'+str(save_num))
        os.mkdir(path+'\\'+str(save_num)+'\\'+'weights')
        os.mkdir(path+'\\'+str(save_num)+'\\'+'analysis')

    StoreWeights(C_best, path+'\\'+str(save_num)+r'\weights\C_best.txt')
    StoreWeights(params_best, path+'\\'+str(save_num)+r'\weights\params_best.txt')
    StoreWeights(elmk_best, path+'\\'+str(save_num)+r'\weights\elmk_best.txt')
    StoreWeights(test_error_aver, path+'\\'+str(save_num)+r'\weights\test_error_aver.txt')
    StoreWeights(test_error_C_L, path+'\\'+str(save_num)+r'\weights\test_error_C_L.txt')

def Grab_kernel_elmdata(save_num):
    path = r'save\result'
    C_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\C_best.txt')
    params_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\params_best.txt')
    elmk_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\elmk_best.txt')
    test_error_aver = GrabWeights(path+'\\'+str(save_num)+r'\weights\test_error_aver.txt')
    test_error_C_L = GrabWeights(path+'\\'+str(save_num)+r'\weights\test_error_C_L.txt')

    return C_best, params_best, elmk_best, test_error_aver, test_error_C_L

#/
#参数图形绘制
#/
def Figure_kernel_elm(save_num, function):
	path = r'save\result'
	test_error_C_L = GrabWeights(path+'\\'+str(save_num)+r'\weights\test_error_C_L.txt')
	C_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\C_best.txt')
	params_best = GrabWeights(path+'\\'+str(save_num)+r'\weights\params_best.txt')
	test_error_aver = GrabWeights(path+'\\'+str(save_num)+r'\weights\test_error_aver.txt')

	print('C_best:', C_best, 'params_best:', params_best)
	print('test_error_aver:', test_error_aver)

	if function == 'linear':
		x = np.array(test_error_C_L)[:, 0]
		y = np.array(test_error_C_L)[:, 1]

		plt.plot(x, y)
		plt.xlabel('C')
		plt.ylabel('RMSE')

		# 添加坐标轴(顺序是Z, Y, X)
		plt.savefig(path+'\\'+str(save_num)+r'\analysis\linear_figure'+'.png', dpi=1200, format='png')
		plt.show()
	elif function == 'rbf':
		x = np.array(test_error_C_L)[:, 0]
		y = np.array(test_error_C_L)[:, 1]
		z = np.array(test_error_C_L)[:, 2]

		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(x, y, z, s=2)

		# 添加坐标轴(顺序是Z, Y, X)
		ax.set_xlabel('C', fontdict={'size': 15, 'color': 'red'})
		ax.set_ylabel('grmma', fontdict={'size': 15, 'color': 'red'})
		ax.set_zlabel('RMSE', fontdict={'size': 15, 'color': 'red'})
		plt.savefig(path+'\\'+str(save_num)+r'\analysis\rbf_figure'+'.png', dpi=1200, format='png')
		plt.show()
	elif function == 'poly':
		x = np.array(test_error_C_L)[:, 0]
		y = np.array(test_error_C_L)[:, 1]
		z = np.array(test_error_C_L)[:, 3]

		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(x, y, z, s=2)

		# 添加坐标轴(顺序是Z, Y, X)
		ax.set_xlabel('C', fontdict={'size': 15, 'color': 'red'})
		ax.set_ylabel('param1', fontdict={'size': 15, 'color': 'red'})
		ax.set_zlabel('RMSE', fontdict={'size': 15, 'color': 'red'})
		plt.savefig(path+'\\'+str(save_num)+r'\analysis\poly_figure'+'.png', dpi=1200, format='png')
		plt.show()