# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
######载入数据
(x_train,y_train),(x_test,y_test)=mnist.load_data()
number=10000
x_train=x_train[0:number]
y_train=y_train[0:number]
x_train=x_train.reshape(number,28*28)
x_test=x_test.reshape(x_test.shape[0],28*28)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)
x_train=x_train
x_test=x_test
x_train=x_train/255 #做nomalization，因为数据集是0-255来表示
x_test=x_test/255   
######载入数据结束

x_train.shape
x_train[0]
y_train.shape
y_train[0]
###开始
model = Sequential()
model.add(Dense(input_dim=28*28,units=633,activation='sigmoid'
                ))
model.add(Dense(units=633,activation='sigmoid'))
model.add(Dense(units=633,activation='sigmoid'))
model.add(Dense(units=10,activation='softmax'))
model.compile(loss='mse',optimizer=SGD(lr=0.1),metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=100,epochs=20)
result= model.evaluate(x_test,y_test)
print('test acc:',result[1])
#########失败的code，因为loss=mse不太好，不适合test
#还是要看在training set的表现,虽然可能over fitting，但是在trainingset 的结果一定是要好的

result1=model.evaluate(x_train,y_train,batch_size=10000)
print('train set acc:',result1[1]) #所以在training set的时候，已经不太好了

####模型改进
#####mse换成 cross entropy, 
model = Sequential()
model.add(Dense(input_dim=28*28,units=633,activation='sigmoid'
                ))
model.add(Dense(units=633,activation='sigmoid'))
model.add(Dense(units=633,activation='sigmoid'))
model.add(Dense(units=10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1),metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=100,epochs=20)
result= model.evaluate(x_test,y_test)
print('test acc:',result[1])
result1=model.evaluate(x_train,y_train,batch_size=10000)
print('train set acc:',result1[1]) 
#可以batch_size开大，开到10000，但是结果不会好，
#也可以试着增加layer
#也可以sigmoid改成relu试试看，这可以解决vanishing gradient problem，因为layer过多的问题
model = Sequential()
model.add(Dense(input_dim=28*28,units=633,activation='sigmoid'
                ))
model.add(Dense(units=633,activation='relu'))
model.add(Dense(units=633,activation='relu'))
for i in range(10):
    model.add(Dense(units=689,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1),metrics=['accuracy']) #SGD也可以改成adam，加上收敛的惯性
model.fit(x_train,y_train,batch_size=100,epochs=20)
result= model.evaluate(x_test,y_test)
print('test acc:',result[1])
result1=model.evaluate(x_train,y_train,batch_size=10000)
print('train set acc:',result1[1]) 
#可以看到accuracy 变大了，relu解决了layer 过多，gradient vanishing的problem，准确率到92%
#可以加dropout，如果发现test 和train 的差距很大
#model.add((Dropout(0.7)))，在每一层后面加
model = Sequential()
model.add(Dense(input_dim=28*28,units=633,activation='sigmoid'
                ))
model.add(Dense(units=633,activation='relu'))
model.add((Dropout(0.7)))
model.add(Dense(units=633,activation='relu'))
model.add((Dropout(0.7)))
model.add(Dense(units=633,activation='relu'))
model.add((Dropout(0.7)))
model.add(Dense(units=10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) #SGD也可以改成adam，加上收敛的惯性
model.fit(x_train,y_train,batch_size=100,epochs=20)
result= model.evaluate(x_test,y_test)
print('test acc:',result[1])
result1=model.evaluate(x_train,y_train,batch_size=10000)
print('train set acc:',result1[1]) 
#一般是training太好，才会加dropout,也可以考虑增加Units,修改模型


 