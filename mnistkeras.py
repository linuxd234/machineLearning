#########################################################################
# 功能：使用keras实现多层感知的mnist识别
# 2020/9/28 create by linuxd

#########################################################################
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dropout
import numpy as np
np.random.seed(10)
# import os
# os.environ['KERAS_BACKEND']='tensorflow'

#########################################################################
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()
# 通过mnist模块读取训练、测试数据
# print('train data=',len(x_train_image))
# print('test data=',len(x_text_image))
x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')
# 将图像矩阵转化为一维浮点数组

x_Train_normalize = x_Train/255
x_Test_normalize = x_Test/255
# 对取值进行压缩，方便计算

y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)
# 将标签转化为独热编码

#########################################################################
model = Sequential()

# model.add(Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu'))
model.add(Dense(units=500, input_dim=784,
                kernel_initializer='normal', activation='relu'))
# units该层输出维数，input_dim是输入尺寸
model.add(Dropout(0.5))
model.add(Dense(units=500, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# print(model.summary())

##########################################################################

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

##########################################################################

train_history = model.fit(x=x_Train_normalize, y=y_Train_OneHot,
                          validation_split=0.2, epochs=10, batch_size=200, verbose=2)

##########################################################################

# import showplot#将显示图标的函数单独拿出去了，显示训练和验证的变化曲线
# show_train_history(train_history,'accuracy','val_accuracy')#accuracy、val_accuracy是固定的字段

scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
print('\naccuracy1=', scores[1])
# print('accuracy2=',scores[2])不存在这种写法，只有scores[1]才会显示整体测试的正确率

##########################################################################

# prediction=model.predict_classes(x_Test_normalize)
prediction = np.argmax(model.predict(x_Test), axis=-1)
pd.crosstab(y_test_label, prediction, rownames=[
            'label'], colnames=['predict'])  # 显示混淆矩阵

