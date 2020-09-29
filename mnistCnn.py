#########################################################################
# 功能：使用keras实现卷积神经网络的mnist识别
# 2020/9/28 create by linuxd

#########################################################################
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import showplot  # 将显示图标的函数单独拿出去了
import pandas as pd
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dropout
from keras.datasets import mnist
import numpy as np
np.random.seed(10)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_Train4D = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_Test4D = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
# 转化成28x28x1的矩阵

x_Train4D_normalize = x_Train4D/255
x_Test4D_normalize = x_Test4D/255

y_Train_OneHot = np_utils.to_categorical(y_train)
y_Test_OneHot = np_utils.to_categorical(y_test)

#########################################################################


model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same',
                 input_shape=(28, 28, 1), activation='relu'))
# 该层卷积会提取16个特征（图片）权重矩阵为5向 padding不会使得图片变小
model.add(MaxPooling2D(pool_size=(2, 2)))
# 用2x2的权重矩阵进行池化
model.add(Conv2D(filters=36, kernel_size=(5, 5),
                 padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())  # 平缓层是用来承上启下的含有神经元数为上一层输出矩阵的形状
model.add(Dense(128, activation='relu'))  # 隐藏层
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

# print(model.summary())

#########################################################################

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

#########################################################################

train_history = model.fit(x=x_Train4D_normalize, y=y_Train_OneHot,
                          validation_split=0.2, epochs=10, batch_size=300, verbose=2)

#########################################################################

show_train_history(train_history, 'accuracy', 'val_accuracy')

#########################################################################

scores = model.evaluate(x_Test4D_normalize, y_Test_OneHot)
print('\naccuracy1=', scores[1])

#########################################################################


prediction = model.predict_classes(x_Test4D_normalize)
pd.crosstab(y_test, prediction, rownames=['label'], colnames=['predict'])
