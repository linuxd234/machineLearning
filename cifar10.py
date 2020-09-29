#########################################################################
# 功能：使用keras实现卷积神经网络的cifar10图像分类识别
# 2020/9/28 create by linuxd

#########################################################################
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
import pandas as pd
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
np.random.seed(10)

(x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data()

# print('train:',len(x_train_image))
# print('test:',len(x_test_image))

###################################

label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog",
              7: "horse", 8: "ship", 9: "truck"}
# 创建对应关系字典


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    # 该函数用来打印图片、真实标签、预测值，需要参数为图片、标签、预测对象、从第几个开始idx、显示几条num
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)  # 建立子图为5行5列
        ax.imshow(images[idx], cmap='binary')  # 画出子图
        title = str(i)+','+label_dict[labels[i][0]]
        if len(prediction) > 0:
            title += '--->'+label_dict[prediction[i]]
        ax.set_title(title, fontsize=10)  # 将以上的title放入
        ax.set_xticks([])
        ax.set_yticks([])  # 设置不显示刻度
        idx += 1  # 读取下一张
    plt.show()

# plot_images_labels_prediction(x_train_image,y_train_label,[],0)用于显示前十条图片

###################################


x_img_train_normalize = x_train_image.astype('float32')/255.0
x_img_test_normalize = x_test_image.astype('float32')/255.0

y_train_OneHot = np_utils.to_categorical(y_train_label)
y_test_OneHot = np_utils.to_categorical(y_test_label)

# x_train_image.shape
# x_img_train_normalize.shape
# y_train_label.shape
# y_train_OneHot.shape

#########################################################################


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                 input_shape=(32, 32, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 padding='same', activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))

# print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

#########################################################################

try:
    model.load_weights("saveModel/cifarModel.h5")
    print("yes!")
except:
    print("noooooo")
    train_history = model.fit(x=x_img_train_normalize, y=y_train_OneHot,
                              validation_split=0.2, epochs=10, batch_size=128, verbose=2)
    # verbose=2显示训练过程
    model.save_weights("saveModel/cifarModel.h5")
    print("saved")

#########################################################################


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('train_history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(train_history, 'accuracy', 'val_accuracy')  # 显示训练、验证准确率曲线

###################################

scores = model.evaluate(x_img_test_normalize, y_test_OneHot)
print(scores)
# predict_classes与predict的区别
prediction = model.predict_classes(x_img_test_normalize)  # 是返回的分类
predict_probability = model.predict(x_img_test_normalize)  # 是返回的是每个分类的概率值
print(prediction[:10])  # 返回前十条预测结果

###################################

plot_images_labels_prediction(x_train_image, y_train_label, prediction, 0)
# 显示出从0到9的全部图片及预测结果

###################################


def show_Predicted_Probability(real, prediction, x_img, predict_probability, index):
    # real为真实的标签，他是二维数组（可以转化为一维）prediction为分类预言
    # predict_probability是一组概率的预言 index是某一条记录 x_img是输入的测试图片
    print('label:', label_dict[real[index][0]],
          'prediction', label_dict[prediction[index]])
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(x_img[index], (32, 32, 3)))
    plt.show()
    for j in range(0, 10):
        print(label_dict[j]+'probability:%1.9f' %
              (predict_probability[index][j]))


show_Predicted_Probability(y_test_label, prediction,
                           x_test_image, predict_probability, 10)
# 显示第index个的预测信息信息（属于每个的概率）
# show_Predicted_Probability(y_test_label,prediction,x_test_image,predict_probability,1)
# show_Predicted_Probability(y_test_label,prediction,x_test_image,predict_probability,100)

#########################################################################

print(label_dict)
pd.crosstab(y_test_label.reshape(-1), prediction,
            rownames=['label'], colnames=['predict'])
