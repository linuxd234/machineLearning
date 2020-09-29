# -*-coding:utf-8-*-
#########################################################################
# 功能：使用keras实现RNN\LSTM的imdb情感识别
# 2020/9/29 create by linuxd

#########################################################################
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding  # 加入嵌入层
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import urllib.request
import os
import tarfile

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath = "data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('download:', result)
# 下载数据集

if not os.path.exists("data/aclImdb"):
    tfile = tarfile.open("data/aclImdb_v1.tar.gz", 'r:gz')
    result = tfile.extractall('data/')
# 解压数据集


def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)
# 去除html标签


def read_file(filetype):
    # 根据test还是train读取数据
    path = "data/aclImdb/"
    file_list = []

    positive_path = path+filetype+"/pos/"
    for f in os.listdir(positive_path):
        # f应该是文件名或者文件指针
        file_list += [positive_path+f]
    negative_path = path+filetype+"/neg/"
    for f in os.listdir(negative_path):
        # f应该是文件名或者文件指针
        file_list += [negative_path+f]
    print('read', filetype, 'files:', len(file_list))
    all_labels = ([1]*12500+[0]*12500)
    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8')as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
            # 返回的是一个数组，list
    return all_labels, all_texts

##################################


y_train, train_text = read_file("train")
y_test, test_text = read_file("test")

#print (train_text[0],y_train[0])
token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)  # 用训练集录就足够了
# 顺序读取，将最多的单词收录到字典里面

x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
# 将文字根据token字典转成数字

x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test = sequence.pad_sequences(x_test_seq, maxlen=100)
# 将长度固定为100

#########################################################################


model = Sequential()

model.add(Embedding(output_dim=32, input_dim=2000, input_length=100))
# 2000是因为将2000个单词转为oneHot模式了此层作用是2000--->32
model.add(Dropout(0.2))

# model.add(SimpleRNN(units=16))
model.add(LSTM(32))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
# 改用softmax也一样,但是单分类问题多用sigmoid与binary组合

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

#########################################################################

try:
    model.load_weights("saveModel/cifarModel.h5")
    print("yes!loaded")
except:
    print("not existed")
    train_history = model.fit(
        x_train, y_train, validation_split=0.2, epochs=10, batch_size=100, verbose=2)
    model.save_weights("saveModel/cifarModel.h5")
    print("saved")

#########################################################################

scores = model.evaluate(x_test, y_test)
print(scores[1])  # 输出准确率


predict = model.predict_classes(x_test)
predict_classes = predict.reshape(-1)
# model.predict是显示预测结果其中带classes是预测这个属于哪一类，不带的是一组概率值（可max求归属）
# model.envalue是评价准确率的，两者都需要输入测试数据-->都是测试模型的手段

sentimentDict = {1: '正面的', 0: '反面的'}


def display_test_sentiment(i):
    print(test_text[i])
    print('label真实值：', sentimentDict[y_test[i]],
          '预测结果：', sentimentDict[predict_classes[i]])


display_test_sentiment(1)  # 输出预测结果与实际
# display_test_sentiment(100)
# display_test_sentiment(1000)
