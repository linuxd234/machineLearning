import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.chdir("C:\\Users\\46240")

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST/",one_hot=True)

batch_images_xs,batch_labels_ys=mnist.train.next_batch(batch_size=100)


def layer(output_dim,input_dim,inputs,activation=None,names='none'):
    W=tf.Variable(tf.random_normal([input_dim,output_dim]))
    b=tf.Variable(tf.random_normal([1,output_dim]))# 结果是向量
    XWb=tf.matmul(inputs,W)+b# 是x乘w所以inputdim是行outputdim是列
    if activation is None:
        outputs=XWb
    else:
        outputs=activation(XWb,name=names)# 这都行！
    return outputs

X=tf.placeholder("float",[None,784],name='input_img')# 一定是个二维数组
y_label=tf.placeholder("float",[None,10],name='predictions')

h1=layer(input_dim=784,output_dim=256,inputs=X,activation=tf.nn.relu,names='hide_layer')
y_predict=layer(input_dim=256,output_dim=10,inputs=h1,activation=None,names='out_layer')

loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict,labels=y_label))
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

correct_prediction=tf.equal(tf.argmax(y_label,1),tf.argmax(y_predict,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

totalBatchs=550
trainEpochs=15
batchSize=100
loss_list=[];epoch_list=[];accuracy_list=[]
from time import time
startTime=time()

sess=tf.Session()
sess.run(tf.global_variables_initializer())

tf.summary.merge_all()
train_writer=tf.summary.FileWriter('log//area',sess.graph)

for epoch in range(trainEpochs):
    for i in  range(totalBatchs):
        batch_x,batch_y=mnist.train.next_batch(batchSize)
        sess.run(optimizer,feed_dict={X:batch_x,y_label:batch_y})
        loss,acc=sess.run([loss_function,accuracy],feed_dict={X:mnist.validation.images,y_label:mnist.validation.labels})
        epoch_list.append(epoch)
        loss_list.append(loss)
        accuracy_list.append(acc)
        print("Train Epoch:",'%02d'%(epoch+1),"Loss=",'{:.9f}'.format(loss),"Accuracy=",acc)
duration=time()-startTime
print("Train finish in:",duration)

import matplotlib.pyplot as plt
fig=plt.gcf()
fig.set_size_inches(4,2)
plt.plot(epoch_list,loss_list,label='loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'],loc='upper left')
plt.show()

fig=plt.gcf()
fig.set_size_inches(4,2)
plt.plot(epoch_list,accuracy_list,label='accuracy')
plt.ylabel('accuracy')
plt.ylim(0.8,1)
plt.xlabel('epoch')
plt.legend(['loss'],loc='upper left')
plt.show()

accnub=sess.run(accuracy,feed_dict={X:mnist.test.images,y_label:mnist.test.labels})
print("Accuracy:",accnub)