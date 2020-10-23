import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST/",one_hot=True)

#定义模型
x = tf.placeholder(tf.float32, [None, 784])#行不确定，列确定，先声明一个占位对象
W = tf.Variable(tf.zeros([784, 10]))#权值，因为要与n x 784的输入矩阵做乘法得到n x 10的结果
#即这n个数据的概率向量（某个数字可能性的多少）
b = tf.Variable(tf.zeros([10]))#偏项，即wx+b
#因为是可变的所以设置为变量

y = tf.nn.softmax(tf.matmul(x,W) + b)
#y为输出结果为1x10向量，构造好模型(此模型为全连接层，没有神经元)

#训练模型
y_ = tf.placeholder("float", [None,10])#y`是真实的概率分布
cross_entropy = -tf.reduce_sum(y_*tf.log(y))#损失函数，交叉熵    y是预测的概率分布
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#训练步长，梯度下降优化

#初始化变量
init = tf.initialize_all_variables()

#启动
sess = tf.Session()
sess.run(init)

#训练
for i in range(10000):
  batch_xs, batch_ys = mnist.train.next_batch(100)#一匹100
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
#评估
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))#准确率（拿模型预测的与实际标签相比）
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))