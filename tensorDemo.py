import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def layer(output_dim,input_dim,inputs,activation=None):
    W=tf.Variable(tf.random_normal([input_dim,output_dim]))
    b=tf.Variable(tf.random_normal([1,output_dim]))# 结果是向量
    XWb=tf.matmul(inputs,W)+b# 是x乘w所以inputdim是行outputdim是列
    if activation is None:
        outputs=XWb
    else:
        outputs=activation(XWb)# 这都行！
    return outputs


# b=tf.constant(2,name='b')
# a=tf.Variable(b+5,name='a')
# c=tf.placeholder("int32")

X=tf.placeholder("float",[None,4])# 一定是个二维数组
hide=layer(output_dim=3,input_dim=4,inputs=X,activation=tf.nn.relu)
y=layer(output_dim=2,input_dim=3,inputs=hide)

# X=tf.placeholder("float",[None,3])# 后边得加上feed_dict
# X=tf.Variable([[0.4,0.2,0.4]])
# W=tf.Variable([[-0.5,-0.2],[-0.3,0.4],[-0.5,0.2]])
# b=tf.Variable([[0.1,0.2]])
# W=tf.Variable(tf.random_normal([3,2]))
# b=tf.Variable(tf.random_normal([1,2]))
# XWb=tf.matmul(X,W)+b
# y=tf.nn.relu(XWb)

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)

    # X_array=np.array([[0.4,0.2,0.4]])
    X_array=np.array([[0.4,0.2,0.4,0.5]])   
    (layer_hide,layer_x,layer_y)=sess.run((hide,X,y),feed_dict={X:X_array})
    print('input:\n',layer_x)
    print('hide:\n',layer_hide)
    print('output:\n',layer_y)

    # print('XWb:\n',sess.run(XWb,feed_dict={X:X_array}))
    # print('y:\n',sess.run((y),feed_dict={X:X_array}))
    # print('#################')
    # print('b:\n',sess.run(b))
    # print('w:\n',sess.run(W))
    # (_w,_b,_x,_y)=sess.run((W,b,X,y),feed_dict={X:X_array}))一次可以得到三个tensor变量


tf.summary.merge_all()# 整合所有数据
train_writer=tf.summary.FileWriter('log/area',sess.graph)
# 将所有数据写到log中


