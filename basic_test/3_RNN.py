# _*_ coding:utf-8 _*_

# @File  : 3_RNN.py
# @Author: teddyjkwang
# @Date  : 2019/9/6
# @Desc  :

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 4
num_classes = 2
state_size = 4
num_steps = 10
learning_rate = 0.2

def gen_data(size = 1000000):
    """
    X:在时间t，Xt有50%的概率为1；50%的概率为0
    Y:在时间t，Yt有50%的概率为1；50%的概率为0
        除此外，如果Xt-3 = 1,Yt为1的概率增加50%；如果Xt-8 = 1,Yt为1的概率减少25%。
        如果上述两条件同时满足，Yt为1的概率为75%
    :param size:
    :return: 生成数据 Xt的值有50%的概率为1；50%的概率为0
    """
    X = np.array(np.random.choice(2,size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand()>threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X,np.array(Y)

# 两个重要参数 batch_size 和num_steps
# batch_size : 将数据分成多少块
# num_steps : 输入rnn_cell中的窗口大小；

def gen_batch(raw_data,batch_size,num_steps):
    raw_x,raw_y = raw_data
    data_x = raw_x.reshape(-1,batch_size,num_steps)
    data_y = raw_y.reshape(-1,batch_size,num_steps)
    for i in range(data_x.shape[0]):
        yield (data_x[i],data_y[i])

    # data_length = len(raw_x)

    # 首先将数据切分成batch_size份 0-batch_partition_length;batch_partition_length-2*batch_partition_length
    # batch_partition_length = data_length//batch_size # 没份数据的长度
    # data_x = np.zeros([batch_size,batch_partition_length],dtype = tf.int32)
    # data_y = np.zeros([batch_size,batch_partition_length],dtype = tf.int32)
    #

    # for i in range(batch_size):
    #     data_x[i] = raw_x[batch_partition_length*i:batch_partition_length(i+1)]
    #     data_y[i] = raw_x[batch_partition_length*i:batch_partition_length(i+1)]
    #
    #     # RNN每次只处理num_steps个数据，所以讲每个batch_partition_length 切分成epoch_size份，
    #     # ，每份num_steps个数据。注意这里的epoch_size和模型中的epoch不同
    #     epoch_size = batch_partition_length//num_steps
    #
    #     for i in range(epoch_size):
    #         x = data_x[:,i*num_steps:(i+1)*num_steps]
    #         y = data_y[:,i*num_steps:(i+1)*num_steps]
    #         yield(x,y)

def gen_epochs(n,num_steps):
    for i in range(n):
        yield gen_batch(gen_data(),batch_size,num_steps)


x = tf.placeholder(tf.int32,[batch_size,num_steps],name="input_placeholder")
y = tf.placeholder(tf.int32,[batch_size,num_steps],name="labels_placeholder")

# RNN的初始化状态；注意state与input保持一致；接下来会有concat操作，所以这里有batch的维度，即每个样本都要有隐层状态
init_state = tf.zeros([batch_size,state_size])

# 输入转化为onehot编码，两个类别。[batch_size,num_steps,num_classes]
x_one_hot = tf.one_hot(x,num_classes)

# 将输入unstack，在num_steps上解绑；方便给每个循环单元输入、这里RNN每个cell都处理一个barch的输入（即batch个二进制样本输入）
rnn_inputs = tf.unstack(x_one_hot ,axis=1)

#定义rnn_cell的权重参数
with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W',[num_classes+state_size,state_size])
    b = tf.get_variable('b',[state_size],initializer=tf.constant_initializer(0.0))

# 定义为reuse模式，参数保持相同
def rnn_cell(rnn_input,state):
    with tf.variable_scope('rnn_cell',reuse=True):
        W = tf.get_variable('W',[num_classes+state_size,state_size])
        b = tf.get_variable('b',[state_size],initializer=tf.constant_initializer(0.0))
    # 定义rnn_cell具体的操作，这里是用的是最简单的rnn 非lstm
    return tf.tanh(tf.matmul(tf.concat([rnn_input,state],1),W)+b)

state = init_state
rnn_outputs = []

#循环num_steps次，将一个序列输入RNN模型
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input,state)
    rnn_outputs.append(state)

final_state = rnn_outputs[-1]

# softmax层
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

# 将num_steps个输出全部分别进行计算其输出，然后softmaxyuce
logits = [tf.matmul(rnn_output,W)+b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

y_as_list = tf.unstack(y,num=num_steps,axis=1)

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits = logit) for logit,label in zip(logits,y_as_list)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

# 模型训练

def train_network(num_epochs,num_steps,state_size=4,verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        # 得到数据 因为num_epochs==5;外循环只执行五次
        for idx,epoch in enumerate(gen_epochs(num_epochs,num_steps)):
            training_loss = 0
            # 保存每次执行后的最后状态，给下一次执行
            training_state = np.zeros((batch_size,state_size))
            if verbose:
                print("\nEPOCH",idx)
            for step,(X,Y) in enumerate(epoch):
                tr_losses,training_loss_,training_state,_ = sess.run([losses,total_loss,final_state,train_step],feed_dict={x:X,y:Y,init_state:training_state})
            training_loss += training_loss_
            if step%100==0 and step>0:
                if verbose:
                    print("Average loss at step",step,"for last 100 steps:",training_loss/100)
                    training_losses.append(training_loss/100)
                    training_loss=0

    return  training_losses

training_losses = train_network(5,num_steps)
plt.plot(training_losses)
plt.show()
