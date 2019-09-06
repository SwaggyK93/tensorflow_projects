# _*_ coding:utf-8 _*_

# @File  : 1_test.py
# @Author: teddyjkwang
# @Date  : 2019/9/6
# @Desc  :

import tensorflow as tf
import numpy as np

"""
    TODO 1 基础知识1 
"""
# x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data*0.1+0.3
#
# # print(x_data.shape)
# # print(type(x_data))
#
# #定义变量
# weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
# biases = tf.Variable(tf.zeros([1]))
#
# #计算预测值
# y = weights*x_data+biases
#
# # loss function
# loss = tf.reduce_mean(tf.square(y-y_data))
#
# # 梯度优化下降器 定义learning_rate
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
#
# # train的目标是loss最小化
# train = optimizer.minimize(loss)
#
# # 初始化变量，初始化Weights和biases
# init = tf.global_variables_initializer()
#
# # 创建session 参数初始化
# sess = tf.Session()
# sess.run(init)
#
# for step in range(201):
#     sess.run(train)
#     if step%20==0:
#         print(step,sess.run(weights),sess.run(biases))
"""
    TODO 2 tf.Session() 的两种使用方式
"""
#
# matrix1 = tf.constant([[3,3]])
# matrix2 = tf.constant([[2],[2]])
#
# product = tf.matmul(matrix1,matrix2)
#
# sess = tf.Session()
# result = sess.run(product)
#
# print(result)
# sess.close()
#
#
# with tf.Session() as sess:
#     print(sess.run(product))
#
#
"""
    TODO 3 tf.Variable
"""

# state = tf.Variable(0,name="counter")
#
# print(state.name)
#
# one = tf.constant(1)
#
# new_value = tf.add(state,one)
# update = tf.assign(state,new_value)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(3):
#         sess.run(update)
#         print(sess.run(state))

"""
    TODO 4 tf.placeholder
"""

input1 = tf.placeholder(dtype=tf.float32)
input2= tf.placeholder(dtype=tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[3.],input2:[5]}))
