# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 23:52:16 2018

@author: hecongcong
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


####下载数据
mnist=input_data.read_data_sets(r'E:\tensorflow\DATA\MNIST',\
                                one_hot=True)


####设置训练参数
learning_rate=0.01
max_samples=400000  #最大训练样本数
batch_size=128   
display_step=10  


n_input=28   #输入层神经元个数
n_steps=28   #时间步的个数
n_hidden=256 #隐藏神经元个数
n_classes=10  #分类数


####创建输入、输出及各个参数
x=tf.placeholder('float',[None,n_steps,n_input])
y=tf.placeholder('float',[None,n_classes])

weights=tf.Variable(tf.random_normal([2*n_hidden,n_classes]))
biases=tf.Variable(tf.random_normal([n_classes]))



####定义Bidirectional LSTM网络的生成函数.
def BiRNN(x,weights,biases):
    '''
    x的shape:[batch_size,n_steps,n_input]
    '''
    x=tf.transpose(x,[1,0,2])  #[n_steps,batch_size,n_input]
    x=tf.reshape(x,[-1,n_input]) #[batch_size*n_steps,n_input]
    x=tf.split(x,n_steps) #长度为n_steps,元素shape:[batch_size,n_input]的元组
    '''
    split 'x' along axis=0:
    把x沿0轴分割成shape相等的n_steps个sub tensor.
    返回的是一个长度为n_steps的tuple.
    '''
    
    lstm_fw_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    lstm_bw_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    
    outputs,_,_=tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,\
                                        lstm_bw_cell,x,\
                                        dtype=tf.float32)
    '''
    return:(outputs,output_state_fw,output_state_bw)
    outputs:长度为n_steps的list,其中元素为:[batch_size,\
    cell_fw.output_size+cell_bw.output_size]
    '''
    return(tf.matmul(outputs[-1],weights)+biases)
    
    
####定义损失函数，优化器,准确率.
pred=BiRNN(x,weights,biases)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                            logits=pred,\
                            labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1)) ##return:bool
accuracy=tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32))


init=tf.global_variables_initializer()

'''
####执行训练和测试操作.
with tf.Session() as sess:
    sess.run(init)
    step=1
    while(step*batch_size<max_samples):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape((batch_size,n_steps,n_input))
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y}
        if(step % display_step==0):
            acc=sess.run(accuracy,feed_dict={x:batch_x,\
                                             y:batch_y})
            loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print('Iter '+ str(step*batch_size) + ', Minibatch \
                  Loss= ' + '{:.6f}'.format(loss) + ', Training \
                  Accuracy= ' + '{:.5f}'.format(acc))
        step+=1
        
    print('Optimization Finished!')
    
    
    ####训练结束后,对测试数据进行预测   
    test_len=10000
    test_data=mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
    test_label=mnist.test.labels[:test_len]
    print('Testing Accuracy:' , sess.run(accuracy,feed_dict=\
                                         {x:test_data,y:test_label}))
    
'''

















