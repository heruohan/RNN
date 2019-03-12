# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 01:29:17 2018

@author: hecongcong
"""

###################Tensorflow 实现基于LSTM的语言模型.
import time
import numpy as np
import tensorflow as tf
from tensorflow.models.tutorials.rnn.ptb import reader


###########定义语言模型处理输入数据的class,PTBInput.
class PTBInput:
    
    def __init__(self,config,data,name=None):
        self.batch_size=batch_size=config.batch_size
        self.num_steps=num_steps=config.num_steps
        self.epoch_size=((len(data)//batch_size)-1)//num_steps
        self.input_data,self.targets=reader.ptb_producer(data,batch_size,\
                                                    num_steps,name=name)
        
        '''
        batch=reader.ptb_producer(train_data,4,5)
        with tf.Session() as sess:
            corrd=tf.train.Coordinator()
            threads=tf.train.start_queue_runners(coord=coord)
            for i in range(2):
                x,y=sess.run(batch)
                print('x:',x)
                print('y:',y)
            coord.request_stop()
            coord.join(threads)
        
        returns：
        x: [[9970 9971 9972 9974 9975]
         [ 332 7147  328 1452 8595]
         [1969    0   98   89 2254]
         [   3    3    2   14   24]]
        y: [[9971 9972 9974 9975 9976]
         [7147  328 1452 8595   59]
         [   0   98   89 2254    0]
         [   3    2   14   24  198]]
        
        '''

###############定义语言模型的class,PTBModel.
class PTBModel:
    
    def __init__(self,is_training,config,input_):
        self._input=input_
        
        batch_size=input_.batch_size
        num_steps=input_.num_steps
        size=config.hidden_size
        vocab_size=config.vocab_size
        
        ###创建LSTM单元.
        def lstm_cell():
            return(tf.contrib.rnn.BasicLSTMCell(size,forget_bias=0.0,\
                                state_is_tuple=True))
        attn_cell=lstm_cell
        if(is_training and config.keep_prob<1):
            def attn_cell():
                return(tf.contrib.rnn.DropoutWrapper(lstm_cell(),\
                            output_keep_prob=config.keep_prob))
        
        cell=tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)],\
                                         state_is_tuple=True)
        self._initial_state=cell.zero_state(batch_size,tf.float32)
        
        
        
        
        ###创建网络的词嵌入embedding部分，embedding即为将one-hot编码
        ##格式的单词转化为向量表达形式.
        with tf.device('/cpu:0'):
            embedding=tf.get_variable('embedding',[vocab_size,size],\
                                      dtype=tf.float32)
            inputs=tf.nn.embedding_lookup(embedding,input_.input_data)
        
        if(is_training and config.keep_prob<1):
            inputs=tf.nn.dropout(inputs,config.keep_prob)
        
        
        ###定义输出outputs.
        outputs=[]
        state=self._initial_state
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if(time_step>0):
                    tf.get_variable_scope().reuse_variables()
                (cell_output,state)=cell(inputs[:,time_step,:],state)
                outputs.append(cell_output)
        
        ##
        output=tf.reshape(tf.concat(outputs,1),[-1,size])
        softmax_w=tf.get_variable('softmax_w',[size,vocab_size],\
                                  dtype=tf.float32)
        softmax_b=tf.get_variable('softmax_b',[vocab_size],\
                                  dtype=tf.float32)
        logits=tf.matmul(output,softmax_w)+softmax_b
        loss=tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],\
                            [tf.reshape(input_.targets,[-1])],\
                            [tf.ones([batch_size*num_steps],\
                                    dtype=tf.float32)])
        
        self._cost=cost=tf.reduce_sum(loss)/batch_size
        self._final_state=state
        
        
        if(not is_training):
            return
        
        
        ######
        self._lr=tf.Variable(0.0,trainable=False)  #学习速率设置为不可训练.
        tvars=tf.trainable_variables()
        grads,_=tf.clip_by_global_norm(tf.gradients(cost,tvars),\
                                       config.max_grad_norm)
        
        optimizer=tf.train.GradientDescentOptimizer(self._lr)
        self._train_op=optimizer.apply_gradients(zip(grads,tvars),\
                        global_step=tf.train.get_global_step())
        
        
        #########控制学习速率lr.
        self._new_lr=tf.placeholder(tf.float32,shape=[],\
                                    name='new_learning_rate')
        self._lr_update=tf.assign(self._lr,self._new_lr)
        
    ###
    def assign_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict=\
                    {self._new_lr:lr_value})
    
    ###模型定义部分完成.
    
    ##定义PTBModel class的一些property.
    @property
    def input(self):
        return(self._input)
    
    @property
    def initial_state(self):
        return(self._initial_state)
    
    @property
    def cost(self):
        return(self._cost)
    
    @property
    def final_state(self):
        return(self._final_state)
    
    @property
    def lr(self):
        return(self._lr)
    
    @property
    def train_op(self):
        return(self._train_op)
    
    
    

###########定义几种不同大小的模型参数.
####SmallConfig:
class SmallConfig:
    init_scale=0.1
    learning_rate=1.0
    max_grad_norm=5
    num_layers=2
    num_steps=20
    hidden_size=200
    max_epoch=4
    max_max_epoch=13
    keep_prob=1.0
    lr_decay=0.5
    batch_size=20
    vocab_size=10000
    
'''
####MediumConfig:        
class MediumConfig:
    init_scale=0.05
    learning_rate=1.0
    max_grad_norm=5
    num_layers=2
    num_steps=35
    hidden_size=650
    max_epoch=6
    max_max_epoch=39
    keep_prob=0.5
    lr_decay=0.8
    batch_size=20
    vocab_size=10000


####LargeConfig:
class LargeConfig:
    init_scale=0.04
    learning_rate=1.0
    max_grad_norm=10
    num_layers=2
    num_steps=35
    hidden_size=1500
    max_epoch=14
    max_max_epoch=55
    keep_prob=0.35
    lr_decay=1/1.15
    batch_size=20
    vocab_size=10000


####TestConfig:
class TestConfig:
    init_scale=0.1
    learning_rate=1.0
    max_grad_norm=1
    num_layers=1
    num_steps=2
    hidden_size=2
    max_epoch=1
    max_max_epoch=1
    keep_prob=1.0
    lr_decay=0.5
    batch_size=20
    vocab_size=1000


'''

#############定义训练一个epoch数据的函数run_epoch.
def run_epoch(session,model,eval_op=None,verbose=False):
    start_time=time.time()
    costs=0.0
    iters=0
    state=session.run(model.initial_state)
    
    fetches={'cost':model.cost,'final_state':model.final_state}
    if(eval_op is not None):
        fetches['eval_op']=eval_op
    
    for step in range(model.input.epoch_size):
        feed_dict={}
        for i, (c,h) in enumerate(model.initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h
        
        vals=session.run(fetches,feed_dict)
        cost=vals['cost']
        state=vals['final_state']
        
        costs+=cost
        iters+=model.input.num_steps
        
        if(verbose and (step%(model.input.epoch_size//10)==0)):
            print('%.3f perplexity: %.3f speed: %.0f wps' % \
                  (step/model.input.epoch_size,np.exp(costs/iters),\
                   iters*model.input.batch_size/(time.time()-start_time)))
    
    return(np.exp(costs/iters))  #cost是平均过batch_size的.
    
    
     
##############解压PTB数据.
raw_data=reader.ptb_raw_data(r'E:\tensorflow\RNN\simple-examples\data')
train_data,valid_data,test_data,_=raw_data


config=SmallConfig()

eval_config=SmallConfig()
eval_config.batch_size=1
eval_config.num_steps=1


###########创建默认Graph.
with tf.Graph().as_default():
    initializer=tf.random_uniform_initializer(-config.init_scale,\
                                              config.init_scale)
    
    ####
    with tf.name_scope('Train'):
        train_input=PTBInput(config=config,data=train_data,\
                             name='TrainInput')
        with tf.variable_scope('Model',reuse=None,initializer=initializer):
            m=PTBModel(is_training=True,config=config,input_=train_input)
    
    ####
    with tf.name_scope('Valid'):
        valid_input=PTBInput(config=config,data=valid_data,\
                             name='ValidInput')
        with tf.variable_scope('Model',reuse=True,initializer=initializer):
            mvalid=PTBModel(is_training=False,config=config,\
                            input_=valid_input)
    
    ####
    with tf.name_scope('Test'):
        test_input=PTBInput(config=eval_config,data=test_data,\
                            name='TestInput')
        with tf.variable_scope('Model',reuse=True,initializer=initializer):
            mtest=PTBModel(is_training=False,config=eval_config,\
                           input_=test_input)
    
    #################
    sv=tf.train.Supervisor()
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
            lr_decay=config.lr_decay**max(i+1-config.max_epoch,0.0)
            m.assign_lr(session,config.learning_rate*lr_decay)
            
            
            ###
            print('Epoch: %d Learning rate: %.3f' % (i+1,\
                                            session.run(m.lr)))
            train_perplexity=run_epoch(session,m,eval_op=m.train_op,\
                                       verbose=True)
            print('Epoch: %d Train Perlexity: %.3f' % (i+1,train_perplexity))
            valid_perplexity=run_epoch(session,mvalid)
            print('Epoch: %d Valid Perlexity: %.3f' % (i+1,valid_perplexity))
            
        test_perplexity=run_epoch(session,mtest)
        print('Test Perlexity: %.3f' % test_perplexity)
        

        
        
        
        
        
    
    
        
        
        
        














