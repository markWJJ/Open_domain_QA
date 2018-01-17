'''
普通的seq2seq 问答模型
'''

import  tensorflow as tf
import sys
import os
sys.path.append("../")
import Data_deal
from seq2seq_basic import Seq2SeqBasic
import numpy as np
import logging
from seq2seqTF_ops import embedding_encoder,embedding_attention_decoder,embedding_attention_seq2seq,attention_decoder
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("seq2seq")

class Config(object):
    '''
    默认配置
    '''
    learning_rate=0.01
    num_samples=5000
    batch_size=168
    encoder_len=15
    decoder_len=20
    embedding_dim=50
    hidden_dim=100
    train_dir='/baidu_zd_500.txt'
    dev_dir='/dev.txt'
    test_dir='/test.txt'
    model_dir='./save_model/seq2seq.ckpt'
    train_num=50
    use_cpu_num=8
    summary_write_dir="./tmp/seq2seq_my.log"
    epoch=2000
    encoder_mod="lstmTF" # Option=[bilstm lstm lstmTF cnn ]
    use_sample=False
    beam_size=5
    use_MMI=False
config=Config()
tf.app.flags.DEFINE_float("learning_rate", config.learning_rate, "学习率")
tf.app.flags.DEFINE_integer("num_samples", config.num_samples, "采样损失函数的采样的样本数")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "批处理的样本数量")
tf.app.flags.DEFINE_integer("encoder_len", config.encoder_len, "编码数据的长度")
tf.app.flags.DEFINE_integer("decoder_len", config.decoder_len, "解码数据的长度")
tf.app.flags.DEFINE_integer("embedding_dim", config.embedding_dim, "词嵌入惟独.")
tf.app.flags.DEFINE_integer("hidden_dim", config.hidden_dim, "中间节点维度.")
tf.app.flags.DEFINE_integer("train_num", config.train_num, "训练次数，每次训练加载上次的模型.")
tf.app.flags.DEFINE_integer("use_cpu_num", config.use_cpu_num, "限定使用cpu的个数")
tf.app.flags.DEFINE_integer("epoch", config.epoch, "每轮训练迭代次数")
tf.app.flags.DEFINE_integer("beam_size", config.beam_size, "束搜索规模")
tf.app.flags.DEFINE_string("summary_write_dir", config.summary_write_dir, "训练数据过程可视化文件保存地址")
tf.app.flags.DEFINE_string("train_dir", config.train_dir, "训练数据的路径")
tf.app.flags.DEFINE_string("dev_dir", config.dev_dir, "验证数据文件路径")
tf.app.flags.DEFINE_string("test_dir", config.test_dir, "测试数据文件路径")
tf.app.flags.DEFINE_string("model_dir", config.model_dir, "模型保存路径")
tf.app.flags.DEFINE_string("encoder_mod", config.encoder_mod, "编码层使用的模型 lstm bilstm cnn")
tf.app.flags.DEFINE_boolean("sample_loss", False, "是否采用采样的损失函数") # true for prediction
tf.app.flags.DEFINE_string("mod", "predict", "默认为训练") # train or predict
tf.app.flags.DEFINE_boolean('use_MMI',config.use_MMI,"是否使用最大互信息来增加解码的多样性")
FLAGS = tf.app.flags.FLAGS


class Seq2Seq(Seq2SeqBasic):

    def __init__(self,hidden_dim,init_dim,content_len,title_len,con_vocab_len,ti_vocab_len,batch_size):
        self.hidden_dim=hidden_dim
        self.init_dim=init_dim
        self.content_len=content_len
        self.title_len=title_len
        self.content_vocab_len=con_vocab_len
        self.title_vocab_len=ti_vocab_len
        self.batch_size=batch_size
        self.num_class=con_vocab_len
        self.__init_ops__()
        # self.__train_ops__()
        # self.__decoder_ops__()

    def __init_ops__(self):
        '''
        定义变量和占位输入
        :return: 
        '''
        self.content_input = tf.placeholder(dtype=tf.int32, shape=(None, self.content_len))
        self.content_decoder = tf.placeholder(dtype=tf.int32, shape=(None, self.content_len))
        self.mod = "beam_decoder"
        self.title = tf.placeholder(dtype=tf.int32, shape=(None, self.title_len))
        self.content_seq_vec = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.title_seq_vec = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.init_loss = 9999
        self.best_loss = 9999
        self.best_iter = 0
        # 构建encoder层词向量矩阵变量
        self.embeding_content = tf.Variable(tf.random_uniform(shape=(self.content_vocab_len, self.init_dim),
                                                              dtype=tf.float32))
        # 构建decoder层词向量矩阵变量
        self.embeding_title = tf.Variable(tf.random_uniform(shape=(self.title_vocab_len, self.init_dim),
                                                            dtype=tf.float32))
        # 定义lstm 单元
        self.cell = tf.contrib.rnn.LSTMCell(self.hidden_dim,
                                            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                            state_is_tuple=True)

        self.content_emb_input = tf.nn.embedding_lookup(self.embeding_content, self.content_input)
        self.content_emb_decoder = tf.nn.embedding_lookup(self.embeding_content, self.content_decoder)
        self.title_emb = tf.nn.embedding_lookup(self.embeding_title, self.title)

        with tf.variable_scope('decoder_para'):
            if FLAGS.encoder_mod=='bilstm':
                self.decoder_w = tf.get_variable(name='decoder_w', shape=(self.num_class, 2*self.hidden_dim),
                                                 initializer=tf.random_normal_initializer())
                self.decoder_b = tf.get_variable(name='decoder_b', shape=(self.num_class,), initializer=tf.random_normal_initializer())

                # self.decoder_w = tf.get_variable(tf.random_uniform(shape=(self.num_class, 2 * self.hidden_dim), dtype=tf.float32),name='decoder_w')
                # self.decoder_b = tf.Variable(tf.random_uniform(shape=(self.num_class,), dtype=tf.float32),name='decoder_b')
            elif FLAGS.encoder_mod in ['lstm','lstmTF'] :
                self.decoder_w = tf.get_variable(name='decoder_w',shape=(self.num_class,self.hidden_dim),initializer=tf.random_normal_initializer())
                self.decoder_b = tf.get_variable(name='decoder_b',shape=(self.num_class,),initializer=tf.random_normal_initializer())
                # self.decoder_w = tf.Variable(tf.random_uniform(shape=(self.num_class, self.hidden_dim), dtype=tf.float32),name='decoder_w')
                # self.decoder_b = tf.Variable(tf.random_uniform(shape=(self.num_class,), dtype=tf.float32),name='decoder_b')

    def __encoder__(self,encoder_inputs,encoder_len,**kwargs):
        '''
        编码层
        :return: 
        '''
        hidden_dim=kwargs['hidden_dim']
        encoder_sequence_length=kwargs['encoder_sequence_length']
        lstm_input=tf.unstack(encoder_inputs,encoder_len,1)
        cell=kwargs['cell']
        init_dim=kwargs['init_dim']
        encoder_orgion=kwargs['encoder_orgion']
        encoder_vocab_size=kwargs['encoder_vocab_size']

        if FLAGS.encoder_mod=="bilstm":
            cell_f=tf.contrib.rnn.LSTMCell(hidden_dim,
                                            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                            state_is_tuple=True)
            cell_b=tf.contrib.rnn.LSTMCell(hidden_dim,
                                            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                            state_is_tuple=True)
            (out, fw_state, bw_state) = tf.contrib.rnn.static_bidirectional_rnn(cell_f, cell_b, lstm_input,
                                                                         dtype=tf.float32,
                                                                         sequence_length=encoder_sequence_length)

            encoder_state_c=tf.concat((fw_state[0],bw_state[0]),1)
            encoder_state_h=tf.concat((fw_state[1],bw_state[1]),1)
            outs=tf.stack(out)
            outs=tf.reshape(outs,[-1,encoder_len,2*hidden_dim])

        elif FLAGS.encoder_mod=="lstm":

            out, state = tf.contrib.rnn.static_rnn(cell, lstm_input,
                                                    dtype=tf.float32,
                                                    sequence_length=encoder_sequence_length)
            top_states = [
                tf.reshape(e, [-1, 1, hidden_dim]) for e in out
            ]
            outs = tf.concat(top_states, 1)
            encoder_state_c=state[0]
            encoder_state_h=state[1]

        elif FLAGS.encoder_mod=="cnn":
            # convd=[height,width,in_channels,out_channels]
            # 第一层卷积层的size [4,embedding_dim,1,10]
            convd_w=tf.Variable(tf.random_uniform(shape=(4,init_dim,1,20),minval=-0.1,maxval=0.1),dtype=tf.float32)
            convd_b=tf.Variable(tf.random_uniform(shape=(20,),dtype=tf.float32))
            #strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离 strides.shape=inputs.shape [batch_size,height,width,channels]
            #padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
            # 转换shape cnn层的标准输入：[batch_size,height,width,channels]
            cnn_input=tf.reshape(encoder_len,[-1,encoder_len,init_dim,1])
            convd=tf.nn.conv2d(cnn_input,convd_w,strides=[1,1,1,1],padding="SAME") #若滑动stride为1 代表输出维度和输入一致
            convd_1=tf.nn.relu(tf.add(convd,convd_b)) #[batch_size,self.title_len,self.init_dim,out_channels]
            convd_pool_1=tf.nn.max_pool(convd_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") # size=[batch_size,title_len/2,init_din/2,20]
            # 第二层 cnn
            convd_w_2=tf.Variable(tf.random_uniform(shape=(4,2,20,32),minval=-0.1,maxval=0.1),dtype=tf.float32)
            convd_b_2=tf.Variable(tf.random_uniform(shape=(32,),dtype=tf.float32))
            convd_2=tf.nn.conv2d(convd_pool_1,convd_w_2,strides=[1,1,1,1],padding="SAME")
            convd_2=tf.nn.relu(tf.add(convd_2,convd_b_2))
            convd_out=tf.nn.max_pool(convd_2,ksize=[1,2,1,1],strides=[1,2,1,1],padding="SAME")
            outs=tf.reshape(convd_out,[-1,100,32])
            outs=tf.transpose(outs,[0,2,1]) #[batch_size,32,100]
            encoder_state_c=tf.reduce_mean(outs,axis=1)
            encoder_state_h=tf.reduce_mean(outs,axis=1)

        elif FLAGS.encoder_mod=="lstmTF":
            encoder_inputs=tf.unstack(encoder_orgion,encoder_len,1)
            attention_states, encoder_state=embedding_encoder(
                encoder_inputs=encoder_inputs,
                              cell=cell,
                              num_encoder_symbols=encoder_vocab_size,
                              embedding_size=init_dim)
            outs=attention_states
            encoder_state_c=encoder_state
            encoder_state_h=[]
        else:
            _logger.error("please input correct encoder_mod!!")

        return outs,encoder_state_c,encoder_state_h

    def __decoder__(self,encoder_outs,encoder_state_c,encoder_state_h,**kwargs):
        '''
        
        :param encoder_outs: 
        :param encoder_state_c: 
        :param encoder_state_h: 
        :return: 
        '''
        cell=kwargs['cell']
        decoder_inputs_emb=kwargs['decoder_inputs_emb']
        decoder_len=kwargs['decoder_len']
        decoder_inputs=kwargs['decoder_inputs']
        decoder_vocab_size=kwargs['decoder_vocab_size']
        init_dim=kwargs['init_dim']
        if FLAGS.encoder_mod in ['lstm','bilstm','cnn']:
            decoder_list = tf.unstack(decoder_inputs_emb, decoder_len, 1)
            encoder_state = (encoder_state_c, encoder_state_h)
            decoder_out, decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(
                decoder_inputs=decoder_list,
                initial_state=encoder_state,
                attention_states=encoder_outs,
                cell=cell,
                output_size=None,
            )
        else:
            decoder_inputs=tf.unstack(decoder_inputs,decoder_len,1)
            decoder_out, decoder_state=embedding_attention_decoder(
                decoder_inputs=decoder_inputs,
                initial_state=encoder_state_c,
                attention_states=encoder_outs,
                cell=cell,
                num_symbols=decoder_vocab_size,
                embedding_size=init_dim,
                num_heads=1,
                output_size=None,
                output_projection=None,
                feed_previous=False,
                initial_state_attention=False)
        decoder_out=tf.stack(decoder_out,1)
        return decoder_out,decoder_state

    def __loss__(self,decoder_outs,label,**kwargs):
        '''
        
        :param decoder_outs: 
        :param label_outs: 
        :param kwargs: 
        :return: 
        '''
        num_class=kwargs["num_class"]
        hidden_dim=kwargs['hidden_dim']
        decoder_vocab_size=kwargs['decoder_vocab_size']
        decoder_len=kwargs['decoder_len']
        logit_list = tf.unstack(decoder_outs, decoder_len, 1)
        if FLAGS.encoder_mod == "bilstm":
            logits = decoder_outs
            ll = tf.einsum('ijk,kl->ijl', logits, tf.transpose(self.decoder_w))
            softmax_logit = tf.nn.softmax(tf.add(ll, self.decoder_b), 1)
            if not FLAGS.sample_loss:
                label_one_hot = tf.one_hot(label, decoder_vocab_size, 1, 0, 2)
                loss = tf.losses.softmax_cross_entropy(logits=softmax_logit, onehot_labels=label_one_hot)
            else:
                labels = tf.unstack(label, decoder_len, 1)
                losses = []
                for logit, label in zip(logit_list, labels):
                    label = tf.reshape(label, (-1, 1))
                    loss = tf.nn.sampled_softmax_loss(weights=self.decoder_w,
                                                      biases=self.decoder_b,
                                                      labels=label,
                                                      inputs=logit,
                                                      num_sampled=1000,
                                                      num_classes=num_class)
                    losses.append(loss)
                losses = tf.stack(losses)
                loss = tf.reduce_mean(losses)
        else:

            logits = tf.stack(logit_list, 1)
            ll = tf.einsum('ijk,kl->ijl', logits, tf.transpose(self.decoder_w))
            softmax_logit = tf.add(ll, self.decoder_b)
            if not FLAGS.sample_loss:
                label_one_hot = tf.one_hot(label, decoder_vocab_size, 1, 0, 2)
                loss = tf.losses.softmax_cross_entropy(logits=softmax_logit, onehot_labels=label_one_hot)
            else:
                def sampled_loss_func(labels, inputs):
                    labels = tf.reshape(labels, [-1, 1])
                    return tf.nn.sampled_softmax_loss(
                        weights=w, biases=b, labels=labels, inputs=inputs,
                        num_sampled=FLAGS.num_samples, num_classes=decoder_vocab_size)
                logits = logit_list
                labels = tf.unstack(label, decoder_len, 1)
                # loss_weights = tf.unstack(self.loss_weight, self.content_len, 1)
                loss = tf.contrib.legacy_seq2seq.sequence_loss(
                    logits=logits,
                    targets=labels,
                    weights=None,
                    average_across_timesteps=True,
                    average_across_batch=True,
                    softmax_loss_function=sampled_loss_func,
                    name=None)
        return loss,softmax_logit

    def __train_ops__(self):
        '''
        
        :return: 
        '''
        # self.Encoder_Decoder()
        kwargs_encoder={"hidden_dim":self.hidden_dim,
                        "encoder_sequence_length":self.title_seq_vec,
                        "cell":self.cell,
                        "init_dim":self.init_dim,
                        "encoder_orgion":self.title,
                        "encoder_vocab_size":self.title_vocab_len}
        encoder_outs, encoder_state_c, encoder_state_h = self.__encoder__(self.title_emb,self.title_len,**kwargs_encoder)

        kwargs_decoder={"cell":self.cell,
                        "decoder_inputs_emb":self.content_emb_input,
                        "decoder_len":self.content_len,
                        "decoder_inputs":self.content_input,
                        "decoder_vocab_size":self.content_vocab_len,
                        "init_dim":self.init_dim
                        }
        decoder_out, decoder_state = self.__decoder__(encoder_outs, encoder_state_c, encoder_state_h,**kwargs_decoder)

        loss_kwargs={"num_class":self.num_class,
                     "hidden_dim":self.hidden_dim,
                     "decoder_vocab_size":self.content_vocab_len,
                     "decoder_len":self.content_len}
        self.loss,_ = self.__loss__(decoder_out, self.content_decoder,**loss_kwargs)

        tf.summary.scalar("loss_my", self.loss)
        self.opt = tf.train.AdamOptimizer(0.8).minimize(self.loss)
        self.merge_summary = tf.summary.merge_all()

    def __decoder_ops__(self):
        '''
        构建 束搜索解码 图
        :return: 
        '''
        kwargs_encoder = {"hidden_dim": self.hidden_dim,
                          "encoder_sequence_length": self.title_seq_vec,
                          "cell": self.cell,
                          "init_dim": self.init_dim,
                          "encoder_orgion": self.title,
                          "encoder_vocab_size": self.title_vocab_len}
        self.encoder_outs, self.encoder_state_c, self.encoder_state_h = self.__encoder__(self.title_emb, self.title_len,
                                                                          **kwargs_encoder)

        self.beam_state_c = tf.placeholder(shape=(None, self.hidden_dim), dtype=tf.float32)
        self.beam_state_h = tf.placeholder(shape=(None, self.hidden_dim), dtype=tf.float32)
        self.beam_inputs = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.beam_encoder = tf.placeholder(shape=(None, self.title_len, self.hidden_dim), dtype=tf.float32)
        emb_beam_decoder = tf.nn.embedding_lookup(self.embeding_content, self.beam_inputs)
        # with tf.variable_scope('decoder_para',reuse=True):
        #
        #     decoder_w = tf.get_variable(name='decoder_w', shape=(self.num_class, self.hidden_dim))
        #     decoder_b = tf.get_variable(name='decoder_b',shape=(self.num_class,))
        beam_state = (self.beam_state_c, self.beam_state_h)
        # with tf.get_variable_scope().reuse_variables():
        outs, states = attention_decoder(
            decoder_inputs=[emb_beam_decoder],
            initial_state=beam_state,
            attention_states=self.beam_encoder,
            cell=self.cell,
        )
        outs = tf.stack(outs, 1)
        ll = tf.einsum('ijk,kl->ijl', outs, tf.transpose(self.decoder_w))
        beam_softmax = tf.nn.softmax(tf.add(ll, self.decoder_b))
        self.state = states
        self.beam_softmax = beam_softmax

    def Train(self,*args):
        '''
        
        :param args: 
        :return: 
        '''
        dd=args[0]
        self.__train_ops__()
        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False)
        id2content = dd.id2content
        saver = tf.train.Saver()
        summary_write = tf.summary.FileWriter(FLAGS.summary_write_dir)
        train_flag = 'train'
        with tf.Session(config=config) as sess:
            if os.path.exists(FLAGS.model_dir):
                _logger.info("load model from %s" % FLAGS.model_dir)
                saver.restore(sess, FLAGS.model_dir)
            else:
                sess.run(tf.global_variables_initializer())
            for i in range(FLAGS.epoch):
                return_content_input, return_content_decoder, return_title, return_content_len, return_title_len, loss_weight = dd.next_batch()

                if train_flag == 'train':
                    su_merge, loss, _ = sess.run([self.merge_summary, self.loss, self.opt], feed_dict={
                        self.content_input: return_content_input,
                        self.content_decoder: return_content_decoder,
                        self.title: return_title,
                        self.content_seq_vec: return_content_len,
                        self.title_seq_vec: return_title_len
                    })

                    summary_write.add_summary(su_merge, i)
                    if loss < self.init_loss:
                        self.init_loss = loss
                        self.best_iter = i
                        self.best_loss = self.init_loss
                        saver.save(sess, FLAGS.model_dir)
                        _logger.info(
                            "这是第%s次训练,误差为%s save best_loss:%s best_iter:%s" % (i, loss, self.best_loss, self.best_iter))

                    else:
                        _logger.info(
                            "这是第%s次训练,误差为%s best_loss:%s best_iter:%s" % (i, loss, self.best_loss, self.best_iter))

                elif train_flag == 'test':
                    loss, soft_logit = sess.run([self.loss, self.softmax_logit], feed_dict={
                        self.content_input: return_content_input,
                        self.content_decoder: return_content_decoder,
                        self.title: return_title,
                        self.content_seq_vec: return_content_len,
                        self.title_seq_vec: return_title_len,
                    })
                    _logger.info("loss is %s" % loss)
                    softs = np.argmax(soft_logit, 2)
                    for soft, content in zip(softs, return_content_decoder):
                        pre = "".join(list(map(lambda x: id2content[int(x)], soft))).replace("NONE", "")
                        content = "".join(list(map(lambda x: id2content[int(x)], content))).replace('NONE', "")
                        print(pre, "---", content)
                        print('\n')

    def Beam_Decoder(self,*args):
        '''
        束搜索 解码
        :param args: 
        :return: 
        '''

        self.__decoder_ops__()
        dd=args[0]
        self.id2content = dd.id2content
        self.beam_size = FLAGS.beam_size
        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=True)

        saver = tf.train.Saver()
        self.mod = "beam_decoder"
        with tf.Session(config=config) as sess:
            saver.restore(sess, FLAGS.model_dir)
            for _ in range(1):
                return_content_input, return_content_decoder, return_title, return_content_len, return_title_len, loss_weight = dd.next_batch()

                for batch_index in range(return_title.shape[0]):
                    self.socre = [0.0] * self.beam_size
                    self.beam_path = [[1]] * self.beam_size

                    title_input = [return_title[batch_index]] * self.beam_size
                    title_seq_input = [return_title_len[batch_index]] * self.beam_size
                    title_input = np.array(title_input)
                    title_seq_input = np.array(title_seq_input)

                    init_encoder_out, init_encoder_state_c, init_encoder_state_h = sess.run(
                        [self.encoder_outs, self.encoder_state_c, self.encoder_state_h],
                        feed_dict={
                            self.title: title_input,
                            self.title_seq_vec: title_seq_input
                        })
                    if FLAGS.encoder_mod=="lstmTF":
                        stateTuple=init_encoder_state_c[:]
                        init_encoder_state_c=stateTuple[0]
                        init_encoder_state_h=stateTuple[1]
                    init_beam_input = np.ones(shape=(self.beam_size,), dtype=np.int32)
                    beam_inputs = [init_beam_input]
                    beam_state_c = [init_encoder_state_c]
                    beam_state_h = [init_encoder_state_h]

                    for j in range(return_content_input.shape[1]):
                        state_, beam_softmax_ = sess.run([self.state, self.beam_softmax],
                                                         feed_dict={self.beam_state_c: beam_state_c[-1],
                                                                    self.beam_state_h: beam_state_h[-1],
                                                                    self.beam_inputs: beam_inputs[-1],
                                                                    self.beam_encoder: init_encoder_out})
                        if j == 0:

                            path, socre, next_input = self.__array_convert__(beam_softmax_, self.socre, self.beam_path,
                                                                           self.beam_size, beam_flag="Beg")
                            self.socre = socre
                            self.beam_path = path
                        else:
                            path, socre, next_input = self.__array_convert__(beam_softmax_, self.socre, self.beam_path,
                                                                           self.beam_size, beam_flag="Med")
                            self.socre = socre
                            self.beam_path = path

                        beam_state_c.append(state_[0])
                        beam_state_h.append(state_[1])
                        beam_inputs.append(next_input)

                    print(self.socre)
                    print(self.beam_path)
                    print(return_content_decoder[batch_index])
                    print('*' * 10)

    def __encoder_decoder__(self):
        '''
        编码+解码
        :return: 
        '''

        encoder_inputs=tf.unstack(self.title,self.title_len,1)
        decoder_inputs=tf.unstack(self.content_input,self.content_len,1)
        out_w = tf.Variable(tf.random_uniform(shape=(self.hidden_dim, self.content_vocab_len), maxval=1.0, minval=-1.0),
                            dtype=tf.float32)
        out_b = tf.Variable(tf.random_uniform(shape=(self.content_vocab_len,)), dtype=tf.float32)

        outs,state=embedding_attention_seq2seq(
            encoder_inputs,
            decoder_inputs,
            self.cell,
            num_encoder_symbols=self.title_vocab_len,
            num_decoder_symbols=self.content_vocab_len,
            embedding_size=self.hidden_dim,
            output_projection=(out_w,out_b),
            feed_previous=False,
            dtype=tf.float32)
        self.loss=self.Loss(outs,self.content_decoder)
        self.opt=tf.train.AdamOptimizer(0.9).minimize(self.loss)

    def __array_convert__(self,beam_data,score,beam_path,beam_size,beam_flag="Med"):
        '''
        beam_data 矩阵转换函数 取最大的前5个概率
        :param beam_data: 
        :return: 
        '''
        beam_data=np.reshape(beam_data,(beam_data.shape[0],beam_data.shape[2]))
        allsocre=score
        # print("allsocre", allsocre)
        # print("beampath", beam_path)
        res=[]
        if beam_flag=="Beg":

            for index,ele in enumerate(beam_data):
                if index < beam_size:
                    eles=[[i,e] for i,e in enumerate(ele)]
                    eles.sort(key=lambda x:x[1],reverse=True)
                    res.append(eles[index][0])
                    path_=beam_path[index][:]
                    path_.append(eles[index][0])
                    beam_path[index]=path_
                    socre_=allsocre[index]
                    socre_+=eles[index][1]
                    allsocre[index]=socre_
            next_input=np.array([ e[-1] for e in beam_path])
            return beam_path,allsocre,next_input

        else:
            ss_score=[] # 5X5的分数
            ss_path=[] # 5X5的路径
            for index,ele in enumerate(beam_data):
                eles = [[i, e] for i, e in enumerate(ele)]
                eles.sort(key=lambda x: x[1], reverse=True)
                # eles=eles[:beam_size] #前一个解码后取beam_size个最大概率的解码
                eles = [eles[0]]
                socre_=[float(e[1])+allsocre[index] for e in eles]
                path_=[e[0] for e in eles]

                for e in path_:
                    ss=beam_path[index][:]
                    ss.append(e)
                    ss_path.append(ss)
                ss_score.extend(socre_)
            all_res=[[socre,path] for socre,path in zip(ss_score,ss_path)] # 全部的解码输出
            all_res.sort(key=lambda x:x[0],reverse=True)
            all_res=all_res[:beam_size] # 总共取beam_size个解码输出
            path=[e[1] for e in all_res]
            socre=[e[0] for e in all_res]
            next_input=np.array([ e[-1] for e in path])
            return path,socre,next_input

    def __show_result__(self,logit,label):
        for soft, content in zip(logit, label):
            pre = "".join(list(map(lambda x: self.id2content[int(x)], soft))).replace("NONE", "")
            content = "".join(list(map(lambda x: self.id2content[int(x)], content))).replace('NONE', "")
            print(pre, "---", content)
            print('\n')


def main(_):
    # 本模型为目的是构建问答系统 因此问句为encoder 答案为decoder
    _logger.info("训练/验证/测试 数据预处理.....")
    dd = Data_deal.DataDealSeq(train_path=FLAGS.train_dir, test_path=FLAGS.test_dir,
                               dev_path=FLAGS.dev_dir,
                               dim=FLAGS.embedding_dim,
                               batch_size=FLAGS.batch_size,
                               content_len=FLAGS.decoder_len,
                               title_len=FLAGS.encoder_len,
                               flag="train_new")
    content_vocab_size, title_vocab_size = dd.get_vocab_size()
    _logger.info("数据处理完毕")
    _logger.info("参数列表\n"
                 "train_dir:%s\nbatch_size:%s\nembedding_dim:%s\nEncoder_len:%s\n"
                 "Decoder_len:%s\nencoder_vocab_size:%s\ndecoder_vocab_size:%s\nhidden_dim%s "
                 % (FLAGS.train_dir, FLAGS.batch_size, FLAGS.embedding_dim, FLAGS.encoder_len,
                    FLAGS.decoder_len, title_vocab_size, content_vocab_size, FLAGS.hidden_dim))
    _logger.info('*' * 50 + "构建模型" + '*' * 50)
    model = Seq2Seq(hidden_dim=FLAGS.hidden_dim, init_dim=FLAGS.embedding_dim, content_len=FLAGS.decoder_len,
                    title_len=FLAGS.encoder_len,
                    con_vocab_len=content_vocab_size, ti_vocab_len=title_vocab_size, batch_size=FLAGS.batch_size)

    if FLAGS.mod=='train':
        for i in range(FLAGS.train_num):
            _logger.info("开始第%s轮训练"%i)

            model.Train(dd)
    elif FLAGS.mod=='predict':
        _logger.info("预测")
        sentence=" "
        model.Beam_Decoder(dd,sentence)

if __name__ == '__main__':
    tf.app.run()




