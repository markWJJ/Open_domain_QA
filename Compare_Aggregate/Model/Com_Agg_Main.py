import tensorflow as tf
import numpy as np
from Compare_Aggregate.Data import Data_deal
import logging
import os
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("com_agg")
Path=os.path.split(os.path.realpath(__file__))[0]
Path_up=os.path.split(Path)[0]
class Config(object):
    '''
    默认配置
    '''
    learning_rate=0.01
    num_class=2
    batch_size=168
    Q_len=10
    A_len=50
    embedding_dim=50
    hidden_dim=100
    train_dir=Path_up+'/Data/output.txt'
    dev_dir='/dev.txt'
    test_dir='/test.txt'
    model_dir=Path_up+'/save_model/com_agg.ckpt'
    train_num=50
    use_cpu_num=8
    summary_write_dir=Path_up+"/tmp/com_agg.log"
    epoch=100
    beam_size=168

config=Config()
tf.app.flags.DEFINE_float("learning_rate", config.learning_rate, "学习率")
tf.app.flags.DEFINE_integer("num_class", config.num_class, "采样损失函数的采样的样本数")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "批处理的样本数量")
tf.app.flags.DEFINE_integer("Q_len", config.Q_len, "问句长度")
tf.app.flags.DEFINE_integer("A_len", config.A_len, "答案长度")
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
tf.app.flags.DEFINE_boolean("sample_loss", False, "是否采用采样的损失函数") # true for prediction
tf.app.flags.DEFINE_string("mod", "train", "默认为训练") # true for prediction
FLAGS = tf.app.flags.FLAGS

class CA(object):
    '''
    compare_aggregate模型
    '''
    def __init__(self,init_dim,hidden_dim,Q_len,A_len,num_class,vocab_size):
        self.init_dim=init_dim
        self.Q_len=Q_len
        self.A_len=A_len
        self.num_class=num_class
        self.vocab_size=vocab_size
        # 论文中的L维
        self.hidden_dim=hidden_dim
        self.Q=tf.placeholder(dtype=tf.int32,shape=(None,self.Q_len))
        self.A=tf.placeholder(dtype=tf.int32,shape=(None,self.A_len))
        self.label=tf.placeholder(dtype=tf.int32,shape=(None,))
        self.embedding=tf.Variable(tf.random_normal(shape=(self.vocab_size,self.init_dim),dtype=tf.float32))
        self.Q=tf.nn.embedding_lookup(self.embedding,self.Q)
        self.A=tf.nn.embedding_lookup(self.embedding,self.A)
        self.Q_ = self.process(flag="N", x=self.Q, scope="Q_process",reuse=False) # 使用lstm进行预处理
        self.A_ = self.process(flag="N", x=self.A, scope="A_process",reuse=False)
        #注意力矩阵
        self.H=self.attention(self.Q_,self.A_,scope="attention",reuse=False)
        # 合并A_和H
        self.T=self.compare(self.A_,self.H)

        #经lstm进行整合
        self.out=self.Lstm(self.T)

        #经softmax进行分类，分为两类：0 不匹配 1匹配
        softmax_w=tf.Variable(tf.random_normal((self.hidden_dim,2),dtype=tf.float32))
        softmax_b=tf.Variable(tf.random_normal((2,),dtype=tf.float32))

        logit=tf.add(tf.matmul(self.out,softmax_w),softmax_b)

        self.softmax_out=tf.nn.softmax(logit)
        self.loss=tf.losses.sparse_softmax_cross_entropy(self.label,self.softmax_out)

        self.train_op=tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

    def process(self,flag,x,scope,reuse):
        '''
        preocess layer
        :return: 
        '''
        with tf.variable_scope(name_or_scope=scope,reuse=reuse):
            if flag=="LSTM":
                if scope=="Q_process":
                    self.lstm_input = tf.unstack(x, self.Q, 1)
                elif type=="A_process":
                    self.lstm_input=tf.unstack(x,self.A,1)

                cell = tf.contrib.rnn.LSTMCell(
                    self.hidden_dim,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                    state_is_tuple=False)
                outputs, states = tf.contrib.rnn.static_rnn(cell, self.lstm_input, dtype=tf.float32)
                outputs=tf.concat(outputs,1)
                return outputs
            else:
                w_i=tf.Variable(tf.random_normal(shape=(self.init_dim,self.hidden_dim)))
                w_u=tf.Variable(tf.random_normal(shape=(self.init_dim,self.hidden_dim)))
                b_i=tf.Variable(tf.random_normal(shape=(self.hidden_dim,)))
                b_u=tf.Variable(tf.random_normal(shape=(self.hidden_dim,)))

                outputs=tf.sigmoid(tf.add(tf.einsum("ijk,kl->ijl",x,w_i),b_i))*tf.tanh(tf.add(tf.einsum("ijk,kl->ijl",x,w_u),b_u))
                return outputs

    def attention(self,Q_,A_,scope,reuse):
        '''
        标准softmax注意力机制模块
        :param Q_: 
        :param A_: 
        :return: 
        '''
        with tf.variable_scope(name_or_scope=scope,reuse=reuse):
            w_g = tf.Variable(tf.random_normal(shape=(self.hidden_dim, self.hidden_dim)))
            b_g = tf.Variable(tf.random_normal(shape=(self.hidden_dim,)))
            logit=tf.add(tf.einsum("ijk,kl->ijl",Q_,w_g),b_g)#(None,Q,l)
            A_T=tf.transpose(A_,perm=[0,2,1])#(None,l,A)
            logit_=tf.einsum("ijk,ikl->ijl",logit,A_T)
            G=tf.nn.softmax(logit_)#(None,Q,A)
            G_T=tf.transpose(G,perm=[0,2,1])
            H=tf.einsum("ikj,ijl->ikl",G_T,Q_)#(None,A,l)
            return H

    def compare(self,A_,H):
        '''
        合并A_和注意力矩阵H,采用论文中的S UB M ULT +NN方式合并
        :param A_: 
        :param H: 
        :return: 
        '''
        flag="submult_nn"
        if flag=="submult_nn":
            w = tf.Variable(tf.random_normal(shape=(2*self.hidden_dim, self.hidden_dim)))
            b = tf.Variable(tf.random_normal(shape=(self.hidden_dim,)))
            sub=(A_-H)*(A_-H)
            mult=A_*H
            con=tf.concat((sub,mult),2)#(None,A,2l)
            logit_1=tf.einsum("ijk,kl->ijl",con,w)
            logit=tf.add(logit_1,b)
            T=tf.nn.relu(logit)
            return T
        elif flag=="concat":
            T=tf.concat((A_,H),2)
            return T

    def Lstm(self,T):
        '''
        lstm层，将T组合为句向量进行分类
        :param T: 
        :return: 
        '''
        lstm_input = tf.unstack(T, self.A_len, 1)
        cell=tf.contrib.rnn.LSTMCell(self.hidden_dim,
                                     initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                     state_is_tuple=False)
        outputs,states=tf.contrib.rnn.static_rnn(cell,lstm_input,dtype=tf.float32)
        return outputs[-1]

    def Train(self,dd):
        def accuracy(predictions, labels):
            predict=np.argmax(predictions, 1)
            length=sum(1 for i,j in  zip(predict.flatten(),labels.flatten()) if int(i)==1 or int(j)==1)
            ss=sum([ 1 for i,j in zip(predict.flatten(),labels.flatten()) if int(i)==int(j) and int(i)!=0])
            if float(length)==0.0:
                return 0.0
            else:
                return 100.0*(float(ss)/float(length))

        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False)
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            if os.path.exists(FLAGS.model_dir):
                _logger.info("load model from ",FLAGS.model_dir)
                saver.restore(sess,FLAGS)
            else:
                _logger.info("inital paramter")
                sess.run(tf.global_variables_initializer())
            ini_acc=0.0
            for i in range(FLAGS.epoch):
                Q,A,label=dd.next_batch()
                train_softmax_out,train_loss,_=sess.run([self.softmax_out,self.loss,self.train_op],feed_dict={self.Q:Q,
                                                                                                              self.A:A,
                                                                                                             self.label:label})

                train_acc = accuracy(train_softmax_out, label)
                _logger.info("第%s次迭代"%i)
                if train_acc>ini_acc:
                    ini_acc=train_acc
                    saver.save(sess,FLAGS.model_dir)
                    _logger.info("训练误差%s ,训练准确率%s, save model, best_acc:%s"%(train_loss,train_acc,ini_acc))
                else:
                    _logger.info("训练误差%s ,训练准确率%s"%(train_loss,train_acc))


    def predict(self,Q,A):
        saver=tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess=sess,save_path="./model.ckpt")
            softmax_out= sess.run([self.softmax_out],feed_dict={self.Q: Q, self.A: A})
            print (softmax_out)

def main(_):
    print(FLAGS.model_dir)
    _logger.info("读取数据.....")
    dd = Data_deal.DataDeal(train_path="../Data/output.txt", test_path="../Data/WikiQA-test.txt",
                            dev_path="../Data/WikiQA-dev.txt",
                            dim=FLAGS.embedding_dim, batch_size=FLAGS.batch_size, Q_len=FLAGS.Q_len, A_len=FLAGS.A_len, flag="train_new")
    vocab_size=dd.vocab_size
    _logger.info("数据读取完毕")
    _logger.info("图模型构建.....")
    ca = CA(init_dim=FLAGS.embedding_dim, hidden_dim=FLAGS.hidden_dim,
            Q_len=FLAGS.Q_len, A_len=FLAGS.A_len,num_class=FLAGS.num_class,vocab_size=vocab_size)
    _logger.info('模型构建完毕')
    if FLAGS.mod=="train":
        _logger.info("开始训练模型....")
        ca.Train(dd)
    elif FLAGS.mod=='predict':
        _logger.info("开始模型预测")
    else:
        _logger.error("请选择正确的模式 train 或者 predict")
if __name__ == '__main__':
    tf.app.run()




