import tensorflow as tf
from Data import Data_deal
import numpy as np


class LSTM(object):


    def __init__(self, init_dim, hidden_dim, Q_len, A_len,num_class):
        self.init_dim = init_dim
        self.Q_len = Q_len
        self.A_len = A_len
        self.num_class=num_class
        # 论文中的L维
        self.hidden_dim = hidden_dim
        self.Q = tf.placeholder(dtype=tf.float32, shape=(None, self.Q_len, self.init_dim))
        self.Q_seq_vec=tf.placeholder(dtype=tf.int32, shape=(None,))
        self.A = tf.placeholder(dtype=tf.float32, shape=(None, self.A_len, self.init_dim))
        self.A_seq_vec=tf.placeholder(dtype=tf.int32, shape=(None,))

        self.label = tf.placeholder(dtype=tf.int32, shape=(None,))

        out_Q=self.Bi_lstm_Q(self.Q,self.Q_len,self.Q_seq_vec)#[batch_size,2*hidden_dim]
        out_A=self.Bi_lstm_A(self.A,self.A_len,self.A_seq_vec)#[batch_size,2*hidden_dim]

        out=tf.concat((out_Q,out_A),1)

        softmax_w=tf.Variable(tf.random_uniform((4*self.hidden_dim,self.num_class)))
        softmax_b=tf.Variable(tf.random_uniform((self.num_class,)))

        self.softmax_out=tf.add(tf.matmul(out,softmax_w),softmax_b)
        #label=tf.reshape(self.label,[-1,1])
        label_hot=tf.one_hot(self.label,self.num_class,1,0)

        self.loss_op=tf.losses.softmax_cross_entropy(label_hot,self.softmax_out)
        self.train_op=tf.train.AdadeltaOptimizer(0.5).minimize(self.loss_op)

    def Bi_lstm_Q(self,lstm_input,seq_len,seq_vec):
        with tf.variable_scope("Bi_lstm_Q",reuse=False):
            lstm_input=tf.transpose(lstm_input,[1,0,2])
            lstm_input=tf.unstack(lstm_input,seq_len,0)
            cell_f=tf.contrib.rnn.LSTMCell(self.hidden_dim,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                    state_is_tuple=False)

            cell_b=tf.contrib.rnn.LSTMCell(self.hidden_dim,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                    state_is_tuple=False)

            (outputs,_,_)=tf.contrib.rnn.static_bidirectional_rnn(cell_f,cell_b,lstm_input,
                                                                  dtype=tf.float32,
                                                                  sequence_length=seq_vec)
            return outputs[-1]

    def Bi_lstm_A(self,lstm_input,seq_len,seq_vec):
        with tf.variable_scope("Bi_lstm_A", reuse=False):
            lstm_input=tf.transpose(lstm_input,[1,0,2])
            lstm_input=tf.unstack(lstm_input,seq_len,0)
            cell_fA=tf.contrib.rnn.LSTMCell(self.hidden_dim,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                    state_is_tuple=False)

            cell_bA=tf.contrib.rnn.LSTMCell(self.hidden_dim,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                    state_is_tuple=False)

            (outputs,_,_)=tf.contrib.rnn.static_bidirectional_rnn(cell_fA,cell_bA,lstm_input,
                                                                  dtype=tf.float32,
                                                                  sequence_length=seq_vec)
            return outputs[-1]

    def train(self,dd):
        def accuracy(predictions, labels):
            predict=np.argmax(predictions, 1)
            length=sum(1 for i,j in  zip(predict.flatten(),labels.flatten()) if int(i)==1 or int(j)==1)
            ss=sum([ 1 for i,j in zip(predict.flatten(),labels.flatten()) if int(i)==int(j) and int(i)!=0])

            #return 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]
            if float(length)==0.0:
                return 0.0
            else:
                return 100.0*(float(ss)/float(length))

        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False)
        #dev_Q,dev_A,dev_label=dd.get_dev()
        #test_Q,test_A,test_label=dd.get_test()
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            #saver.restore(sess,"./model.ckpt")
            sess.run(tf.global_variables_initializer())
            ini_acc=0.0
            for i in range(10000):
                Q,A,label,Q_seq,A_seq=dd.nex_batch_()

                train_softmax_out,train_loss,_=sess.run([self.softmax_out,self.loss_op,self.train_op],feed_dict={self.Q:Q,
                                                                                                              self.A:A,
                                                                                                              self.label:label,
                                                                                                                self.Q_seq_vec:Q_seq,
                                                                                                                 self.A_seq_vec:A_seq})

                '''
                dev_softmax_out, dev_loss = sess.run([self.softmax_out, self.loss],
                                                 feed_dict={self.Q: dev_Q, self.A: dev_A, self.label: dev_label})
                test_softmax_out, test_loss  = sess.run([self.softmax_out, self.loss],
                                                 feed_dict={self.Q: test_Q, self.A: test_A, self.label: test_label})
                test_acc=accuracy(test_softmax_out,test_label)
                                    dev_acc=accuracy(dev_softmax_out,dev_label)
                                    print("验证误差%s ,验证准确率%s"%(dev_loss,dev_acc))
                print("测试误差%s ,测试准确率%s"%(test_loss,test_acc))
                '''

                train_acc = accuracy(train_softmax_out, label)
                print("第%s次迭代"%i)

                print("误差",train_loss)
                print("Acc",train_acc)
                if train_acc>ini_acc:
                    ini_acc=train_acc
                    saver.save(sess,"./model.ckpt")

                    print("训练误差%s ,训练准确率%s"%(train_loss,train_acc))

                    print("\n")

if __name__ == '__main__':

    init_dim=100
    batch_size=128
    Q_len=10
    A_len=50
    hidden_dim=100
    dd=Data_deal.DataDeal(train_path="../Data/output.txt",test_path="../Data/WikiQA-test.txt",dev_path="../Data/WikiQA-dev.txt",
                          dim=init_dim,batch_size=batch_size,Q_len=Q_len,A_len=A_len,flag="train_new")
    ca=LSTM(init_dim=init_dim,hidden_dim=hidden_dim,Q_len=Q_len,A_len=A_len,num_class=2)
    ca.train(dd)