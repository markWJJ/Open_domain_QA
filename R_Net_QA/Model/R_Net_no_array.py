import numpy as np
import tensorflow as tf
import sys
#sys.path.append("../Data_1/Data_deal_1")
from R_Net_QA.Data_r_net import Data_deal_no_array


class R_Net(object):
    '''
    R_net 智能问答模型
    '''
    def __init__(self,init_dim,hidden_dim,Q_len,P_len,batch_size,vocab_size):
        self.init_dim=init_dim
        self.Q_len=Q_len
        self.P_len=P_len
        self.num_class=5
        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size
        self.vocab=tf.Variable(tf.random_uniform(shape=(self.vocab_size,self.init_dim)))
        self.Q=tf.placeholder(dtype=tf.int32,shape=(None,self.Q_len))
        self.P=tf.placeholder(dtype=tf.int32,shape=(None,self.P_len))

        self.Q_array=tf.nn.embedding_lookup(self.vocab,self.Q)
        self.P_array=tf.nn.embedding_lookup(self.vocab,self.P)

        #print(self.Q_array)
        #print(self.P_array)

        self.Q_seq_vec=tf.placeholder(dtype=tf.int32,shape=(None,))
        self.P_seq_vec=tf.placeholder(dtype=tf.int32,shape=(None,))
        self.out_num=2 #输出数量 为2 则输出为 起始位置和结束位置
        #self.batch_size=self.Q.get_shape().as_list()[0]
        self.batch_size=batch_size

        self.Q_,_ = self.process(input_=self.Q_array,seq_len=self.Q_len,seq_vec=self.Q_seq_vec,scope="Q_encoder")
        self.P_,_ = self.process(input_=self.P_array,seq_len=self.P_len,seq_vec=self.P_seq_vec,scope="P_encoder")

        #gated_attention
        self.P_=self.gated_attention(self.Q_,self.P_) #[None,P_len,hidden_dim]

        # self_match_attention
        self.H=self.self_match_attention(self.P_)
        mode="pointer_net"
        if mode=="pointer_net":
            self.soft_logits,self.ids=self.pointer_network(self.Q_,self.H)
        else:
            pass
        self.label = tf.placeholder(dtype=tf.int32, shape=(None, self.out_num))
        label_one_hot = tf.one_hot(self.label, self.P_len, 1, 0, 2)
        # logit=tf.reshape(self.soft_logits,(-1,self.out_num,self.P_len))
        logit=self.soft_logits
        self.loss_op=tf.losses.softmax_cross_entropy(logits=logit,onehot_labels=label_one_hot,weights=1.0)

        self.optimizer=tf.train.AdadeltaOptimizer(0.7).minimize(self.loss_op)

    def process(self,input_,seq_len,seq_vec,scope):
        '''
        preocess layer
        :return: 
        '''
        state=[]
        with tf.variable_scope(scope):
            lstm_input=tf.transpose(input_,[1,0,2])
            lstm_input = tf.unstack(lstm_input, seq_len, 0)
            cell_f = tf.contrib.rnn.LSTMCell(
                self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                state_is_tuple=False)

            cell_b = tf.contrib.rnn.LSTMCell(
                self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                state_is_tuple=False)
            #static_bidirectional_rnn
            (out,fw_state,_)=tf.contrib.rnn.static_bidirectional_rnn(cell_f,cell_b,lstm_input,dtype=tf.float32,
                                                               sequence_length=seq_vec)
            state.append(fw_state)
            out=tf.stack(out,0)
            out=tf.transpose(out,[1,0,2])
            return out,state[-1]

    def gated_attention(self,Q_,P_):
        '''
        gated attention模块
        :return: RNN的输出结果
        '''
        P_P = tf.transpose(P_, [1, 0, 2])
        P_list=tf.unstack(P_P,self.P_len,0)
        #init_c=tf.zeros(shape=(self.batch_size,self.hidden_dim))
        init_v=tf.zeros(shape=(self.batch_size,self.hidden_dim))
        #C=[init_c]

        lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim,
                                       initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                       state_is_tuple=True)
        V = [init_v]
        #C = [init_v]
        init_state=lstm_cell.zero_state(self.batch_size, tf.float32)
        #print("init_state",init_state)
        state=[init_state]
        with tf.variable_scope("gated_attention"):
            for t in range(self.P_len):
                if t>0:
                    tf.get_variable_scope().reuse_variables()
                w_g = tf.Variable(tf.random_normal(shape=(4 * self.hidden_dim, 4 * self.hidden_dim)))

                u_P_t=tf.reshape(P_list[t],[-1,1,2*self.hidden_dim])
                v_t_=tf.reshape(V[-1],[-1,1,self.hidden_dim])

                c_t=self.gated_attention_ops(Q_,u_P_t=u_P_t,v_t_=v_t_)#[None,2*hidden_dim]
                u_p_t_c_t=tf.concat((P_list[t],c_t),1)
                u_p_t_c_t_=tf.sigmoid(tf.matmul(u_p_t_c_t,w_g))*u_p_t_c_t
                #v
                (v_t,new_state)=lstm_cell(u_p_t_c_t_,state[-1])# v_t=[None,self.hidden_dim]

                V.append(v_t)
                state.append(new_state)
            out=V[1::]
            out=tf.stack(out)
            out=tf.transpose(out,[1,0,2])#[None,P_len,hidden_dim]
            return out

    def gated_attention_ops(self,u_Q,u_P_t,v_t_):
        '''
        u_Q=[None,Q_len,2*hidden_dim]
        u_P=[None,1,2*hidden_dim]
        v_t_=[None,1,hidden_dim]
        :param u_Q: 
        :param u_P: 
        :param v_t_: 
        :return: 
        '''
        with tf.variable_scope("atteion",reuse=True):
            w_u_Q = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, self.hidden_dim)))
            w_u_P = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, self.hidden_dim)))
            w_v_P = tf.Variable(tf.random_normal(shape=(self.hidden_dim, self.hidden_dim)))
            V = tf.Variable(tf.random_normal(shape=(self.hidden_dim,1)))

            QQ=tf.einsum("ijk,kl->ijl",u_Q,w_u_Q)
            PP=tf.einsum("ijk,kl->ijl",u_P_t,w_u_P)
            PP_V=tf.einsum("ijk,kl->ijl",v_t_,w_v_P)
            logit=tf.tanh(QQ+PP+PP_V)
            logit=tf.einsum("ijk,kl->ijl",logit,V)
            soft_logit=tf.nn.softmax(logit,1)

            c_t=tf.einsum("ijk,ijl->ikl",soft_logit,u_Q)#(None,2*hideen_dim,1)
            c_t=tf.reshape(c_t,[-1,2*self.hidden_dim])#[None,2*hidden_dim]

            return c_t

    def self_match_attention_ops(self,P_,P_t):
        '''
        P_[None,P_len,hidden_dim],P_t[None,1,hidden_dim]
        :param P_: 
        :param P_t: 
        :return: 
        '''
        with tf.variable_scope("self_attention", reuse=True):
            w_u_P = tf.Variable(tf.random_normal(shape=(self.hidden_dim, self.hidden_dim)))
            w_u_P_ = tf.Variable(tf.random_normal(shape=(self.hidden_dim, self.hidden_dim)))
            V = tf.Variable(tf.random_normal(shape=(self.hidden_dim, 1)))

            PP = tf.einsum("ijk,kl->ijl", P_, w_u_P)
            PP_ = tf.einsum("ijk,kl->ijl", P_t, w_u_P_)
            logit = tf.tanh(PP+PP_)
            logit = tf.einsum("ijk,kl->ijl", logit, V)
            soft_logit = tf.nn.softmax(logit, 1)

            c_t = tf.einsum("ijk,ijl->ikl", soft_logit, P_)  # (None,hideen_dim,1)
            c_t = tf.reshape(c_t, [-1, self.hidden_dim])  # [None,hidden_dim]
            return c_t

    def self_match_attention(self,P_):
        '''
        自匹配注意力机制 
        :param P_: 
        :return: 
        '''
        P_P = tf.transpose(P_, [1, 0, 2])
        P_list = tf.unstack(P_P, self.P_len, 0)
        init_h = tf.zeros(shape=(self.batch_size, self.hidden_dim))
        lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim,
                                            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                                            state_is_tuple=True)
        H = [init_h]
        init_state=(init_h,init_h)
        state=[init_state]
        w_g = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, 2 * self.hidden_dim)))
        with tf.variable_scope("self_attention"):
            for t in range(self.P_len):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                P_t = tf.reshape(P_list[t], [-1, 1 , self.hidden_dim])

                c_t = self.self_match_attention_ops(P_,P_t)  # [None,hidden_dim]
                u_p_t_c_t = tf.concat((P_list[t], c_t), 1)
                u_p_t_c_t_ = tf.sigmoid(tf.matmul(u_p_t_c_t, w_g)) * u_p_t_c_t
                h_t, new_state = lstm_cell(u_p_t_c_t_, state[-1])  # v_t=[None,self.hidden_dim]
                H.append(h_t)
                state.append(new_state)
            out = H[1::]
            out = tf.stack(out)
            out = tf.transpose(out, [1, 0, 2])  # [None,P_len,hidden_dim]
            return out

    def pre_Q_attention(self,Q_):
        '''
        answer的pointer netwrok的init_state
        :param Q_: 
        :return: 
        '''

        Q_list=tf.unstack(Q_,self.Q_len,1)
        w_Q_1 = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, self.hidden_dim)))
        r_Q_1=tf.matmul(Q_list[-1],w_Q_1)
        # r_Q_1=Q_list[-1]
        with tf.variable_scope("pre_init_Q"):
            V_r_Q =  tf.zeros_like(tf.reshape(Q_list[0],(-1,1,2*self.hidden_dim)))
            w_u_Q = tf.Variable(tf.random_normal(shape=(2*self.hidden_dim, self.hidden_dim)))
            w_v_Q = tf.Variable(tf.random_normal(shape=(2*self.hidden_dim, self.hidden_dim)))
            V = tf.Variable(tf.random_normal(shape=(self.hidden_dim, 1)))

            u_Q_Q = tf.einsum("ijk,kl->ijl", Q_, w_u_Q)  # None,Q_len,hidden_dim
            v_Q_Q = tf.einsum("ijk,kl->ijl", V_r_Q, w_v_Q) #None,1,hidden_dim
            logit = tf.tanh(u_Q_Q + v_Q_Q) #None,Q_len,hidden_dim
            logit = tf.einsum("ijk,kl->ijl", logit, V)
            soft_logit = tf.nn.softmax(logit, 1)

            r_Q = tf.einsum("ijk,ijl->ikl", soft_logit, Q_)  # (None,2*hideen_dim,1)
            r_Q = tf.reshape(r_Q, [-1, 2*self.hidden_dim])  # [None,2*hidden_dim]
            w_Q = tf.Variable(tf.random_normal(shape=(2*self.hidden_dim, self.hidden_dim)))
            r_Q=tf.matmul(r_Q,w_Q)
            return r_Q,r_Q_1

    def pointer_attention(self,H,h_a_t_):
        '''
        H passage的输入 None,len_P,hidden_dim  h_a_t:None,hidden
        :param H: 
        :param h_a_t_: 
        :return: 
        '''
        H_list=tf.unstack(H,self.P_len,1)
        h_a_t_=tf.reshape(h_a_t_,(-1,1,self.hidden_dim))
        self.HHH=h_a_t_
        with tf.variable_scope("pointer_net_attention"):
            w_h_p = tf.Variable(tf.random_normal(shape=(self.hidden_dim, self.hidden_dim)))
            w_a_p = tf.Variable(tf.random_normal(shape=(self.hidden_dim, self.hidden_dim)))
            V = tf.Variable(tf.random_normal(shape=(self.hidden_dim, 1)))

            h_P_P=tf.einsum("ijk,kl->ijl", H, w_h_p)  # None,P_len,hidden_dim
            h_a_a=tf.einsum("ijk,kl->ijl", h_a_t_, w_a_p)  # None,1,hidden_dim
            logit = tf.tanh(h_P_P + h_a_a) #None,P_len,hidden_dim
            logit = tf.einsum("ijk,kl->ijl", logit, V)
            soft_logit = tf.nn.softmax(logit, 1) # None,P_len,1
            soft_logit_ = tf.reshape(soft_logit,(-1,self.P_len))
            id=tf.argmax(soft_logit_,1) # (None,)
            id_flatten=tf.reshape(id,(-1,1))
            H_flatten=tf.reshape(H,(-1,self.hidden_dim))
            ss=tf.gather(H_flatten,id_flatten)
            # print(ss)
            c_t=tf.einsum("ijk,ijl->ilk",soft_logit,H)#[None,hidden_dim,1]
            c_t=tf.reshape(c_t,(-1,self.hidden_dim))
            #c_t=tf.reshape(ss,(-1,self.hidden_dim))
            return soft_logit_,id,c_t

    def pointer_network(self,Q_,H):
        '''
        指针网络 pointer_network
        :return: 
        '''
        with tf.variable_scope("pointer_net"):
            init_c_state,init_input=self.pre_Q_attention(Q_) # [None,hidden_dim] [None,2*hidden_dim]
            # init_c_state = tf.transpose(H, [1, 0, 2])[-1]
            # cell=tf.contrib.rnn.LSTMCell(self.hidden_dim,
            #                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
            #                             state_is_tuple=True)

            cell=tf.contrib.rnn.BasicRNNCell(self.hidden_dim)
            # init_input=tf.zeros(shape=(self.batch_size,self.hidden_dim))
            # init_input=tf.transpose(H,[1,0,2])[-1]

            # init_input = tf.transpose(Q_, [1, 0, 2])[-1]
            # init_h_state=init_input
            # init_state=(init_c_state,init_h_state)
            # state=[init_state]
            state=[init_c_state]
            H_a=[init_input]
            soft_logits=[]
            ids=[]

            for t in range(self.out_num):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                soft_logit,id,c_t=self.pointer_attention(H,H_a[-1])#soft_logit [None,P_len] id [None]
                (h_t,new_state)=cell(c_t,state[-1])
                state.append(new_state)
                H_a.append(c_t)
                soft_logits.append(soft_logit)
                ids.append(id)
            soft_logits=tf.stack(soft_logits,1) #[None,pre_num,P_len]
            ids=tf.stack(ids,1)   #[batch_size,pre_num]
            return soft_logits,ids

    def compare(self,A_,H):
        '''
        合并A_和注意力矩阵H,采用论文中的S UB M ULT +NN方式合并
        :param A_: 
        :param H: 
        :return: 
        '''
        flag="concat"
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

    def Bi_Lstm(self,T,seq_len,seq_vec):
        '''
        lstm层，将T组合为句向量进行分类
        :param T: 
        :return: 
        '''
        with tf.variable_scope("out_lstm",reuse=False):
            lstm_input=tf.transpose(T,[1,0,2])
            lstm_input = tf.unstack(lstm_input, seq_len, 0)
            cell_f = tf.contrib.rnn.LSTMCell(
                self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                state_is_tuple=False)

            cell_b = tf.contrib.rnn.LSTMCell(
                self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                state_is_tuple=False)
            #static_bidirectional_rnn
            (out,_,_)=tf.contrib.rnn.static_bidirectional_rnn(cell_f,cell_b,lstm_input,dtype=tf.float32,
                                                               sequence_length=seq_vec)

            out=tf.stack(out,0)
            out=tf.transpose(out,[1,0,2])
            return out

    def train(self,dd):
        def accuracy(predictions, labels):
            predict=np.argmax(predictions, 2)
            print(predict.shape)
            length=sum(1 for i,j in  zip(predict.flatten(),labels.flatten()) if int(i)!=0 or int(j)!=0)
            ss=sum([ 1 for i,j in zip(predict.flatten(),labels.flatten()) if int(i)==int(j) and int(i)!=0])
            if float(length)==0.0:
                return 0.0
            else:
                return 100.0*(float(ss)/float(length))

        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False)
        # dev_Q,dev_A,dev_label=dd.get_dev()
        # test_Q,test_A,test_label=dd.get_test()
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            saver.restore(sess,"./R_Net_Model.ckpt")
            # sess.run(tf.global_variables_initializer())
            ini_acc=0.0
            init_loss=999.99
            for i in range(10000):
                Q,A,label,Q_len,A_len=dd.next_batch()
                Q_seq_vec=np.array(Q_len)
                A_seq_vec=np.array(A_len)
                HHH,train_loss,_,ids=sess.run([self.HHH,self.loss_op,self.optimizer,self.ids],feed_dict={self.Q:Q,
                                              self.P:A,
                                              self.Q_seq_vec:Q_seq_vec,
                                              self.P_seq_vec:A_seq_vec,
                                              self.label:label
                                              })
                print("迭代次数%s"%i)
                print("训练误差：%s ,START_END_ID:%s"%(train_loss,ids))
                print("\n")
                if train_loss<init_loss:
                    init_loss=train_loss
                    saver.save(sess,"./R_Net_Model.ckpt")
                    print("save_model")



                # dev_softmax_out, dev_loss = sess.run([self.softmax_out, self.loss],
                #                                  feed_dict={self.Q: dev_Q, self.A: dev_A, self.label: dev_label})
                # test_softmax_out, test_loss  = sess.run([self.softmax_out, self.loss],
                #                                  feed_dict={self.Q: test_Q, self.A: test_A, self.label: test_label})
                # test_acc=accuracy(test_softmax_out,test_label)
                #                     dev_acc=accuracy(dev_softmax_out,dev_label)
                #                     print("验证误差%s ,验证准确率%s"%(dev_loss,dev_acc))
                # print("测试误差%s ,测试准确率%s"%(test_loss,test_acc))

    def test(self,Q,P,Q_len,P_len):
        def accuracy(predictions, labels):
            predict=np.argmax(predictions, 2)
            print(predict.shape)
            length=sum(1 for i,j in  zip(predict.flatten(),labels.flatten()) if int(i)!=0 or int(j)!=0)
            ss=sum([ 1 for i,j in zip(predict.flatten(),labels.flatten()) if int(i)==int(j) and int(i)!=0])
            if float(length)==0.0:
                return 0.0
            else:
                return 100.0*(float(ss)/float(length))
        Q_len=np.array(Q_len)
        P_len=np.array(P_len)
        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False)
        # dev_Q,dev_A,dev_label=dd.get_dev()
        # test_Q,test_A,test_label=dd.get_test()
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            saver.restore(sess,"./R_Net_Model.ckpt")
            # sess.run(tf.global_variables_initializer())
            ini_acc=0.0
            init_loss=999.99

            ids=sess.run(self.ids,feed_dict={self.Q:Q,
                                              self.P:P,
                                              self.Q_seq_vec:Q_len,
                                              self.P_seq_vec:P_len,
                                              })
            print(ids)





if __name__ == '__main__':

    init_dim=100
    batch_size=16
    Q_len=30
    P_len=100
    hidden_dim=100
    dd= Data_deal_no_array.DataDealRNet(train_path="/SQUQA_train.txt", test_path="/test1.txt",
                               dev_path="/dev1.txt",
                               dim=init_dim, batch_size=batch_size, Q_len=Q_len, P_len=P_len, flag="train")
    ca=R_Net(init_dim=init_dim,hidden_dim=hidden_dim,Q_len=Q_len,P_len=P_len,batch_size=batch_size,vocab_size=dd.get_vocab_size())
    ca.train(dd)


