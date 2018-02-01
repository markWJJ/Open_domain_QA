from R_Net_QA.Data_r_net import Data_deal_no_array
import R_Net_no_array

init_dim = 100
batch_size = 1
Q_len = 30
P_len = 100
hidden_dim = 100

dd= Data_deal_no_array.DataDealRNet(train_path="/SQUQA_train.txt", test_path="/test1.txt",
                               dev_path="/dev1.txt",
                               dim=init_dim, batch_size=batch_size, Q_len=Q_len, P_len=P_len, flag="train")
ca=R_Net_no_array.R_Net(init_dim=init_dim,hidden_dim=hidden_dim,Q_len=Q_len,P_len=P_len,batch_size=batch_size,vocab_size=dd.get_vocab_size())
while True:
    sentence=input("输入：")
    Q_sentence=sentence.split("\t\t")[0]
    A_sentence=sentence.split("\t\t")[1]

    Q_array,Q_len=dd.get_Q_array(Q_sentence)
    A_array,A_len=dd.get_A_array(A_sentence)

    print(Q_array.shape,Q_len.shape)
    print(A_array.shape,A_len.shape)
    Q_array=Q_array.reshape((1,Q_array.shape[0]))
    A_array=A_array.reshape((1,A_array.shape[0]))
    print(Q_array.shape,A_array.shape)
    print(Q_len,A_len)
    ca.test(Q_array,A_array,Q_len,A_len)
