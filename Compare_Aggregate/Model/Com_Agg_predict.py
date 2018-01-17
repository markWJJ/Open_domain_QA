import sys
sys.path.append("../")
from R_Net_QA.Data import Data_deal
# from Data import Data_deal
import Com_Agg_v

dd=Data_deal.DataDeal(train_path="../Data/WikiQA-train.txt",test_path="../Data/WikiQA-test.txt",dev_path="../Data/WikiQA-dev.txt",
                dim=100,batch_size=64,Q_len=10,A_len=50,flag="test")

model=Com_Agg_v.CA(init_dim=100,hidden_dim=100,Q_len=10,A_len=50)
while True:
    Q_sentence=input("输入问句：")
    A_sentence=input("输入候选答案：")

    Q_array=dd.get_Q_array(Q_sentence)
    A_array=dd.get_A_array(A_sentence)

    Q_array=Q_array.reshape((1,Q_array.shape[0],Q_array.shape[1]))
    A_array=A_array.reshape((1,A_array.shape[0],A_array.shape[1]))
    print(Q_array.shape,A_array.shape)
    model.test(Q_array,A_array)
