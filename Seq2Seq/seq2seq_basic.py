'''
seq2seq 模型基类
'''
from abc import ABCMeta,abstractmethod

class Seq2SeqBasic(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def __init_ops__(self):
        '''
        设定变量和占位输入
        :return: 
        '''
        pass


    @abstractmethod
    def __encoder__(self,encoder_inputs,encoder_len,**kwargs):
        '''
        编码层
        :param encoder_inputs: 编码输入 type=tensor
        :param encoder_seq_len: 编码句子程度 type=tensor
        :param kwargs: 
        :return: 
        '''
        pass

    @abstractmethod
    def __decoder__(self,encoder_outs,encoder_state_c,encoder_state_h,**kwargs):
        pass

    @abstractmethod
    def __loss__(self,decoder_outs,label_outs,**kwargs):
        pass

    @abstractmethod
    def __train_ops__(self):
        '''
        构建训练图
        :return: 
        '''
        pass

    @abstractmethod
    def __decoder_ops__(self):
        '''
        构建 解码 图
        :return: 
        '''
        pass

    @abstractmethod
    def Train(self,*args):
        '''
        模型训练接口
        :param args: 
        :return: 
        '''
        pass


    @abstractmethod
    def Beam_Decoder(self,*args):
        '''
        模型预测接口
        :param args: 
        :return: 
        '''
        pass