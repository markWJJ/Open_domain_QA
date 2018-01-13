from abc import ABCMeta,abstractmethod



class basicExtract(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def extract(self,sentence,**kwargs):
        '''
        
        :param traceid: 
        :param sentence: 
        :param kwargs: 
        :return: 
        '''
        pass

