from abc import abstractmethod

class TextClassifier(object):
    def __init__(self):
        pass

    @abstractmethod
    def load_model(self,load_name):
        pass

    @abstractmethod
    def train(self,train_data,val_data,text_col_name,label_col_name,save_name):
        pass

    @abstractmethod
    def test(self,data,text_col_name):
        pass

    @abstractmethod
    def predict(self,data):
        pass

