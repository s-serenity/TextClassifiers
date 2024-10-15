from abc import abstractmethod


class TextClassifier(object):
    @abstractmethod
    def train(self,data,text_col_name,label_col_name):
        pass

    @abstractmethod
    def test(self,data,text_col_name):
        pass
    @abstractmethod
    def predict(self,data):
        pass

