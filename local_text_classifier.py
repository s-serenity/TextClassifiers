from text_classifier import TextClassifier
import numpy as np
from skorch.llm import ZeroShotClassifier,FewShotClassifier
from common import categories

class LocalTextClassifier(TextClassifier):
    def __int__(self,model_name,device,prompt_way):
        self.model_name = model_name
        self.prompt_way = prompt_way
        if prompt_way=="zeroshot":
            self.model = ZeroShotClassifier(self.model_name, device=device, use_caching=False)
        else:
            self.model = FewShotClassifier(model=self.model_name, tokenizer=self.model_name, max_samples=5, use_caching=False)

    def train(self, train_data, val_data, text_col_name, label_col_name, save_name):
        # train_labels = np.array(['positive','negative'])[train_data[label_col_name]]
        if self.prompt_way == "zeroshot":
            self.model.fit(X=None, y=categories)
        else:
            self.model.fit(train_data[text_col_name], train_data[label_col_name])

    def test(self,data,text_col_name):
        y_pred = self.model.predict(data[text_col_name])
        return y_pred

    def predict(self,data):
        y_pred = self.model.predict(data)
        return y_pred

