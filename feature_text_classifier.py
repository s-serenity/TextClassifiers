from text_classifier import TextClassifier
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
from TweetNormalizer import normalizeTweet
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import pickle
import joblib

class FeatureTextClassifier(TextClassifier):
    def __init__(self, feature_transformer_name, classifier_name, plm_name):
        super().__init__()
        self.feature_transformer_name = feature_transformer_name
        self.classifier_name = classifier_name
        self.plm_name = plm_name
        if self.feature_transformer_name == "tfidf":
            self.feature_transformer = TfidfVectorizer()
        if self.feature_transformer_name == "plm":
            self.model = AutoModel.from_pretrained(plm_name,torch_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(plm_name)
        if self.classifier_name == "LR":
            self.classifier = LogisticRegression()
        if self.classifier_name == "RF":
            self.classifier = RandomForestClassifier(n_estimators=100)
        if self.classifier_name == "catboost":
            self.classifier = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.05)
        if self.classifier_name == "lightgbm":
            self.classifier = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31,verbosity=1)

    def load_model(self,load_name):
        if ".pkl" in load_name:
            with open(load_name, 'rb') as f:
                self.classifier = pickle.load(f)

        if ".cbm" in load_name:
            self.classifier = CatBoostClassifier().load_model(load_name)

        if ".txt" in load_name:
            self.classifier = lgb.Booster(model_file=load_name)

    def train(self,train_data,val_data,text_col_name,label_col_name,save_name):
        train_x = None
        val_x = None
        if self.feature_transformer_name == "tfidf":
            train_x = self.feature_transformer.fit_transform(train_data[text_col_name]).toarray()
            joblib.dump(self.feature_transformer, 'tfidf_vectorizer.joblib')
            val_x = self.feature_transformer.transform(val_data[text_col_name]).toarray()

        if self.feature_transformer_name == "plm":
            train_data[text_col_name] = train_data[text_col_name].map(lambda x: normalizeTweet(x))
            val_data[text_col_name] = val_data[text_col_name].map(lambda x: normalizeTweet(x))
            train_inputs = self.tokenizer(train_data[text_col_name].tolist(), padding='max_length',max_length=47, truncation=True, return_tensors="pt")
            val_inputs = self.tokenizer(val_data[text_col_name].tolist(), padding='max_length', max_length=47,truncation=True, return_tensors="pt")
            with torch.no_grad():
                train_x = self.model(train_inputs["input_ids"])
                val_x = self.model(val_inputs["input_ids"])

        train_y = train_data[label_col_name]
        val_y = val_data[label_col_name]
        if train_x is not None:
            if self.classifier_name in ["catboost","lightgbm"]:
                if self.classifier_name == "catboost":
                    self.classifier.fit(train_x, train_y, eval_set=[(val_x, val_y)], early_stopping_rounds=10,
                                        verbose=True)

                    self.classifier.save_model(save_name)
                if self.classifier_name == "lightgbm":
                    self.classifier.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(stopping_rounds=10)])

                    self.classifier.booster_.save_model(save_name)
            else:
                self.classifier.fit(train_x, train_y)
                with open(save_name, 'wb') as f:
                    pickle.dump(self.classifier, f)

    def test(self,data,text_col_name):
        self.feature_transformer = joblib.load('tfidf_vectorizer.joblib')
        test_x = None
        if self.feature_transformer_name == "tfidf":
            test_x = self.feature_transformer.transform(data[text_col_name]).toarray()
        if self.feature_transformer_name == "plm":
            data[text_col_name] = data[text_col_name].map(lambda x:normalizeTweet(x))
            test_inputs = self.tokenizer(data[text_col_name].tolist(), padding='max_length',max_length=47, truncation=True, return_tensors="pt")
            with torch.no_grad():
                test_x = self.model(test_inputs["input_ids"])
        predict_y = self.classifier.predict(test_x)
        data["predict_answer"] = predict_y
        return data

    def predict(self,data):
        if self.feature_transformer_name == "tfidf":
            test_x = self.feature_transformer.fit_transform(data).toarray()
        if self.feature_transformer_name == "plm":
            test_inputs = self.tokenizer(data, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                test_x = self.model(test_inputs["input_ids"])
        predict_y = self.classifier.predict(test_x)
        return predict_y




