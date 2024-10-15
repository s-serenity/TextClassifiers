from text_classifier import TextClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class FeatureTextClassifier(TextClassifier):
    def __int__(self,feature_transformer_name,classifier_name):
        self.feature_transformer_name = feature_transformer_name
        if feature_transformer_name=="tfidf":
            self.feature_transformer = TfidfVectorizer
        self.classifier_name = classifier_name
        if classifier_name=="LR":
            self.classifier = LogisticRegression()

    def train(self,data,text_col_name,label_col_name):
        x = self.feature_transformer.fit_transform(data[text_col_name]).toarray()
        y = data[label_col_name].toarray()
        self.classifier.fit(x, y)
        return self.classifier

    def test(self,data,text_col_name):
        x = self.feature_transformer.fit_transform(data[text_col_name]).toarray()
        return self.classifier.predict(x)

    def predict(self,data):
        x = self.feature_transformer.fit_transform(data).toarray()
        return self.classifier.predict(x)




