import os
import pandas as pd
from common import categories
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from feature_text_classifier import FeatureTextClassifier
from api_text_classifier import ApiTextClassifier
from finetune_text_classifier import FinetuneTextClassifier

def split_dataset(data:pd.DataFrame,label_name:str):
    train_val_data, test_data = train_test_split(data, test_size=0.2, stratify=data[label_name])
    train_data, val_data = train_test_split(train_val_data, test_size=0.125, stratify=train_val_data[label_name])
    return train_data,val_data,test_data

def main():
    model_path = "./models/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    file_path = "./data/twitter_hate/train_E6oV3lV.csv"
    train_file_path = file_path[:-4]+"_train.csv"
    val_file_path = file_path[:-4]+"_val.csv"
    test_file_path = file_path[:-4]+"_test.csv"
    data = pd.read_csv(file_path,encoding="utf-8-sig")
    if not os.path.exists(train_file_path):
        train_data, val_data, test_data = split_dataset(data, "label")
        train_data.to_csv(train_file_path,index=False, encoding="utf-8-sig")
        val_data.to_csv(val_file_path,index=False, encoding="utf-8-sig")
        test_data.to_csv(test_file_path,index=False, encoding="utf-8-sig")
    else:
        train_data = pd.read_csv(train_file_path)
        val_data = pd.read_csv(val_file_path)
        test_data = pd.read_csv(test_file_path)
        print(len(train_data))
    feature_test_lists = ["tfidf","plm"]
    plm_name = "vinai/bertweet-base"
    classifier_test_lists = ["RL","RF","catboost","lightgbm"]
    for feature in feature_test_lists:
        for classifier in classifier_test_lists:
            extension = ""
            if classifier in ["RL","RF"]:
                extension = ".pkl"
            if classifier == "catboost":
                extension = ".cbm"
            if classifier == "lightgbm":
                extension = ".txt"
            model_name = feature+"_"+classifier+extension
            save_name = model_path+model_name
            feature_classifier = FeatureTextClassifier(feature,classifier,plm_name)
            if not os.path.exists(save_name):
                feature_classifier.train(train_data,val_data,"text","label",save_name)
                test_results = feature_classifier.test(test_data,"text")
            else:
                feature_classifier.load_model(save_name)
                test_results = feature_classifier.test(test_data, "text")
            class_report = classification_report(y_true=test_results["label"], y_pred=test_results["predict_answer"], target_names=categories,
                                                 labels=list(range(len(categories))))
            print('\nClassification Report for %s ,%s and %s:'%(feature,classifier,plm_name))
            print(class_report)

            # Generate confusion matrix
            conf_matrix = confusion_matrix(y_true=test_results["label"], y_pred=test_results["predict_answer"], labels=list(range(len(categories))))
            print('\nConfusion Matrix for %s ,%s and %s:'%(feature,classifier,plm_name))
            print(conf_matrix)


if __name__ == "__main__":
    main()