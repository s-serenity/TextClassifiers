import os
import pandas as pd
from common import categories, OPENAI_KEY, TOGETHER_KEY
# import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
# from feature_text_classifier import FeatureTextClassifier
# from api_text_classifier import ApiTextClassifier
from finetune_text_classifier import FinetuneTextClassifier

def split_dataset(data:pd.DataFrame,label_name:str):
    train_val_data, test_data = train_test_split(data, test_size=0.2, stratify=data[label_name])
    train_data, val_data = train_test_split(train_val_data, test_size=0.125, stratify=train_val_data[label_name])
    return train_data,val_data,test_data

def main():
    model_path = "./models/"
    test_path = "./tests/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
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
        matches = val_data["tweet"].isin(train_data["tweet"])
        matching_count = matches.sum()
        print(len(val_data))
        print(matching_count)
        matches = test_data["tweet"].isin(train_data["tweet"])
        matching_count = matches.sum()
        print(len(test_data))
        print(matching_count)
        # print(len(train_data))
    classifier_types = ["api","finetune","feature"]
    # classifier_types = ["api"]
    for classifier_type in classifier_types:
        if classifier_type == "finetune":
            finetune_ways = ["classifier","llm-sft"]
            base_models = ["distilbert/distilbert-base-uncased","NousResearch/Llama-2-7b-chat-hf"]
            for base_model,finetune_way in zip(base_models,finetune_ways):
                if "/" in base_model:
                    save_model_name = base_model.split("/")[-1]
                else:
                    save_model_name = base_model
                if finetune_way=="llm-sft":
                    train_data["label"] = train_data["label"].map(lambda x: categories[int(x)])
                    val_data["label"] = val_data["label"].map(lambda x:categories[int(x)])
                    test_data["label"] = test_data["label"].map(lambda x:categories[int(x)])
                finetune_classifier = FinetuneTextClassifier(base_model,finetune_way)
                model_name = save_model_name + "_" + finetune_way
                save_name = model_path+model_name
                test_name = save_model_name + "_" + finetune_way + ".csv"
                test_save_name = test_path + test_name
                if not os.path.exists(test_save_name):
                    finetune_classifier.train(train_data,val_data,"tweet","label",save_name)
                    if not os.path.exists(save_name):
                        finetune_classifier.train(train_data,val_data,"tweet","label",save_name)
                        test_results = finetune_classifier.test(test_data,"tweet")
                    else:
                        finetune_classifier.load_model(save_name)
                        test_results = finetune_classifier.test(test_data, "tweet")
        # if classifier_type == "feature":
        #     feature_test_lists = ["tfidf","plm"]
        #     plm_name = "vinai/bertweet-base"
        #     classifier_test_lists = ["LR","RF","catboost","lightgbm"]
        #     for feature in feature_test_lists:
        #         for classifier in classifier_test_lists:
        #             extension = ""
        #             if classifier in ["LR","RF"]:
        #                 extension = ".pkl"
        #             if classifier == "catboost":
        #                 extension = ".cbm"
        #             if classifier == "lightgbm":
        #                 extension = ".txt"
        #             model_name = feature+"_"+classifier+extension
        #             save_name = model_path+model_name
        #             test_name = feature + "_" + classifier +".csv"
        #             test_save_name = test_path + test_name
        #             if not os.path.exists(test_save_name):
        #                 feature_classifier = FeatureTextClassifier(feature,classifier,plm_name)
        #                 if not os.path.exists(save_name):
        #                     feature_classifier.train(train_data,val_data,"tweet","label",save_name)
        #                     test_results = feature_classifier.test(test_data,"tweet")
        #                 else:
        #                     feature_classifier.load_model(save_name)
        #                     test_results = feature_classifier.test(test_data, "tweet")
        #                 test_results.to_csv(test_save_name, index=False, encoding="utf-8-sig")
        #             else:
        #                 test_results = pd.read_csv(test_save_name)
        #             class_report = classification_report(y_true=test_results["label"], y_pred=test_results["predict_answer"], target_names=categories,
        #                                                  labels=list(range(len(categories))))
        #             print('\nClassification Report for %s ,%s and %s:'%(feature,classifier,plm_name))
        #             print(class_report)
        #
        #             # Generate confusion matrix
        #             conf_matrix = confusion_matrix(y_true=test_results["label"], y_pred=test_results["predict_answer"], labels=list(range(len(categories))))
        #             print('\nConfusion Matrix for %s ,%s and %s:'%(feature,classifier,plm_name))
        #             print(conf_matrix)
        # if classifier_type == "api":
        #     api_names = ["openai","together"]
        #     model_names = ["gpt-4o-mini","mistralai/Mixtral-8x22B-Instruct-v0.1"]
        #     prompt_ways = ["fewshot","zeroshot"]
        #     train_data["label"] = train_data["label"].map(lambda x:categories[int(x)])
        #     val_data["label"] = val_data["label"].map(lambda x:categories[int(x)])
        #     test_data["label"] = test_data["label"].map(lambda x:categories[int(x)])
        #     for api_name,model_name in zip(api_names,model_names):
        #         for prompt_way in prompt_ways:
        #             if "/" in model_name:
        #                 save_model_name = model_name.split("/")[-1]
        #             else:
        #                 save_model_name = model_name
        #             test_name = api_name + "_" + save_model_name + "_"+prompt_way+".csv"
        #             test_save_name = test_path + test_name
        #             if api_name == "openai":
        #                 key = OPENAI_KEY
        #             else:
        #                 key = TOGETHER_KEY
        #             if not os.path.exists(test_save_name):
        #                 api_classifier = ApiTextClassifier(api_name,model_name,key,prompt_way)
        #                 api_classifier.train(train_data,val_data,"tweet","label",save_name=None)
        #                 test_results = api_classifier.test(test_data,"tweet")
        #                 test_results.to_csv(test_save_name,index=False, encoding="utf-8-sig")
        #             else:
        #                 test_results = pd.read_csv(test_save_name)
        #             class_report = classification_report(y_true=test_results["label"],
        #                                                  y_pred=test_results["predict_answer"], target_names=categories,
        #                                                  labels=list(range(len(categories))))
        #             print('\nClassification Report for %s ,%s and %s:' % (api_name, model_name, prompt_way))
        #             print(class_report)
        #
        #             # Generate confusion matrix
        #             conf_matrix = confusion_matrix(y_true=test_results["label"], y_pred=test_results["predict_answer"],
        #                                            labels=list(range(len(categories))))
        #             print('\nConfusion Matrix for %s ,%s and %s:' % (api_name, model_name, prompt_way))
        #             print(conf_matrix)


if __name__ == "__main__":
    main()