import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
def split_dataset(data:pd.DataFrame,label_name:str):
    train_val_data, test_data = train_test_split(data, test_size=0.2, stratify=data[label_name])
    train_data, val_data = train_test_split(train_val_data, test_size=0.125, stratify=train_val_data[label_name])
    return train_data,val_data,test_data
def main():
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

if __name__ == "__main__":
    main()