from text_classifier import TextClassifier
import pandas as pd
import openai
from openai import OpenAI
from together import Together
from common import categories,generate_prompt,generate_test_prompt,generate_test_prompt_with_examples
from rank_bm25 import BM25Okapi
import traceback
from tqdm import tqdm

SUPPORT_APIS = ["openai","together"]
SUPPORT_API_MODELS_DICT = {"openai":["gpt-4o-mini","gpt-4o","gpt-4-turbo","gpt-3.5-turbo"],
                           "together":["meta-llama/Llama-3-8b-chat-hf",
                                       "meta-llama/Llama-3-70b-chat-hf",
                                       "mistralai/Mixtral-8x22B-Instruct-v0.1",
                                       "Qwen/Qwen2.5-72B-Instruct-Turbo"]}

def best_examples(query, bm25, examples, k):
    tokenized_query = query.split(" ")
    top_machings = bm25.get_top_n(tokenized_query, examples, k)
    return top_machings

class ApiTextClassifier(TextClassifier):
    def __init__(self, api_name, model_name, api_key,prompt_way):
        super().__init__()
        self.api_name = api_name
        self.model_name = model_name
        self.api_key = api_key
        self.prompt_way = prompt_way
        if self.api_name == "openai":
            self.client = OpenAI(api_key=self.api_key)
            # openai.api_key = api_key
            # self.client = None
        else:
            self.client = Together(api_key=self.api_key)

    def train(self,train_data,val_data,text_col_name,label_col_name,save_name):
        self.text_col_name = text_col_name
        self.label_col_name = label_col_name
        self.train_text_labels = train_data[[text_col_name,label_col_name]]
        self.train_text_labels_0 = train_data[train_data[label_col_name]==categories[0]]
        self.train_text_labels_1 = train_data[train_data[label_col_name]==categories[1]]
        tokenized_corpus_0 = [doc.split(" ") for doc in self.train_text_labels_0[text_col_name].tolist()]
        tokenized_corpus_1 = [doc.split(" ") for doc in self.train_text_labels_1[text_col_name].tolist()]
        self.bm25_0 = BM25Okapi(tokenized_corpus_0)
        self.bm25_1 = BM25Okapi(tokenized_corpus_1)

    def test(self,data,text_col_name):
        for index, row in tqdm(data.iterrows()):
            try:
                text = row[text_col_name]
                if self.prompt_way == "zeroshot":
                    prompt = generate_test_prompt(text)
                else:
                    k = 2
                    example_text_labels_0 = best_examples(text,self.bm25_0,self.train_text_labels_0[self.text_col_name].tolist(),k)
                    example_text_labels_1 = best_examples(text,self.bm25_1,self.train_text_labels_1[self.text_col_name].tolist(),k)
                    example_df_0 = self.train_text_labels_0[self.train_text_labels[self.text_col_name].isin(example_text_labels_0)]
                    example_df_1 = self.train_text_labels_1[self.train_text_labels[self.text_col_name].isin(example_text_labels_1)]
                    example_df = pd.concat([example_df_0,example_df_1])
                    examples = []
                    for i, row in example_df.iterrows():
                        examples.append({"text":row[self.text_col_name],"label":row[self.label_col_name]})
                    prompt = generate_test_prompt_with_examples(text,examples)
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                if self.api_name == "openai":
                    answer = completion.choices[0].message.content
                else:
                    answer = completion.choices[0].message.content
                final_answer = categories[0]
                for category in categories:
                    if category.lower() in answer.lower():
                        final_answer = category
                data.at[index, "predict_answer"] = final_answer
            except:
                traceback.print_exc()
                break
        return data

    def predict(self,data):
        if self.prompt_way == "zeroshot":
            prompt = generate_test_prompt(data)
        else:
            k = 2
            example_text_labels_0 = best_examples(data, self.bm25_0,
                                                  self.train_text_labels_0[self.text_col_name].tolist(), k)
            example_text_labels_1 = best_examples(data, self.bm25_1,
                                                  self.train_text_labels_1[self.text_col_name].tolist(), k)
            example_df_0 = self.train_text_labels_0[
                self.train_text_labels[self.text_col_name].isin(example_text_labels_0)]
            example_df_1 = self.train_text_labels_1[
                self.train_text_labels[self.text_col_name].isin(example_text_labels_1)]
            example_df = pd.concat([example_df_0, example_df_1])
            examples = []
            for i, row in example_df.iterrows():
                examples.append({"text": row[self.text_col_name], "label": row[self.label_col_name]})
            prompt = generate_test_prompt_with_examples(data, examples)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        if self.api_name == "openai":
            answer = completion.choices[0].message.content
        else:
            answer = completion.choices[0].message.content
        final_answer = categories[0]
        for category in categories:
            if category.lower() in answer.lower():
                final_answer = category
        return final_answer



