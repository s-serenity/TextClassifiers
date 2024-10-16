from text_classifier import TextClassifier
from openai import OpenAI
from together import Together
from common import categories,generate_prompt,generate_test_prompt
import traceback

SUPPORT_APIS = ["openai","together"]
SUPPORT_API_MODELS_DICT = {"openai":["gpt-4o-mini","gpt-4o","gpt-4-turbo","gpt-3.5-turbo"],
                           "together":["meta-llama/Llama-3-8b-chat-hf",
                                       "meta-llama/Llama-3-70b-chat-hf",
                                       "mistralai/Mixtral-8x22B-Instruct-v0.1",
                                       "Qwen/Qwen2.5-72B-Instruct-Turbo"]}

class ApiTextClassifier(TextClassifier):
    def __init__(self, api_name, model_name, api_key):
        super().__init__()
        self.api_name = api_name
        self.model_name = model_name
        self.api_key = api_key
        if self.api_name == "openai":
            self.client = OpenAI(self.api_key)
        else:
            self.client = Together(api_key=self.api_key)

    def train(self,train_data,val_data,text_col_name,label_col_name,save_name):
        self.train_data = train_data

    def test(self,data,text_col_name):
        for index, row in data.iterrows():
            try:
                text = row[text_col_name]
                prompt = generate_test_prompt(text)
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
                    answer = completion.choices[0].message
                else:
                    answer = completion.choices[0].delta.content
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
        prompt = generate_test_prompt(data)
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
            answer = completion.choices[0].message
        else:
            answer = completion.choices[0].delta.content
        final_answer = categories[0]
        for category in categories:
            if category.lower() in answer.lower():
                final_answer = category
        return final_answer



