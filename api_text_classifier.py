from text_classifier import TextClassifier
from openai import OpenAI
from together import Together

SUPPORT_APIS = ["openai","together"]
SUPPORT_API_MODELS_DICT = {"openai":["gpt-4o-mini","gpt-4o","gpt-4-turbo","gpt-3.5-turbo"],
                           "together":["meta-llama/Llama-3-8b-chat-hf",
                                       "meta-llama/Llama-3-70b-chat-hf",
                                       "mistralai/Mixtral-8x22B-Instruct-v0.1",
                                       "Qwen/Qwen2.5-72B-Instruct-Turbo"]}
PROMPT_TEMPLATE = """"""
class ApiTextClassifier(TextClassifier):
    def __init__(self,api_name,model_name,api_key):
        self.api_name = api_name
        self.model_name = model_name
        self.api_key = api_key
        if self.api_name == "openai":
            self.client = OpenAI(self.api_key)
        else:
            self.client = Together(api_key=self.api_key)
    def train(self,data,text_col_name,label_col_name):
        self.data = data

    def test(self,data,text_col_name,label_col_name):
        pass

    def predict(self,data,text_col_name):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Write a haiku about recursion in programming."
                }
            ]
        )
        if self.api_name=="openai":
            return completion.choices[0].message
        else:
            return completion.choices[0].delta.content



