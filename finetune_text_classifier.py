from text_classifier import TextClassifier
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
class FinetuneTextClassifier(TextClassifier):
    def __int__(self,base_model_name,finetune_way):
        self.base_model_name = base_model_name
        self.finetune_way = finetune_way
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name, quantization_config=self.bnb_config, device_map="auto"
        )

    def train(self,data,text_col_name,label_col_name):
        pass


