from text_classifier import TextClassifier
from common import categories,generate_prompt,generate_test_prompt
import traceback
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import evaluate
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import evaluate
import numpy as np
import bitsandbytes as bnb
metric = evaluate.load("accuracy")

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def compute_metrics(eval_pred):
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   return metric.compute(predictions=predictions, references=labels)


class FinetuneTextClassifier(TextClassifier):
    def __init__(self,base_model_name,finetune_way):
        super().__init__()
        self.base_model_name = base_model_name
        self.finetune_way = finetune_way
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        if finetune_way == "classifier":
            id2label = {0: categories[0], 1: categories[1]}
            label2id = {categories[0]: 0, categories[1]: 1}
            self.model = AutoModelForSequenceClassification.from_pretrained(base_model_name,num_labels=2,
                                                                            id2label=id2label, label2id=label2id)

        if finetune_way == "llm-sft":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name, quantization_config=self.bnb_config, device_map="auto")
            self.modules = find_all_linear_names(self.model)

    def train(self,train_data,val_data,text_col_name,label_col_name,save_name):
        if self.finetune_way == "llm-sft":
            train_data[text_col_name] = train_data.apply(lambda x:generate_prompt(x[text_col_name],x[label_col_name]), axis=1)
            val_data[text_col_name] = val_data.apply(lambda x:generate_prompt(x[text_col_name],x[label_col_name]), axis=1)

            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0,
                r=64,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.modules,
            )

            training_arguments = TrainingArguments(
                output_dir=save_name,
                num_train_epochs=5,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                gradient_checkpointing=True,
                optim="paged_adamw_32bit",
                logging_steps=1,
                learning_rate=2e-4,
                weight_decay=0.001,
                fp16=True,
                bf16=False,
                max_grad_norm=0.3,
                max_steps=-1,
                warmup_ratio=0.03,
                group_by_length=False,
                lr_scheduler_type="cosine",
                eval_steps=0.2
            )

            trainer = SFTTrainer(
                model=self.model,
                args=training_arguments,
                train_dataset=train_data,
                eval_dataset=val_data,
                peft_config=peft_config,
                dataset_text_field="text",
                tokenizer=self.tokenizer,
                max_seq_length=512,
                packing=False,
                dataset_kwargs={
                    "add_special_tokens": False,
                    "append_concat_token": False,
                }
            )
            trainer.train()
            trainer.save_model(save_name)
            # self.tokenizer.save_pretrained(save_name)
        if self.finetune_way == "classifier":
            train_dataset = Dataset.from_pandas(train_data)
            val_dataset = Dataset.from_pandas(val_data)
            def tokenize_function(examples):
                return self.tokenizer(examples[text_col_name], padding="max_length", truncation=True)
            tokenized_train_data = train_dataset.map(tokenize_function, batched=True)
            tokenized_val_data = val_dataset.map(tokenize_function, batched=True)
            training_args = TrainingArguments(
                output_dir=save_name,
                num_train_epochs=5,
                per_device_train_batch_size = 16,
                per_device_eval_batch_size = 16,
                evaluation_strategy = "epoch",
                logging_strategy = "epoch",
                save_strategy = "epoch",
                learning_rate =2e-5,
                load_best_model_at_end = True,
                metric_for_best_model = "auc",
                optim = "adam"
            )
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_train_data,
                eval_dataset=tokenized_val_data,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )
            trainer.train()
            trainer.save_model(save_name)
            # self.tokenizer.save_pretrained(save_name)

    def test(self,data,text_col_name):
        pipe = pipeline(task="text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_new_tokens=2)
        for index, row in data.iterrows():
            try:
                text = row[text_col_name]
                if self.finetune_way=="llm-sft":
                    text = generate_test_prompt(text)
                result = pipe(text)
                answer = result[0]['generated_text'].split("label:")[-1].strip()
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
        pipe = pipeline(task="text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_new_tokens=2)
        if self.finetune_way == "llm-sft":
            data = generate_test_prompt(data)
        result = pipe(data)
        answer = result[0]['generated_text'].split("label:")[-1].strip()
        final_answer = categories[0]
        for category in categories:
            if category.lower() in answer.lower():
                final_answer = category
        return final_answer


