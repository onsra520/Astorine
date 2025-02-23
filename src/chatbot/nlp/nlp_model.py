import sys
import os
from pathlib import Path
import pandas as pd
import json5
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
import numpy as np
import evaluate

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from utils.qgene import generate_text

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

paths = {
    "processed": os.path.abspath(f"{project_root}/data/storage/processed"),
    "qfragments": os.path.abspath(f"{project_root}/intents/qfragments.json"),
    "questions": os.path.abspath(f"{project_root}/intents/questions.csv"),
    "model": os.path.abspath(f"{project_root}/models/t5-base"),
    "tokenizer": os.path.abspath(f"{project_root}/models/tokenizer"),
    "results": os.path.abspath(f"{project_root}/training/results"),
}

os.makedirs(paths["model"], exist_ok=True)
os.makedirs(paths["tokenizer"], exist_ok=True)
os.makedirs(paths["results"], exist_ok=True)

class NLPModel:
    def __init__(self, datanum: int = 1000):
        self.spec = pd.read_csv(os.path.join(paths["processed"], "final_cleaning.csv"))
        self.qfragments = json5.load(open(paths["qfragments"]))
        self.questions = generate_text(datanum)
        self.model_checkpoint = "t5-small"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint)
        self.rouge = evaluate.load("rouge")
        self.tokenized_dataset = self.dataset().map(
            self.tokenize_function, batched=True
        )
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

    def preprocess_data(self, row: pd.Series) -> dict:
        return {
            "input_text": f"{row['question']}",
            "target_text": f"""
                DISPLAY: {row['display']}; 
                RAM: {row['ram']}; 
                GPU: {row['gpu']}; 
                CPU: {row['cpu']}; 
                REFRESH_RATE: {row['refresh rate']}; 
                BRAND: {row['brand']}; 
                PRICE: {row['price']}
                """,
        }

    def dataset(self):
        processed = self.questions.apply(self.preprocess_data, axis=1)
        df_processed = pd.DataFrame(list(processed))
        return Dataset.from_pandas(df_processed)

    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(
            examples["input_text"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )

        labels = self.tokenizer(
            text_target=examples["target_text"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(
            predictions != -100, predictions, self.tokenizer.pad_token_id
        )
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_output = self.rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        return {
            "rouge1": rouge_output["rouge1"].mid.fmeasure,
            "rouge2": rouge_output["rouge2"].mid.fmeasure,
            "rougeL": rouge_output["rougeL"].mid.fmeasure,
        }

    def save_model(self):
        self.trainer.save_model(paths["model"])
        self.tokenizer.save_pretrained(paths["tokenizer"])

    def train(self, save_model: bool = True):
        training_args = TrainingArguments(
            output_dir=paths["results"],
            eval_strategy="no",
            learning_rate=3e-5,
            per_device_train_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            do_eval=False,
            gradient_accumulation_steps=2,
            eval_accumulation_steps=2,
            fp16=True,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()
        if save_model:
            self.save_model()
