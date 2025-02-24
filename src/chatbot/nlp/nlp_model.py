import sys, os, shutil, warnings
import json5
from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.qgene import generate_text

paths = {
    "processed": os.path.abspath(f"{project_root}/data/storage/processed"),
    "qfragments": os.path.abspath(f"{project_root}/intents/qfragments.json"),
    "questions": os.path.abspath(f"{project_root}/intents/questions.csv"),
    "trained_questions": os.path.abspath(f"{project_root}/intents/trained_questions.csv"),
    "models": os.path.abspath(f"{project_root}/models/t5-small"),
    "results": os.path.abspath(f"{project_root}/training/results"),
}

os.makedirs(paths["models"], exist_ok=True) 
os.makedirs(paths["results"], exist_ok=True)

class TokenizerSaver(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self.tokenizer.save_pretrained(ckpt_dir)
        return control

class NLPModel:
    def __init__(self, datanum: int = 1000) -> None:
        # Load data
        self.qfragments = json5.load(open(paths["qfragments"]))
        self.questions = self._dataset(datanum)

        # Model setup
        self.model_checkpoint = self._load_checkpoint()
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # Initialize datasets
        self._prepare_datasets()

    def _dataset(self, datanum: int = 1000) -> pd.DataFrame:
        if os.path.exists(paths["trained_questions"]):
            old_data = pd.read_csv(paths["trained_questions"])
            new_questions = pd.concat([old_data, generate_text(datanum)])
        else:
            new_questions = generate_text(datanum)
        
        new_questions.to_csv(paths["trained_questions"], index=False)
        return new_questions
    
    def _rows_preprocessor(self, row: pd.Series) -> dict:
        return {
            "input_text": str(row.get("question", "")),
            "target_text": (
                f"DISPLAY: {row.get('display', '')}; "
                f"RAM: {row.get('ram', '')}; "
                f"GPU: {row.get('gpu', '')}; "
                f"CPU: {row.get('cpu', '')}; "
                f"REFRESH_RATE: {row.get('refresh rate', '')}; "
                f"BRAND: {row.get('brand', '')}; "
                f"PRICE: {row.get('price', '')}"
            ),
        } 

    def _tokenize_function(self, examples):
        tokenized = self.tokenizer(
            examples["input_text"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["target_text"],
                max_length=128,
                truncation=True,
                padding="max_length",
            )
        tokenized["labels"] = labels["input_ids"]
        return tokenized

    def _prepare_datasets(self):
        """Prepare and tokenize datasets"""
        processed_data = self.questions.apply(self._rows_preprocessor, axis=1).tolist()
        df_processed = pd.DataFrame(processed_data)
        full_dataset = Dataset.from_pandas(df_processed)
        split_datasets = full_dataset.train_test_split(test_size=0.2, seed=42)
        self.tokenized_train = split_datasets["train"].map(
            self._tokenize_function,
            batched=True,
            remove_columns=split_datasets["train"].column_names,
        )

        self.tokenized_eval = split_datasets["test"].map(
            self._tokenize_function,
            batched=True,
            remove_columns=split_datasets["test"].column_names,
        )

    def _load_checkpoint(self, resume_checkpoint: str = "auto") -> str:
        if resume_checkpoint == "auto":
            checkpoint_dirs = list(Path(paths["results"]).glob("checkpoint-*"))
            valid_ckpts = [
                ckpt for ckpt in checkpoint_dirs 
                if ckpt.is_dir() and ckpt.name.split("-")[-1].isdigit()
            ]
            valid_ckpts = sorted(valid_ckpts, key=lambda x: int(x.name.split("-")[-1]))
            if valid_ckpts:
                resume_checkpoint = str(valid_ckpts[-1])
            else:
                print("No valid checkpoints found.")
                resume_checkpoint = "t5-small"
        else:
            if resume_checkpoint not in os.listdir(paths["results"]):
                resume_checkpoint = "t5-small"
        
        if resume_checkpoint != "t5-small":
            required_files = ["tokenizer_config.json", "special_tokens_map.json"]
            ckpt_path = Path(resume_checkpoint)
            if not all((ckpt_path / f).exists() for f in required_files):
                print("Missing tokenizer files in checkpoint, using default")
                resume_checkpoint = "t5-small"
        return resume_checkpoint

    def _del_checkpoint(self) -> None:
        checkpoint_dirs = list(Path(paths["results"]).glob("checkpoint-*"))
        checkpoints = sorted(
            [
                ckpt
                for ckpt in checkpoint_dirs
                if ckpt.is_dir() and ckpt.name.split("-")[-1].isdigit()
            ],
            key=lambda x: int(x.name.split("-")[-1]),
        )
        if len(checkpoints) > 3:
            for checkpoint in checkpoints[:-3]:
                shutil.rmtree(checkpoint)
                print(f"Removed old checkpoint: {checkpoint}")

    def _nlptraining(
        self,
        early_stopping_patience: int = 3,
        resume_checkpoint: str = None,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
        num_train_epochs: int = 1,
    ) -> None:

        training_args = TrainingArguments(
            output_dir=str(paths["results"]),
            overwrite_output_dir=True,            
            logging_dir=str(os.path.join(paths["results"],"logs")),
            report_to=["tensorboard"],            
            logging_first_step=True,
            logging_steps=500,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            gradient_accumulation_steps=2,
            fp16=True,
            load_best_model_at_end=True,
            greater_is_better=True,
            eval_accumulation_steps=1,
            resume_from_checkpoint=resume_checkpoint
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_eval,
            data_collator=self.data_collator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
                TokenizerSaver(self.tokenizer)
            ],
        )
            
        self.trainer.train()
        self.model.save_pretrained(paths["models"])
        self.tokenizer.save_pretrained(paths["models"])
        