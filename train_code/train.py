import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import wandb
import os
from VLDataset import VQA_Dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import torch
from sklearn.model_selection import KFold,train_test_split

def train_qa_model(model, processor, train_df, val_df, img_folder, project_name="train"): 
    train_dataset = [s[-1] for s in VQA_Dataset(train_df, processor, img_folder, True)]
    val_dataset = [s[-1] for s in VQA_Dataset(val_df, processor, img_folder, True)]
    print(len(train_dataset))
    print(len(val_dataset))
    
    # LoRA config 
    peft_config = LoraConfig( 
        lora_alpha=16, 
        r=8,
        target_modules=["q_proj", "v_proj"], 
        task_type="CAUSAL_LM",
        lora_dropout=0.05 
    ) 
    # model = get_peft_model(model, peft_config) 
    training_args = SFTConfig( 
        output_dir="./checkpoints", 
        per_device_train_batch_size=10, 
        per_device_eval_batch_size=10,
        gradient_accumulation_steps=1, 
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        num_train_epochs=10, # Large max epochs 
        fp16=True, 
        # logging_steps=1,
        # eval_steps=1,
        # save_steps=250,
        optim="adamw_torch_fused",
        eval_strategy="epoch", # Evaluate every epoch 
        save_strategy="epoch", 
        load_best_model_at_end=True, 
        greater_is_better=False, 
        report_to="wandb", 
        save_total_limit=2,
        run_name=project_name ) 
    trainer = SFTTrainer( 
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        peft_config=peft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
        ) 
    trainer.train() 
    trainer.save_model(training_args.output_dir)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct") 
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", 
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)
# adapter_path = "/home/s2510447/Study/term21/Advanced_Machine_learning/vl_qa_checkpoints_2/checkpoint-94"
# model.load_adapter(adapter_path)
wandb.login() 
project = "train" 
config = { 'epochs' : 10, 'lr' : 0.01 } 
project_folder_path = "/home/s2510447/Study/term21/AML/data"
df = pd.read_csv(f'{project_folder_path}/custom_dataset/custom_dataset/train_labels.csv') 
img_folder = f"{project_folder_path}/custom_dataset/custom_dataset/train" 

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)
print("=== Training ===")
train_qa_model(
    model=model,
    processor=processor,
    train_df=train_df,
    val_df=val_df,
    img_folder=img_folder,
    project_name="train"
)