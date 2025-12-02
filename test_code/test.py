import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import wandb
import os

from source.preprocessing import VQA_Dataset
from source.model import create_model, generate_text_from_sample
from source.postprocessing import extract_text
from sklearn.model_selection import KFold,train_test_split

import csv
import re


model_id = "Qwen/Qwen3-VL-2B-Instruct"

model, processor = create_model(model_id)
adapter_path = "/home/s2510447/Study/term21/Advanced_Machine_learning/checkpointsfull/checkpoint-1126"
model.load_adapter(adapter_path)


df = pd.read_csv('/home/s2510447/Study/term21/Advanced_Machine_learning/data/custom_dataset/custom_dataset/test_non_labels.csv')
img_folder = "/home/s2510447/Study/term21/Advanced_Machine_learning/data/custom_dataset/custom_dataset/test" 


val_dataset = VQA_Dataset(df, processor, img_folder, False) 

output_csv = ".test/gen_reasoning_1126.csv"

# Nếu file chưa tồn tại → tạo header
if not os.path.exists(output_csv):
    # pd.DataFrame(columns=["image", "question", "true_answer", "true_explaination", "prediction","p_answer","p_explain"]).to_csv(output_csv, index=False)
    pd.DataFrame(columns=["id", "answer", "explanation"]).to_csv(output_csv, index=False)

for sample in val_dataset:
    id = sample[0]
    image_name = sample[1]
    question = sample[2]
    is_done = False
    max_ire = 5
    while is_done == False and max_ire > 0:
        result = generate_text_from_sample(model=model, processor=processor,sample=sample,is_labeled=False)
        # ---- SAVE TO CSV ----
        explanation, answer, is_done = extract_text(result)
        if is_done == False:
            max_ire = max_ire - 1
    explanation = result
    entry = pd.DataFrame([{
        "id": id,
        "answer": answer,
        "explanation": explanation   # lấy lần predict cuối
    }])

    entry.to_csv(output_csv, mode="a", header=False, index=False)


        
