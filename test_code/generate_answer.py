import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import wandb
import os
from source.model import create_model
from source.preprocessing import VQA_Dataset_2
from qwen_vl_utils import process_vision_info
from source.postprocessing import extract_text
from sklearn.model_selection import KFold,train_test_split

import csv
import re
def generate_text_from_sample(model, processor, sample, max_new_tokens=200, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[4]['messages'][:2],  # Use the sample without the system message
        tokenize=False,
        add_generation_prompt=True
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample[4]['messages'])        
    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    print(generated_ids)
    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(output_text)
    print(output_text[0])
    # return output_text[0]  # Return the first decoded output text
    return output_text[0]

model_id = "Qwen/Qwen3-VL-2B-Instruct"

model, processor = create_model(model_id)
adapter_path = "./checkpoint/checkpoint-2815"
model.load_adapter(adapter_path)



# df = pd.read_csv('/home/s2510447/Study/term21/Advanced_Machine_learning/data/custom_dataset/custom_dataset/train_labels.csv')
df = pd.read_csv('/home/s2510447/Study/term21/Advanced_Machine_learning/data/custom_dataset/custom_dataset/test_non_labels.csv')
add_df = pd.read_csv('./test/gen_reasoning_1126.csv')
img_folder = "/home/s2510447/Study/term21/Advanced_Machine_learning/data/custom_dataset/custom_dataset/test" 

val_df = df.merge(add_df, on="id")[["id", "file","question", "explanation", "answer"]]

print(val_df.head(5))
val_dataset = VQA_Dataset_2(val_df, processor, img_folder) 


output_csv = "./test/test_check_point2252_gen_answer.csv"

if not os.path.exists(output_csv):
    pd.DataFrame(columns=["id", "question", "explanation", "old_answer", "new_answer"]).to_csv(output_csv, index=False)

for sample in val_dataset:
    id = sample[0]
    question = sample[1]
    explanation = sample[2]
    old_answer = sample[3]
    result = generate_text_from_sample(model=model, processor=processor,sample=sample)
        # ---- SAVE TO CSV ----
    print(f"result: {result}")
    entry = pd.DataFrame([{
        "id": id,
        "question": question,
        "explanation": explanation,   # lấy lần predict cuối
        "old_answer": old_answer,
        "new_answer": result
    }])

    entry.to_csv(output_csv, mode="a", header=False, index=False)


        
