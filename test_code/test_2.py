import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import wandb
import os
from collections import Counter
from source.preprocessing import VQA_Dataset_2
from source.model import create_model, generate_text_from_sample
from source.postprocessing import extract_text
from sklearn.model_selection import KFold,train_test_split
from qwen_vl_utils import process_vision_info
import csv
import re


model_id = "Qwen/Qwen3-VL-2B-Instruct"

model, processor = create_model(model_id)
adapter_path = "/home/s2510447/Study/term21/Advanced_Machine_learning/checkpoints_gen_answer/checkpoint-1126"
model.load_adapter(adapter_path)



# df = pd.read_csv('/home/s2510447/Study/term21/Advanced_Machine_learning/data/custom_dataset/custom_dataset/train_labels.csv')
df = pd.read_csv('/home/s2510447/Study/term21/Advanced_Machine_learning/data/custom_dataset/custom_dataset/test_non_labels.csv')
reasoning_df = pd.read_csv('/home/s2510447/Study/term21/Advanced_Machine_learning/test/temp/test_check_point2252_fixed.csv')
img_folder = "/home/s2510447/Study/term21/Advanced_Machine_learning/data/custom_dataset/custom_dataset/test" 

# img_folder = "/home/s2510447/Study/term21/Advanced_Machine_learning/data/custom_dataset/custom_dataset/train" 
# _, val_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

# val_dataset = VQA_Dataset(val_df, processor, img_folder, True) 

merge_df = reasoning_df.merge(df, on="id")[["id", "file","question", "explanation", "answer"]]

val_dataset = VQA_Dataset_2(merge_df, processor, img_folder) 

output_csv = "/home/s2510447/Study/term21/Advanced_Machine_learning/test/temp/test_check_point2252_fixed_with_1126_2.csv"
# output_csv = "./test/results_log.csv"
# Nếu file chưa tồn tại → tạo header
if not os.path.exists(output_csv):
    # pd.DataFrame(columns=["image", "question", "true_answer", "true_explaination", "prediction","p_answer","p_explain"]).to_csv(output_csv, index=False)
    pd.DataFrame(columns=["id", "question", "answer", "explanation", "old_answer"]).to_csv(output_csv, index=False)

# for sample in val_dataset:
#     id = sample[0]
#     question = sample[1]
#     explanation = sample[2]
#     old_answer = sample[3]
#     is_done = False
#     max_ire = 5
#     print(f"ID: {id}")
#     # Prepare the text input by applying the chat template
#     text_input = processor.apply_chat_template(
#         sample[4]['messages'],  
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     # Process the visual input from the sample
#     image_inputs, _ = process_vision_info(sample[4]['messages'])
      
#     # Prepare the inputs for the model
#     model_inputs = processor(
#         text=[text_input],
#         images=image_inputs,
#         return_tensors="pt",
#     ).to('cuda')  # Move inputs to the specified device

#     # Generate text with the model
#     generated_ids = model.generate(**model_inputs, max_new_tokens=10)
#     print(generated_ids)
#     # Trim the generated ids to remove the input ids
#     trimmed_generated_ids = [
#         out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     # Decode the output text
#     output_text = processor.batch_decode(
#         trimmed_generated_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )
#     print(output_text)
#     print(output_text[0])        
#     answer = output_text[0]
#     entry = pd.DataFrame([{
#         "id": id,
#         "question": question,
#         "answer": answer,
#         "explanation": explanation,
#         "old_answer": old_answer   # lấy lần predict cuối
#     }])

#     entry.to_csv(output_csv, mode="a", header=False, index=False)

for sample in val_dataset:
    id = sample[0]
    question = sample[1]
    explanation = sample[2]
    old_answer = sample[3]

    print(f"ID: {id}")

    # Lưu tất cả kết quả predict
    answers = []

    for _ in range(5):   # chạy 5 lần
        # Prepare the text input by applying the chat template
        text_input = processor.apply_chat_template(
            sample[4]['messages'],  
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
        ).to('cuda')

        # Generate text with the model
        generated_ids = model.generate(**model_inputs, 
                                       max_new_tokens=10, 
                                       do_sample=True, # Must be True to use temperature for sampling
                                        temperature=0.5,
                                        # top_k=50, # Often used alongside temperature for further control
                                        top_p=0.9)

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

        answer = output_text[0]
        answers.append(answer)
        print("Run answer:", answer)

    # Lấy kết quả xuất hiện nhiều nhất
    final_answer = Counter(answers).most_common(1)[0][0]
    print("Final answer:", final_answer)

    # Lưu vào CSV
    entry = pd.DataFrame([{
        "id": id,
        "question": question,
        "answer": final_answer,
        "explanation": explanation,
        "old_answer": old_answer
    }])

    entry.to_csv(output_csv, mode="a", header=False, index=False)


        
