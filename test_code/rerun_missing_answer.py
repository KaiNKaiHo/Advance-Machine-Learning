# from test import generate_text_from_sample, extract_text
from source.model import create_model, generate_text_from_sample
from source.postprocessing import extract_text, extract_text_2
from source.preprocessing import VQA_Dataset
import pandas as pd 
import torch


input_csv = "./test/test_check_point2252_gen_answer.csv"
output_csv_new = "./test/test_check_point2252_gen_answer_fixed.csv"

df = pd.read_csv('/home/s2510447/Study/term21/Advanced_Machine_learning/data/custom_dataset/custom_dataset/test_non_labels.csv')
input_df = pd.read_csv(input_csv)
img_folder = "/home/s2510447/Study/term21/Advanced_Machine_learning/data/custom_dataset/custom_dataset/test"
missing_ids = input_df[input_df["answer"].isna() | (input_df["answer"].astype(str).str.strip() == "")]["id"].tolist()
print(f"🔍 Found {len(missing_ids)} samples missing answers: {missing_ids}")

re_run_df = df[df["id"].isin(missing_ids)]
print(len(re_run_df))
print(re_run_df.iloc[0])

model, processor = create_model("Qwen/Qwen3-VL-2B-Instruct")
adapter_path = "./checkpoint/checkpoint-1126"
model.load_adapter(adapter_path)

val_dataset = VQA_Dataset(re_run_df, processor, img_folder, False) 

updated_rows = []
for sample in val_dataset:
    sample_id = sample[0]

    if sample_id not in missing_ids:
        continue

    print(f"⟲ Rerunning missing sample id = {sample_id}")

    retries = 5
    is_done = False

    while not is_done and retries > 0:
        output = generate_text_from_sample(
            model=model,
            processor=processor,
            sample=sample,
            is_labeled=False,
            device="cuda",
            max_new_tokens=800
        )

        explanation, answer, is_done = extract_text(output)

        if not is_done:
            retries -= 1
    if not is_done:
        explanation = output
        answer = ""
    print(f"id: {sample_id}")
    print(f"answer: {answer}")
    print(f"explanation: {explanation}")
    updated_rows.append({
        "id": sample_id,
        "answer": answer,
        "explanation": explanation
    })



# Make a copy of old df and update values
updated_df = input_df.copy()

for row in updated_rows:
    updated_df.loc[updated_df["id"] == row["id"], ["answer", "explanation"]] = \
        [row["answer"], row["explanation"]]

# Save into new CSV file
updated_df.to_csv(output_csv_new, index=False)

print(f"✔ Missing predictions recomputed and saved to {output_csv_new}")

