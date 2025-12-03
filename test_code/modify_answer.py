import pandas as pd
import ast
import re
def modify(df):


    df = df.copy()

    for index, row in df.iterrows():
        answer = row['answer']

        # Skip if answer is a number or NaN
        if isinstance(answer, (int, float)) or pd.isna(answer) or row['answer'] == '':
            continue
        if row['answer'] == 'S' or ((pd.isna(answer) or  row['answer'] == '') and any(w in row['explanation'] for w in ['There is no', 'There are no','there is no', 'there are no'])):
            df.at[index, 'answer'] = "no"
            continue
        answer = row['answer'].lower()
        words_of_materal = ['material', 'made of'] 
        words_of_color = ['color'] 
        words_of_size = ['size', 'how big', 'How big '] 
        words_of_shape = ['shape'] 
        match answer:
            case 'n':
                df.at[index, 'answer'] = "no"
            case 'y':
                df.at[index, 'answer'] = "yes"
            case 'true':
                df.at[index, 'answer'] = "yes"
            case 'false':
                df.at[index, 'answer'] = "no"
            case 'one':
                df.at[index, 'answer'] = "1"
            case 'two':
                df.at[index, 'answer'] = "2"
            case 'three':
                df.at[index, 'answer'] = "3"
            case 'four':
                df.at[index, 'answer'] = "4"
            case 'five':
                df.at[index, 'answer'] = "5"
            case 'six':
                df.at[index, 'answer'] = "6"       
            case 'seven':
                df.at[index, 'answer'] = "7"
            case 'eight':
                df.at[index, 'answer'] = "8" 
            case 'nine':
                df.at[index, 'answer'] = "9"
            case 'ten':
                df.at[index, 'answer'] = "10"   
            case 'matte':
                df.at[index, 'answer'] = "metal"    
            case 'ball':
                df.at[index, 'answer'] = "sphere"   
            case 'big':
                df.at[index, 'answer'] = "large"          
            case 'm':
                df.at[index, 'answer'] = "metal"     
            case 't':
                df.at[index, 'answer'] = "small" 
            case 'p':
                df.at[index, 'answer'] = "purple"  
            case 'shiny':
                df.at[index, 'answer'] = "metal"    
            case 's':
                if any(w in row['question'] for w in words_of_materal):
                    df.at[index, 'answer'] = "metal"
                elif any(w in row['question'] for w in words_of_size):
                    df.at[index, 'answer'] = "small" 
                elif any(w in row['question'] for w in words_of_shape):   
                    df.at[index, 'answer'] = "sphere" 
            case 'tiny':
                df.at[index, 'answer'] = "small"   
            case 'l':
                df.at[index, 'answer'] = "large"   
            case 'r':
                if any(w in row['question'] for w in words_of_materal):
                    df.at[index, 'answer'] = "rubber"
                elif any(w in row['question'] for w in words_of_color):
                    df.at[index, 'answer'] = "red"
            case 'b':
                if any(w in row['question'] for w in words_of_size):
                    df.at[index, 'answer'] = "large"
                elif any(w in row['question'] for w in words_of_color):
                    df.at[index, 'answer'] = "brown"       
            case 'c':
                if any(w in row['question'] for w in words_of_color):
                    df.at[index, 'answer'] = "cyan"  
                # elif any(w in row['question'] for w in words_of_shape):
                #     df.at[index, 'answer'] = "cylinder"
                    

                            
    return df

def merge_df(df1,df2):
    return df1.merge(df2, on="id")[["id", "file","question", "explanation", "answer"]]

def remove_duplicates(df):
    """
    Remove all duplicate rows in a DataFrame and return the cleaned DataFrame.
    """
    df = df.copy()                    # safe copy
    df = df.drop_duplicates()         # remove exact duplicate rows
    df = df.reset_index(drop=True)    # reset index after removal
    return df



def clean_explanation(value):
    if pd.isna(value):
        return value

    original = value

    # ----- 1. Cố gắng đọc list nếu chuỗi là dạng list -----
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and len(parsed) > 0:
                value = parsed[0]
        except:
            # Nếu không parse được (trường hợp dòng 112), loại bỏ ngoặc thô
            value = value.strip("[]")

    # ----- 2. Nếu vẫn là list thật (csv đọc sai kiểu) -----
    if isinstance(value, list) and len(value) > 0:
        value = value[0]

    # ----- 3. Bảo đảm value là string -----
    if not isinstance(value, str):
        value = str(value)

    # ----- 4. Loại bỏ ngoặc còn sót lại -----
    value = value.strip().strip("[]").strip()

    # ----- 5. Chỉ lấy câu đầu tiên -----
    sentences = re.split(r'[.!?]\s*', value)
    first_sentence = sentences[0].strip()

    # ----- 6. Viết hoa chữ cái đầu -----
    if first_sentence:
        first_sentence = first_sentence[0].upper() + first_sentence[1:]

    # ----- 7. Kết thúc bằng dấu chấm -----
    if not first_sentence.endswith("."):
        first_sentence += "."

    return first_sentence

def merge_answer(df1, df2):
    """
    Nếu cùng id và df1.explanation != df2.explanation
    và df1.explanation là số → dùng giá trị df1 thay df2.
    """
    df2 = df2.copy()

    for idx, row in df2.iterrows():
        id_val = row["id"]

        # Lấy dòng tương ứng trong df1
        row1 = df1[df1["id"] == id_val]
        if row1.empty:
            continue

        exp1 = row1["answer"]
        exp2 = row["answer"]

        # Nếu explanation khác nhau và exp1 là số → thay thế
        try:
            if str(exp1).isdigit():  # check số
                if str(exp1) != str(exp2):
                    df2.at[idx, "answer"] = exp1
                    continue
        except:
            pass

    return df2

# --- Đọc input ---
input_path = "./test/test_check_point2252_gen_answer.csv"
# test_data_path = "/home/s2510447/Study/term21/Advanced_Machine_learning/data/custom_dataset/custom_dataset/test_non_labels.csv"
output_path = "/home/s2510447/Study/term21/Advanced_Machine_learning/test/full/final.csv"

df = pd.read_csv(input_path)
# df2 = pd.read_csv(test_data_path)
# # --- Chuẩn hóa answer ---
# df_fixed = merge_df(df1,df2)
# df_fixed = modify(df_fixed)
# df_fixed = remove_duplicates(df_fixed)
# df_fixed = modify(df1)
df["explanation"] = df["explanation"].apply(clean_explanation)
# --- Ghi ra output ---
df.to_csv(output_path, index=False)

print(f"✔ Đã sửa answer và lưu vào: {output_path}")

