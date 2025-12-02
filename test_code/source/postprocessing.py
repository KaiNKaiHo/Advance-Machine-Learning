import re
def extract_text(text):
    """
    Trích xuất Explanation và Answer từ danh sách câu và lưu vào CSV.
    """
    pattern = r"Because (.*?)[\.,]\s*So the answer is ([a-zA-Z0-9]+)\."
    # pattern = r"Because\s+(.*?)\s*[\.,]\s*So the answer is\s+(.+?)\s*\."
    match = re.search(pattern, text)
    if match:
        explanation = match.group(1).strip()
        answer = match.group(2).strip()
        return explanation, answer, True
    return "", "", False

def extract_text_2(text):
    return text, "", True


def normalize_yes_no(df):
    """
    Chuẩn hóa cột 'answer':
        y -> yes
        n -> no
    Không phân biệt hoa thường, giữ nguyên các giá trị khác.
    """

    # Tạo bản copy an toàn
    df = df.copy()

    # Chuẩn hóa bằng map có điều kiện
    df["answer"] = df["answer"].astype(str).str.strip().str.lower().map(
        lambda x: "yes" if x == "y"
        else "no" if x == "n"
        else x
    )

    return df