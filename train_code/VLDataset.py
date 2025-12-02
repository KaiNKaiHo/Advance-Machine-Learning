from torch.utils.data import Dataset
from PIL import Image
import torch

class VQA_Dataset(Dataset):
    def __init__(self, df, processor, img_folder, is_labeled):
        self.df = df
        self.processor = processor
        self.img_folder = img_folder
        self.is_labeled = is_labeled

    def __len__(self):
        return len(self.df)
    
    def format_data(self, sample):
        image = Image.open(f"{self.img_folder}/{sample['file']}")
        question = sample['question']
        system_message = "You are an effective Question Answering Model. Answer the question with only 2 sentences with format: \"Because <Explain>. So the answer is <Answer>.\". Additionally the <Answer> is only a word or a number."
        # system_message = "You are an effective Question Answering Model. Answer the question with only a word or a number."
        # user_prompt = f"Question: {question}\nExplanation: Because {sample['explanation'][0]}. So the answer is "

        if self.is_labeled == True:
            answer = f"Because {sample['explanation'][0]}. So the answer is {sample['answer']}."
            # answer = sample['answer']
            return [
                sample['id'],
                sample['file'],
                question,
                sample['answer'][0],
                sample['explanation']
                ,{
                "images": [image],
                "messages": [

                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system_message
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {
                                "type": "text",
                                "text": question,
                                # "text": user_prompt,
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": answer
                            }
                        ],
                    },
                ]
            }       
            ]
        return [
                sample['id'],
                sample['file'],
                question,
                {
                "images": [image],
                "messages": [

                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system_message
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {
                                "type": "text",
                                "text": question,
                            }
                        ],
                    }
                ]
            }       
        ]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = self.format_data(row)
        return data
   