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
        system_message = "You are an effective Question Answering Model. Answer the question with only 2 sentences with format: \"Because <Explain>. So the answer is <Answer>."
        if self.is_labeled == True:
            answer = f"Because {sample['explanation']}. So the answer is {sample['answer'][0]}."
            return [
                sample['id'],
                sample['file'],
                question,
                sample['answer'],
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


class VQA_Dataset_2(Dataset):
    def __init__(self, df, processor, img_folder):
        self.df = df
        self.processor = processor
        self.img_folder = img_folder

    def __len__(self):
        return len(self.df)
    
    def format_data(self, sample):
        image = Image.open(f"{self.img_folder}/{sample['file']}")
        question = sample['question']
        explanation = sample['explanation'][0] if isinstance(sample['explanation'], list) else sample['explanation']
        old_answer = sample['answer']
        system_message = "You are an effective Question Answering Model. Answer the question with only a word or a number."
        return [
                sample['id'],
                question,
                explanation,
                old_answer,
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
                                "text": f"Question: {question}\nExplanation: Because {explanation}. So the answer is ",
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
   