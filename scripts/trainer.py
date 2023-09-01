import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import pandas as pd
from torch.utils.data import Dataset

def train_gpt(data_filepath):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = CustomTextDataset(tokenizer=tokenizer, file_path=data_filepath, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print(f"Number of samples in train_dataset: {len(train_dataset)}")

    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=10,
        output_dir="../models",
    )

    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()

def csv_to_text(file_path, output_path):
    df = pd.read_csv(file_path)

    # Concatenate all columns into a single string
    df['combined'] = df.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    
    # Save the combined column to a text file
    df['combined'].to_string(output_path, index=False, header=False)

# Use the function
csv_to_text("../data/cleaned/cleaned_data.csv", "../data/cleaned/cleaned_data.txt")

class CustomTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        with open(file_path, 'r') as f:
            self.text = f.read()

        self.tokenizer = tokenizer
        self.block_size = block_size

        # Tokenize the text
        self.tokens = tokenizer.tokenize(self.text)

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        tokenized_text = self.tokenizer.convert_tokens_to_ids(self.tokens[start_idx:end_idx])
        return torch.tensor(tokenized_text)
    
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token

with open("../data/cleaned/cleaned_data.txt", 'r') as f:
    content = f.read()
    tokens = tokenizer.encode(content)
    print(f"Number of tokens: {len(tokens)}")

