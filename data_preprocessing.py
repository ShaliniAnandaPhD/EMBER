import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

class WildfireDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['text'][index]
        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

def preprocess_data(data_path, tokenizer, max_length, batch_size):
    try:
        # Load and parse the dataset
        data = pd.read_csv(data_path)
        
        # Perform data cleaning
        data['text'] = data['text'].apply(lambda x: x.lower())  # Convert text to lowercase
        data['text'] = data['text'].apply(lambda x: x.strip())  # Remove leading/trailing whitespaces
        
        # Remove rows with missing values
        data = data.dropna()
        
        # Split the data into training, validation, and test sets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
        
        # Create dataset instances
        train_dataset = WildfireDataset(train_data, tokenizer, max_length)
        val_dataset = WildfireDataset(val_data, tokenizer, max_length)
        test_dataset = WildfireDataset(test_data, tokenizer, max_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' does not exist.")
        return None, None, None
    
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{data_path}' is empty.")
        return None, None, None
    
    except Exception as e:
        print(f"Error: An unexpected error occurred - {str(e)}")
        return None, None, None

def save_preprocessed_data(data_loader, save_path):
    # Save the preprocessed data
    torch.save(data_loader, save_path)
    print(f"Preprocessed data saved at: {save_path}")

# Example usage 
if __name__ == '__main__':
    data_path = 'path/to/wildfire/dataset.csv'
    save_path = 'path/to/save/preprocessed/data.pt'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 512
    batch_size = 16

    train_loader, val_loader, test_loader = preprocess_data(data_path, tokenizer, max_length, batch_size)
    
    if train_loader is not None:
        save_preprocessed_data(train_loader, save_path)
