from collections import Counter
import os
import json
import pandas as pd
import re
import torch
from nltk.tokenize import word_tokenize
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

nlp = English()
PAD_IDX = 0

class CustomDataset(Dataset):
    def __init__(self, data, vocab=None, task_type='seq2seq', transform=None):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")
        if not isinstance(vocab, dict):
            print(f"vocab: {vocab}")
            raise TypeError("Vocab must be a dictionary.")
        self.data = data
        self.vocab = vocab
        self.task_type = task_type
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def numericalize(self, text):
        if self.vocab is None:
            raise ValueError("Vocab not initialized.")
        tokens = word_tokenize(text.lower())
        numericalized_tokens = [self.vocab.get('<start>', self.vocab.get('<unk>'))]
        numericalized_tokens += [self.vocab.get(token, self.vocab.get('<unk>')) for token in tokens]
        numericalized_tokens += [self.vocab.get('<end>', self.vocab.get('<unk>'))]
        return numericalized_tokens


    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        if self.task_type == 'seq2seq':
            input_seq = self.numericalize(item['input'])
            target_seq = self.numericalize(item['label'])
            return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
        else:
            item = self.data.iloc[idx]
            text = item['cleaned_input']
            label = item['label']
            if self.transform:
                text = self.transform(text)
            return text, label


def build_vocab(data):
    token_freqs = Counter()
    for row in data.itertuples(index=False):
        tokens = word_tokenize(row.input.lower()) + word_tokenize(row.label.lower())
        token_freqs.update(tokens)
    vocab = {token: idx for idx, (token, _) in enumerate(token_freqs.items(), start=4)}
    vocab['<pad>'] = 0
    vocab['<start>'] = 1
    vocab['<end>'] = 2
    vocab['<unk>'] = 3
    return vocab

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_IDX).long()
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX).long()
    return inputs_padded, targets_padded


def standardize_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

def clean_text(text):
    text = text.lower() 
    text = re.sub(r"[^a-zA-Z0-9\s]", ' ', text) 
    tokens = word_tokenize(text)
    cleaned_text = " ".join(tokens)
    return cleaned_text


def tokenize_text(df, column_name):
    vocab = set()
    df[f'{column_name}_tokens'] = df[column_name].apply(lambda x: word_tokenize(x))
    for tokens in df[f'{column_name}_tokens']:
        vocab.update(tokens)
    return df, vocab

def load_and_preprocess_data(directory_path, include_entities_intents=False):
    all_dfs = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                df = pd.DataFrame(data)
                if include_entities_intents:
                    pass
                df.dropna(subset=['input', 'label'], inplace=True)
                df['standardized_input'] = df['input'].apply(standardize_text)
                df['cleaned_input'] = df['standardized_input'].apply(clean_text)
                all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def adjust_indices_to_vocab(dataset, vocab):
    unk_index = vocab.get('<unk>') 
    adjusted_dataset = []
    for text, label in dataset:
        adjusted_text = [vocab[token] if token in vocab else unk_index for token in text]
        adjusted_label = [vocab[token] if token in vocab else unk_index for token in label]
        adjusted_dataset.append((adjusted_text, adjusted_label))
    return adjusted_dataset

def update_counts_json(vocab_size, num_entities=0, num_intents=0):
    counts = {'num_entities': num_entities, 'num_intents': num_intents, 'vocab_size': vocab_size}
    with open('src\\learning_techniques\\entity_intent_counts.json', 'w') as f:
        json.dump(counts, f)
    print("Updated entity_intent_counts.json with the latest vocabulary size.")

def extract_entities_intents(combined_df):
    unique_entities = set()
    unique_intents = set()
    for item in combined_df.itertuples(index=False):
        if hasattr(item, 'entities'):
            unique_entities.update(item.entities)
        if hasattr(item, 'intent'):
            unique_intents.add(item.intent)
    return len(unique_entities), len(unique_intents)

def run_preprocess(general_conv_directory, python_qa_directory=None):
    print("Starting preprocessing...")
    general_conv_data = load_and_preprocess_data(general_conv_directory)
    python_qa_data = pd.DataFrame()
    if python_qa_directory is not None:
        python_qa_data = load_and_preprocess_data(python_qa_directory, include_entities_intents=True)
    all_data = pd.concat([general_conv_data, python_qa_data], ignore_index=True) if not python_qa_data.empty else general_conv_data
    num_entities, num_intents = extract_entities_intents(all_data)
    _, vocab = tokenize_text(all_data, 'cleaned_input') 
    vocab = build_vocab(all_data) 
    update_counts_json(vocab_size=len(vocab), num_entities=num_entities, num_intents=num_intents)
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab, vocab_file)
    print("Actual vocabulary size (including special tokens):", len(vocab))
    print("Preprocessing completed.")
    return all_data, vocab

if __name__ == "__main__":
    general_conv_directory = 'src\learning_techniques\data'
    python_qa_directory = 'path/to/python/qa/data'
    all_data, vocab = run_preprocess(general_conv_directory, python_qa_directory)
    train_df, test_df = train_test_split(all_data, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(train_df, vocab)
    test_dataset = CustomDataset(test_df, vocab)
        