import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from learn2learn.algorithms import MAML
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from model_architecture import GeneralModel
import os
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("training_log.txt"),
                              logging.StreamHandler()])
nltk.download('punkt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MetaLearningDataset(Dataset):
    def __init__(self, file_path, task_size=10):
        self.data = self._load_data(file_path)
        self.vocab = self._build_vocab()
        self.task_size = task_size

    def _load_data(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        if file_extension == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_extension == '.csv':
            return pd.read_csv(file_path).to_dict('records')
        else:
            raise ValueError("Unsupported file type")

    def _build_vocab(self):
        all_text = " ".join([item['input'] + " " + item['label'] for item in self.data])
        tokens = word_tokenize(all_text.lower())
        vocab = {word: i+1 for i, word in enumerate(set(tokens))}
        return vocab

    def __len__(self):
        return len(self.data) // self.task_size

    def __getitem__(self, idx):
        task_data = self.data[idx * self.task_size:(idx + 1) * self.task_size]
        inputs = [self.text_to_tensor(item['input']) for item in task_data]
        labels = [self.text_to_tensor(item['label']) for item in task_data]
        inputs_tensor = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        labels_tensor = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        return inputs_tensor, labels_tensor

    def text_to_tensor(self, text):
        tokens = word_tokenize(text.lower())
        indices = [self.vocab.get(token, 0) for token in tokens]
        return torch.tensor(indices, dtype=torch.long)

def collate_fn(batch):
    inputs, labels = zip(*batch)
    # Log shapes before padding
    inputs_shapes = [input.shape for input in inputs]
    labels_shapes = [label.shape for label in labels]
    logging.info(f"Input shapes before padding: {inputs_shapes}")
    logging.info(f"Label shapes before padding: {labels_shapes}")

    inputs_padded = pad_sequence(inputs, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True)

    # Log shapes after padding
    logging.info(f"Input tensor shape after padding: {inputs_padded.shape}")
    logging.info(f"Label tensor shape after padding: {labels_padded.shape}")

    return inputs_padded, labels_padded



def load_entity_intents_counts(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['num_entities'], data['num_intents'], data['vocab_size']


def meta_run():
    initial_dataset = MetaLearningDataset(file_path='src/learning_techniques/data/test.json', task_size=10)
    file_path = 'src/learning_techniques/data/test.json'
    dataset = MetaLearningDataset(file_path=file_path, task_size=10)
    counts_json_path = 'src/learning_techniques/entity_intent_counts.json'
    num_entities, num_intents, vocab_size = load_entity_intents_counts(counts_json_path)

    # Now, you can use these values to initialize your GeneralModel
    base_model = GeneralModel(vocab_size=vocab_size, embed_dim=256, num_layers=3, heads=8, ff_dim=512,
                              num_entities=num_entities, num_intents=num_intents,
                              dropout_rate=0.1).to(device)
    model_path = 'models/model.pth'
    try:
        base_model.load_state_dict(torch.load(model_path))
        logging.info("Loaded pre-trained model.")
    except FileNotFoundError:
        logging.info("Pre-trained model not found, initializing a new model.")
    meta_model = MAML(base_model, lr=0.01)
    optimizer = optim.Adam(meta_model.parameters(), lr=0.001)
    epochs = 20
    task_varieties = ['general_conversation', 'python_programming', 'other_domains']
    for epoch in range(epochs):
        optimizer.zero_grad()
        logging.info(f"Starting epoch {epoch+1}/{epochs}")
        for domain in task_varieties:
            # Adjusted file_path for CSV file
            file_path = 'processed_data.csv'
            train_dataset = MetaLearningDataset(file_path=file_path, task_size=10)
            logging.info(f"Training on domain: {domain}")
            for task_data, task_labels in DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn):
                logging.info(f"Task data shape: {task_data.shape}")
                logging.info(f"Task labels shape: {task_labels.shape}")
                learner = meta_model.clone()
                prediction = learner(task_data.squeeze(0))
                loss = nn.CrossEntropyLoss()(prediction, task_labels.squeeze(0))
                learner.adapt(loss)
            optimizer.step()
            eval_dataset = MetaLearningDataset(file_path=file_path, task_size=10)
            logging.info(f"Evaluating on domain: {domain}")
            total_eval_loss = 0
            total_eval_samples = 0
            with torch.no_grad():
                for eval_data, eval_labels in DataLoader(eval_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn):
                    eval_prediction = learner(eval_data.squeeze(0))
                    eval_loss = nn.CrossEntropyLoss()(eval_prediction, eval_labels.squeeze(0))
                    total_eval_loss += eval_loss.item()
                    total_eval_samples += 1
            average_eval_loss = total_eval_loss / total_eval_samples if total_eval_samples > 0 else 0
            logging.info(f"Average evaluation loss for domain {domain}: {average_eval_loss:.4f}")
        logging.info(f"Epoch {epoch+1} completed.")
    torch.save(meta_model.state_dict(), model_path)
    logging.info("Updated model saved.")

