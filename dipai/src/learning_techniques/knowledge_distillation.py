import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from optuna.pruners import MedianPruner
import optuna.visualization as vis
import optuna
import json
import pandas as pd
import os
from train_general import calculate_seq2seq_loss
from model_architecture import GeneralModel, PythonModel
from preprocess import CustomDataset, load_and_preprocess_data, extract_entities_intents, tokenize_text, collate_fn, build_vocab
from sklearn.metrics import accuracy_score


model_path = Path('models/model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
general_conv_directory = 'src\\learning_techniques\\data'
all_data_preprocessed = load_and_preprocess_data(general_conv_directory)
train_df, test_df = train_test_split(all_data_preprocessed, test_size=0.2, random_state=42)
all_data, _ = tokenize_text(train_df, 'cleaned_input')
vocab = build_vocab(all_data)
with open('src\\learning_techniques\\entity_intent_counts.json', 'r') as f:
    dynamic_params = json.load(f)
vocab_size = dynamic_params['vocab_size']
print(f"Vocab size: {vocab_size}")
num_entities = dynamic_params['num_entities']
num_intents = dynamic_params['num_intents']
counts = {'num_entities': num_entities, 'num_intents': num_intents, 'vocab_size': vocab_size}
train_dataset = CustomDataset(train_df, vocab)
test_dataset = CustomDataset(test_df, vocab)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
#python_dataset_path = 'path/to/preprocessed/python_specialization_processed/'
criterion = nn.CrossEntropyLoss(reduction='sum') 
distillation_criterion = nn.KLDivLoss(reduction='batchmean')
is_seq2seq = True

def mutual_learning_loss(outputs, targets, all_outputs):
    """Calculate mutual learning loss."""
    gt_loss = sum(criterion(output, targets) for output in outputs) / len(outputs)
    avg_outputs = sum(F.softmax(output, dim=1) for output in outputs) / len(outputs)
    dl_loss = sum(distillation_criterion(F.log_softmax(output, dim=1), avg_outputs.detach()) for output in outputs) / len(outputs)
    return gt_loss + dl_loss

def evaluate_model(model, test_loader, criterion, is_seq2seq=True):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if is_seq2seq:
                outputs = model(x=inputs, mode='seq2seq', tgt=targets[:, :-1])
                targets_for_loss = targets[:, 1:].reshape(-1)
            else:
                outputs = model(inputs)
                targets_for_loss = targets.reshape(-1)
            outputs_reshaped = outputs.view(-1, outputs.shape[-1])
            if outputs_reshaped.shape[0] != targets_for_loss.shape[0]:
                min_size = min(outputs_reshaped.shape[0], targets_for_loss.shape[0])
                outputs_reshaped = outputs_reshaped[:min_size, :]
                targets_for_loss = targets_for_loss[:min_size]
            loss = criterion(outputs_reshaped, targets_for_loss)
            total_loss += loss.item()
            _, predicted = torch.max(outputs_reshaped.data, 1)
            total_accuracy += (predicted == targets_for_loss).sum().item()
            total_samples += targets_for_loss.size(0)
    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
    return avg_loss, avg_accuracy


def objective(trial):
    embed_dim = 256
    num_layers = 3
    heads = 8
    ff_dim = 512
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.7)
    models = [GeneralModel(vocab_size, embed_dim, num_layers, heads, ff_dim, num_entities, num_intents, dropout_rate=dropout_rate).to(device) for _ in range(3)]
    pretrained_state = torch.load(model_path)
    for model in models:
        model.load_state_dict(pretrained_state)
        model.train()
    optimizers = [optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) for model in models]
    num_epochs = 10
    early_stopping_counter = 0 
    patience_threshold = 3 
    best_loss = float('inf') 
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if is_seq2seq:
                teacher_forcing_inputs = targets[:, :-1]
                targets = targets[:, 1:] 
            all_outputs = []
            for model in models:
                model.train()
                if is_seq2seq:
                    outputs = model(inputs, mode='seq2seq', tgt=teacher_forcing_inputs)
                else:
                    outputs = model(inputs, mode='non_seq2seq')
                all_outputs.append(outputs)
            for optimizer, model_outputs in zip(optimizers, all_outputs):
                if is_seq2seq:
                    loss = calculate_seq2seq_loss(model_outputs, targets, all_outputs)
                else:
                    loss = mutual_learning_loss(model_outputs, targets, all_outputs)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            evaluate_model_loss, evaluate_model_accuracy = evaluate_model(model, test_loader, criterion, is_seq2seq=True)
            trial.report(evaluate_model_loss, epoch) 
            if trial.should_prune(): 
                raise optuna.exceptions.TrialPruned() 
            if evaluate_model_loss < best_loss:
                print(f"New best loss {evaluate_model_loss} found, previous best was {best_loss}")
                best_loss = evaluate_model_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"No improvement, early stopping counter: {early_stopping_counter}")
                if early_stopping_counter > patience_threshold:
                    print("Early stopping triggered.")
                    break
        print(f'Epoch {epoch+1}, Total Loss: {total_loss / len(train_loader)}')
        print(f"Epoch {epoch+1}, Evaluation Loss: {evaluate_model_loss:.4f}, Evaluation Accuracy: {evaluate_model_accuracy:.4f}")
    print(f"Objective - Trial Evaluation: Loss {evaluate_model_loss}, Accuracy {evaluate_model_accuracy}")
    return evaluate_model_loss


def run():
    pruner = MedianPruner() 
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=10) 
    vis.plot_optimization_history(study)
    vis.plot_param_importances(study)
    print("Best hyperparameters:", study.best_trial.params)
    best_lr = study.best_trial.params['learning_rate']
    best_batch_size = 64
    embed_dim = 256
    num_layers = 3
    heads = 8
    ff_dim = 512
    best_params = study.best_trial.params
    models = [
        GeneralModel(vocab_size, embed_dim, num_layers, heads, ff_dim, num_entities, num_intents, dropout_rate=best_params['dropout_rate']).to(device), 
        GeneralModel(vocab_size, embed_dim, num_layers, heads, ff_dim, num_entities, num_intents, dropout_rate=best_params['dropout_rate']).to(device)
        #PythonModel(dropout_rate=dropout_rate).to(device) 
    ]
    optimizers = [optim.Adam(model.parameters(), lr=study.best_trial.params['learning_rate'], weight_decay=study.best_trial.params['weight_decay']) for model in models]
    #general_loader = DataLoader(CustomDataset(general_conv_directory), batch_size=64, shuffle=True)
    #python_loader = DataLoader(CustomDataset(python_dataset_path), batch_size=study.best_trial.params['batch_size'], shuffle=True)
    evaluation_scores = [evaluate_model(m, test_loader, criterion, is_seq2seq=True) for m in models]
    best_model_idx = evaluation_scores.index(min(evaluation_scores))
    best_model = models[best_model_idx]
    best_params = study.best_trial.params
    model_configurations = {
        'vocab_size': vocab_size, 
        'num_entities': num_entities, 
        'num_intents': num_intents, 
        **best_params
    }
    config_path = 'model_hyperparameters.json'
    with open(config_path, 'w') as f:
        json.dump(model_configurations, f)
    torch.save(best_model.state_dict(), model_path)
