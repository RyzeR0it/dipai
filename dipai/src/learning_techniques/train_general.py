import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model_architecture import GeneralModel
from preprocess import CustomDataset, collate_fn, load_and_preprocess_data, train_test_split, run_preprocess
import optuna
from optuna.pruners import MedianPruner
from pathlib import Path
import json
from sklearn.metrics import f1_score, precision_score, recall_score

print("In train_general.py")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
general_conv_directory = "src\\learning_techniques\\data"
all_data, vocab = run_preprocess(general_conv_directory)
train_df, test_df = train_test_split(all_data, test_size=0.2, random_state=42)
train_dataset = CustomDataset(train_df, vocab=vocab)
test_dataset = CustomDataset(test_df, vocab=vocab)
sentiment_loss_fn = nn.CrossEntropyLoss()
entity_recognition_loss_fn = nn.CrossEntropyLoss()
intent_classification_loss_fn = nn.CrossEntropyLoss()
PAD_IDX = 0
seq2seq_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(device)

def calculate_seq2seq_loss(output, target, pad_idx=PAD_IDX):
    _, batch_size, vocab_size = output.shape
    target = target.reshape(-1)
    output_flat = output.reshape(-1, vocab_size)  
    if output_flat.size(0) != target.size(0):
        min_len = min(output_flat.size(0), target.size(0))
        output_flat = output_flat[:min_len, :]
        target = target[:min_len]
    loss = seq2seq_loss_fn(output_flat, target)
    return loss


def calculate_f1_score(true_labels, predictions, average='macro'):
    return f1_score(true_labels, predictions, average=average)

def train_model(trial):
    print("In train model in train general .py")
    with open('src\\learning_techniques\\entity_intent_counts.json', 'r') as f:
        counts = json.load(f)
    num_entities = counts['num_entities']
    num_intents = counts['num_intents']
    vocab_size = counts['vocab_size']
    batch_size = 64
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.7)
    epochs = trial.suggest_int('epochs', 5, 20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)
    model = GeneralModel(vocab_size=vocab_size, embed_dim=256, num_layers=3, heads=8, ff_dim=512, num_entities=num_entities, num_intents=num_intents, dropout_rate=dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    is_seq2seq = hasattr(model, 'seq2seq_mode') and model.seq2seq_mode
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            is_seq2seq = model.seq2seq_mode
            if is_seq2seq:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                output = model(x=inputs, tgt=targets[:-1], mode='seq2seq')
                loss = calculate_seq2seq_loss(output, targets[1:], pad_idx=PAD_IDX)
            else:
                inputs, sentiment_targets, entity_targets, intent_targets = data
                inputs = inputs.to(device)
                sentiment_targets = sentiment_targets.to(device)
                entity_targets = entity_targets.to(device)
                intent_targets = intent_targets.to(device)
                optimizer.zero_grad()
                sentiment_pred, _, entity_pred, intent_pred = model(inputs, mode='non_seq2seq')
                sentiment_loss = sentiment_loss_fn(sentiment_pred, sentiment_targets)
                entity_loss = entity_recognition_loss_fn(entity_pred, entity_targets)
                intent_loss = intent_classification_loss_fn(intent_pred, intent_targets)
                loss = sentiment_loss + entity_loss + intent_loss
                pass

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Total Loss: {total_loss / len(train_loader)}')
        validation_loss = validate_model(model, val_loader, is_seq2seq, total_loss, epoch, train_loader, trial)
    return validation_loss
    
def validate_model(model, val_loader, is_seq2seq, total_loss, epoch, train_loader, trial=None):
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        num_batches = 0
        patience = 3
        best_loss = float('inf')
        epochs_no_improve = 0
        for data in val_loader:
            try:
                if is_seq2seq:
                    inputs, targets = data
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(x=inputs, tgt=targets[:-1], mode='seq2seq')
                    loss = calculate_seq2seq_loss(output, targets[1:], pad_idx=PAD_IDX)
                else:
                    inputs, sentiment_targets, entity_targets, intent_targets = data
                    inputs = inputs.to(device)
                    sentiment_targets = sentiment_targets.to(device)
                    entity_targets = entity_targets.to(device)
                    intent_targets = intent_targets.to(device)
                    sentiment_pred, _, entity_pred, intent_pred = model(inputs, mode='non_seq2seq')
                    sentiment_loss = sentiment_loss_fn(sentiment_pred, sentiment_targets)
                    entity_loss = entity_recognition_loss_fn(entity_pred, entity_targets)
                    intent_loss = intent_classification_loss_fn(intent_pred, intent_targets)
                    loss = sentiment_loss + entity_loss + intent_loss
                total_val_loss += loss.item()
                num_batches += 1
            except ValueError:
                print("Data loading issue: Non-seq2seq mode expects four items per batch during validation.")
        if num_batches > 0:
            average_val_loss = total_val_loss / num_batches
            if trial:
                trial.report(average_val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            if average_val_loss < best_loss:
                best_loss = average_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"No improvement in validation loss for {epochs_no_improve} epochs, stopping early.")
        else:
            print("No valid data was processed; skipping average calculation.")
            average_val_loss = float('inf')

        print(f'Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {average_val_loss:.4f}')
        return average_val_loss

def retrain_model_with_best_params(best_params):
    print("In retrain model with best params in train general .py")
    with open('src\\learning_techniques\\entity_intent_counts.json', 'r') as f:
        counts = json.load(f)
    num_entities = counts['num_entities']
    num_intents = counts['num_intents']
    vocab_size = counts['vocab_size']
    batch_size = 64
    learning_rate = best_params['learning_rate']
    weight_decay = best_params['weight_decay']
    dropout_rate = best_params['dropout_rate']
    epochs = best_params['epochs']
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)
    model = GeneralModel(vocab_size=vocab_size, embed_dim=256, num_layers=3, heads=8, ff_dim=512, num_entities=num_entities, num_intents=num_intents, dropout_rate=dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    is_seq2seq = hasattr(model, 'seq2seq_mode') and model.seq2seq_mode
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            if is_seq2seq:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                output = model(x=inputs, tgt=targets[:-1], mode='seq2seq')
                loss = calculate_seq2seq_loss(output, targets[1:], pad_idx=PAD_IDX)
            else:
                inputs, sentiment_targets, entity_targets, intent_targets = data
                inputs = inputs.to(device)
                sentiment_targets = sentiment_targets.to(device)
                entity_targets = entity_targets.to(device)
                intent_targets = intent_targets.to(device)
                optimizer.zero_grad()
                sentiment_pred, _, entity_pred, intent_pred = model(inputs, mode='non_seq2seq')
                sentiment_loss = sentiment_loss_fn(sentiment_pred, sentiment_targets)
                entity_loss = entity_recognition_loss_fn(entity_pred, entity_targets)
                intent_loss = intent_classification_loss_fn(intent_pred, intent_targets)
                loss = sentiment_loss + entity_loss + intent_loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Total Loss: {total_loss / len(train_loader)} in retrain model')
        validate_model(model, val_loader, is_seq2seq, total_loss, epoch, train_loader, None)
    model_save_path = Path('models')
    model_save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path / 'model.pth')


def evaluate_model(model, val_loader):
    print("In evalaute model in train general .py")
    model.eval()
    all_sentiment_preds, all_sentiment_targets = [], []
    all_entity_preds, all_entity_targets = [], []
    all_intent_preds, all_intent_targets = [], []
    with torch.no_grad():
        for data in val_loader:
            inputs, sentiment_targets, entity_targets, intent_targets = [d.to(device) for d in data]
            sentiment_pred, entity_pred, intent_pred = model(inputs)
            sentiment_labels = torch.argmax(sentiment_pred, dim=1)
            entity_labels = torch.argmax(entity_pred, dim=1)
            intent_labels = torch.argmax(intent_pred, dim=1)
            all_sentiment_preds.extend(sentiment_labels.tolist())
            all_sentiment_targets.extend(sentiment_targets.tolist())
            all_entity_preds.extend(entity_labels.tolist())
            all_entity_targets.extend(entity_targets.tolist())
            all_intent_preds.extend(intent_labels.tolist())
            all_intent_targets.extend(intent_targets.tolist())
    sentiment_f1 = f1_score(all_sentiment_targets, all_sentiment_preds, average='macro')
    entity_f1 = f1_score(all_entity_targets, all_entity_preds, average='macro')
    intent_f1 = f1_score(all_intent_targets, all_intent_preds, average='macro')
    print(f'Sentiment F1: {sentiment_f1:.4f}, Entity F1: {entity_f1:.4f}, Intent F1: {intent_f1:.4f}')
    return sentiment_f1, entity_f1, intent_f1

def main():
    study = optuna.create_study(direction='minimize', pruner=MedianPruner())
    study.optimize(train_model, n_trials=10) 
    best_params = study.best_trial.params
    print("Best hyperparameters:", best_params)
    retrain_model_with_best_params(best_params)

if __name__ == "__main__":
    main()
