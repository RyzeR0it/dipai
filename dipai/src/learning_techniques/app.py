from flask import jsonify, Flask, request, render_template
import torch.nn.functional as F
import torch
from model_architecture import GeneralModel
import json

app = Flask(__name__)
with open('vocab.json', 'r') as vocab_file:
    vocab = json.load(vocab_file)
    inv_vocab = {v: k for k, v in vocab.items()}
with open('model_hyperparameters.json', 'r') as f:
    hyperparams = json.load(f)
model_path = 'models/model.pth'
model = GeneralModel(
    vocab_size=hyperparams['vocab_size'],
    embed_dim=256,
    num_layers=3,
    heads=8,
    ff_dim=512,
    num_entities=hyperparams['num_entities'],
    num_intents=hyperparams['num_intents'],
    dropout_rate=hyperparams.get('dropout_rate', 0.1) 
)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval() 

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    print(f"Received user input: {user_input}")
    model_response = generate_seq2seq_response(user_input)
    print(f"Final model response to be sent: {model_response}")
    return jsonify({'user_input': user_input, 'DipAI': model_response})


def text_to_token_ids(text):
    tokens = text.lower().split()
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

def token_ids_to_text(token_ids):
    words = [inv_vocab.get(id, '') for id in token_ids if id not in [vocab['<start>'], vocab['<end>'], vocab['<pad>']]]
    response_text = ' '.join(words)
    print(f"Translated response text: {response_text}")
    return response_text

def generate_seq2seq_response(input_text, temperature=1.0):
    assert temperature > 0, "Temperature must be positive"
    device = torch.device('cpu')
    input_ids = text_to_token_ids(input_text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    start_token_id = vocab['<start>']
    decoder_input_ids = torch.tensor([[start_token_id]], dtype=torch.long).to(device)
    print(f"Input text: {input_text}")
    print(f"Input token IDs: {input_ids}")
    generated_sequence = []
    with torch.no_grad():
        for _ in range(100):
            output_logits = model(x=input_tensor, mode='seq2seq', tgt=decoder_input_ids)
            next_token_logits = output_logits[:, -1, :]
            scaled_logits = next_token_logits / temperature
            probabilities = F.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
            generated_token_id = next_token_id.squeeze().item()
            if generated_token_id == vocab['<end>']:
                break
            if generated_token_id not in [vocab['<start>'], vocab['<end>'], vocab['<pad>']]:
                generated_sequence.append(generated_token_id)
    print(f"Generated token IDs: {generated_sequence}")
    response_text = token_ids_to_text(generated_sequence)
    print(f"Generated response text: {response_text}")
    return response_text


def start_app():
    app.run(debug=True)

if __name__ == "__main__":
    start_app()