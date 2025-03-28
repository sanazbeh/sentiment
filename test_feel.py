import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F

# 1. Download IMDB dataset
train_iter = IMDB(split='train')

# 2. Text processing
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 3. Define the model
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, _) = self.rnn(embedded)
        return self.fc(hidden[-1])

# 4. Setup model
vocab_size = len(vocab)
embed_dim = 100
hidden_dim = 256
output_dim = 1

model = SentimentModel(vocab_size, embed_dim, hidden_dim, output_dim)

# Define `text_to_tensor` function
def text_to_tensor(text):
    tokens = tokenizer(text)  # Tokenize text
    numericalized = vocab(tokens)  # Convert tokens to numbers
    return torch.tensor(numericalized, dtype=torch.long)  # Convert to tensor

# Example input text (can be changed)
sample_text = "this file is very horrible!"

# Convert the text to tensor (you need your text preprocessing here)
text_tensor = text_to_tensor(sample_text)
text_tensor = text_tensor.unsqueeze(0)  # Add batch dimension if needed

# Model evaluation (run the model to get the prediction)
model.eval()  # Put model in evaluation mode
with torch.no_grad():  # Don't track gradients during inference
    logit_output = model(text_tensor)

# Apply sigmoid activation to get the probability
probability = torch.sigmoid(logit_output).item()

# Prediction logic based on probability
sentiment = "Positive" if probability > 0.5 else "Negative"

# Output the result
print(f"Probability: {probability:.4f}")
print(f"Predicted Sentiment: {sentiment}")