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

# 5. Test the model with a sample sentence
sample_text = "This movie is amazing and I loved it!"
text_tensor = text_to_tensor(sample_text)
text_tensor = text_tensor.unsqueeze(0)  # Add batch dimension

model.eval()  # Put model in evaluation mode
with torch.no_grad():
    prediction = model(text_tensor)

# Apply sigmoid to convert logits to probability
probability = torch.sigmoid(prediction)

# Print the probability
print(f"Probability: {probability.item()}")

# Determine sentiment
if probability.item() > 0.5:
    print("Predicted Sentiment: Positive")
else:
    print("Predicted Sentiment: Negative")