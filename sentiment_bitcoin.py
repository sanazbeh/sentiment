import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import praw
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random

# 1. Setup Reddit client
reddit = praw.Reddit(
    client_id="G8QP3NeAIdAzrNixB0keFA",
    client_secret="kb9cP3ERWm5PjU9zEEaezKO2eg8f7Q",
    user_agent="sentimentAnalyzerBot/0.1 by u/Sani_class1980"
)

# 2. Tokenizer
tokenizer = get_tokenizer("basic_english")

# 3. Collect Bitcoin-related posts from Reddit
subreddit = reddit.subreddit("Bitcoin")
reddit_posts = []
for post in subreddit.hot(limit=50):  # You can change the limit
    text = post.title + " " + (post.selftext or "")
    reddit_posts.append(text)
    print(f"Number of posts received: {len(reddit_posts)}")

# 4. Build vocab from Reddit posts
def yield_tokens(data):
    for text in data:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(reddit_posts), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 5. Text-to-tensor
def text_to_tensor(text):
    tokens = tokenizer(text)
    numericalized = vocab(tokens)
    return torch.tensor(numericalized, dtype=torch.long)

# 6. Sentiment model
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

# 7. Model setup
vocab_size = len(vocab)
embed_dim = 100
hidden_dim = 256
output_dim = 1
model = SentimentModel(vocab_size, embed_dim, hidden_dim, output_dim)

# (Optional) Load pretrained weights here
# model.load_state_dict(torch.load("bitcoin_sentiment.pt"))

# 8. Analyze a Reddit post
input_text = reddit_posts[0]  # Example: first post
text_tensor = text_to_tensor(input_text).unsqueeze(0)

model.eval()
with torch.no_grad():
    logit_output = model(text_tensor)
    probability = torch.sigmoid(logit_output).item()

sentiment = "Positive" if probability > 0.5 else "Negative"

# 9. Output
print("Reddit Post:\n", input_text)
print(f"Predicted Sentiment: {sentiment}")
print(f"Probability: {probability:.4f}")



labeled_data = [
    ("Bitcoin is amazing and will change the world", 1),
    ("I hate how volatile Bitcoin is", 0),
    ("Bitcoin to the moon!", 1),
    ("Bitcoin is a scam", 0),
    ("Love the decentralization idea", 1),
    ("Mining consumes too much energy", 0),
    # Add more examples here...
]

# Shuffle for training
random.shuffle(labeled_data)

# Create dataset tensors
def collate_batch(batch):
    text_list, label_list = [], []
    for text, label in batch:
        text_tensor = text_to_tensor(text)
        text_list.append(text_tensor)
        label_list.append(torch.tensor(label, dtype=torch.float))
    # Pad sequences to same length
    text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    label_list = torch.tensor(label_list).unsqueeze(1)
    return text_list, label_list

train_loader = DataLoader(labeled_data, batch_size=2, shuffle=True, collate_fn=collate_batch)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()

# Training loop
num_epochs = 10
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_texts, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_texts)
        loss = loss_fn(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
