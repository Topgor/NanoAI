import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from model import NanoAIModel

class TextDataset(Dataset):
    def __init__(self, data_path, seq_length=50):
        self.seq_length = seq_length

        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Простая токенизация по словам
        words = text.lower().split()
        unique_words = list(set(words))
        self.vocab_size = min(len(unique_words), 5000)

        self.word2idx = {w: i for i, w in enumerate(unique_words[:self.vocab_size])}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        # Конвертируем текст в числа
        self.tokens = []
        for w in words:
            if w in self.word2idx:
                self.tokens.append(self.word2idx[w])

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Total tokens: {len(self.tokens)}")

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_length - 1)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_length])
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_length + 1])
        return x, y

def train(epochs=10, batch_size=16, lr=0.001):
    # Путь к данным
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.txt')

    if not os.path.exists(data_path):
        print("ERROR: No training data found!")
        print(f"Please add text data to: {data_path}")
        return

    # Загружаем данные
    dataset = TextDataset(data_path)

    if len(dataset) == 0:
        print("ERROR: Not enough data to train!")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Создаём модель
    model = NanoAIModel(vocab_size=dataset.vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"Training for {epochs} epochs...\n")

    # Тренировка
    for epoch in range(epochs):
        total_loss = 0
        batches = 0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.view(-1, dataset.vocab_size), batch_y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # Сохраняем модель
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'nano_model.pt')
    torch.save({
        'model_state': model.state_dict(),
        'vocab': dataset.word2idx,
        'vocab_size': dataset.vocab_size
    }, save_path)

    print(f"\nModel saved to {save_path}")
    print("Training complete!")

if __name__ == "__main__":
    train()
