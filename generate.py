import torch
import torch.nn.functional as F
import os
import sys
from model import NanoAIModel

def generate(prompt="hello", max_length=50, temperature=0.8):
    # Загружаем модель
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'nano_model.pt')

    if not os.path.exists(model_path):
        print("ERROR: No trained model found!")
        print("Run train.py first!")
        return

    # Загружаем чекпоинт
    checkpoint = torch.load(model_path, map_location='cpu')
    vocab = checkpoint['vocab']
    vocab_size = checkpoint['vocab_size']
    idx2word = {i: w for w, i in vocab.items()}

    # Создаём модель
    model = NanoAIModel(vocab_size=vocab_size)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"Model loaded! Vocab size: {vocab_size}")
    print(f"Prompt: {prompt}")
    print(f"Generating {max_length} words...\n")

    # Токенизируем промпт
    words = prompt.lower().split()
    tokens = []
    for w in words:
        if w in vocab:
            tokens.append(vocab[w])
        else:
            tokens.append(0)

    generated_words = list(words)

    # Генерация
    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor([tokens[-50:]])  # последние 50 токенов
            output = model(x)
            logits = output[0, -1, :] / temperature

            # Сэмплируем следующее слово
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            tokens.append(next_token)

            if next_token in idx2word:
                generated_words.append(idx2word[next_token])
            else:
                generated_words.append("[UNK]")

    result = ' '.join(generated_words)
    print("=" * 50)
    print("GENERATED TEXT:")
    print("=" * 50)
    print(result)
    print("=" * 50)

    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
    else:
        prompt = "the world is"

    generate(prompt=prompt)
