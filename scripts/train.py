import torch
import torch.nn.functional as F
import sys, os, time, math, json
sys.path.insert(0, os.path.dirname(__file__))
from model import NanoAI

class TextDataset:
    def __init__(self, data_path, seq_len=256):
        self.seq_len = seq_len
        if not os.path.isfile(data_path):
            text = self._gen_data()
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Generated: {data_path}")
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        self.i2c = {i: c for c, i in self.c2i.items()}
        self.data = torch.tensor([self.c2i[c] for c in text], dtype=torch.long)
        print(f"Dataset: {len(self.data):,} tokens, vocab: {self.vocab_size}")

    def _gen_data(self):
        knowledge = [
            "Искусственный интеллект это область компьютерных наук создающая умные системы.",
            "Нейронные сети состоят из слоев нейронов обрабатывающих информацию.",
            "Трансформер это архитектура нейронной сети основанная на механизме внимания.",
            "Машинное обучение позволяет компьютерам учиться на данных.",
            "Python самый популярный язык для машинного обучения.",
            "Linux свободная операционная система.",
            "Алгоритм это последовательность шагов для решения задачи.",
            "Обучение нейросети это настройка весов для минимизации ошибки.",
            "Градиентный спуск метод оптимизации нейронных сетей.",
            "Данные основа любого искусственного интеллекта.",
        ]
        dialogs = [
            "Вопрос: Что такое ИИ?\nОтвет: Искусственный интеллект это система способная решать задачи требующие человеческого интеллекта.",
            "Вопрос: Как работает нейросеть?\nОтвет: Нейросеть пропускает данные через слои нейронов и выдает результат.",
            "Вопрос: Что такое обучение?\nОтвет: Обучение это процесс улучшения предсказаний модели на основе данных.",
            "Вопрос: Кто ты?\nОтвет: Я NanoAI компактный искусственный интеллект созданный для работы на слабом железе.",
            "Вопрос: Чем ты лучше больших моделей?\nОтвет: Я использую Mixture of Experts и Sparse Attention для эффективности при малом размере.",
            "Вопрос: Что такое трансформер?\nОтвет: Трансформер использует механизм внимания для обработки последовательностей.",
            "Вопрос: Зачем нужен Python?\nОтвет: Python удобный язык с библиотеками для данных и ИИ.",
        ]
        logic = [
            "Если идет дождь нужен зонт. Идет дождь. Значит нужен зонт.",
            "Все люди смертны. Сократ человек. Значит Сократ смертен.",
            "Два плюс два равно четыре.",
            "Программа работает если нет ошибок.",
        ]
        parts = []
        for item in knowledge * 20:
            parts.append(item)
        for item in dialogs * 30:
            parts.append(item)
        for item in logic * 20:
            parts.append(item)
        return "\n\n".join(parts)

    def get_batch(self, batch_size=2):
        ix = torch.randint(0, len(self.data) - self.seq_len - 1, (batch_size,))
        x = torch.stack([self.data[i:i+self.seq_len] for i in ix])
        y = torch.stack([self.data[i+1:i+self.seq_len+1] for i in ix])
        return x, y

    def encode(self, text):
        return [self.c2i.get(c, 0) for c in text]

    def decode(self, indices):
        return ''.join([self.i2c.get(i, '?') for i in indices])

def train():
    print("=" * 50)
    print("NANO-AI TRAINING")
    print("=" * 50)
    SEQ_LEN = 256
    BATCH = 2
    EPOCHS = 50
    LR = 3e-4

    ds = TextDataset('../data/train.txt', SEQ_LEN)
    model = NanoAI(vocab_size=ds.vocab_size, dim=256, n_layers=8, n_heads=8, n_kv_heads=2, max_seq=SEQ_LEN)
    t, _ = model.count_params()
    print(f"Params: {t:,} (~{t*4/1024/1024:.1f} MB)")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    steps = max(len(ds.data) // (SEQ_LEN * BATCH), 100)
    total_steps = EPOCHS * steps

    model.train()
    best = float('inf')
    gs = 0

    for ep in range(EPOCHS):
        eloss = 0
        t0 = time.time()
        for s in range(steps):
            warmup = 100
            if gs < warmup:
                lr = LR * gs / warmup
            else:
                lr = LR * 0.5 * (1 + math.cos(math.pi * (gs - warmup) / (total_steps - warmup)))
            for pg in opt.param_groups:
                pg['lr'] = lr

            x, y = ds.get_batch(BATCH)
            logits, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            eloss += loss.item()
            gs += 1

        avg = eloss / steps
        print(f"Epoch {ep+1:3d}/{EPOCHS} | Loss: {avg:.4f} | LR: {lr:.6f} | Time: {time.time()-t0:.1f}s")

        if (ep + 1) % 5 == 0:
            model.eval()
            tok = torch.tensor([ds.encode("Вопрос:")], dtype=torch.long)
            with torch.no_grad():
                out = model.generate(tok, max_new_tokens=100, temperature=0.7)
            print(f"  Gen: {ds.decode(out[0].tolist())[:200]}")
            model.train()

        if avg < best:
            best = avg
            model.save('../model/nano-ai-best.pt')

    model.save('../model/nano-ai-final.pt')
    with open('../model/ds_config.json', 'w') as f:
        json.dump({'c2i': ds.c2i, 'i2c': {str(k): v for k, v in ds.i2c.items()}, 'vocab_size': ds.vocab_size}, f, ensure_ascii=False)
    print("=" * 50)
    print(f"DONE! Best loss: {best:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    train()
