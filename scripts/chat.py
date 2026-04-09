import torch, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import NanoAI

def chat():
    print("=" * 50)
    print("NANO-AI CHAT")
    print("=" * 50)
    cfg_path = '../model/ds_config.json'
    if not os.path.exists(cfg_path):
        print("Train first: python3 train.py")
        return
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    c2i = cfg['c2i']
    i2c = {int(k): v for k, v in cfg['i2c'].items()}

    md = '../model'
    mf = 'nano-ai-final.pt' if os.path.exists(f'{md}/nano-ai-final.pt') else 'nano-ai-best.pt'
    print(f"Loading: {mf}")

    model = NanoAI(vocab_size=cfg['vocab_size'], dim=256, n_layers=8, n_heads=8, n_kv_heads=2, max_seq=256)
    ckpt = torch.load(f'{md}/{mf}', map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    t, _ = model.count_params()
    print(f"Params: {t:,}")
    print("Type message (exit to quit)")
    print("-" * 50)

    while True:
        try:
            inp = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if inp.lower() in ['exit', 'quit']:
            break
        prompt = f"Вопрос: {inp}\nОтвет:"
        tokens = [c2i[c] for c in prompt if c in c2i]
        if not tokens:
            print("AI: ???")
            continue
        x = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=200, temperature=0.7)
        resp = ''.join([i2c.get(i, '?') for i in out[0].tolist()])
        if "Ответ:" in resp:
            resp = resp.split("Ответ:", 1)[-1]
        if "Вопрос:" in resp:
            resp = resp.split("Вопрос:")[0]
        print(f"AI: {resp.strip()}")

if __name__ == "__main__":
    chat()
