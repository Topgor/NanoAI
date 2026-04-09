import torch
import torch.nn as nn
import math

class NanoAIModel(nn.Module):
    def __init__(self, vocab_size=5000, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding(positions)
        x = self.transformer(x)
        x = self.output_head(x)
        return x

if __name__ == "__main__":
    model = NanoAIModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"NanoAI Model created!")
    print(f"Total parameters: {total_params:,}")
    test_input = torch.randint(0, 5000, (1, 20))
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test PASSED!")
