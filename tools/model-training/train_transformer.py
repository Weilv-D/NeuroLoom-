"""
Train a tiny 4-token, 128-dim, 4-head, 1-layer GPT-style transformer on next-token prediction.
Exports to ONNX with attention weight intermediates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from pathlib import Path
import math


class TinyTransformer(nn.Module):
    def __init__(self, n_tokens=4, d_model=128, n_heads=4, d_ff=256, vocab_size=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_tokens = n_tokens

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(n_tokens, d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device)

        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # Self-attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch_size = x.shape[0]
        head_dim = self.d_model // self.n_heads

        q = q.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)

        x = self.norm1(x + attn_out)

        # FFN
        ff_out = self.ff2(F.gelu(self.ff1(x)))
        x = self.norm2(x + ff_out)

        logits = self.head(x)
        return logits, attn_weights


def make_sequence_data(n=50, seq_len=4, vocab_size=64, seed=42):
    rng = np.random.RandomState(seed)
    # Simple next-token prediction: token[i+1] = (token[i] + 3) % vocab_size
    inputs = []
    targets = []
    for _ in range(n):
        start = rng.randint(0, vocab_size - seq_len)
        tokens = [(start + i * 3) % vocab_size for i in range(seq_len)]
        inputs.append(tokens[:-1])
        targets.append(tokens[1:])
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


def train_and_export(output_dir: str = "apps/studio/public/models"):
    torch.manual_seed(42)
    model = TinyTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    x_data, y_data = make_sequence_data()

    step_outputs = []

    for step_idx in range(22):
        model.train()
        optimizer.zero_grad()
        logits, attn = model(x_data)
        loss = criterion(logits.view(-1, logits.size(-1)), y_data.view(-1))
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(-1) == y_data).float().mean().item()
        step_outputs.append({
            "step": step_idx,
            "loss": loss.item(),
            "accuracy": acc,
        })

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save intermediates
    with torch.no_grad():
        model.eval()
        logits, attn = model(x_data[:5])
        np.savez(
            out_path / "transformer_intermediates.npz",
            attn_weights=attn.numpy(),
            logits=logits.numpy(),
            losses=np.array([s["loss"] for s in step_outputs]),
            accuracies=np.array([s["accuracy"] for s in step_outputs]),
        )

    # Export to ONNX
    dummy = torch.tensor([[1, 5, 9, 13]], dtype=torch.long)
    torch.onnx.export(
        model,
        dummy,
        out_path / "tiny-gpt.onnx",
        input_names=["input_ids"],
        output_names=["logits", "attn_weights"],
        dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}, "attn_weights": {0: "batch"}},
        opset_version=17,
    )

    # Verify
    sess = ort.InferenceSession(str(out_path / "tiny-gpt.onnx"))
    result = sess.run(None, {"input_ids": dummy.numpy()})
    print(f"Transformer ONNX exported: logits={result[0].shape}, attn={result[1].shape}")
    print(f"  Loss: {step_outputs[0]['loss']:.4f} → {step_outputs[-1]['loss']:.4f}, Acc: {step_outputs[-1]['accuracy']:.2%}")
    print(f"  File size: {(out_path / 'tiny-gpt.onnx').stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    train_and_export()
