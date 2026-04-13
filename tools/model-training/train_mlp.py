"""
Train a tiny 2-input → 16-hidden → 1-output MLP on spiral data.
Exports the trained model to ONNX with intermediate layer outputs captured.
"""
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper
from pathlib import Path


class SpiralMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        out = torch.sigmoid(self.fc3(h2))
        return out, h1, h2


def make_spiral(n=200, seed=42):
    rng = np.random.RandomState(seed)
    x = []
    y = []
    for i in range(n):
        r = i / n * 5
        t = 1.75 * i / n * 2 * np.pi + rng.randn() * 0.15
        x.append([r * np.cos(t), r * np.sin(t)])
        y.append(0)
        x.append([r * np.cos(t + np.pi), r * np.sin(t + np.pi)])
        y.append(1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)


def train_and_export(output_dir: str = "apps/studio/public/models"):
    torch.manual_seed(42)
    model = SpiralMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.BCELoss()

    x_data, y_data = make_spiral()

    # Collect intermediate outputs across training steps
    step_outputs = []

    for step_idx in range(24):
        model.train()
        optimizer.zero_grad()
        out, h1, h2 = model(x_data)
        loss = criterion(out, y_data)
        loss.backward()
        optimizer.step()

        step_outputs.append({
            "step": step_idx,
            "output": out.detach().numpy(),
            "h1": h1.detach().numpy(),
            "h2": h2.detach().numpy(),
            "loss": loss.item(),
        })

    # Save intermediate data as numpy for reuse
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path / "mlp_intermediates.npz",
        steps=np.array([s["step"] for s in step_outputs]),
        losses=np.array([s["loss"] for s in step_outputs]),
        h1=np.stack([s["h1"] for s in step_outputs]),
        h2=np.stack([s["h2"] for s in step_outputs]),
        outputs=np.stack([s["output"] for s in step_outputs]),
    )

    # Export to ONNX
    dummy = torch.randn(1, 2)
    torch.onnx.export(
        model,
        dummy,
        out_path / "spiral-mlp.onnx",
        input_names=["input"],
        output_names=["output", "hidden1", "hidden2"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}, "hidden1": {0: "batch"}, "hidden2": {0: "batch"}},
        opset_version=17,
    )

    # Verify
    sess = ort.InferenceSession(str(out_path / "spiral-mlp.onnx"))
    result = sess.run(None, {"input": dummy.numpy()})
    print(f"MLP ONNX exported: output shape={result[0].shape}, h1 shape={result[1].shape}, h2 shape={result[2].shape}")
    print(f"  Loss range: {step_outputs[0]['loss']:.4f} → {step_outputs[-1]['loss']:.4f}")
    print(f"  File size: {(out_path / 'spiral-mlp.onnx').stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    train_and_export()
