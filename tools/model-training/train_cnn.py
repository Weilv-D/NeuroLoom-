"""
Train a tiny 1-channel 28x28 CNN on synthetic data ( Fashion-MNIST-like patterns).
Exports to ONNX with feature map intermediates.
"""
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from pathlib import Path


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 7 * 7, 3)

    def forward(self, x):
        c1 = torch.relu(self.conv1(x))
        p1 = self.pool1(c1)
        c2 = torch.relu(self.conv2(p1))
        p2 = self.pool2(c2)
        flat = p2.view(p2.size(0), -1)
        logits = self.fc(flat)
        return logits, c1, p1, c2, p2


def make_synthetic_images(n=100, seed=42):
    rng = np.random.RandomState(seed)
    # Generate simple pattern images: 3 classes based on stripe direction
    x = np.zeros((n, 1, 28, 28), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cls = i % 3
        y[i] = cls
        base = rng.randn(1, 28, 28).astype(np.float32) * 0.1
        if cls == 0:  # horizontal stripes
            base[0, ::3, :] += 0.8
        elif cls == 1:  # vertical stripes
            base[0, :, ::3] += 0.8
        else:  # checkerboard
            base[0, ::2, ::2] += 0.8
        x[i] = base
    return torch.tensor(x), torch.tensor(y)


def train_and_export(output_dir: str = "apps/studio/public/models"):
    torch.manual_seed(42)
    model = TinyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    x_data, y_data = make_synthetic_images()

    step_outputs = []

    for step_idx in range(20):
        model.train()
        optimizer.zero_grad()
        logits, c1, p1, c2, p2 = model(x_data)
        loss = criterion(logits, y_data)
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(1) == y_data).float().mean().item()
        step_outputs.append({
            "step": step_idx,
            "loss": loss.item(),
            "accuracy": acc,
            "conv1_mean": c1.abs().mean().item(),
            "conv2_mean": c2.abs().mean().item(),
        })

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save intermediates
    with torch.no_grad():
        model.eval()
        logits, c1, p1, c2, p2 = model(x_data[:10])
        np.savez(
            out_path / "cnn_intermediates.npz",
            conv1_feature=c1.numpy(),
            pool1_feature=p1.numpy(),
            conv2_feature=c2.numpy(),
            pool2_feature=p2.numpy(),
            losses=np.array([s["loss"] for s in step_outputs]),
            accuracies=np.array([s["accuracy"] for s in step_outputs]),
        )

    # Export to ONNX
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy,
        out_path / "fashion-cnn.onnx",
        input_names=["input"],
        output_names=["logits", "conv1", "pool1", "conv2", "pool2"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # Verify
    sess = ort.InferenceSession(str(out_path / "fashion-cnn.onnx"))
    result = sess.run(None, {"input": dummy.numpy()})
    print(f"CNN ONNX exported: logits={result[0].shape}, conv1={result[1].shape}, conv2={result[3].shape}")
    print(f"  Loss: {step_outputs[0]['loss']:.4f} → {step_outputs[-1]['loss']:.4f}, Acc: {step_outputs[-1]['accuracy']:.2%}")
    print(f"  File size: {(out_path / 'fashion-cnn.onnx').stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    train_and_export()
