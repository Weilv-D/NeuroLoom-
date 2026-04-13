import torch
import numpy as np
from pathlib import Path
from transformers import Qwen2Config, Qwen2Model

def export_hf_transformer(output_dir="apps/studio/public/models"):
    print(f"Creating a Tiny Qwen2 (SOTA) model locally without downloading weights")
    
    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        output_attentions=True,
    )
    
    model = Qwen2Model(config)
    model.eval()
    
    input_ids = torch.randint(0, 1000, (1, 8))
    attention_mask = torch.ones((1, 8), dtype=torch.long)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    attentions = outputs.attentions[-1].numpy() if outputs.attentions else np.zeros((1, 1, 1, 1))
    last_hidden_state = outputs.last_hidden_state.numpy()
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        out_path / "tiny_qwen2_intermediates.npz",
        attn_weights=attentions,
        hidden_states=last_hidden_state
    )
    
    onnx_path = out_path / "tiny-qwen2.onnx"
    
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        opset_version=14,
        dynamic_axes=None
    )
    
    print(f"成功导出 Tiny Qwen2 到 {onnx_path}")
    
if __name__ == "__main__":
    export_hf_transformer()
