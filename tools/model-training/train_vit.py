import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time

class TinyViT(nn.Module):
    def __init__(self, image_size=28, patch_size=7, num_classes=3, dim=64, depth=2, heads=4):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 1 * patch_size ** 2
        
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.transformer = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(dim),
                'attn': nn.MultiheadAttention(dim, heads, batch_first=True),
                'norm2': nn.LayerNorm(dim),
                'mlp': nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
            }) for _ in range(depth)
        ])
        
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.patch_size = patch_size
        
    def forward(self, img):
        p = self.patch_size
        B, C, H, W = img.shape
        x = img.unfold(2, p, p).unfold(3, p, p).contiguous()
        x = x.view(B, C, -1, p, p).permute(0, 2, 1, 3, 4).reshape(B, -1, p*p*C)
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for i, layer in enumerate(self.transformer):
            norm_x = layer['norm1'](x)
            attn_out, attn_weights = layer['attn'](norm_x, norm_x, norm_x)
            x = x + attn_out
            x = x + layer['mlp'](layer['norm2'](x))
            
        return self.mlp_head(x[:, 0]), attn_weights

def export_vit(output_dir="apps/studio/public/models"):
    print("Creating and exporting Tiny SOTA ViT on GPU...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = TinyViT().to(device)
    model.eval()
    
    dummy = torch.randn(1, 1, 28, 28).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy)
        
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        logits, attn = model(dummy)
        
    torch.cuda.synchronize()
    print(f"Inference took {time.time() - start_time:.4f} seconds")
        
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        out_path / "vit_intermediates.npz",
        attn_weights=attn.cpu().numpy(),
        logits=logits.cpu().numpy()
    )
    
    onnx_path = out_path / "tiny-vit.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits", "attn_weights"],
        opset_version=14,
        do_constant_folding=True
    )
    print(f"ViT exported to {onnx_path}")

if __name__ == "__main__":
    export_vit()
