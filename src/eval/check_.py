# analyze_class_mels.py

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.train import AudioProcessor

AUDIO_DIR = Path("/scratch/vladimir_albrekht/projects/i2m/large_files/ILSVRC_images_10_class/audios_10_class")

def main():
    audio_processor = AudioProcessor(target_sr=24000, target_duration=3.0, device="cpu")
    
    # Load all mels
    class_mels = {}
    for class_dir in sorted(AUDIO_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        audio_file = class_dir / "description.wav"
        if audio_file.exists():
            mel = audio_processor.process_file(str(audio_file))
            if mel.shape[-1] < 280:
                mel = F.pad(mel, (0, 280 - mel.shape[-1]))
            class_mels[class_dir.name] = mel
    
    class_names = sorted(class_mels.keys())
    
    # Compute statistics
    print("=" * 60)
    print("MEL STATISTICS PER CLASS")
    print("=" * 60)
    print(f"{'Class':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Energy':>10}")
    print("-" * 70)
    
    for name in class_names:
        mel = class_mels[name].numpy().flatten()
        energy = np.sum(mel ** 2)
        print(f"{name:<25} {mel.mean():>8.2f} {mel.std():>8.2f} {mel.min():>8.2f} {mel.max():>8.2f} {energy:>10.1f}")
    
    # Compute pairwise distances
    print("\n" + "=" * 60)
    print("PAIRWISE MSE BETWEEN CLASS MELS")
    print("=" * 60)
    
    # Average mel (what model might converge to)
    all_mels = torch.stack([class_mels[n] for n in class_names])
    avg_mel = all_mels.mean(dim=0)
    
    print(f"\n{'Class':<25} {'MSE to Average':>15}")
    print("-" * 42)
    
    for name in class_names:
        mse_to_avg = F.mse_loss(class_mels[name], avg_mel).item()
        print(f"{name:<25} {mse_to_avg:>15.2f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, name in enumerate(class_names):
        mel_np = class_mels[name].squeeze().numpy()
        axes[i].imshow(mel_np, aspect='auto', origin='lower', cmap='viridis')
        axes[i].set_title(name[:15])
        axes[i].set_ylabel('Mel bins')
    
    plt.suptitle('Mel Spectrograms for All 10 Classes')
    plt.tight_layout()
    plt.savefig('class_mels_comparison.png', dpi=150)
    print(f"\n✓ Saved visualization to class_mels_comparison.png")

if __name__ == "__main__":
    main()
# ```

# ---

# ## Корень Проблемы и Решение

# ### Проблема
# ```
# Cross-attention не работает потому что:

# Training видит:
#   Image_A (tench)     + noise → target: denoise to mel_tench
#   Image_B (goldfish)  + noise → target: denoise to mel_goldfish
#   Image_C (shark)     + noise → target: denoise to mel_shark
  
# Но! Image features не содержат информации о классе для модели.
# Модель учит: "игнорируй image, просто выучи 'средний' mel"

# И "средний" mel оказывается ближе всего к electric_ray.