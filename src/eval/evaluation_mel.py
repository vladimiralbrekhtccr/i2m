# evaluate_with_confusion.py
"""
Evaluation script for Image-to-Audio DiT model.

Compares generated mels against the ACTUAL eval audio files (from pairs_eval.json),
not some separate reference audios.
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random
import numpy as np
from collections import defaultdict

from src.train import MelAdapter, NoiseScheduler, SimpleDiT, AudioProcessor

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "/scratch/vladimir_albrekht/projects/i2m/src/output/dit-h768-l12-cfg0.1-20251225_040630/checkpoint_epoch50.pt"
EVAL_JSON = "/scratch/vladimir_albrekht/projects/i2m/large_files/ILSVRC_images_10_class/eval_data_10_class_10_percent/pairs_eval.json"

DEVICE = "cuda"
NUM_INFERENCE_STEPS = 50
CFG_SCALE = 1.0

SAMPLES_PER_CLASS = 10
RANDOM_SEED = 42

# ============================================================
# HELPER FUNCTIONS
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def patchify_image(image: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    C, H, W = image.shape
    P = patch_size
    x = image.reshape(C, H // P, P, W // P, P)
    x = x.permute(1, 3, 0, 2, 4)
    x = x.reshape((H // P) * (W // P), C * P * P)
    return x

def load_model(checkpoint_path: str, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model = SimpleDiT(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('epoch', '?'), config, checkpoint.get('val_loss', '?')

def sample_per_class(eval_pairs: list, samples_per_class: int, seed: int = 42) -> list:
    random.seed(seed)
    by_class = defaultdict(list)
    for pair in eval_pairs:
        by_class[pair['class']].append(pair)
    
    sampled = []
    for class_name in sorted(by_class.keys()):
        n = min(samples_per_class, len(by_class[class_name]))
        sampled.extend(random.sample(by_class[class_name], n))
    return sampled

# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate_mel(model, scheduler, mel_adapter, image_patches, 
                 num_steps=50, cfg_scale=1.0, device="cuda"):
    latents = torch.randn((1, 875, 32), device=device)
    uncond_patches = torch.zeros_like(image_patches) if cfg_scale != 1.0 else None
    
    step_size = scheduler.num_timesteps // num_steps
    timesteps = list(range(scheduler.num_timesteps - 1, -1, -step_size))[:num_steps]
    if timesteps[-1] != 0:
        timesteps.append(0)
    
    for i in range(len(timesteps) - 1):
        t, prev_t = timesteps[i], timesteps[i + 1]
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        
        if cfg_scale != 1.0 and uncond_patches is not None:
            noise_cond = model(latents, image_patches, t_tensor)
            noise_uncond = model(latents, uncond_patches, t_tensor)
            predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            predicted_noise = model(latents, image_patches, t_tensor)
        
        latents = scheduler.step_ddim(predicted_noise, t, latents, prev_timestep=prev_t)
    
    return mel_adapter.unpack(latents, H=100, W=280)

# ============================================================
# REFERENCE MEL EXTRACTION FROM EVAL DATA
# ============================================================

def extract_class_reference_mels(eval_pairs: list, audio_processor, device: str) -> dict:
    """
    Extract ONE reference mel per class from eval data.
    Prefer samples with augmentation='none' (original audio).
    
    Returns: {class_name: mel_tensor}
    """
    # Group by class
    by_class = defaultdict(list)
    for pair in eval_pairs:
        by_class[pair['class']].append(pair)
    
    class_mels = {}
    
    print("\n  Extracting reference mels from eval data:")
    for class_name in sorted(by_class.keys()):
        class_pairs = by_class[class_name]
        
        # Try to find 'none' augmentation first (original audio)
        none_pairs = [p for p in class_pairs if p.get('augmentation', {}).get('type') == 'none']
        
        if none_pairs:
            # Use first 'none' augmentation sample
            chosen = none_pairs[0]
            aug_info = "none (original)"
        else:
            # Fallback to any sample
            chosen = class_pairs[0]
            aug_info = chosen.get('augmentation', {}).get('type', 'unknown')
        
        # Load mel
        mel = audio_processor.process_file(chosen['audio']).unsqueeze(0)
        if mel.shape[-1] < 280:
            mel = F.pad(mel, (0, 280 - mel.shape[-1]))
        else:
            mel = mel[..., :280]
        
        class_mels[class_name] = mel.to(device)
        
        speaker = chosen.get('speaker', 'unknown')
        print(f"    {class_name}: speaker={speaker}, aug={aug_info}")
    
    return class_mels

def find_closest_class(generated_mel: torch.Tensor, class_mels: dict) -> tuple:
    mse_dict = {}
    for class_name, class_mel in class_mels.items():
        mse = F.mse_loss(generated_mel, class_mel).item()
        mse_dict[class_name] = mse
    
    predicted_class = min(mse_dict, key=mse_dict.get)
    return predicted_class, mse_dict

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("IMAGE-TO-AUDIO EVALUATION")
    print("=" * 70)
    
    # -------------------- Load Eval Data --------------------
    print("\n[1/4] Loading evaluation data...")
    
    with open(EVAL_JSON, 'r') as f:
        all_eval_pairs = json.load(f)
    
    print(f"  Total pairs: {len(all_eval_pairs)}")
    
    # Show augmentation distribution
    aug_counts = defaultdict(int)
    for p in all_eval_pairs:
        aug_type = p.get('augmentation', {}).get('type', 'unknown')
        aug_counts[aug_type] += 1
    
    print(f"\n  Augmentation distribution in eval:")
    for aug_type, count in sorted(aug_counts.items()):
        print(f"    {aug_type}: {count}")
    
    # Sample subset
    if SAMPLES_PER_CLASS:
        eval_pairs = sample_per_class(all_eval_pairs, SAMPLES_PER_CLASS, RANDOM_SEED)
        print(f"\n  Sampled {SAMPLES_PER_CLASS} per class = {len(eval_pairs)} total")
    else:
        eval_pairs = all_eval_pairs
    
    # -------------------- Load Model --------------------
    print("\n[2/4] Loading model...")
    
    model, epoch, config, val_loss = load_model(MODEL_PATH, DEVICE)
    print(f"  Checkpoint: {Path(MODEL_PATH).name}")
    print(f"  Epoch: {epoch}, Val Loss: {val_loss}")
    print(f"  Hidden: {config['hidden_size']}, Layers: {config['num_layers']}")
    
    # -------------------- Extract Reference Mels --------------------
    print("\n[3/4] Extracting reference mels from eval data...")
    
    audio_processor = AudioProcessor(target_sr=24000, target_duration=3.0, device="cpu")
    mel_adapter = MelAdapter(patch_freq=4, patch_time=8)
    scheduler = NoiseScheduler(num_timesteps=1000, device=DEVICE)
    
    # Get reference mels from eval data (prefer 'none' augmentation)
    class_mels = extract_class_reference_mels(all_eval_pairs, audio_processor, DEVICE)
    class_names = sorted(class_mels.keys())
    
    # -------------------- Run Evaluation --------------------
    print(f"\n[4/4] Evaluating (steps={NUM_INFERENCE_STEPS}, cfg={CFG_SCALE})...")
    
    n_classes = len(class_names)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    correct = 0
    total = 0
    
    for pair in tqdm(eval_pairs, desc="  Generating"):
        true_class = pair['class']
        
        # Load image
        image = Image.open(pair['image']).convert('RGB')
        image = image_transform(image)
        image_patches = patchify_image(image).unsqueeze(0).to(DEVICE)
        
        # Generate mel
        generated_mel = generate_mel(
            model, scheduler, mel_adapter, image_patches,
            num_steps=NUM_INFERENCE_STEPS, cfg_scale=CFG_SCALE, device=DEVICE
        )
        
        # Classify
        predicted_class, _ = find_closest_class(generated_mel, class_mels)
        
        # Update stats
        true_idx = class_to_idx[true_class]
        pred_idx = class_to_idx[predicted_class]
        confusion[true_idx, pred_idx] += 1
        
        if predicted_class == true_class:
            correct += 1
        total += 1
    
    # -------------------- Results --------------------
    accuracy = correct / total * 100
    
    print("\n" + "=" * 70)
    print(f"ACCURACY: {accuracy:.1f}% ({correct}/{total})")
    print("=" * 70)
    
    # Confusion matrix
    print("\nCONFUSION MATRIX:")
    short_names = [n.split('_')[-1][:6] for n in class_names]
    
    header = "True\\Pred".ljust(15) + "".join([s.center(7) for s in short_names])
    print(header)
    print("-" * len(header))
    
    for i, name in enumerate(class_names):
        short = name.split('_')[-1][:12]
        row = short.ljust(15)
        for j in range(n_classes):
            val = confusion[i, j]
            row += (f"[{val:2d}]" if i == j else f" {val:2d} ").center(7)
        print(row)
    
    # Per-class
    print("\nPER-CLASS:")
    for i, name in enumerate(class_names):
        total_i = confusion[i].sum()
        correct_i = confusion[i, i]
        acc_i = correct_i / total_i * 100 if total_i > 0 else 0
        print(f"  {name:<25} {correct_i:2d}/{total_i:2d} = {acc_i:5.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"  Random baseline: {100/n_classes:.1f}%")
    print(f"  Model accuracy:  {accuracy:.1f}%")
    print(f"  Improvement:     {accuracy - 100/n_classes:+.1f}%")
    
    if accuracy <= 15:
        print("\n  ⚠️ Mode collapse - model ignores image condition")
    elif accuracy <= 30:
        print("\n  ⚠️ Weak learning - try more epochs or higher CFG")
    else:
        print("\n  ✅ Model differentiates classes!")

if __name__ == "__main__":
    main()