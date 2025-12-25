# compare_generation_vs_denoising.py

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
from vocos import Vocos
import torchaudio

from debug_training_code import MelAdapter, NoiseScheduler, SimpleDiT, AudioProcessor

# Config
MODEL_PATH = "/home/vladimir_albrekht/projects/img_to_spec/src/output/test_1/dit_checkpoint.pt"
DEVICE = "cuda"
OUTPUT_DIR = Path("/home/vladimir_albrekht/projects/img_to_spec/src/inference_output")

IMAGE_PATH = "/home/vladimir_albrekht/projects/img_to_spec/large_files/ILSVRC/images_10_class/000_tench/00000.jpg"
AUDIO_PATH = "/home/vladimir_albrekht/projects/img_to_spec/large_files/ILSVRC/audio_10_class/000_tench/description.wav"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessing
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def patchify_image(image, patch_size=16):
    C, H, W = image.shape
    P = patch_size
    x = image.reshape(C, H // P, P, W // P, P)
    x = x.permute(1, 3, 0, 2, 4)
    x = x.reshape((H // P) * (W // P), C * P * P)
    return x

def load_model(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleDiT(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Load everything
model = load_model(MODEL_PATH)
mel_adapter = MelAdapter(patch_freq=4, patch_time=8)
scheduler = NoiseScheduler(num_timesteps=1000, device=DEVICE)

# Load original mel
audio_processor = AudioProcessor(target_sr=24000, target_duration=3.0, device="cpu")
original_mel = audio_processor.process_file(AUDIO_PATH).unsqueeze(0)
if original_mel.shape[-1] < 280:
    original_mel = F.pad(original_mel, (0, 280 - original_mel.shape[-1]))
else:
    original_mel = original_mel[..., :280]
original_mel = original_mel.to(DEVICE)

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
image = image_transform(image)
image_patches = patchify_image(image).unsqueeze(0).to(DEVICE)

print("=" * 60)
print("COMPARISON: Generation from Scratch vs Single-Step Denoising")
print("=" * 60)

# ============== 1. Full Generation (from pure noise) ==============
@torch.no_grad()
def full_generation(model, scheduler, mel_adapter, image_patches, num_steps=100):
    latents = torch.randn((1, 875, 32), device=DEVICE)
    
    step_size = scheduler.num_timesteps // num_steps
    timesteps = list(range(scheduler.num_timesteps - 1, -1, -step_size))[:num_steps]
    if timesteps[-1] != 0:
        timesteps.append(0)
    
    for i in range(len(timesteps) - 1):
        t, prev_t = timesteps[i], timesteps[i + 1]
        t_tensor = torch.tensor([t], device=DEVICE, dtype=torch.long)
        predicted_noise = model(latents, image_patches, t_tensor)
        latents = scheduler.step_ddim(predicted_noise, t, latents, prev_timestep=prev_t)
    
    return mel_adapter.unpack(latents, H=100, W=280)

# ============== 2. Single-Step Denoising (from noisy original) ==============
@torch.no_grad()
def single_step_denoise(model, scheduler, mel_adapter, original_mel, image_patches, t_val=300):
    clean_patches = mel_adapter.pack(original_mel)
    t = torch.tensor([t_val], device=DEVICE)
    
    noisy_patches, true_noise = scheduler.add_noise(clean_patches, t)
    predicted_noise = model(noisy_patches, image_patches, t)
    
    # Recover
    alpha_cumprod = scheduler.alphas_cumprod[t_val]
    sqrt_alpha = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod)
    
    recovered = (noisy_patches - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha
    return mel_adapter.unpack(recovered, H=100, W=280)

# Generate both
print("Generating from scratch (100 steps)...")
generated_full = full_generation(model, scheduler, mel_adapter, image_patches, num_steps=100)

print("Single-step denoising from t=300...")
denoised_t300 = single_step_denoise(model, scheduler, mel_adapter, original_mel, image_patches, t_val=300)

print("Single-step denoising from t=500...")
denoised_t500 = single_step_denoise(model, scheduler, mel_adapter, original_mel, image_patches, t_val=500)

print("Single-step denoising from t=800...")
denoised_t800 = single_step_denoise(model, scheduler, mel_adapter, original_mel, image_patches, t_val=800)

# ============== Plot Comparison ==============
fig, axes = plt.subplots(5, 1, figsize=(14, 16))

mels = [
    (original_mel, "Original (Ground Truth)"),
    (generated_full, "Generated from Scratch (100 steps)"),
    (denoised_t300, "Single-Step Denoised (t=300, light noise)"),
    (denoised_t500, "Single-Step Denoised (t=500, medium noise)"),
    (denoised_t800, "Single-Step Denoised (t=800, heavy noise)"),
]

# Use same colorbar range for fair comparison
vmin = original_mel.min().item()
vmax = original_mel.max().item()

for i, (mel, title) in enumerate(mels):
    mel_np = mel.squeeze().cpu().numpy()
    mse = F.mse_loss(mel.squeeze(), original_mel.squeeze()).item()
    
    im = axes[i].imshow(mel_np, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[i].set_title(f"{title}\nMSE: {mse:.4f} | Range: [{mel_np.min():.2f}, {mel_np.max():.2f}]", fontsize=10)
    axes[i].set_ylabel('Mel Bins')
    plt.colorbar(im, ax=axes[i])

axes[-1].set_xlabel('Time Frames')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "generation_vs_denoising.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved to {OUTPUT_DIR / 'generation_vs_denoising.png'}")

# ============== Save Audio Files ==============
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(DEVICE)
vocos.eval()

audio_files = [
    (original_mel, "original.wav"),
    (generated_full, "generated_full.wav"),
    (denoised_t300, "denoised_t300.wav"),
    (denoised_t500, "denoised_t500.wav"),
    (denoised_t800, "denoised_t800.wav"),
]

print("\nSaving audio files...")
for mel, filename in audio_files:
    with torch.no_grad():
        audio = vocos.decode(mel.squeeze(1))
    torchaudio.save(str(OUTPUT_DIR / filename), audio.cpu(), 24000)
    print(f"  ✓ {filename}")

# ============== Print Summary ==============
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for mel, title in mels:
    mse = F.mse_loss(mel.squeeze(), original_mel.squeeze()).item()
    mel_np = mel.squeeze().cpu().numpy()
    print(f"{title[:40]:40s} | MSE: {mse:8.4f} | Mean: {mel_np.mean():6.2f}")

@torch.no_grad()
def run_inference_cfg(
    model, scheduler, mel_adapter, image_patches, 
    cfg_scale=3.0,  # Попробуй 1.0, 3.0, 5.0, 7.5
    num_steps=100,
    device="cuda"
):
    model.eval()
    latents = torch.randn((1, 875, 32), device=device)
    
    # Unconditional: нулевые патчи
    uncond_patches = torch.zeros_like(image_patches)
    
    step_size = scheduler.num_timesteps // num_steps
    timesteps = list(range(scheduler.num_timesteps - 1, -1, -step_size))[:num_steps]
    if timesteps[-1] != 0:
        timesteps.append(0)
    
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        prev_t = timesteps[i + 1]
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        
        # Conditional prediction
        noise_cond = model(latents, image_patches, t_tensor)
        
        # Unconditional prediction
        noise_uncond = model(latents, uncond_patches, t_tensor)
        
        # CFG formula
        predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        
        # DDIM step
        latents = scheduler.step_ddim(predicted_noise, t, latents, prev_timestep=prev_t)
        
        if i % 20 == 0:
            print(f"Step {i}: t={t}, range=[{latents.min():.2f}, {latents.max():.2f}]")
    
    return mel_adapter.unpack(latents, H=100, W=280)

# Test different CFG scales
for cfg in [1.0, 3.0, 5.0, 7.5]:
    print(f"\n=== CFG Scale: {cfg} ===")
    generated = run_inference_cfg(model, scheduler, mel_adapter, image_patches, cfg_scale=cfg)
    mse = F.mse_loss(generated.squeeze(), original_mel.squeeze()).item()
    print(f"MSE: {mse:.4f}")