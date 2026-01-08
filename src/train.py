import os
from dotenv import load_dotenv
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


import math
import torch
import torchaudio
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import wandb


from vocos import Vocos
from pathlib import Path

from datetime import datetime
import matplotlib.pyplot as plt

load_dotenv()

### Utils:
def save_loss_log(losses: list, save_path: str):
    """Save losses to text file."""
    with open(f"{save_path}/loss.log", 'w') as f:
        # Header
        f.write("type,epoch,batch,value\n")
        
        for entry in losses:
            epoch = entry.get('epoch', '')
            
            if 'loss' in entry:
                # Batch loss entry
                batch = entry.get('batch', '')
                f.write(f"batch,{epoch},{batch},{entry['loss']:.6f}\n")
            
            if 'avg_loss' in entry:
                # Epoch average entry
                f.write(f"epoch_avg,{epoch},,{entry['avg_loss']:.6f}\n")
    
    print(f"✓ Saved loss log to {save_path}/loss.log")


def plot_loss(losses: list, save_path: str):
    """Plot and save loss curves."""
    # Extract batch losses
    all_losses = [entry['loss'] for entry in losses if 'loss' in entry]
    
    # Extract epoch average losses  
    epoch_losses = [entry['avg_loss'] for entry in losses if 'avg_loss' in entry]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: All batch losses
    if all_losses:
        axes[0].plot(all_losses, alpha=0.7, linewidth=0.5)
        axes[0].set_xlabel('Batch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss (All Batches)')
        if min(all_losses) > 0:  # Only log scale if all positive
            axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No batch data', ha='center', va='center')
    
    # Plot 2: Epoch average losses
    if epoch_losses:
        axes[1].plot(epoch_losses, marker='o', markersize=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Average Loss')
        axes[1].set_title('Training Loss (Epoch Average)')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No epoch data', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/loss_plot.png", dpi=150)
    plt.close()
    print(f"✓ Saved loss plot to {save_path}/loss_plot.png")


### AudioProcessor Processing
class AudioProcessor:
    """
    Handles audio loading, resampling, and Mel spectogram extraction.
    Uses Vocos for consistent Mel generation and reconstruction.
    """
    def __init__(
        self,
        target_sr: int = 24000,
        target_duration: float = 3.0,
        device: str = "cpu"
    ):
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)
        self.device = device

        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.vocos.eval()
        self.vocos.to(device)

        self._resamplers = {}

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        """Cache resemplers to avoid recreating time"""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.target_sr
            )
        return self._resamplers[orig_sr]

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio using soundfile instead of torchaudio."""
        # Читаем через soundfile
        data, original_sr = sf.read(audio_path, dtype='float32')
        
        # Конвертируем в torch tensor
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)  # [1, samples]
        else:
            # Stereo -> mono
            waveform = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
        # ??? maybe         assert sr == self.target_sr, f"Expected {self.target_sr} Hz, got {sr}"
        # Resample если нужно
        if original_sr != self.target_sr:
            resampler = self._get_resampler(original_sr)
            waveform = resampler(waveform)
        
        # Pad/trim
        if waveform.shape[1] < self.target_samples:
            waveform = F.pad(waveform, (0, self.target_samples - waveform.shape[1]))
        elif waveform.shape[1] > self.target_samples:
            waveform = waveform[:, :self.target_samples]
        
        return waveform
    
    def waveform_to_mel(self, waveform:torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel using Vocos.
        
        Pipe:
            Input: [1, samples] or [B, 1, samples]
            Output: [1, 100, frames] or [B,100, frames]
        """
        with torch.no_grad():
            if waveform.dim() == 3:
                waveform = waveform.squeeze(1)
            elif waveform.dim() == 2 and waveform.shape[0] == 1:
                pass
            waveform = waveform.to(self.device)
            mel_spec = self.vocos.feature_extractor(waveform)
        return mel_spec
    
    def mel_to_waveform(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct waveform from Mel spectogram using Vocos

        Pipe:
            Input: [B, 100, frames] or [100, frames]
            Output: [B, samples] or [samples]
        """
        with torch.no_grad():
            mel_spec = mel_spec.to(self.device)
            waveform = self.vocos.decode(mel_spec)

        return waveform
    
    def process_file(self, audio_path: str) -> torch.Tensor:
        """
        Full pipeline: audio file -> Mel Spec.
        Pipe:
            Input: Path to audio file
            Output: [1, 100, frames] Mel spectogram
        """
        waveform = self.load_audio(audio_path)
        mel_spec = self.waveform_to_mel(waveform)

        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)
        
        return mel_spec
    

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Optional
import json

class ImageAudioDataset(Dataset):
    """
    Dataset that pairs images with audio files.

    Expected directory structure:
    data/
    ├── pairs.json  (or infer from filenames)
    ├── images/
    │   ├── 001.png
    │   ├── 002.png
    │   └── ...
    └── audio/
        ├── 001.wav
        ├── 002.wav
        └── ...
    """
    def __init__(
        self,
        data_dir: str,
        audio_processor: AudioProcessor,
        image_size: int = 512,
        patch_size: int = 16, 
        pairs_file: Optional[str] = "pairs.json",
        device: str = "cuda"
    ):
        self.data_dir = Path(data_dir)
        self.audio_processor = audio_processor
        self.device = device
        self.patch_size = patch_size

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # [0, 255] -> [0, 1], HWC -> CHW
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5]) # [-1. 1]
        ])
        
        if pairs_file and (self.data_dir / pairs_file).exists():
            with open(self.data_dir / pairs_file, 'r') as f:
                self.pairs = json.load(f)
        else:
            raise AttributeError(f"Please specify data_dir properly! Current value [{self.data_dir}]")
    
    def _patchify_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patches for transformer input.

        Input: [3, 512, 512]
        Output: [1024, 768] (32 * 32 patches, each 16*16*3)
        """
        C, H, W =image.shape
        P = self.patch_size

        # [3, 512, 512] -> [3, 32, 16, 32, 16]
        x = image.reshape(C, H // P, P, W // P, P)

        # -> [32, 32, 3, 16, 16]
        x = x.permute(1, 3, 0, 2, 4)

        # -> [1024, 768]
        x = x.reshape((H // P) * (W // P), C * P * P)

        return x

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mel_spec: [1, 100, 280] - Mel spec.
            image_patches: [1024, 768]
        """
        pair = self.pairs[idx]

        image = Image.open(pair["image"]).convert("RGB")
        image = self.image_transform(image) # [3, 512, 512]
        image_patches = self._patchify_image(image) # [1024, 768]

        mel_spec = self.audio_processor.process_file(pair["audio"])

        target_frames = 280 
        if mel_spec.shape[-1] < target_frames:
            mel_spec = F.pad(mel_spec, (0, target_frames - mel_spec.shape[-1]))
        elif mel_spec.shape[-1] > target_frames:
            mel_spec = mel_spec[..., :target_frames]

        return mel_spec, image_patches

def create_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    device: str = "cuda"
) -> DataLoader:
    """Factory function to create ready to use dataloader."""
    
    audio_processor = AudioProcessor(
        target_sr=24000,
        target_duration=3.0,
        device="cpu"
    )
    
    dataset = ImageAudioDataset(
        data_dir=data_dir,
        audio_processor=audio_processor,
        image_size=512,
        patch_size=16,
        device="cpu"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader



### MEL_to_Patches block
class MelAdapter:
    def __init__(self, patch_freq=4, patch_time=8):
        #TODO: because we have as input not equally sized MEL_spectogram[1, 100, 256] it's hard to find a good spot with single patch_size. We use 2 patches one for bins and one for freq. I'm curious if that a good idea or not. One option is to try using VAE?
        """        
        patch_freq: patch_size along (100 bins)\n
        patch_time: patch_size along (256 frames)
        """
        self.pf = patch_freq
        self.pt = patch_time
    
    def pack(self, mel_spec):
        """
        Input: [B, 1, 100, 256] (Mel)
        Output: [B, seq_len, patch_size] (DiT input)
        """
        if mel_spec.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B,C,H,W], got {mel_spec.dim()}D")

        B, C, H, W = mel_spec.shape
        PF = self.pf
        PT = self.pt

        # [B, 1, 100, 256] -> [B, C, 50, 2, 141, 2]
        x = mel_spec.reshape(B, C, H//PF, PF, W//PT, PT)
        # [1, 1, 50, 2, 141, 2] -> [1, 50, 141, 1, 2, 2]
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, (H//PF) * (W//PT), C*PF*PT)
        return x
    
    def unpack(self, patches, H=100, W=280):
        """
        Input: [Bs, seq_len, C*PF*PT]
        Output: [B, 1, 100, 256] (Mel)
        """
        if patches.dim() != 3:
            raise ValueError(f"Expected 3D Tensor [B,seq,dim], got{patches.dim()}D")
        
        B = patches.shape[0]
        C = 1 
        PF = self.pf
        PT = self.pt

        x = patches.reshape(B, H//PF, W//PT, C, PF, PT)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(B, C, H, W)
        return x

### Noise Scheduler or Anime Destroyer block
class NoiseScheduler:
    def __init__(self, num_timesteps=1000, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device 

        # std linear schedule #TODO: later on implement 'FlowMatchEulerDiscreteScheduler'
        beta_start, beta_end = 0.0001, 0.02
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas 
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, clean_patches, t):
        """
        Input: Clean Patches from adapter [bs, seq_len, 4]
        Output: Noisy Patches [bs, seq_len, 4], The Noise [bs, seq_len, 4]
        """
        noise = torch.randn_like(clean_patches).to(self.device)

        # [500] --> sqrt(0.42%) --> 0.6481 and we convert it to [1,1,1] where first value is == 0.6481 
        sqrt_alpha = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1)
        
        # same but 1.0 - 0.42 = 0.58 --> sqrt(0.58%) --> 0.7616 and we convert it to [1,1,1] where first value is == 0.7616
        sqrt_one_minus_alpha = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1)
        
        # so it will be ([1,1,1] * clean_patches) + ([1,1,1] * noise)
        noisy_patches = (sqrt_alpha * clean_patches) + (sqrt_one_minus_alpha * noise)

        return noisy_patches, noise
    
    def step(self, model_output, timestep, sample):
        """
        INFERENCE: Heals noisy data (One step: t -> 1-t):
            model_output: The predicted noise (output from DiT)
            timestep: current t (integer)
            sample: current noisy image (x_t)
        """
        t = timestep 
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]

        #TODO: write yourself. DDPM formula: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * noise_pred)
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)

        prev_sample_mean = coef1 * (sample - coef2 * model_output)

        if t > 0:
            noise = torch.randn_like(sample)
            sigma_t = torch.sqrt(beta_t)
            prev_sample = prev_sample_mean + sigma_t * noise
        else:
            prev_sample = prev_sample_mean

        return prev_sample

    def get_timesteps(self, num_inference_steps):
        """
        Get evenly spaced timesteps for inference.

        Args:
            num_inference_steps: Number of steps (e.g., 50, 100)

        Returns:
            List of timesteps [999, 979, 959, ...] for num_steps=50
        """
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = list(range(self.num_timesteps - 1, -1, -step_ratio))
        return timesteps[:num_inference_steps]

    def step_ddim(self, model_output, timestep, sample, prev_timestep, eta=0.0):
        """
        DDIM step: x_t -> x_{prev_t}
        
        Args:
            model_output: predicted noise
            timestep: current t
            sample: current x_t
            prev_timestep: target t (must be provided!)
            eta: noise scale (0 = deterministic)
        """
        t = timestep
        prev_t = prev_timestep
        
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=self.device)
        
        # Predict x0 from x_t and noise prediction
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        # x0 = (x_t - sqrt(1-alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        x0_pred = (sample - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t
        
        # Clip x0 for stability
        x0_pred = torch.clamp(x0_pred, -10, 10)
        
        # Direction pointing to x_t
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)
        
        # DDIM: x_{t-1} = sqrt(alpha_bar_{t-1}) * x0 + sqrt(1 - alpha_bar_{t-1}) * noise_pred
        pred_sample = sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * model_output
        
        # Optional: add noise for stochastic sampling
        if eta > 0 and prev_t > 0:
            variance = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            noise = torch.randn_like(sample)
            pred_sample = pred_sample + variance * noise
        
        return pred_sample

### DiT block
# Here are the 4 ingredients we need to write:
# Timestep Embedder: Turns the integer 500 into a vector.
# AdaLayerNorm: The "Fuel Injector" that injects Time into the layers.
# DiT Block: The actual brain (Attention + MLP).
# Final Model: Puts it all together.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSON = PROJECT_ROOT / "train.json"
EVAL_JSON  = PROJECT_ROOT / "eval.json"

class TimestepEmbedder(nn.Module):
    """
    Input: Time integer (e.g., 500)
    Output: Vector of size `hidden_size`
    Logic: Uses Sine/Cosine waves so the model understands `order`.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half_dim = self.frequency_embedding_size // 2 # 128
        emb = math.log(10000) / (half_dim - 1) # 
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        # [128 sin + 128 cos]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
        t_emb = self.mlp(emb)
        return t_emb
    
class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Norm.
    The magic: it looks at the `TIME` and decides how to Scale and Shift the data.
    If t=1000 (Noise), it might scale everything down.
    If t=0 (Clean), it might scale everything up.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.silu = nn.SiLU()

        self.linear = nn.Linear(hidden_size, 2 * hidden_size, bias=True)

    def forward(self, x, t_emb):
        emb = self.linear(self.silu(t_emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        x = self.norm(x)
        x = x * (1 + scale) + shift
        return x 


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

        # self-attention for mel
        self.norm1 = AdaLayerNorm(hidden_size)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # cross-attention for condition-image
        self.norm2 = AdaLayerNorm(hidden_size)
        self.attn2 = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.norm3 = AdaLayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4,  hidden_size)
        )
    
    def forward(self, x, context, t_emb):
        """
        x: Mel Features [batch, seq_len, hidden]
        context: Image Features [batch, ???, hidden]
        t_emb: Time-step embedding [batch, hidden]
        """
        ### Self-attention
        x_norm = self.norm1(x, t_emb)
        attn_output, _ = self.attn1(x_norm, x_norm, x_norm)
        x = x + attn_output # Residual
        
        ### Cross-attention
        x_norm = self.norm2(x, t_emb)
        attn_output, _ = self.attn2(query=x_norm, key=context, value=context)
        x = x + attn_output # Residual

        ### MLP
        x_norm = self.norm3(x, t_emb)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output # Residual

        return x 

class SimpleDiT(nn.Module):
    def __init__(
            self,
            mel_input_dim=32, # patch_freq(4) * patch_time(8) * channels(1)
            img_input_dim=768, # 16 * 16 * 3
            hidden_size=512,
            mel_seq_len=875, # (100/4) * (280/8) = 25 * 35
            img_seq_len=1024, # (512/16) * (512/16) = 32 * 32
            num_layers=6,
            num_heads=8):
        super().__init__()

        self.mel_seq_len = mel_seq_len
        self.img_seq_len = img_seq_len

        ### Adapters for Mel and Image 
        self.mel_proj = nn.Linear(mel_input_dim, hidden_size)
        self.img_proj = nn.Linear(img_input_dim, hidden_size)

        ### Time Embedder
        self.time_embed = TimestepEmbedder(hidden_size)

        ### Pos Embeddings
        self.pos_embed_mel = nn.Parameter(torch.zeros(1, mel_seq_len, hidden_size))
        self.pos_embed_img = nn.Parameter(torch.zeros(1, img_seq_len, hidden_size))

        ### Stack of Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])

        ### Output Projection
        self.final_norm = AdaLayerNorm(hidden_size)
        self.final_linear = nn.Linear(hidden_size, mel_input_dim)

    def forward(self, x, context, t):
        """
        x: Noisy Mel Patches [B, 875, 32]
        context: Image Patches [B, 1024, 768]
        t: Timesteps [B]

        Returns: Predicted Noise [B, 875, 32]
        """

        ### Input preparaiton
        x = self.mel_proj(x) # [B, 875, 32] -> [B, 875, 512]
        context = self.img_proj(context) # [B, 1024,768] -> [B, 1024, 512]

        x = x + self.pos_embed_mel
        context = context + self.pos_embed_img

        t_emb = self.time_embed(t) #[B] -> [B, 512]

        for block in self.blocks:
            x = block(x, context, t_emb)
        
        x = self.final_norm(x, t_emb)
        prediction = self.final_linear(x) # [B, 875, 512] -> [B, 875, 32]

        return prediction

def setup_distributed():
    """Initialize distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="gloo")
    
    return local_rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

@torch.no_grad()
def validate(model, val_dataloader, mel_adapter, noise_scheduler, device):
    """Run validation and return metrics."""
    model.eval()
    
    total_loss = 0
    total_samples = 0
    
    # Losses at specific timesteps
    timestep_losses = {100: [], 300: [], 500: [], 800: []}
    
    for mel_spec, image_patches in val_dataloader:
        mel_spec = mel_spec.to(device)
        image_patches = image_patches.to(device)
        batch_size = mel_spec.shape[0]
        
        clean_patches = mel_adapter.pack(mel_spec)
        
        # Random timesteps for general loss
        t = torch.randint(0, 1000, (batch_size,), device=device)
        noisy_patches, noise = noise_scheduler.add_noise(clean_patches, t)
        predicted_noise = model(noisy_patches, image_patches, t)
        loss = F.mse_loss(predicted_noise, noise)
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Losses at specific timesteps (for first sample in batch)
        for t_val in timestep_losses.keys():
            t_fixed = torch.tensor([t_val], device=device)
            noisy, noise = noise_scheduler.add_noise(clean_patches[:1], t_fixed)
            pred = model(noisy, image_patches[:1], t_fixed)
            timestep_losses[t_val].append(F.mse_loss(pred, noise).item())
    
    model.train()
    
    avg_loss = total_loss / total_samples
    avg_timestep_losses = {k: sum(v) / len(v) for k, v in timestep_losses.items()}
    
    return {
        'val_loss': avg_loss,
        'val_loss_t100': avg_timestep_losses[100],
        'val_loss_t300': avg_timestep_losses[300],
        'val_loss_t500': avg_timestep_losses[500],
        'val_loss_t800': avg_timestep_losses[800],
    }


def train():
    ### DISTRIBUTED SETUP
    local_rank, world_size = setup_distributed()
    is_main = (local_rank == 0)
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    ### CONF
    DEVICE = f'cuda:{local_rank}'
    BATCH_SIZE = 32  # Per GPU
    GRAD_ACCUM_STEPS = 1
    NUM_EPOCHS = 500
    SAVE_EPOCHS_STEP = 50
    LR = 1e-4
    TRAIN_DATA_PATH = TRAIN_JSON

    # Validation config
    VAL_EVERY_N_EPOCHS = 10
    VAL_DATA_PATH = EVAL_JSON

    ### MODEL PARAMS:
    HIDDEN_SIZE = 768
    NUM_LAYERS = 12
    NUM_HEADS = 12
    #### MEL
    PATCH_FREQ_H = 4
    PATCH_TIME_W = 8

    ### TRAINING ADJUSTMENTS
    CONDITION_DROPOUT = 0.1 # TODO: read about it
    


    # Wandb config
    USE_WANDB = False
    WANDB_PROJECT = "i2m-diffusion"
    
    WANDB_RUN_NAME = f"20_a_dit-h{HIDDEN_SIZE}-l{NUM_LAYERS}-cfg{CONDITION_DROPOUT}-{start_time}"
    SAVE_PATH = f"./output/20_a_dit-h{HIDDEN_SIZE}-l{NUM_LAYERS}-cfg{CONDITION_DROPOUT}-{start_time}"

    if is_main:
        Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        if USE_WANDB:
            wandb.init(
                project=WANDB_PROJECT,
                name=WANDB_RUN_NAME,
                config={
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "num_heads": NUM_HEADS,
                    "batch_size": BATCH_SIZE,
                    "world_size": world_size,
                    "effective_batch_size": BATCH_SIZE * world_size,
                    "learning_rate": LR,
                    "num_epochs": NUM_EPOCHS,
                    "condition_dropout": CONDITION_DROPOUT,
                }
            )

    ### DATALOADERS
    audio_processor = AudioProcessor(
        target_sr=24000,
        target_duration=3.0,
        device="cpu"
    )
    
    # Train dataset
    train_dataset = ImageAudioDataset(
        data_dir=TRAIN_DATA_PATH,
        audio_processor=audio_processor,
        image_size=512,
        patch_size=16,
        pairs_file="pairs.json",
        device="cpu"
    )
    
    # Validation dataset
    val_dataset = ImageAudioDataset(
        data_dir=VAL_DATA_PATH,
        audio_processor=audio_processor,
        image_size=512,
        patch_size=16,
        pairs_file="pairs_eval.json",
        device="cpu"
    )
    
    if is_main:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    # Distributed sampler for training
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Validation dataloader (no distributed sampler, run only on main)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    ### MODEL
    mel_adapter = MelAdapter(patch_freq=PATCH_FREQ_H, patch_time=PATCH_TIME_W)
    noise_scheduler = NoiseScheduler(num_timesteps=1000, device=DEVICE)

    model = SimpleDiT(
        mel_input_dim=32,
        img_input_dim=768,
        hidden_size=HIDDEN_SIZE,
        mel_seq_len=875,
        img_seq_len=1024,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS
    ).to(DEVICE)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    effective_batch_size = BATCH_SIZE * GRAD_ACCUM_STEPS * world_size
    scaled_lr = LR * (effective_batch_size / 32)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Track best validation loss
    best_val_loss = float('inf')
    
    if is_main:
        print("=" * 60)
        print(f"Training started at: {datetime.now()}")
        print(f"World size: {world_size} GPUs")
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Batch size: {BATCH_SIZE} x {world_size} = {effective_batch_size} effective")
        print(f"Learning rate: {scaled_lr:.2e}")
        print(f"Condition dropout: {CONDITION_DROPOUT}")
        print(f"Validation every {VAL_EVERY_N_EPOCHS} epochs")
        print("=" * 60)

    ### TRAINING LOOP
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0 
        num_batches = 0
        uncond_batches = 0

        optimizer.zero_grad()

        for batch_idx, (mel_spec, image_patches) in enumerate(train_dataloader):
            mel_spec = mel_spec.to(DEVICE)
            image_patches = image_patches.to(DEVICE)

            # Condition dropout
            is_unconditional = torch.rand(1).item() < CONDITION_DROPOUT
            if is_unconditional:
                image_patches = torch.zeros_like(image_patches)
                uncond_batches += 1

            clean_patches = mel_adapter.pack(mel_spec)
            actual_batch_size = mel_spec.shape[0]
            t = torch.randint(0, 1000, (actual_batch_size,), device=DEVICE)
            
            noisy_patches, noise = noise_scheduler.add_noise(clean_patches, t)
            predicted_noise = model(noisy_patches, image_patches, t)
            loss = F.mse_loss(predicted_noise, noise)
            
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item() * GRAD_ACCUM_STEPS
            epoch_loss += batch_loss
            num_batches += 1
            global_step += 1

            # Log to wandb every 20 steps
            if is_main and USE_WANDB and global_step % 1 == 0:
                wandb.log({
                    "train/loss": batch_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/is_uncond": 1 if is_unconditional else 0,
                }, step=global_step)

            print(f"Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(train_dataloader)-1} | Loss {batch_loss:.6f} | LR {scheduler.get_last_lr()[0]:.2e} | Uncond: {uncond_batches}/{num_batches}")
        
        # Handle remaining gradients
        if (batch_idx + 1) % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = epoch_loss / num_batches
        scheduler.step()

        if is_main:
            print(f">>> Epoch {epoch:3d} Avg Train Loss: {avg_train_loss:.6f}")
            
            # Log epoch metrics
            if USE_WANDB:
                wandb.log({
                    "train/epoch_loss": avg_train_loss,
                    "train/uncond_batches": uncond_batches,
                }, step=global_step)
            
            # VALIDATION
            if (epoch + 1) % VAL_EVERY_N_EPOCHS == 0:
                print("Running validation...")
                
                # Get base model for validation
                val_model = model.module if world_size > 1 else model
                
                val_metrics = validate(
                    val_model, val_dataloader, mel_adapter, noise_scheduler, DEVICE
                )
                
                print(f"    Val Loss: {val_metrics['val_loss']:.6f}")
                print(f"    Val Loss t=100: {val_metrics['val_loss_t100']:.6f}")
                print(f"    Val Loss t=300: {val_metrics['val_loss_t300']:.6f}")
                print(f"    Val Loss t=500: {val_metrics['val_loss_t500']:.6f}")
                print(f"    Val Loss t=800: {val_metrics['val_loss_t800']:.6f}")
                
                if USE_WANDB:
                    wandb.log({
                        "val/loss": val_metrics['val_loss'],
                        "val/loss_t100": val_metrics['val_loss_t100'],
                        "val/loss_t300": val_metrics['val_loss_t300'],
                        "val/loss_t500": val_metrics['val_loss_t500'],
                        "val/loss_t800": val_metrics['val_loss_t800'],
                    }, step=global_step)
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
                    
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': best_val_loss,
                        'config': {
                            'mel_input_dim': 32,
                            'img_input_dim': 768,
                            'hidden_size': HIDDEN_SIZE,
                            'mel_seq_len': 875,
                            'img_seq_len': 1024,
                            'num_layers': NUM_LAYERS,
                            'num_heads': NUM_HEADS
                        }
                    }, f"{SAVE_PATH}/best_model.pt")
                    print(f"    ✓ New best model saved! Val loss: {best_val_loss:.6f}")
            
            print("=" * 60)
            
            # Regular checkpoint saving
            if (epoch + 1) % SAVE_EPOCHS_STEP == 0:
                model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'config': {
                        'mel_input_dim': 32,
                        'img_input_dim': 768,
                        'hidden_size': HIDDEN_SIZE,
                        'mel_seq_len': 875,
                        'img_seq_len': 1024,
                        'num_layers': NUM_LAYERS,
                        'num_heads': NUM_HEADS
                    }
                }
                torch.save(checkpoint, f"{SAVE_PATH}/checkpoint_epoch{epoch+1}.pt")
                print(f"✓ Saved checkpoint epoch {epoch+1}")

    ### FINISH
    if is_main:
        # Save final model
        model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
        
        torch.save({
            'epoch': NUM_EPOCHS,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'config': {
                'mel_input_dim': 32,
                'img_input_dim': 768,
                'hidden_size': HIDDEN_SIZE,
                'mel_seq_len': 875,
                'img_seq_len': 1024,
                'num_layers': NUM_LAYERS,
                'num_heads': NUM_HEADS
            }
        }, f"{SAVE_PATH}/final_model.pt")
        
        print("=" * 60)
        print(f"Training finished at {datetime.now()}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print("=" * 60)
        
        if USE_WANDB:
            wandb.finish()
    
    cleanup_distributed()


if __name__ == "__main__":
    train()
