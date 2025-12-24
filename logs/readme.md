
# Frist thoughts and text

## The "Generative Classification" Experiment

Goal: Train a Diffusion Transformer (DiT) to generate the Mel Spectrogram of the class name, conditioned on the image.



## Architecture:

ViT (Vision Transformer)
DiT
VaE (what about taking Qwen3VL as a base for VAE?)



### Summary 1 from Gemini3.0

Here is the **Master Summary** of the project we have architected. Use this prompt to kickstart your next session with Gemini 3.0.

---

### **Project Title: Image-to-Audio (Mel Spectrogram) via DiT**

**Goal:** Train a Diffusion Transformer (DiT) to generate a 3-second Audio Clip (Mel Spectrogram) conditioned on an Image.

#### **1. Data Pipeline (The "Truth")**
*   **Input Image:** $512 \times 512$ RGB.
    *   **Processing:** Patchify into $16 \times 16$ chunks $\to$ Flatten to sequence.
    *   **Tensor Shape:** `[Batch, 1024, 768]` (Sequence Length = 1024).
*   **Target Audio:** 3.0 seconds at 24,000 Hz.
    *   **Processing:** Resample to 24k $\to$ Mel Spectrogram via `Vocos`.
    *   **Mel Config:** 100 Bands, 1024 FFT, 256 Hop.
    *   **Tensor Shape:** `[Batch, 1, 100, 282]` (100 Freqs, 282 Time Frames).

#### **2. Model Architecture (The "Engine")**
We are building a **Latent Diffusion Transformer** with a custom adapter.

*   **Adapter (Mel $\to$ Tokens):**
    *   Takes Mel `[B, 1, 100, 282]`.
    *   Patchifies ($4 \times 4$ patches).
    *   Outputs Sequence `[B, 1750, 16]`.
*   **Input Projection:**
    *   Projects Mel Patches `16` $\to$ `Hidden_Size` (e.g., 512).
    *   Projects Image Patches `768` $\to$ `Hidden_Size` (e.g., 512).
*   **Backbone (DiT Block):**
    *   Standard Transformer Block with **AdaLayerNorm** (Time Modulation).
    *   **Self-Attention:** Mel attends to Mel (Structure).
    *   **Cross-Attention:** Mel attends to Image (Conditioning).
*   **Output:** Predicts the Noise `[B, 1750, 16]`.

#### **3. The Diffusion Process**
*   **Scheduler:** Linear Noise Scheduler.
*   **Mechanism:** `alphas_cumprod` lookup to add noise in one step.
*   **Loss Function:** MSE Loss between `Predicted_Noise` and `Actual_Noise`.

#### **4. Inference & Reconstruction**
*   **Sampler:** DDIM or Euler Discrete (Reverse diffusion loop).
*   **Vocoder:** **Vocos (24kHz)**.
    *   Takes the clean Mel Spectrogram from the DiT.
    *   Converts it to high-fidelity Waveform.

---

### **Current Status**
*   ✅ **Understood:** Audio Physics (Hz, Sampling Rate), FFT, Mel Scale, Matrix Math.
*   ✅ **Solved:** Resampling logic (GCD), Mel Generation (Vocos match), Patching Logic.
*   ✅ **Ready:** Training Loop logic, DiT Block architecture.

**Next Step for Gemini 3.0:** "Help me write the `Dataset` class and the `Training Loop` to connect these blocks."


# Init env

0. Env

```bash
uv venv --python 3.12 .venv
uva

uv pip install torch==2.8.0
uv pip install torchaudio==2.8.0
uv pip install torchvision==0.23.0
uv pip install matplotlib

uv pip install ipykernel
python -m ipykernel install --user --name diffusers_env

```

1. Download dataset:

```python
from huggingface_hub import snapshot_download
from huggingface_hub import login
import dotenv
import os
from pathlib import Path
dotenv.load_dotenv()
login(token=os.getenv("HF_TOKEN"))

repo_id = "ILSVRC/imagenet-1k"
REPO_TYPE = "model"
local_dir = Path("/home/vladimir_albrekht/projects/img_to_spec/large_files/ILSVRC/imagenet-1k").resolve()

def download_model_repo(repo_id, local_dir):
    print(f"Downloading repository {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        # allow_patterns=["kk/kk_KZ/*"],
        local_dir=str(local_dir),
        repo_type=REPO_TYPE
        
    )
    print(f"Repository downloaded to {local_dir}")

download_model_repo(repo_id, local_dir)
```

2. 


git clone https://github.com/huggingface/diffusers.git