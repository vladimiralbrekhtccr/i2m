# i2m

## uv

```bash
uv venv --python 3.13 --seed .venv
uv pip install huggingface_hub
uv pip install python-dotenv
python -m ipykernel install --user --name i2m
uv pip install \
        torch==2.9.0 \
        torchvision==0.24.0 \
        torchaudio==2.9.0 \
        transformers==4.57.3 \
        pandas==2.3.2 \
        python-dotenv==1.2.1 \
        matplotlib==3.10.8 \
        accelerate==1.12.0 \
        ipykernel \
        piper-tts \
        vocos

```


1. tar

`tar -czf images_10_class.tar.gz -C /home/vladimir_albrekht/projects/img_to_spec/large_files/ILSVRC images_10_class`

### TODO:
1. Let's add Flash-attention if that possible for 'self_attention' and 'cross-attention'
2. 


## README Content

---

### 📊 Dataset

```
DATASET STRUCTURE
=================

10 ImageNet Classes:
├── tench, goldfish, great white shark, tiger shark, hammerhead
├── electric ray, stingray, rooster, hen, ostrich
└── ~1,300 images per class = 13,000 total images

Audio Generation:
├── 20 TTS speakers (Piper TTS)
├── Speed augmentation: 0.75x - 1.35x
├── Pitch augmentation: -4 to +4 semitones
└── 13,000 unique audio files (3 sec, 24kHz)

Train/Eval Split:
├── Train: 11,700 pairs (90%)
└── Eval:   1,300 pairs (10%)

Each pair:
┌─────────────────┐      ┌─────────────────┐
│  Image (512×512)│  →   │ Mel Spectrogram │
│    (fish.jpg)   │      │   (100×280)     │
└─────────────────┘      └─────────────────┘
```

---

### 🏗️ Architecture

```
SimpleDiT: Image-to-Audio Diffusion Transformer
===============================================

INPUT PROCESSING:
                                                    
 Image (512×512×3)          Mel Spectrogram (100×280)
        │                            │
        ▼                            ▼
   ┌─────────┐                 ┌───────────┐
   │Patchify │                 │  Patchify │
   │ 16×16   │                 │  4×8      │
   └────┬────┘                 └─────┬─────┘
        │                            │
        ▼                            ▼
  [1024, 768]                   [875, 32]
  1024 patches                  875 patches
        │                            │
        ▼                            ▼
   ┌─────────┐                 ┌───────────┐
   │Linear   │                 │  Linear   │
   │Projection                 │ Projection│
   └────┬────┘                 └─────┬─────┘
        │                            │
        ▼                            ▼
  [1024, 768]                   [875, 768]
   + pos_embed                   + pos_embed
        │                            │
        └──────────┐    ┌────────────┘
                   │    │
                   ▼    ▼
            ┌─────────────────┐
            │   DiT Block ×12 │
            │                 │
            │ ┌─────────────┐ │
            │ │ Self-Attn   │◄── Mel attends to Mel
            │ │ (Mel→Mel)   │ │
            │ └─────────────┘ │
            │        │        │
            │ ┌─────────────┐ │
            │ │ Cross-Attn  │◄── Mel attends to Image
            │ │ (Mel→Image) │ │
            │ └─────────────┘ │
            │        │        │
            │ ┌─────────────┐ │        ┌───────────┐
            │ │    MLP      │ │◄───────│ Timestep  │
            │ └─────────────┘ │        │ Embedding │
            │                 │        │  (t=500)  │
            └────────┬────────┘        └───────────┘
                     │
                     ▼
              ┌─────────────┐
              │   Linear    │
              │  Projection │
              └──────┬──────┘
                     │
                     ▼
              Predicted Noise
                [875, 32]
                     │
                     ▼
               ┌───────────┐
               │ Unpatchify│
               └─────┬─────┘
                     │
                     ▼
              Mel (100×280)


MODEL CONFIG (150M params):
===========================
• Hidden size: 768
• Layers: 12
• Attention heads: 12
• Mel patches: 875 (25×35)
• Image patches: 1024 (32×32)


TRAINING:
=========
• Noise schedule: Linear β (0.0001 → 0.02)
• Timesteps: 1000
• Condition dropout: 10% (for CFG)
• Optimizer: AdamW
• Scheduler: Cosine Annealing


INFERENCE:
==========
• Sampler: DDIM (50-100 steps)
• CFG scale: 1.0 - 7.5
• Audio reconstruction: Vocos vocoder
```

---

### 🎯 Project Summary

```
GOAL
====
Generate audio descriptions from images using diffusion models.

Image of a fish  →  "A goldfish, bright orange fish" (spoken audio)


CURRENT STATUS
==============
✅ Single-step denoising works well (MSE ~0.007)
❌ Full generation produces wrong class (mode collapse)
❌ Model ignores image condition


ROOT CAUSE
==========
Original dataset: 13,000 images → only 10 unique audio files
Model learned: "ignore image, generate average mel"


SOLUTION IN PROGRESS
====================
New dataset with diversity:
• 20 speakers × 10 classes = 200 base audios
• Speed/pitch augmentation = 13,000 unique mels
• Condition dropout (10%) for CFG training
```

---

### 🗺️ Roadmap

```
TODO LIST
=========

[✅] DONE:
    • Basic DiT architecture
    • Training pipeline with DDP
    • Vocos mel extraction/reconstruction
    • Single-step denoising validation
    • Train/eval split

[🔄] IN PROGRESS:
    • Multi-speaker audio generation (20 speakers)
    • Speed/pitch augmentations
    • Condition dropout for CFG

[📋] NEXT:
    □ Train on augmented dataset (500 epochs)
    □ Evaluate with CFG (scale 1.0, 3.0, 5.0, 7.5)
    □ Confusion matrix evaluation
    □ Compare: random baseline (10%) vs model accuracy

[🔮] FUTURE IMPROVEMENTS:
    □ EMA weights tracking
    □ Cosine noise schedule
    □ Larger model (300M+ params)
    □ More classes (100+)
    □ Real image-audio pairs (video datasets)
    □ CLIP image encoder instead of raw patches


EXPERIMENTS TO TRY
==================
1. CFG scale sweep: Find optimal guidance strength
2. More speakers: 50+ voices for more diversity  
3. Text variations: Multiple descriptions per class
4. Noise schedule: Try cosine instead of linear
5. Architecture: Add CLIP encoder for better image features
```

---

### 📈 Metrics

```
EVALUATION METRICS
==================

1. Denoising Quality (should be low):
   • MSE at t=300: ~0.007 ✅
   • MSE at t=500: ~0.03  ✅
   • MSE at t=800: ~0.25  ✅

2. Generation Quality (currently broken):
   • MSE vs ground truth: ~45-50 ❌
   • Classification accuracy: 10% (random) ❌

3. Target After Fix:
   • Classification accuracy: >50%
   • Distinct mel per class
```