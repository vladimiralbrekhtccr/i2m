## Start

Task_1: Generate a audio files based on [those](https://huggingface.co/datasets/CCRss/ILSVRC_images_10_class) 10 classes. 

### What I'm looking for

1. How you will follow the instructions.
2. You will need understand the logic of data that we currently using for training.
3. You will learn how to use PiperTTS what will be valuable for us.


## Task pipeline

0. Open GoogleCollab/LocalPC your choice.
1. Download the dataset
2. Untar
3. Download the [pipe_tts](https://github.com/OHF-Voice/piper1-gpl) models

Simple inference script

```python 
# 1. download
from huggingface_hub import snapshot_download
from huggingface_hub import login
import dotenv
import os
from pathlib import Path
dotenv.load_dotenv()
login(token=os.getenv("HF_TOKEN"))


repo_id = "rhasspy/piper-voices"
local_dir = Path("piper_tts").resolve()

def download_model_repo(repo_id, local_dir):
    print(f"Downloading repository {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=["kk/kk_KZ/*"],
        local_dir=str(local_dir)
    )
    print(f"Repository downloaded to {local_dir}")

download_model_repo(repo_id, local_dir)

# 2. generate speech
import wave
from piper import PiperVoice

# Simple CPU TTS inference

voice_path = "piper_tts/kk/kk_KZ/issai/high/kk_KZ-issai-high.onnx"
text = "Сәлеметсіз бе, бұл мысал сөйлем."
output_wav = "audio.wav"

voice = PiperVoice.load(voice_path, use_cuda=False)

with wave.open(output_wav, "wb") as wav_file:
    voice.synthesize_wav(text, wav_file)

print("Done:", output_wav)
```

And you goal is:to take not kazakh but english speaker and based on `class_info` in HF repo card make audio files. We will need to have the audios with SR==24000, please resample if using `torchaudio` library


And save it in the format of:

```text
dataset_1/
|----audios/
|----images/
|----meta_file.json 
```

Example of `meta_file.json` that I want to receive:

```json
{
    "image": "images_10_classes/001_goldfish/00517.jpg",
    "audio": "audios_10_classes/001_goldfish/reza_ibrahim.wav",
    "class": "001_goldfish",
    "text": "A goldfish, bright orange fish",
    "speaker": "reza_ibrahim"
}
...
...
```

4. Create new HF and upload to the HF in same tar format, such that it will be easy to use but we will have audios ready.
5. Upload your scripts to HF as well how you did it, I will check.