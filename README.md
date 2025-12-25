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