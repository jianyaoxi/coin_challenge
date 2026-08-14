mkdir coin_env && cd $_ && python -m venv . && source bin/activate && cd .. 
pip install retrying flask attrs gymnasium colorama accelerate transformers==4.43.1 Pillow opencv-python dotenv qwen-vl-utils huggingface_hub google-genai openai
