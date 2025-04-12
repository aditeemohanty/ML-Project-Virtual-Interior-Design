from pathlib import Path
from transformers import TRANSFORMERS_CACHE

cache_dir = Path(TRANSFORMERS_CACHE)
if cache_dir.exists():
    for model in cache_dir.glob("*"):
        print(model)
else:
    print("No cache found.")
