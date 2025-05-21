import torch
import numpy as np
import json
from pathlib import Path
import sys

# Import the GloVeModel definition from your train_glove.py
sys.path.append(str(Path('src/models').resolve()))
from train_glove import GloVeModel, save_embeddings

# Set paths
models_dir = Path('models')
checkpoint_path = max(models_dir.glob('glove_checkpoint_epoch_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
print(f"Loading checkpoint: {checkpoint_path}")

# Load vocab
vocab_path = models_dir / 'glove_vocab.json'
if vocab_path.exists():
    with open(vocab_path, 'r', encoding='utf-8') as f:
        word_to_idx = json.load(f)
else:
    # If vocab not saved, try to find and load from training artifacts
    print("glove_vocab.json not found. Please provide the vocabulary dictionary used during training.")
    sys.exit(1)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')
vocab_size = len(word_to_idx)
embedding_dim = checkpoint['model_state_dict']['word_embeddings.weight'].shape[1]

# Reconstruct model
model = GloVeModel(vocab_size, embedding_dim)
model.load_state_dict(checkpoint['model_state_dict'])

# Save embeddings and vocab
save_embeddings(model, word_to_idx, models_dir)
print("Embeddings and vocab saved!") 