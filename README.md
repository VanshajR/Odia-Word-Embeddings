# Odia Word Embeddings

This project implements and compares different word embedding models (Word2Vec and GloVe) for the Odia language. The models are trained on a combination of Odia literature and Wikipedia data.

## Project Structure

```
├── data/
│   ├── raw/          # Raw data files (not in repo)
│   ├── processed/    # Processed data files (not in repo)
│   └── interim/      # Intermediate data files (not in repo)
├── models/           # Trained models (not in repo)
├── notebooks/        # Jupyter notebooks for analysis
├── scripts/          # Utility scripts
└── src/             # Source code
    ├── data/        # Data processing modules
    └── models/      # Model training modules
```

## Data and Models

Due to size limitations, the data and model files are hosted on Hugging Face:

### Data
All data is available in the [odia-word-embeddings-data](https://huggingface.co/datasets/VanshajR/odia-word-embeddings-data) dataset:

#### Raw Data
- `odia_wiki_scraped.txt` - Scraped Odia Wikipedia articles
- `odia_literature.txt` - Odia literature corpus

#### Processed Data
- `odia_wiki_scraped.csv` - Processed Wikipedia articles
- `odia_literature.csv` - Processed literature corpus

#### Additional Data Used
- Monolingual Odia corpus from [OdiEnCorp 1.0](https://github.com/odiencorp/OdiEnCorp) (used for training)

### Models
All models are available in the [odia-word-embeddings](https://huggingface.co/VanshajR/odia-word-embeddings) repository:

#### Word2Vec
- `word2vec.model` - Trained Word2Vec model

#### GloVe
- `glove_embeddings.npy` - Trained GloVe embeddings
- `glove_vocab.json` - Vocabulary mapping for GloVe model

## Setup

1. Clone the repository:
```bash
git clone https://github.com/VanshajR/Odia-Word-Embeddings.git
cd Odia-Word-Embeddings
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download data and models:
```bash
# Install huggingface-hub
pip install huggingface-hub

# Login to Hugging Face
huggingface-cli login

# Download data
huggingface-cli download VanshajR/odia-word-embeddings-data --local-dir data/

# Download models
huggingface-cli download VanshajR/odia-word-embeddings --local-dir models/
```

## Usage

### Training Models

1. Train Word2Vec:
```bash
python scripts/train_word2vec.py
```

2. Train GloVe:
```bash
python scripts/train_glove.py
```

### Evaluating Models

```bash
python scripts/evaluate_embeddings.py
```

## Results

[Add your evaluation results and comparisons here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OdiEnCorp team for providing the monolingual Odia corpus
- [Add other acknowledgments here]