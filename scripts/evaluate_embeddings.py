import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingEvaluator:
    def __init__(self, model_path: str, model_type: str = 'word2vec'):
        """
        Initialize the evaluator with a model path and type.
        
        Args:
            model_path: Path to the model file
            model_type: Either 'word2vec', 'word2vec_sg', or 'glove'
        """
        self.model_type = model_type
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        """Load the appropriate model based on type."""
        if self.model_type in ['word2vec', 'word2vec_sg']:
            model = Word2Vec.load(model_path)
            return model.wv  # Return the word vectors directly
        elif self.model_type == 'glove':
            # Load GloVe embeddings and vocabulary
            embeddings = np.load(model_path)
            vocab_path = str(Path(model_path).parent / 'glove_vocab.json')
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            
            # Create a KeyedVectors object for GloVe
            kv = KeyedVectors(embeddings.shape[1])
            kv.add_vectors(list(vocab.keys()), embeddings)
            return kv
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def get_most_similar(self, word: str, topn: int = 5) -> List[Tuple[str, float]]:
        """Get most similar words for a given word."""
        try:
            return self.model.most_similar(word, topn=topn)
        except KeyError:
            logger.warning(f"Word '{word}' not found in vocabulary")
            return []

    def evaluate_analogy(self, word1: str, word2: str, word3: str) -> List[Tuple[str, float]]:
        """
        Evaluate word analogy: word1 is to word2 as word3 is to ?
        Returns top 5 most likely words.
        """
        try:
            return self.model.most_similar(positive=[word2, word3], negative=[word1], topn=5)
        except KeyError as e:
            logger.warning(f"One of the words not found in vocabulary: {e}")
            return []

    def evaluate_similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between two words."""
        try:
            return self.model.similarity(word1, word2)
        except KeyError as e:
            logger.warning(f"One of the words not found in vocabulary: {e}")
            return 0.0

    def evaluate_word_pairs(self, word_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Evaluate a list of word pairs and return average similarity.
        
        Args:
            word_pairs: List of tuples containing word pairs to evaluate
        """
        similarities = []
        for word1, word2 in word_pairs:
            sim = self.evaluate_similarity(word1, word2)
            similarities.append(sim)
        
        return {
            'mean_similarity': np.mean(similarities),
            'median_similarity': np.median(similarities),
            'std_similarity': np.std(similarities),
            'raw_similarities': similarities
        }

def calculate_percentage_score(similarity: float) -> float:
    """Convert similarity score to percentage (0-100)."""
    # Convert from [-1, 1] range to [0, 100] range
    return ((similarity + 1) / 2) * 100

def get_similarity_rating(percentage: float) -> str:
    """Get a rating based on the similarity percentage."""
    if percentage >= 90:
        return "Excellent"
    elif percentage >= 75:
        return "Very Good"
    elif percentage >= 60:
        return "Good"
    elif percentage >= 45:
        return "Fair"
    else:
        return "Poor"

def main():
    # Example usage
    model_paths = {
        'word2vec': 'models/word2vec.model',  # CBOW model
        'word2vec_sg': 'models/word2vec_sg.model',  # Skip Gram model
        'glove': 'models/glove_embeddings.npy'
    }
    
    # Example word pairs for evaluation (you should replace these with actual Odia word pairs)
    test_pairs = [
        ('ବାପା', 'ମା'),  # father-mother
        ('ବାପା', 'ପୁଅ'),  # father-son
        ('ମା', 'ଝିଅ'),  # mother-daughter
        ('ବାପା', 'ଝିଅ'),  # father-daughter
        ('ମା', 'ପୁଅ'),  # mother-son
    ]
    
    # Example analogies (you should replace these with actual Odia analogies)
    test_analogies = [
        ('ବାପା', 'ମା', 'ପୁଅ'),  # father:mother::son:?
        ('ବାପା', 'ପୁଅ', 'ମା'),  # father:son::mother:?
    ]
    
    results = {}
    
    for model_type, model_path in model_paths.items():
        logger.info(f"Evaluating {model_type} model...")
        evaluator = EmbeddingEvaluator(model_path, model_type)
        
        # Evaluate word pairs
        pair_results = evaluator.evaluate_word_pairs(test_pairs)
        results[model_type] = {
            'word_pairs': pair_results
        }
        
        # Evaluate analogies
        analogy_results = []
        for word1, word2, word3 in test_analogies:
            analogy = evaluator.evaluate_analogy(word1, word2, word3)
            analogy_results.append({
                'query': f"{word1}:{word2}::{word3}:?",
                'results': analogy
            })
        results[model_type]['analogies'] = analogy_results
    
    # Print results with percentage conclusions
    print("\n=== EVALUATION RESULTS ===\n")
    
    for model_type, model_results in results.items():
        print(f"\n{'='*20} {model_type.upper()} MODEL {'='*20}")
        
        # Word Pair Evaluation
        print("\nWord Pair Evaluation:")
        print("-" * 50)
        
        # Calculate and print individual pair scores
        similarities = model_results['word_pairs']['raw_similarities']
        for (word1, word2), similarity in zip(test_pairs, similarities):
            percentage = calculate_percentage_score(similarity)
            rating = get_similarity_rating(percentage)
            print(f"{word1}-{word2}: {percentage:.1f}% ({rating})")
        
        # Overall word pair statistics
        mean_percentage = calculate_percentage_score(model_results['word_pairs']['mean_similarity'])
        rating = get_similarity_rating(mean_percentage)
        print(f"\nOverall Word Pair Similarity: {mean_percentage:.1f}% ({rating})")
        
        # Analogy Evaluation
        print("\nAnalogy Evaluation:")
        print("-" * 50)
        
        for analogy in model_results['analogies']:
            print(f"\nQuery: {analogy['query']}")
            if analogy['results']:
                top_result = analogy['results'][0]
                percentage = calculate_percentage_score(top_result[1])
                rating = get_similarity_rating(percentage)
                print(f"Top match: {top_result[0]} ({percentage:.1f}% confidence, {rating})")
            else:
                print("No results found")
        
        # Model Summary
        print(f"\nModel Summary:")
        print("-" * 50)
        print(f"Vocabulary Coverage: {len(evaluator.model.key_to_index):,} words")
        print(f"Embedding Dimension: {evaluator.model.vector_size}")
        print(f"Overall Performance Rating: {rating}")

if __name__ == "__main__":
    main() 