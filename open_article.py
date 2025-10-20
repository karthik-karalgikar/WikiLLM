import numpy as np
import pickle

# Load one article's tokens
tokens = np.load('article_000000.npy')

print(f"Article 0 tokens: {tokens}")
print(f"Number of tokens: {len(tokens)}")
print(f"Data type: {tokens.dtype}")

with open('tokenizer.pkl', 'rb') as f:
    tokenizer_data = pickle.load(f)
vocab = tokenizer_data['vocab']

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

# Load and decode an article
article_tokens = np.load('article_000000.npy')
article_text = decode(article_tokens.tolist())

print("Original article text:")
print(article_text[:500])  # First 500 characters
print("\n...")