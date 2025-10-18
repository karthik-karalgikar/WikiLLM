from datasets import load_from_disk
import pickle
import numpy as np

print("Loading tokenizer...")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer_data = pickle.load(f)

merges = tokenizer_data['merges']
vocab = tokenizer_data['vocab']

print(f"Vocabulary size: {len(vocab)}")

# Helper functions
def get_stats(tokens):
    counts = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):                             
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# Encode function
def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

print("\nLoading Wikipedia data...")
dataset = load_from_disk("./wikipedia_data")
dataset = dataset.select(range(10000))  # Using only 10,000 articles
print(f"Using subset: {len(dataset)} articles")

# Tokenize all articles
print("\nTokenizing articles...")
all_tokens = []

for i in range(len(dataset)):
    text = dataset[i]['text']
    if text.strip():  # Skip empty articles
        tokens = encode(text)
        all_tokens.extend(tokens)
        all_tokens.append(2)  # Add <EOS> token between articles
    
    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(dataset)} articles, {len(all_tokens):,} tokens so far")

print(f"\nTotal tokens: {len(all_tokens):,}")

# Create training sequences
sequence_length = 64  # Context window for our model
print(f"\nCreating training sequences (length={sequence_length})...")

input_sequences = []
target_sequences = []

for i in range(0, len(all_tokens) - sequence_length):
    input_seq = all_tokens[i:i+sequence_length]
    target_seq = all_tokens[i+1:i+sequence_length+1]
    
    input_sequences.append(input_seq)
    target_sequences.append(target_seq)
    
    if (i + 1) % 100000 == 0:
        print(f"Created {i+1:,} sequences...")

print(f"Created {len(input_sequences):,} training examples")

# Convert to numpy arrays and save
print("\nSaving training data...")
train_data = {
    'inputs': np.array(input_sequences, dtype=np.int32),
    'targets': np.array(target_sequences, dtype=np.int32),
    'vocab_size': len(vocab),
    'sequence_length': sequence_length
}

np.savez_compressed('train_data.npz', **train_data)

print("\n Training data saved to train_data.npz")
print(f"\nDataset info:")
print(f"  - Training examples: {len(input_sequences):,}")
print(f"  - Sequence length: {sequence_length}")
print(f"  - Vocabulary size: {len(vocab)}")
print(f"  - Approximate file size: {len(input_sequences) * sequence_length * 4 / 1024 / 1024:.1f} MB")

print("\n Dataset preparation complete!")