from google.colab import drive
from datasets import load_from_disk
import pickle
import numpy as np
import os
import json
import shutil
import gc

# MOUNT GOOGLE DRIVE FIRST
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Set working directory to Google Drive
work_dir = '/content/drive/MyDrive/WikiLLM'
os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)

print(f"✓ Working directory: {work_dir}")
print("✓ All files will persist across sessions!\n")

# Load tokenizer
print("Loading tokenizer...")
if not os.path.exists('tokenizer.pkl'):
    print("ERROR: tokenizer.pkl not found!")
    exit()

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

# Load dataset
print("\nLoading Wikipedia data...")
if not os.path.exists('wikipedia_data'):
    print("ERROR: wikipedia_data not found!")
    exit()

dataset = load_from_disk("./wikipedia_data")
dataset = dataset.select(range(10000))
print(f"Total articles: {len(dataset)}")

sequence_length = 64

# Create directories
os.makedirs('tokenized_articles', exist_ok=True)

# Check progress
progress_file = 'tokenized_articles/progress.json'
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress = json.load(f)
        start_idx = progress['last_completed'] + 1
        print(f"\n✓ Resuming from article {start_idx}")
else:
    start_idx = 0
    print(f"\nStarting from article 0")

# Tokenize articles
print(f"\nTokenizing articles {start_idx} to {len(dataset)-1}...")

for i in range(start_idx, len(dataset)):
    text = dataset[i]['text']
    
    if text.strip():
        tokens = encode(text)
        tokens.append(2)
        article_file = f'tokenized_articles/article_{i:06d}.npy'
        np.save(article_file, np.array(tokens, dtype=np.int32))
    else:
        article_file = f'tokenized_articles/article_{i:06d}.npy'
        np.save(article_file, np.array([2], dtype=np.int32))
    
    # Update progress
    with open(progress_file, 'w') as f:
        json.dump({'last_completed': i, 'total': len(dataset)}, f)
    
    # Print progress
    if (i + 1) % 10 == 0:
        print(f"✓ {i + 1}/{len(dataset)} articles ({(i+1)/len(dataset)*100:.1f}%)", end='\r')
    
    if (i + 1) % 100 == 0:
        print(f"\n✓ Tokenized {i + 1}/{len(dataset)} articles")

print(f"\n\n✓ All {len(dataset)} articles tokenized!")

# Check if train_data already exists
if os.path.exists('train_data.npz'):
    print("\n train_data.npz already exists!")
    
    # Verify the data
    data = np.load('train_data.npz')
    print(f"\nDataset info:")
    print(f"  - Training examples: {data['inputs'].shape[0]:,}")
    print(f"  - Sequence length: {data['sequence_length']}")
    print(f"  - Vocabulary size: {data['vocab_size']}")
    
else:
    print("\n--- Creating training sequences in chunks ---")
    
    chunk_size = 100
    os.makedirs('sequence_chunks', exist_ok=True)
    
    chunk_num = 0
    total_sequences = 0
    
    # THIS FOR LOOP PROCESSES ALL CHUNKS
    for start_idx in range(0, len(dataset), chunk_size):
        end_idx = min(start_idx + chunk_size, len(dataset))
        
        print(f"\nChunk {chunk_num + 1}: articles {start_idx}-{end_idx-1}...")
        
        # Load tokens for this chunk
        chunk_tokens = []
        for i in range(start_idx, end_idx):
            article_file = f'tokenized_articles/article_{i:06d}.npy'
            tokens = np.load(article_file).tolist()
            chunk_tokens.extend(tokens)
        
        print(f"  Loaded {len(chunk_tokens):,} tokens")
        
        # Create sequences
        if len(chunk_tokens) <= sequence_length:
            print(f"  Chunk too small, skipping...")
            del chunk_tokens
            continue
        
        print(f"  Creating sequences...")
        input_sequences = []
        target_sequences = []
        
        for i in range(0, len(chunk_tokens) - sequence_length):
            input_seq = chunk_tokens[i:i+sequence_length]
            target_seq = chunk_tokens[i+1:i+sequence_length+1]
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
        
        print(f"  Created {len(input_sequences):,} sequences")
        
        # Save chunk
        if len(input_sequences) > 0:
            chunk_data = {
                'inputs': np.array(input_sequences, dtype=np.int32),
                'targets': np.array(target_sequences, dtype=np.int32),
            }
            
            chunk_filename = f'sequence_chunks/chunk_{chunk_num:04d}.npz'
            np.savez_compressed(chunk_filename, **chunk_data)
            print(f"  ✓ Saved {chunk_filename}")
            
            total_sequences += len(input_sequences)
            chunk_num += 1
        
        # Clear memory
        del chunk_tokens, input_sequences, target_sequences
        gc.collect()
    
    # ⭐ THIS IS OUTSIDE THE LOOP - RUNS ONCE AFTER ALL CHUNKS
    print(f"\n Created {total_sequences:,} sequences across {chunk_num} chunks")
    
    # Combine chunks
    print("\n--- Combining chunks ---")
    
    chunk_files = sorted([
        f'sequence_chunks/{f}' 
        for f in os.listdir('sequence_chunks') 
        if f.endswith('.npz')
    ])
    
    print(f"Loading {len(chunk_files)} chunks...")
    
    all_inputs = []
    all_targets = []
    
    for i, chunk_file in enumerate(chunk_files):
        print(f"Loading chunk {i+1}/{len(chunk_files)}...", end='\r')
        data = np.load(chunk_file)
        all_inputs.append(data['inputs'])
        all_targets.append(data['targets'])
    
    print("\nConcatenating...")
    final_inputs = np.concatenate(all_inputs, axis=0)
    final_targets = np.concatenate(all_targets, axis=0)
    
    print("Saving train_data.npz...")
    train_data = {
        'inputs': final_inputs,
        'targets': final_targets,
        'vocab_size': len(vocab),
        'sequence_length': sequence_length
    }
    
    np.savez_compressed('train_data.npz', **train_data)
    
    print(f"\n✓train_data.npz saved!")
    print(f"\nDataset info:")
    print(f"  - Training examples: {len(final_inputs):,}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Vocabulary size: {len(vocab)}")
    
    # Cleanup
    print("\nCleaning up chunks...")
    shutil.rmtree('sequence_chunks')
    print(" Done!")

print("\n Complete! Location:", work_dir)


#TESTING 

import numpy as np

# Load file (read-only)
data = np.load('train_data.npz')

print("Keys inside NPZ file:", list(data.keys()))
print("Inputs shape:", data['inputs'].shape)
print("Targets shape:", data['targets'].shape)
print("Input dtype:", data['inputs'].dtype)
print("Target dtype:", data['targets'].dtype)
print("Vocab size:", int(data['vocab_size']))
print("Sequence length:", int(data['sequence_length']))

'''
Keys inside NPZ file: ['inputs', 'targets', 'vocab_size', 'sequence_length']
Inputs shape: (25775541, 64)
Targets shape: (25775541, 64)
Input dtype: int32
Target dtype: int32
Vocab size: 5000
Sequence length: 64
'''

#------------------------------------------------------------------------------


import numpy as np

data = np.load('/content/drive/MyDrive/WikiLLM/train_data.npz')

print(" File loaded successfully!\n")
print("Dataset contents:")
print(f"  - Inputs shape: {data['inputs'].shape}")
print(f"  - Targets shape: {data['targets'].shape}")
print(f"  - Vocab size: {data['vocab_size']}")
print(f"  - Sequence length: {data['sequence_length']}")
print(f"\nTotal training examples: {data['inputs'].shape[0]:,}")

# Check first example
print(f"\nFirst training example:")
print(f"  Input:  {data['inputs'][0][:10]}...")
print(f"  Target: {data['targets'][0][:10]}...")

'''
File loaded successfully!

Dataset contents:
  - Inputs shape: (25775541, 64)
  - Targets shape: (25775541, 64)
  - Vocab size: 5000
  - Sequence length: 64

Total training examples: 25,775,541

First training example:
  Input:  [ 456  909 1077  862 2034 1786 1214 4676 3522 1005]...
  Target: [ 909 1077  862 2034 1786 1214 4676 3522 1005  493]...
'''