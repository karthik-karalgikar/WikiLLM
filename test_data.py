import numpy as np

print("Loading train_data.npz...")
try:
    data = np.load('train_data.npz')
    
    inputs = data['inputs']
    targets = data['targets']
    vocab_size = data['vocab_size']
    sequence_length = data['sequence_length']
    
    print(f"✓ File loaded successfully!")
    print(f"\nDataset info:")
    print(f"  - Inputs shape: {inputs.shape}")
    print(f"  - Targets shape: {targets.shape}")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Total training examples: {len(inputs):,}")
    
    # Test first example
    print(f"\nFirst input sequence (first 10 tokens): {inputs[0][:10]}")
    print(f"First target sequence (first 10 tokens): {targets[0][:10]}")
    
    print("\n✓ Data is valid and ready to use!")
    
except Exception as e:
    print(f"Error loading file: {e}")
    print("The file may be corrupted. Need to regenerate with fewer articles.")