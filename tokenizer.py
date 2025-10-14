from datasets import load_from_disk
import pickle

dataset = load_from_disk("/.wikipedia_data")

num_articles = 10000
texts = [dataset[i]['text'] for i in range(min(num_articles, len(dataset)))]
text = "\n".join(texts)

print("total text length = ", len(text))

tokens = text.encode("utf-8") # raw bytes 
tokens = list(map(int, tokens)) # convert into a list of integers in range 0-255

def get_stats(ids):
    count = {}
    for pair in zip(ids, ids[1:]): #iterating through consecutive integers through the list(check diary for explanation)
        count[pair] = count.get(pair, 0) + 1 #frequency DSA

    return count

def merge(ids, pair, idx):
    #in the list of ids(tokens or integers), replace all consecutive occurrences of pair with the new int idx(256)
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i = i + 2
        else:
            newids.append(ids[i])
            i = i + 1
    return newids

vocab_size = 10000
num_merges = vocab_size - 256
ids = list(tokens) #copy so that we don't destroy the original list

merges = {} # (int, int) -> int =>  mapping the merges, like a tree but starting with a leaf node and then merging
vocab = { idx : bytes([idx]) for idx in range(256)} # initial vocabulary -> a mapping from token IDs to their actual byte values
'''
vocab = {
    0: b'\x00',    # byte 0
    1: b'\x01',    # byte 1
    2: b'\x02',    # byte 2
    ...
    65: b'A',      # byte 65 (letter A)
    97: b'a',      # byte 97 (letter a)
    ...
    255: b'\xff'   # byte 255
}


Why is this needed?
when we decode tokens back to text, we need to know what each token ID represents. 
As we merge pairs, new tokens are formed 
token 256 might be b'e ' (e + whitespace)

'''


for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get) #refer the tokenizer with text for explanation(most common pair)
    idx = 256 + i # new token 
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]] 

print("tokens length = ", len(tokens))
print("length of ids = ", len(ids))
print("compression ratio = ", len(tokens) / len(ids))

# Save tokenizer
print("\nSaving tokenizer...")
tokenizer_data = {
    'merges': merges,
    'vocab': vocab
}

#Why are we saving the tokenizer?
'''
So that we can use the same tokenizer for: 
1. Training the model -> convert Wikipedia text to tokens
2. Testing the model -> encode prompts and decode generated text
3. Future use

Pickle is a way of saving Python objects to disk 
It serializes(saves) dictionaries, lists and other data structures. 

It can also be saves using JSON but that cannot handle bytes easily. 
Without saving, we need to train the tokenizer every time we need to use it. 

'''

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer_data, f)

'''
Note, the Tokenizer is a completely separate, independent module from the LLM. 
It has its own training dataset of text (which could be different from that of the LLM), 
on which you train the vocabulary using the Byte Pair Encoding (BPE) algorithm. 
It then translates back and forth between raw text and sequences of tokens. 
The LLM later only ever sees the tokens and never directly deals with any text.

Order - 

LLM 
token sequence
tokenizer
raw tokens

The LLM only interacts with the token sequence which is given by the tokenizer
'''
#NOTE - Now, we have created the merges and added them into the vocabulary. The size of the tokens is less, but the size of 
#vocabulary has been increased.

#Next step is Encoding and Decoding 
'''
Why do we have to do this?

Encode = text -> numbers
Decode = numbers -> text 

to make the model understand the prompts, we need to encode our text into numbers.
and the output of the model will also be in numbers. For us to understand, we need to decode it back to text.
'''


def decode(ids):
    #given ids (list of integers), return python strings
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text 







