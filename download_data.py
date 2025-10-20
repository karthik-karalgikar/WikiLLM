from datasets import load_dataset, Dataset
import os

print("Downloading Wikipedia dataset")

dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, trust_remote_code=True)

# print(f"Total articles in the subset = {len(dataset)}")

subset_size = 50000
articles = []
for i, article in enumerate(dataset):
    if i >= subset_size:
        break
    articles.append(article)
    if (i + 1) % 5000 == 0:
        print(f"collected {i + 1} articles")

print(f"\n Collected {len(articles)} articles!")

dataset_subset = Dataset.from_list(articles)

print("Saving locally")
dataset_subset.save_to_disk("./wikipedia_data")

print(f"location = ./wikipedia_data")


#comment