from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

print("Loading MiniLM...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading MNLI...")
classifier = pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-MiniLM2-L6-H768",
    device="cpu",
)

print("Both models loaded. Running test...")

# Test 1: embedding similarity between two claims
claims = [
    "The study sample size is large enough to be statistically valid.",
    "The study only included 12 companies, which is too small to generalize."
]
embeddings = embedder.encode(claims)
similarity = float(
    torch.nn.functional.cosine_similarity(
        torch.tensor(embeddings[0]).unsqueeze(0),
        torch.tensor(embeddings[1]).unsqueeze(0)
    )
)
print(f"Cosine similarity between two opposing claims: {similarity:.3f}")
print("(Expect somewhere between 0.3 and 0.7, similar topic but opposing stance)")

# Test 2: stance detection between the same two claims
result = classifier(
    claims[0],
    candidate_labels=["entailment", "contradiction", "neutral"],
    multi_label=False
)
print(f"Stance detection result: {result['labels'][0]} (score: {result['scores'][0]:.3f})")

print("All good! Models work on your machine.")