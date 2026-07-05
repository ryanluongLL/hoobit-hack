import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import networkx as nx
from dataclasses import dataclass

# ── Data shapes ──────────────────────────────────────────────────────────────


@dataclass
class Claim:
    id: str
    text: str
    agent: str        # "advocate" or "skeptic"
    claim_type: str   # "empirical", "causal", "value", "definitional"
    depends_on: str   # id of another claim this one builds on, or ""

@dataclass
class StancePair:
    claim_a_id: str
    claim_b_id: str
    stance: str       # "contradiction", "entailment", "neutral"
    score: float

@dataclass
class CruxResult:
    claim_a: Claim
    claim_b: Claim
    stance: str
    score: float
    centrality_a: float
    centrality_b: float

# ── Model loader (call once at startup) ──────────────────────────────────────

class Models:
    def __init__(self):
        print("Loading MiniLM...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Loading MNLI...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-MiniLM2-L6-H768",
            device="cpu",
        )
        print("Models ready.")

# ── Claim type classifier (embedding nearest-neighbor) ────────────────────────

CLAIM_TYPE_EXAMPLES = {
    "empirical": [
        "The study found a statistically significant result.",
        "Data shows a 20 percent increase in output.",
        "The sample included 500 participants.",
    ],
    "causal": [
        "Working fewer hours causes employees to be more focused.",
        "The policy led to a reduction in emissions.",
        "Stress increases the likelihood of burnout.",
    ],
    "value": [
        "Workers deserve more time with their families.",
        "Productivity should not come at the cost of wellbeing.",
        "Economic growth is less important than quality of life.",
    ],
    "definitional": [
        "Productivity here means output per hour worked.",
        "A four day week is defined as 32 hours with full pay.",
        "Burnout is classified as an occupational phenomenon.",
    ],
}

def classify_claim_type(text: str, models: Models) -> str:
    claim_embedding = models.embedder.encode([text])[0]
    best_type = "empirical"
    best_score = -1.0

    for claim_type, examples in CLAIM_TYPE_EXAMPLES.items():
        example_embeddings = models.embedder.encode(examples)
        scores = [
            float(torch.nn.functional.cosine_similarity(
                torch.tensor(claim_embedding).unsqueeze(0),
                torch.tensor(e).unsqueeze(0)
            ))
            for e in example_embeddings
        ]
        avg_score = sum(scores) / len(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_type = claim_type
        
    return best_type

# ── Stance detection between two claims ──────────────────────────────────────

def detect_stance(claim_a: str, claim_b: str, models: Models) -> tuple[str, float ]:
    combined = f"{claim_a} [SEP] {claim_b}"
    result = models.classifier(
        combined,
        candidate_labels = ["entailment", "contradiction", "neutral"],
        multi_label=False,
    )
    return result["labels"][0], result["scores"][0]

# ── Build dependency graph per agent ─────────────────────────────────────────

def build_graph(claims: list[Claim]) -> nx.DiGraph:
    G = nx.DiGraph()
    for claim in claims:
        G.add_node(claim.id, data=claim)
    for claim in claims:
        if claim.depends_on and claim.depends_on in G.nodes:
            G.add_edge(claim.depends_on, claim.id)
    return G

# ── Main crux finder ─────────────────────────────────────────────────────────

def find_crux(
    advocate_claims: list[Claim],
    skeptic_claims: list[Claim],
    models: Models,
) -> tuple[CruxResult, list[StancePair]]:
    
    # Build dependency graphs
    graph_a = build_graph(advocate_claims)
    graph_b = build_graph(skeptic_claims)

    # Centrality: how load-bearing is each claim in its own argument
    centrality_a = nx.pagerank(graph_a) if len(graph_a.edges) > 0 else {c.id: 1.0 for c in advocate_claims}
    centrality_b = nx.pagerank(graph_b) if len(graph_b.edges) > 0 else {c.id: 1.0 for c in skeptic_claims}

    # Run stance detection across all claim pairs
    all_pairs: list[StancePair] = []
    for ca in advocate_claims:
        for cb in skeptic_claims:
            stance, score = detect_stance(ca.text, cb.text, models)
            all_pairs.append(StancePair(ca.id, cb.id, stance, score))

    # Find the contradiction with the highest combined centrality
    contradictions = [p for p in all_pairs if p.stance == "contradiction"]

    if not contradictions:
        contradiction = sorted(all_pairs, key=lambda p: p.score, reverse=True )

    best = max(
        contradictions,
        key=lambda p: (
            centrality_a.get(p.claim_a_id, 0) +
            centrality_b.get(p.claim_b_id, 0)
        )
    )

    claim_a = next(c for c in advocate_claims if c.id == best.claim_a_id)
    claim_b = next(c for c in skeptic_claims if c.id == best.claim_b_id)

    crux = CruxResult(
        claim_a=claim_a,
        claim_b=claim_b,
        stance=best.stance,
        score=best.score,
        centrality_a=centrality_a.get(claim_a.id, 0),
        centrality_b=centrality_b.get(claim_b.id, 0)   
    )

    return crux, all_pairs