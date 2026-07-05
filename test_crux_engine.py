from app.crux_engine import Models, Claim, classify_claim_type, find_crux

models = Models()

advocate_claims = [
    Claim("a1", "The study found a 20 percent productivity gain.", "advocate", "", ""),
    Claim("a2", "The sample size of 500 companies is statistically adequate.", "advocate", "", "a1"),
    Claim("a3", "These results should generalize to most industries.", "advocate", "", "a2"),
]

skeptic_claims = [
    Claim("s1", "The productivity data was entirely self-reported by participants.", "skeptic", "", ""),
    Claim("s2", "Only companies that volunteered were included, introducing selection bias.", "skeptic", "", "s1"),
    Claim("s3", "The findings cannot generalize beyond self-selected, motivated firms.", "skeptic", "", "s2"),
]

print("Testing claim type classifier...")
for c in advocate_claims + skeptic_claims:
    claim_type = classify_claim_type(c.text, models)
    print(f"  [{claim_type}] {c.text[:60]}")

print("\nFinding crux...")
crux, all_pairs = find_crux(advocate_claims, skeptic_claims, models)

print(f"\nCRUX FOUND:")
print(f"  Advocate: {crux.claim_a.text}")
print(f"  Skeptic:  {crux.claim_b.text}")
print(f"  Stance:   {crux.stance} (score: {crux.score:.3f})")
print(f"  Centrality A: {crux.centrality_a:.3f}")
print(f"  Centrality B: {crux.centrality_b:.3f}")