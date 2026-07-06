

from dotenv import load_dotenv
load_dotenv()

from app.agents import generate_arguments
from app.crux_engine import Models, find_crux

advocate_claims, skeptic_claims = generate_arguments("4 day work week increases productivity")

print("Advocate claims:")
for c in advocate_claims:
    print(f"  [{c.id}] {c.text}")

print("\nSkeptic claims:")
for c in skeptic_claims:
    print(f"  [{c.id}] {c.text}")

print("\nLoading models and finding crux...")
models = Models()
crux, all_pairs = find_crux(advocate_claims, skeptic_claims, models)

print(f"\nCRUX FOUND:")
print(f"  Advocate: {crux.claim_a.text}")
print(f"  Skeptic:  {crux.claim_b.text}")
print(f"  Stance:   {crux.stance} (score: {crux.score:.3f})")