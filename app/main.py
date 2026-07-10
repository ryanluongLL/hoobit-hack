import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

from app.crux_engine import Models, find_crux, classify_claim_type
from app.agents import generate_arguments

models: Models = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    print("Loading models at startup...")
    models = Models()
    print("Models ready.")
    yield

app = FastAPI(title="Crux Detector API", lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    topic: str

class ClaimOut(BaseModel):
    id: str
    text: str
    agent: str
    claim_type: str
    depends_on: str

class StancePairOut(BaseModel):
    claim_a_id: str
    claim_b_id: str
    stance: str
    score: float

class CruxOut(BaseModel):
    claim_a: ClaimOut
    claim_b: ClaimOut
    stance: str
    score: float
    centrality_a: float
    centrality_b: float

class AnalyzeResponse(BaseModel):
    topic: str
    advocate_claims: list[ClaimOut]
    skeptic_claims: list[ClaimOut]
    all_pairs: list[StancePairOut]
    crux: CruxOut

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": models is not None}

@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("5/minute")
async def analyze(request: Request, body: AnalyzeRequest):
    if not body.topic or len(body.topic.strip()) < 10:
        raise HTTPException(status_code=400, detail="Topic must be at least 10 characters.")

    if len(body.topic) > 300:
        raise HTTPException(status_code=400, detail="Topic is too long, keep it under 300 characters.")

    if models is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    advocate_claims, skeptic_claims = generate_arguments(body.topic)

    for claim in advocate_claims:
        if not claim.claim_type:
            claim.claim_type = classify_claim_type(claim.text, models)
    for claim in skeptic_claims:
        if not claim.claim_type:
            claim.claim_type = classify_claim_type(claim.text, models)

    crux, all_pairs = find_crux(advocate_claims, skeptic_claims, models)

    return AnalyzeResponse(
        topic=body.topic,
        advocate_claims=[ClaimOut(**c.__dict__) for c in advocate_claims],
        skeptic_claims=[ClaimOut(**c.__dict__) for c in skeptic_claims],
        all_pairs=[StancePairOut(**p.__dict__) for p in all_pairs],
        crux=CruxOut(
            claim_a=ClaimOut(**crux.claim_a.__dict__),
            claim_b=ClaimOut(**crux.claim_b.__dict__),
            stance=crux.stance,
            score=crux.score,
            centrality_a=crux.centrality_a,
            centrality_b=crux.centrality_b,
        ),
    )