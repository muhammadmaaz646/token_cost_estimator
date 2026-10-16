"""
FastAPI layer — this is what makes it "a production API wrapper" and
not just a script. One route: POST /generate.

Flow for every request:
    1. Estimate cost (before spending anything)
    2. Check budget — refuse the call if it would blow past the limit
    3. Make the real Gemini call
    4. Compute actual cost from the real response
    5. Log it to SQLite
    6. Return the text + full cost breakdown to the caller

Run with:
    uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from google import genai
import sqlite3
from pathlib import Path

from cost_estimator import estimate_cost, actual_cost, DEFAULT_MODEL, PRICING_PER_MILLION
from usage_tracker import init_db, log_usage, get_total_spent, check_budget, get_budget_limit, DB_PATH

app = FastAPI(title="Token Cost Estimator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


class GenerateRequest(BaseModel):
    endpoint: str = Field(
        ...,
        description="Logical name for this call site, e.g. 'chatbot', "
        "'summarizer'. Used to group budget tracking.",
    )
    prompt: str
    model: str = DEFAULT_MODEL
    max_output_tokens: int = 1024


class GenerateResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    actual_cost: float
    total_spent_this_endpoint: float
    budget_limit: float
    budget_status: str


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    # 1. Estimate BEFORE spending anything
    estimate = estimate_cost(
        req.prompt, model=req.model, max_output_tokens=req.max_output_tokens
    )

    # 2. Budget check — block the call entirely if it would exceed
    # the limit. This is the enforcement half of "alerts on budget
    # limits" — a real alert (email/Slack/log aggregator) would also
    # fire here in a production system; this raises an HTTP error,
    # which is the FastAPI-native way to refuse a request.
    status, current_spent, limit = check_budget(
        req.endpoint, estimate.estimated_total_cost
    )
    if status == "exceeded":
        raise HTTPException(
            status_code=402,  # 402 Payment Required — the closest standard code for this
            detail=(
                f"Budget exceeded for endpoint '{req.endpoint}': "
                f"${current_spent:.4f} already spent this month, "
                f"limit is ${limit:.2f}, this call could add "
                f"${estimate.estimated_total_cost:.4f}."
            ),
        )

    # 3. The real call
    client = genai.Client()
    response = client.models.generate_content(
        model=req.model, contents=req.prompt
    )

    # 4. Real cost from the real response
    real = actual_cost(
        model=req.model,
        input_tokens=response.usage_metadata.prompt_token_count,
        output_tokens=response.usage_metadata.candidates_token_count,
    )

    # 5. Log it
    log_usage(
        endpoint=req.endpoint,
        model=req.model,
        input_tokens=real.input_tokens,
        output_tokens=real.output_tokens,
        estimated_cost=estimate.estimated_total_cost,
        actual_cost=real.total_cost,
    )

    # 6. Return everything the caller needs to see, including where
    # this leaves the budget now that the real cost is known.
    new_total = current_spent + real.total_cost
    final_status = (
        "exceeded"
        if new_total > limit
        else "warning"
        if new_total > limit * 0.8
        else "ok"
    )

    return GenerateResponse(
        text=response.text,
        model=req.model,
        input_tokens=real.input_tokens,
        output_tokens=real.output_tokens,
        estimated_cost=estimate.estimated_total_cost,
        actual_cost=real.total_cost,
        total_spent_this_endpoint=new_total,
        budget_limit=limit,
        budget_status=final_status,
    )


@app.get("/usage/{endpoint}")
def usage(endpoint: str) -> dict:
    """Quick spend check for one endpoint without making any model call."""
    spent = get_total_spent(endpoint)
    limit = get_budget_limit(endpoint)
    return {
        "endpoint": endpoint,
        "spent_this_month": round(spent, 6),
        "budget_limit": limit,
        "remaining": round(limit - spent, 6),
        "percent_used": round((spent / limit) * 100, 1) if limit else None,
    }


@app.get("/models")
def list_models() -> dict:
    """Return available models and their pricing."""
    return {
        "models": list(PRICING_PER_MILLION.keys()),
        "pricing": PRICING_PER_MILLION,
        "default_model": DEFAULT_MODEL,
    }


@app.get("/history")
def history(endpoint: str = None, limit: int = 20) -> list:
    """Get recent usage logs."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if endpoint:
            rows = conn.execute(
                "SELECT * FROM usage_log WHERE endpoint = ? ORDER BY id DESC LIMIT ?",
                (endpoint, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM usage_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]


@app.get("/")
def serve_frontend():
    """Serve the frontend single-page app."""
    frontend_path = Path(__file__).parent / "static" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "Token Cost Estimator API is running. static/index.html not found."}
