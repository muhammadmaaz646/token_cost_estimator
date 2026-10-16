# Token Cost Estimator

> **The Problem:** If you're building LLM-backed features, you often don't know what they'll cost until the bill arrives. You can't easily stop a feature from blowing its budget, and tracking usage across different parts of your app is a manual mess. This estimator acts as a governance layer to predict, track, and block excessive spend before it happens.

## 🏗️ Architecture

A production-patterned FastAPI service that wraps Gemini API calls with a "refuse-before-spend" workflow. It calculates projected costs using real tokenizers and enforces monthly budget ceilings per endpoint.

```text
┌─────────────┐      1. estimate_cost()       ┌──────────────────┐
│             │ ─────────────────────────────▶│ cost_estimator.py │
│             │                                └──────────────────┘
│             │      2. check_budget()         ┌──────────────────┐
│  FastAPI    │ ─────────────────────────────▶│ usage_tracker.py  │
│  (main.py)  │      (SQLite Check)            │   (usage.db)      │
│             │                                └──────────────────┘
│             │      3. Real Gemini Call
│             │ ─────────────────────────────▶ Gemini API
│             │
│             │      4. Log Actual Cost        ┌──────────────────┐
│             │ ─────────────────────────────▶│ usage_tracker.py  │
└─────────────┘                                └──────────────────┘
```

---

## 🛠️ Tech Stack

- **Framework:** FastAPI
- **Web Server:** Uvicorn
- **AI SDK:** Google GenAI Python SDK
- **Database:** SQLite (Persistent usage logging)
- **Environment:** python-dotenv
- **Validation:** Pydantic

---

## 💡 Decisions Made & Solved Problems

- **Refuse-before-spend:** Implemented a pre-call estimation check that blocks requests (402 Payment Required) _before_ they hit the Gemini API if the budget is exceeded.
- **Accuracy vs. Reliability:** Prioritized accuracy by using Gemini's real `count_tokens` API for estimation, with a safe character-based heuristic fallback if the API is unreachable.
- **Endpoint Budgeting:** Designed the system to track costs by logical "endpoints" (e.g., `chatbot`, `summarizer`), allowing different features to have independent financial ceilings.
- **Audit Logging:** Implemented a full audit trail in SQLite that reconciles estimated vs. actual costs, identifying discrepancies and tracking spend trends.

---

## Code explanation

- At the start of each code file

## 🚀 Setup & How to Run

### Prerequisites

- Python 3.10+
- A Google Gemini API Key

### Installation

1. **Activate Virtual Environment**

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**

   ```powershell
   pip install -r requirements.txt
   ```

3. **Set API Key**
   Create a `.env` file:

   ```env
   GEMINI_API_KEY="your-actual-api-key-here"
   ```

4. **Run Application**

   ```powershell
   uvicorn main:app --reload
   ```

5. **Access**
   Open your browser and navigate to: `http://127.0.0.1:8000/docs`
