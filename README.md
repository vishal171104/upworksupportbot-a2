# Upwork API Support Bot

A retrieval-augmented (RAG) support assistant for the Upwork API documentation.
Ask a developer question in plain English; the bot answers **only** from the
indexed documentation and shows you the exact chunks it used.

The interesting constraint here is refusal. The system prompt forbids the model
from using its own training data — if the retrieved context doesn't contain the
answer, the bot is required to say so verbatim rather than improvise a
plausible-sounding API that doesn't exist:

> `I'm sorry, but the provided documentation does not contain that information.`

That trade-off — fewer answers, but no invented endpoints — is the whole point
of the project.

## How it works

```
PDF docs ──► PyPDFLoader ──► RecursiveCharacterTextSplitter (500 / 50 overlap)
                                        │
                                        ▼
                       all-MiniLM-L6-v2 sentence embeddings
                                        │
                                        ▼
                            ChromaDB (persistent, on disk)
                                        │
   question ──► similarity_search(k=3) ─┘
                                        │
                                        ▼
              Llama 3.1 8B Instruct (DeepInfra, temperature=0)
                                        │
                                        ▼
                    answer + source chunks + latency
```

| File | Responsibility |
|---|---|
| `ingest.py` | Loads the PDF, chunks it, embeds it, and (re)builds the Chroma collection |
| `rag.py` | Retrieval, the grounding system prompt, and the LLM call |
| `app.py` | Streamlit UI — auto-runs ingestion on first launch, renders answer + sources + latency |
| `config.py` | Env/secrets handling, shared between local `.env` and Streamlit Cloud secrets |

Retrieval is top-3 by cosine similarity. The LLM runs at `temperature=0` so the
same question gives the same answer — necessary if you want to reason about
failures at all.

## Stack

Python · LangChain · ChromaDB · Sentence-Transformers (`all-MiniLM-L6-v2`) ·
Streamlit · Llama 3.1 8B Instruct via DeepInfra

Embeddings run locally on CPU; only the generation step hits an API.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set two values in `.env`:

| Variable | Meaning |
|---|---|
| `DEEPINFRA_API_KEY` | Your DeepInfra API key (used as an OpenAI-compatible endpoint) |
| `DOCS_PATH` | Path to the Upwork API documentation PDF, e.g. `docs/upwork_api_documentation.pdf` |

Then:

```bash
streamlit run app.py
```

The app builds the vector store on first run if `./chroma_db` doesn't exist yet.
To rebuild the index by hand after changing the source PDF:

```bash
python ingest.py
```

### Deploying to Streamlit Cloud

Add `DEEPINFRA_API_KEY` and `DOCS_PATH` in the app's **Secrets** settings.
`config.sync_streamlit_secrets()` copies them into `os.environ` so the
ingestion subprocess inherits them.

## Notes and limitations

- `docs/` ships with a **partial** copy of the API documentation, so coverage is
  deliberately incomplete — expect the refusal path to trigger often. That is
  correct behaviour, not a bug.
- Chunking is fixed at 500 characters with 50 overlap. Tables and long code
  samples in the PDF can straddle a chunk boundary and lose context.
- `k=3` keeps the prompt small and latency low; recall-heavy questions that need
  several sections at once are the main failure mode.
- The Chroma collection is rebuilt from scratch on every `ingest.py` run — there
  is no incremental update path.
