import os
from dotenv import load_dotenv
import re
import pandas as pd
import tiktoken
from sentence_transformers import SentenceTransformer
import faiss
import openai
import json
import numpy as np
from typing import Optional


# ─── Step 0: Configuration & Constants ───────────────────────────────────────
# Load environment variables
load_dotenv()  
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment")

# Define cache paths

CACHE_DIR = "./rag_cache" 
CHUNKS_PATH = os.path.join(CACHE_DIR, "spotify_review_chunks.csv")
EMB_PATH    = os.path.join(CACHE_DIR, "embeddings.npy")
IDX_PATH    = os.path.join(CACHE_DIR, "index.faiss")
MODEL_NAME  = "sentence-transformers/all-mpnet-base-v2"



# ─── Step 1: Data Preprocessing ──────────────────────────────────────────────
def load_reviews_csv(
        filepath: str,
        text_column: str,
        nrows: Optional[int] = None,
        sample_n: Optional[int] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
    """Load reviews, keeping only *text_column*.
       - nrows: read first N rows (mutually exclusive with sample_n)
       - sample_n: draw a random sample of size N after full load
    """
    df = pd.read_csv(filepath, usecols=[text_column], nrows=nrows)
    df = df.dropna(subset=[text_column])

    if sample_n is not None:
        df = df.sample(n=sample_n, random_state=random_state)

    return df.reset_index(drop=True)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─── Step 2: Text Chunking ───────────────────────────────────────────────────
def chunk_text(text: str) -> list:
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
    except:
        enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(text)
    chunks = []
    max_tokens, overlap = 300, 50
    start, n = 0, len(ids)
    while start < n:
        end = min(start + max_tokens, n)
        chunks.append(enc.decode(ids[start:end]))
        if end == n:
            break
        start = end - overlap
    return chunks

# ─── Step 3: Embedding Generation ────────────────────────────────────────────
def embed_texts(texts: list):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(emb)
    return model, emb

# ─── Step 4: Vector Indexing ─────────────────────────────────────────────────
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

# ─── Step 5: Semantic Retrieval ──────────────────────────────────────────────
def retrieve_top_k(query: str, embed_model, index, chunks: list, k: int = 8):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [(chunks[i], float(D[0][j])) for j,i in enumerate(I[0])]


# ─── Step 6: Response Generation ─────────────────────────────────────────────
# ─── Prompt template ────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
### 📄 Prompt Template — *Review-Aware RAG Analyst*

SYSTEM
You are *InsightGPT*, an expert reviewer-analysis assistant. You **only** draw conclusions
that are supported by the review snippets supplied in CONTEXT.
If evidence is missing, reply “Insufficient information in the provided reviews.”
Do **not** hallucinate.

──────────────────────────────────────────────────────────────────────────────
USER REQUEST
──────────────────────────────────────────────────────────────────────────────
QUESTION:
{question}

CONTEXT:
{context_json}

DATASET_META:
{{"name": "{dataset_name}"}}

REQUEST_TYPE: {request_type}
──────────────────────────────────────────────────────────────────────────────

ASSISTANT INSTRUCTIONS
1. Understand the task
   • If REQUEST_TYPE == summary → produce a concise dataset synopsis (§4).
   • Else assume Q&A.

2. Evidence first
   • Skim CONTEXT and select the *minimum* snippets that directly support the answer.
   • Never invent content beyond CONTEXT.

3. Reasoning
   • Briefly explain *why* the chosen snippets answer the question.
   • Use short logical steps; no long essays.

4. Output format
   ANSWER: <single sentence or 4-5 bullets>
   EVIDENCE:
     - [idx 43] “There’s no way to …”
     - [idx 22] “I can’t find …”
   INSIGHTS (optional, ≤3 bullets):
     • <actionable takeaway>

   For REQUEST_TYPE == summary use:
   SUMMARY (max 7 bullets, each with one evidence idx):
     • <theme> – e.g., “Crashes on launch” [idx 37593]
     • <theme> – e.g., “Lag when scrubbing” [idx 46842]
   TOP_EXAMPLES:
     - [idx 165] “…”
     - [idx 404] “…”

5. Style rules
   • Plain English, no jargon.  ≤20 words per bullet.
   • Quote snippets verbatim; trim with “…” if >120 chars.
   • List indices in square brackets exactly as given.
"""

def generate_answer(
        query: str,
        retrieved: list,
        dataset_name: str = "Spotify_reviews",
        request_type: str = "answer",
        openai_model: str = "gpt-4o"   # or "gpt-4"
    ) -> str:

    # Build CONTEXT list limited to 15 snippets
    context_objs = []
    for chunk, _ in retrieved[:15]:
        context_objs.append({
            "idx" : chunk["review_id"],
            "text": chunk["text"].replace("\n", " ").strip()
        })

    # Fill in the big template
    filled_prompt = PROMPT_TEMPLATE.format(
        question      = query.strip(),
        context_json  = json.dumps(context_objs, indent=2, ensure_ascii=False),
        dataset_name  = dataset_name,
        request_type  = request_type
    )

    # Send as a single user message (system role already in template)
    resp = openai.ChatCompletion.create(
        model       = openai_model,
        messages    = [{"role": "user", "content": filled_prompt}],
        temperature = 0.0,
        max_tokens  = 512
    )
    return resp.choices[0].message.content.strip()


# ─── Step 7: Main Pipeline ───────────────────────────────────────────────────
def main():
    """End-to-end RAG workflow execution"""
    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Phase 1: Data Preparation
    if os.path.exists(CHUNKS_PATH) and os.path.exists(EMB_PATH) and os.path.exists(IDX_PATH):
        print("Loading cached chunks, embeddings, and index…")
        chunks_df = pd.read_csv(CHUNKS_PATH)
        chunks     = chunks_df.to_dict("records")        # list[dict] for retrieval

        embeddings  = np.load(EMB_PATH)
        index       = faiss.read_index(IDX_PATH)
        embed_model = SentenceTransformer(MODEL_NAME)           # recreate for queries
    else:
        # Process data from scratch
        # Step 1: Load and clean raw data
        df = load_reviews_csv("Spotify_reviews.csv", text_column="Review", nrows=None)

        if "review_id" not in df:
            df["review_id"] = df.index.astype(str)

        df["clean_text"] = df["Review"].apply(clean_text)

        # Step 2: Chunk text
        chunks = []
        for _, row in df.iterrows():
            for sub in chunk_text(row["clean_text"]):
                chunks.append({"review_id": row["review_id"], "text": sub})
        
        # Step 3: Generate embeddings

        embed_model           = SentenceTransformer(MODEL_NAME)
        texts                 = [c["text"] for c in chunks]
        embeddings            = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(embeddings)

        # Step 4: Build index
        index                 = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # Cache processed data
        pd.DataFrame(chunks).to_csv(CHUNKS_PATH, index=False)
        np.save(EMB_PATH, embeddings)
        faiss.write_index(index, IDX_PATH)
        print("Cached chunks, embeddings, and index.")

    # Phase 2: Interactive Querying
    print("\nIndex ready. Ask me questions about the reviews! (type 'quit' to exit)")
    while True:
        # Get user input
        query = input("\nYour question: ").strip()
        if query.lower() in ("quit", "exit"):
            print("Good-bye!")
            break

        # Determine request type
        # — Detect summary/overview requests by keyword, too
        ql = query.lower()
        if ql.startswith("summary:") or "overview" in ql or "general view" in ql:
            request_type = "summary"
            # strip out leading keywords so the prompt is clean
            query = re.sub(r"^(summary:|overview of|general view of)\s*", "", query, flags=re.I).strip()
        else:
            request_type = "answer"

        # Step 5: Retrieve relevant chunks    
            # Step 5: Retrieve relevant chunks    
        k = 50 if request_type == "summary" else 15
        topk = retrieve_top_k(query, embed_model, index, chunks, k=k)


        # Step 6: Generate and display response
        response  = generate_answer(query, topk,
                                    dataset_name="Spotify_reviews",
                                    request_type=request_type)
        print("\n" + response + "\n")

# ─── Execution Entry Point ───────────────────────────────────────────────────
if __name__ == "__main__":
    main()
