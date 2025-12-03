from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import ollama
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import requests
import certifi
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load environment variables
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "migrated",
    "user": "postgres",
    "password": "1234",
    "port": 5432
}
DB_CONFIG2 = {
    "host": "localhost",
    "database": "eGazette",
    "user": "postgres",
    "password": "1234",
    "port": 5432
}

# API Configuration
API_KEY = os.getenv("API_KEY", "").strip()
BASE_URL = os.getenv("BASE_URL", "https://api.studio.nebius.com/v1/")
API_MODEL = os.getenv("model", "moonshotai/Kimi-K2-Instruct")

# Local Ollama Models
EMBED_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "gemma3:4b"

# Hybrid Search Configuration
VECTOR_WEIGHT = 0.5
BM25_WEIGHT = 0.5

# Determine which mode to use
USE_API = bool(API_KEY)

print(f"🚀 Starting in {'API' if USE_API else 'LOCAL OLLAMA'} mode")
if USE_API:
    print(f"   Using model: {API_MODEL}")
else:
    print(f"   Using local models: {CHAT_MODEL} (chat), {EMBED_MODEL} (embeddings)")
print(f"🔍 Hybrid Search: Vector Weight={VECTOR_WEIGHT}, BM25 Weight={BM25_WEIGHT}")
print(f"📅 Date Filter: ENABLED")

# Global BM25 index (will be loaded on startup)
bm25_index = None
corpus_chunks = []
metadata_cache = {}  # Cache for metadata lookups

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict[str, str]] = []
    start_date: Optional[str] = None  # YYYY-MM-DD format
    end_date: Optional[str] = None    # YYYY-MM-DD format

class ChatResponse(BaseModel):
    response: str
    main_sources: List[Dict]
    other_sources: List[Dict]
    filter_info: Dict  # Info about applied filters

class DateFilterRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class DateFilterResponse(BaseModel):
    start_date: Optional[str]
    end_date: Optional[str]
    total_parent_ids: int
    total_chunks: int
    document_count: int
    documents: List[Dict]

def get_db_connection():
    """Create database connection"""
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    return conn

def get_db_connection2():
    """Create database connection"""
    conn2 = psycopg2.connect(**DB_CONFIG2, cursor_factory=RealDictCursor)
    return conn2

def tokenize_text(text: str) -> List[str]:
    """Tokenize text for BM25"""
    try:
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        return tokens
    except:
        return text.lower().split()

def get_parent_ids_by_date_range(start_date: Optional[str] = None, end_date: Optional[str] = None) -> tuple:
    """
    Get parent_ids (serialno) matching the date range from metadata table
    Returns: (parent_ids_list, metadata_dict)
    """
    conn = get_db_connection()
    conn2 = get_db_connection2()
    try:
        with conn2.cursor() as cur2:
            if start_date and end_date:
                # Convert string dates to proper format
                try:
                    start = datetime.strptime(start_date, "%Y-%m-%d").date()
                    end = datetime.strptime(end_date, "%Y-%m-%d").date()
                except ValueError:
                    raise ValueError("Dates must be in YYYY-MM-DD format")
                
                # Query metadata table for date range
                cur2.execute("""
                    SELECT 
                        serialno,
                        publicationdate
                    FROM gazettes
                    WHERE gazettes.publicationdate::date BETWEEN %s AND %s
                    ORDER BY publicationdate DESC
                """, (start, end))
            else:
                # No date filter - get all
                cur2.execute("""
                    SELECT 
                        serialno,
                        publicationdate
                    FROM gazettes
                    ORDER BY publicationdate DESC
                """)
            
            results = cur2.fetchall()
            metadata_dict = {}
            parent_ids = []
            
            for row in results:
                serialno = row['serialno']
                parent_ids.append(serialno)
                metadata_dict[serialno] = {
                    'serialno': serialno,
                    'publicationdate': row['publicationdate'].isoformat() if row['publicationdate'] else None,
                }
             
            print(f"📅 Date filter: Found {len(parent_ids)} documents in range")
            return parent_ids, metadata_dict
    
    finally:
        conn2.close()

def load_bm25_index(parent_ids: Optional[List[int]] = None):
    """
    Load documents and create BM25 index, optionally filtered by parent_ids
    """
    global bm25_index, corpus_chunks, metadata_cache
    
    print("📚 Loading BM25 index...")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if parent_ids:
                # Use parameterized query instead of string formatting
                placeholders = ','.join(['%s'] * len(parent_ids))
                cur.execute(f"""
                    SELECT 
                        id,
                        content,
                        source,
                        parent_id,
                        chunk_index,
                        total_chunks
                    FROM document_embeddings
                    WHERE parent_id IN ({placeholders})
                    ORDER BY parent_id, chunk_index
                """, parent_ids)
            else:
                # Load all
                cur.execute("""
                    SELECT 
                        id,
                        content,
                        source,
                        parent_id,
                        chunk_index,
                        total_chunks
                    FROM document_embeddings
                    ORDER BY parent_id, chunk_index
                """)
            
            results = cur.fetchall()
            corpus_chunks = [dict(row) for row in results]
            
            # Tokenize all documents
            tokenized_corpus = [tokenize_text(chunk['content']) for chunk in corpus_chunks]
            
            # Create BM25 index
            bm25_index = BM25Okapi(tokenized_corpus)
            
            print(f"✅ BM25 index loaded with {len(corpus_chunks)} chunks")
    finally:
        conn.close()

def get_embedding(text: str) -> List[float]:
    """Generate embedding using Ollama"""
    try:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Error generating embedding")

def search_vector(
    query_embedding: List[float], 
    parent_ids: Optional[List[int]] = None,
    top_k: int = 20
) -> List[Dict]:
    """
    Search for similar chunks using cosine similarity, optionally filtered by parent_ids
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            vector_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            if parent_ids:
                # Use parameterized query
                placeholders = ','.join(['%s'] * len(parent_ids))
                cur.execute(f"""
                    SELECT 
                        id,
                        content,
                        source,
                        parent_id,
                        chunk_index,
                        total_chunks,
                        embedding <=> %s::vector AS distance
                    FROM document_embeddings
                    WHERE parent_id IN ({placeholders})
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (vector_str, *parent_ids, vector_str, top_k))
            else:
                cur.execute("""
                    SELECT 
                        id,
                        content,
                        source,
                        parent_id,
                        chunk_index,
                        total_chunks,
                        embedding <=> %s::vector AS distance
                    FROM document_embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (vector_str, vector_str, top_k))
            
            results = cur.fetchall()
            chunks = []
            for row in results:
                chunk = dict(row)
                chunk['vector_score'] = 1 - chunk['distance']
                chunk['distance'] = chunk['distance']
                chunks.append(chunk)
            
            return chunks
    finally:
        conn.close()

def search_bm25(query: str, parent_ids: Optional[List[int]] = None, top_k: int = 20) -> List[Dict]:
    """Search using BM25 keyword matching, optionally filtered by parent_ids"""
    global bm25_index, corpus_chunks
    
    if bm25_index is None:
        load_bm25_index(parent_ids)
    
    tokenized_query = tokenize_text(query)
    scores = bm25_index.get_scores(tokenized_query)
    
    # If parent_ids filter is provided, zero out scores for excluded documents
    if parent_ids:
        parent_ids_set = set(parent_ids)
        for i, chunk in enumerate(corpus_chunks):
            if chunk['parent_id'] not in parent_ids_set:
                scores[i] = 0
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include positive scores
            chunk = corpus_chunks[idx].copy()
            chunk['bm25_score'] = float(scores[idx])
            results.append(chunk)
    
    return results

def normalize_scores(chunks: List[Dict], score_key: str) -> List[Dict]:
    """Normalize scores to 0-1 range"""
    if not chunks:
        return chunks
    
    scores = [chunk[score_key] for chunk in chunks]
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score - min_score == 0:
        for chunk in chunks:
            chunk[f'{score_key}_normalized'] = 1.0
    else:
        for chunk in chunks:
            chunk[f'{score_key}_normalized'] = (chunk[score_key] - min_score) / (max_score - min_score)
    
    return chunks

def hybrid_search(
    query: str,
    query_embedding: List[float],
    parent_ids: Optional[List[int]] = None,
    top_k: int = 20
) -> List[Dict]:
    """
    Perform hybrid search combining vector and BM25 results, with optional parent_id filtering
    """
    print(f"🔍 Performing hybrid search (Vector: {VECTOR_WEIGHT}, BM25: {BM25_WEIGHT})...")
    if parent_ids:
        print(f"   Filtering by {len(parent_ids)} parent_ids")
    
    # Get results from both methods
    vector_results = search_vector(query_embedding, parent_ids, top_k=top_k)
    bm25_results = search_bm25(query, parent_ids, top_k=top_k)
    
    print(f"   Vector results: {len(vector_results)}")
    print(f"   BM25 results: {len(bm25_results)}")
    
    # Normalize scores
    vector_results = normalize_scores(vector_results, 'vector_score')
    bm25_results = normalize_scores(bm25_results, 'bm25_score')
    
    # Combine results
    combined_results = {}
    
    for chunk in vector_results:
        chunk_id = chunk['id']
        combined_results[chunk_id] = chunk.copy()
        combined_results[chunk_id]['vector_score_normalized'] = chunk['vector_score_normalized']
        combined_results[chunk_id]['has_vector'] = True
        combined_results[chunk_id]['has_bm25'] = False
    
    for chunk in bm25_results:
        chunk_id = chunk['id']
        if chunk_id in combined_results:
            combined_results[chunk_id]['bm25_score'] = chunk['bm25_score']
            combined_results[chunk_id]['bm25_score_normalized'] = chunk['bm25_score_normalized']
            combined_results[chunk_id]['has_bm25'] = True
        else:
            combined_results[chunk_id] = chunk.copy()
            combined_results[chunk_id]['bm25_score_normalized'] = chunk['bm25_score_normalized']
            combined_results[chunk_id]['has_vector'] = False
            combined_results[chunk_id]['has_bm25'] = True
            combined_results[chunk_id]['vector_score_normalized'] = 0.0
    
    # Calculate hybrid scores
    for chunk_id, chunk in combined_results.items():
        vector_score = chunk.get('vector_score_normalized', 0.0)
        bm25_score = chunk.get('bm25_score_normalized', 0.0)
        chunk['hybrid_score'] = (VECTOR_WEIGHT * vector_score) + (BM25_WEIGHT * bm25_score)
        chunk['similarity'] = chunk['hybrid_score']
    
    # Sort by hybrid score
    sorted_chunks = sorted(combined_results.values(), key=lambda x: x['hybrid_score'], reverse=True)
    
    print(f"   Combined results: {len(sorted_chunks)}")
    
    return sorted_chunks

def build_context_with_citations(chunks: List[Dict], metadata_cache: Dict) -> str:
    """Build context string with source citations"""
    context_parts = []
    for chunk in chunks:
        source_label = f"[Source: {chunk['source']}, Chunk {chunk['chunk_index']}/{chunk['total_chunks']}]"
        context_parts.append(f"{source_label}:\n{chunk['content']}")
    
    return "\n\n".join(context_parts)

def generate_response_api(
    query: str,
    context_chunks: List[Dict],
    conversation_history: List[Dict],
    metadata_cache: Dict
) -> str:
    """Generate response using external API"""
    context_text = build_context_with_citations(context_chunks, metadata_cache)
    
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that answers questions based on the provided context.

CITATION RULES:
1. Citation Format: [Source: filename, Chunk X/Y]
2. Answer ONLY based on the provided context.
3. **FORMATTING:** Use Markdown to structure your answer. Use **bold** for key terms, bullet points for lists, and ### Headers for sections.
4. Be concise.
"

Context:
{context_text}"""
        }
    ]
    
    for msg in conversation_history[-6:]:
        messages.append(msg)
    
    messages.append({
        "role": "user",
        "content": query
    })
    
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": API_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1000
        }
        
        response = requests.post(
            f"{BASE_URL.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
            verify=certifi.where()
        )
        
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
    except Exception as e:
        print(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")

def generate_response_ollama(
    query: str,
    context_chunks: List[Dict],
    conversation_history: List[Dict],
    metadata_cache: Dict
) -> str:
    """Generate response using local Ollama"""
    context_text = build_context_with_citations(context_chunks, metadata_cache)
    
    messages = [
        {
            "role": "system",
            "content": f"""
            You are a helpful assistant that answers questions based on the provided context.

CITATION RULES:
1. Citation Format: [Source: filename, Chunk X/Y]
2. Answer ONLY based on the provided context.
3. **FORMATTING:** Use Markdown to structure your answer. Use **bold** for key terms, bullet points for lists, and ### Headers for sections.
4. Be concise.
"

Context:
{context_text}
"""
#[Source: filename, Chunk X/Y]
        }
    ]
    
    for msg in conversation_history[-6:]:
        messages.append(msg)
    
    messages.append({
        "role": "user",
        "content": query
    })
    
    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=messages
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")

def generate_response(
    query: str,
    context_chunks: List[Dict],
    conversation_history: List[Dict],
    metadata_cache: Dict
) -> str:
    """Generate response using either API or Ollama"""
    if USE_API:
        return generate_response_api(query, context_chunks, conversation_history, metadata_cache)
    else:
        return generate_response_ollama(query, context_chunks, conversation_history, metadata_cache)

from dataclasses import dataclass
import re

@dataclass
class IntentResult:
    intent: str  # one of: fact_finding, summarization, monitoring, comparison, general
    needs_retrieval: bool
    raw: dict | None = None

def safe_json_extract(s: str) -> dict:
    """Try to extract a JSON object from string `s`. Returns dict or raises."""
    # fast attempt
    try:
        return json.loads(s)
    except Exception:
        pass
    # fallback: extract first {...} block
    m = re.search(r"(\{(?:[^{}]|(?R))*\})", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # as final fallback, try simple key:value extraction
    # return empty to allow fallback behavior
    return {}

def _build_intent_prompt(user_query: str) -> str:
    """Return the prompt asking the model to return intent JSON only."""
    return f"""
You are an intent classifier for a Gazette Q&A system. Reply with a single JSON object and nothing else.

Output schema:
{{
  "intent": "<one of: fact_finding, summarization, monitoring, comparison, general>",
  "needs_retrieval": <true|false>   // whether the query requires searching local documents/gazettes
}}

Rules:
- intent "general" means the user wants a conversational answer or opinion and does NOT require retrieving gazette content.
- For any question that asks for factual details, dates, official statements, laws, or "what does X say", choose "fact_finding" and set needs_retrieval=true.
- For "summarize", "give a summary", choose "summarization" and set needs_retrieval=true.
- For "compare", "difference between", choose "comparison" and set needs_retrieval=true.
- For "monitor", "latest update", "show changes", choose "monitoring" and set needs_retrieval=true.
- If unsure, set needs_retrieval=true (safer).

Query: \"\"\"{user_query}\"\"\"
Return only the JSON object.
"""

def detect_intent_api(user_query: str) -> IntentResult:
    """Detect intent using external API (if USE_API)."""
    prompt = _build_intent_prompt(user_query)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": "You are an intent classifier for a Gazette Q&A system."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 80
    }
    try:
        resp = requests.post(f"{BASE_URL.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=20, verify=certifi.where())
        resp.raise_for_status()
        data = resp.json()
        text = data['choices'][0]['message']['content']
        parsed = safe_json_extract(text)
        intent = parsed.get("intent", "general")
        needs = bool(parsed.get("needs_retrieval", parsed.get("needsRetrieval", None) or parsed.get("needs_retrieval", False)))
        return IntentResult(intent=intent, needs_retrieval=needs, raw=parsed)
    except Exception as e:
        print(f"Intent detection API failed: {e}")
        # fallback: simple heuristic (short queries -> general)
        if len(user_query.split()) <= 6:
            return IntentResult(intent="general", needs_retrieval=False, raw=None)
        return IntentResult(intent="fact_finding", needs_retrieval=True, raw=None)

def detect_intent_ollama(user_query: str) -> IntentResult:
    """Detect intent using local ollama model."""
    prompt = _build_intent_prompt(user_query)
    try:
        resp = ollama.chat(model=CHAT_MODEL, messages=[{"role": "system", "content": "You are an intent classifier."}, {"role": "user", "content": prompt}])
        text = resp.get('message', {}).get('content', '') or resp.get('content', '')
        parsed = safe_json_extract(text)
        intent = parsed.get("intent", "general")
        needs = bool(parsed.get("needs_retrieval", parsed.get("needsRetrieval", None) or parsed.get("needs_retrieval", False)))
        return IntentResult(intent=intent, needs_retrieval=needs, raw=parsed)
    except Exception as e:
        print(f"Intent detection (ollama) failed: {e}")
        if len(user_query.split()) <= 6:
            return IntentResult(intent="general", needs_retrieval=False, raw=None)
        return IntentResult(intent="fact_finding", needs_retrieval=True, raw=None)

def detect_intent(user_query: str) -> IntentResult:
    """Main wrapper that chooses API or local."""
    if USE_API:
        return detect_intent_api(user_query)
    else:
        return detect_intent_ollama(user_query)

def generate_general_response_api(query: str, conversation_history: List[Dict]) -> str:
    messages = [
        {"role": "system", "content": "You are a friendly assistant. Answer based on your knowledge and do not invent legal citations or claim to read gazettes."}
    ]
    for msg in conversation_history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": query})
    try:
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        payload = {"model": API_MODEL, "messages": messages, "temperature": 0.5, "max_tokens": 800}
        resp = requests.post(f"{BASE_URL.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=60, verify=certifi.where())
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        print(f"General API generation failed: {e}")
        raise HTTPException(status_code=500, detail="General generation failed")

def generate_general_response_ollama(query: str, conversation_history: List[Dict]) -> str:
    messages = [{"role": "system", "content": "You are a friendly assistant. Provide helpful general answers without citing gazettes."}]
    for msg in conversation_history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": query})
    try:
        r = ollama.chat(model=CHAT_MODEL, messages=messages)
        return r['message']['content']
    except Exception as e:
        print(f"General ollama generation failed: {e}")
        raise HTTPException(status_code=500, detail="General generation failed")

def generate_general_response(query: str, conversation_history: List[Dict]) -> str:
    if USE_API:
        return generate_general_response_api(query, conversation_history)
    else:
        return generate_general_response_ollama(query, conversation_history)


@app.on_event("startup")
async def startup_event():
    """Load BM25 index on startup (without date filter)"""
    load_bm25_index()

@app.get("/date-range", response_model=DateFilterResponse)
async def get_available_date_range():
    """Get available date range and document info from metadata"""
    try:
        parent_ids, metadata_dict = get_parent_ids_by_date_range()
        
        # Get document count and chunk info
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                parent_ids_str = ','.join(str(pid) for pid in parent_ids)
                cur.execute(f"""
                    SELECT COUNT(DISTINCT parent_id) as doc_count
                    FROM document_embeddings
                    WHERE parent_id IN ({parent_ids_str})
                """)
                result = cur.fetchone()
                doc_count = result['doc_count'] if result else 0
        finally:
            conn.close()
        
        documents = list(metadata_dict.values())
        
        return DateFilterResponse(
            start_date=None,
            end_date=None,
            total_parent_ids=len(parent_ids),
            total_chunks=len(corpus_chunks),
            document_count=doc_count,
            documents=documents
        )
    except Exception as e:
        print(f"Error getting date range: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/apply-date-filter", response_model=DateFilterResponse)
async def apply_date_filter(request: DateFilterRequest):
    """Apply date filter and return filtered document info"""
    try:
        parent_ids, metadata_dict = get_parent_ids_by_date_range(request.start_date, request.end_date)
        
        # Reload BM25 index with filtered parent_ids
        load_bm25_index(parent_ids)
        
        # Get chunk count
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                if parent_ids:
                    # Use parameterized query
                    placeholders = ','.join(['%s'] * len(parent_ids))
                    cur.execute(f"""
                        SELECT COUNT(*) as chunk_count
                        FROM document_embeddings
                        WHERE parent_id IN ({placeholders})
                    """, parent_ids)
                else:
                    cur.execute("SELECT COUNT(*) as chunk_count FROM document_embeddings")
                
                result = cur.fetchone()
                chunk_count = result['chunk_count'] if result else 0
        finally:
            conn.close()
        
        documents = list(metadata_dict.values())
        
        return DateFilterResponse(
            start_date=request.start_date,
            end_date=request.end_date,
            total_parent_ids=len(parent_ids),
            total_chunks=chunk_count,
            document_count=len(documents),
            documents=documents
        )
    except Exception as e:
        print(f"Error applying date filter: {e}")
        raise HTTPException(status_code=500, detail=str(e))

DISCLAIMER_TEXT = "\n\n> **Note:** AI generated response is not 100% accurate and may need to be rechecked from resource citation for further verification."

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with optional date filtering and intent-first flow"""
    try:
        # 1. date filter (same as before)
        parent_ids = None
        filter_info = {"applied": False, "start_date": None, "end_date": None, "filtered_documents": 0}
        global metadata_cache

        if request.start_date or request.end_date:
            parent_ids, metadata = get_parent_ids_by_date_range(request.start_date, request.end_date)
            filter_info["applied"] = True
            filter_info["start_date"] = request.start_date
            filter_info["end_date"] = request.end_date
            filter_info["filtered_documents"] = len(parent_ids)
            metadata_cache = metadata

        # 2. Detect intent FIRST
        intent_res = detect_intent(request.message)
        print(f"Intent detected: {intent_res.intent}, needs_retrieval={intent_res.needs_retrieval}")

        # 3. If intent says no retrieval, produce a general response and return immediately
        if not intent_res.needs_retrieval or intent_res.intent == "general":
            response_text = generate_general_response(request.message, request.conversation_history)
            return ChatResponse(
                response=response_text + DISCLAIMER_TEXT,
                main_sources=[],
                other_sources=[],
                filter_info=filter_info
            )

        # 4. Otherwise proceed with embedding + hybrid retrieval as before
        query_embedding = get_embedding(request.message)
        hybrid_results = hybrid_search(request.message, query_embedding, parent_ids=parent_ids, top_k=20)

        if not hybrid_results:
            return ChatResponse(response="I couldn't find any relevant information to answer your question." + DISCLAIMER_TEXT,
                                main_sources=[], other_sources=[], filter_info=filter_info)

        main_results = hybrid_results[:6]
        raw_response = generate_response(request.message, main_results, request.conversation_history, metadata_cache)


        final_response = raw_response + DISCLAIMER_TEXT

        # 5. Build main_sources & other_sources (same as your code)
        main_sources = []
        for chunk in main_results:
            source_info = {
                "id": chunk['id'],
                "source": chunk['source'],
                "parent_id": chunk['parent_id'],
                "chunk_index": chunk['chunk_index'],
                "total_chunks": chunk['total_chunks'],
                "content": chunk['content'],
                "similarity": round(chunk['similarity'], 3),
                "hybrid_score": round(chunk.get('hybrid_score', chunk['similarity']), 3),
                "vector_score": round(chunk.get('vector_score', 0), 3) if chunk.get('has_vector') else None,
                "bm25_score": round(chunk.get('bm25_score', 0), 3) if chunk.get('has_bm25') else None,
                "intent": intent_res.intent
            }
            if chunk['parent_id'] in metadata_cache:
                source_info['metadata'] = metadata_cache[chunk['parent_id']]
            main_sources.append(source_info)

        main_source_ids = {chunk['id'] for chunk in main_results}
        other_sources = []
        for chunk in hybrid_results[6:20]:
            if chunk['id'] not in main_source_ids:
                source_info = {
                    "id": chunk['id'],
                    "source": chunk['source'],
                    "parent_id": chunk['parent_id'],
                    "chunk_index": chunk['chunk_index'],
                    "total_chunks": chunk['total_chunks'],
                    "content": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                    "similarity": round(chunk['similarity'], 3),
                    "hybrid_score": round(chunk.get('hybrid_score', chunk['similarity']), 3),
                    "vector_score": round(chunk.get('vector_score', 0), 3) if chunk.get('has_vector') else None,
                    "bm25_score": round(chunk.get('bm25_score', 0), 3) if chunk.get('has_bm25') else None
                }
                if chunk['parent_id'] in metadata_cache:
                    source_info['metadata'] = metadata_cache[chunk['parent_id']]
                other_sources.append(source_info)
                if len(other_sources) >= 10:
                    break

        return ChatResponse(response=final_response, main_sources=main_sources, other_sources=other_sources, filter_info=filter_info)

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/chat", response_model=ChatResponse)
# async def chat(request: ChatRequest):
#     """Main chat endpoint with optional date filtering"""
#     try:
#         # Get filtered parent_ids if date range is provided
#         parent_ids = None
#         filter_info = {
#             "applied": False,
#             "start_date": None,
#             "end_date": None,
#             "filtered_documents": 0
#         }
        
#         if request.start_date or request.end_date:
#             parent_ids, metadata = get_parent_ids_by_date_range(request.start_date, request.end_date)
#             filter_info["applied"] = True
#             filter_info["start_date"] = request.start_date
#             filter_info["end_date"] = request.end_date
#             filter_info["filtered_documents"] = len(parent_ids)
#             global metadata_cache
#             metadata_cache = metadata
        
#         # Generate embedding for the query
#         query_embedding = get_embedding(request.message)
        
#         # Perform hybrid search with optional parent_id filtering
#         hybrid_results = hybrid_search(
#             request.message,
#             query_embedding,
#             parent_ids=parent_ids,
#             top_k=20
#         )
        
#         if not hybrid_results:
#             return ChatResponse(
#                 response="I couldn't find any relevant information to answer your question.",
#                 main_sources=[],
#                 other_sources=[],
#                 filter_info=filter_info
#             )
        
#         # Get top 6 results for main sources
#         main_results = hybrid_results[:6]
        
#         # Generate response
#         response_text = generate_response(
#             request.message,
#             main_results,
#             request.conversation_history,
#             metadata_cache
#         )
        
#         # Build main_sources
#         main_sources = []
#         for chunk in main_results:
#             source_info = {
#                 "id": chunk['id'],
#                 "source": chunk['source'],
#                 "parent_id": chunk['parent_id'],
#                 "chunk_index": chunk['chunk_index'],
#                 "total_chunks": chunk['total_chunks'],
#                 "content": chunk['content'],
#                 "similarity": round(chunk['similarity'], 3),
#                 "hybrid_score": round(chunk.get('hybrid_score', chunk['similarity']), 3),
#                 "vector_score": round(chunk.get('vector_score', 0), 3) if chunk.get('has_vector') else None,
#                 "bm25_score": round(chunk.get('bm25_score', 0), 3) if chunk.get('has_bm25') else None,
#             }
            
#             # Add metadata info if available
#             if chunk['parent_id'] in metadata_cache:
#                 source_info['metadata'] = metadata_cache[chunk['parent_id']]
            
#             main_sources.append(source_info)
        
#         # Build other_sources
#         main_source_ids = {chunk['id'] for chunk in main_results}
#         other_sources = []
        
#         for chunk in hybrid_results[6:20]:
#             if chunk['id'] not in main_source_ids:
#                 source_info = {
#                     "id": chunk['id'],
#                     "source": chunk['source'],
#                     "parent_id": chunk['parent_id'],
#                     "chunk_index": chunk['chunk_index'],
#                     "total_chunks": chunk['total_chunks'],
#                     "content": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
#                     "similarity": round(chunk['similarity'], 3),
#                     "hybrid_score": round(chunk.get('hybrid_score', chunk['similarity']), 3),
#                     "vector_score": round(chunk.get('vector_score', 0), 3) if chunk.get('has_vector') else None,
#                     "bm25_score": round(chunk.get('bm25_score', 0), 3) if chunk.get('has_bm25') else None
#                 }
                
#                 if chunk['parent_id'] in metadata_cache:
#                     source_info['metadata'] = metadata_cache[chunk['parent_id']]
                
#                 other_sources.append(source_info)
#                 if len(other_sources) >= 10:
#                     break
        
#         return ChatResponse(
#             response=response_text,
#             main_sources=main_sources,
#             other_sources=other_sources,
#             filter_info=filter_info
#         )
    
#     except Exception as e:
#         print(f"Error in chat endpoint: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": "api" if USE_API else "local",
        "chat_model": API_MODEL if USE_API else CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "date_filter": "enabled",
        "hybrid_search": True,
        "vector_weight": VECTOR_WEIGHT,
        "bm25_weight": BM25_WEIGHT,
        "bm25_index_loaded": bm25_index is not None,
        "corpus_size": len(corpus_chunks)
    }

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "mode": "api" if USE_API else "local",
        "chat_model": API_MODEL if USE_API else CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "api_configured": USE_API,
        "date_filter_enabled": True,
        "hybrid_search": True,
        "vector_weight": VECTOR_WEIGHT,
        "bm25_weight": BM25_WEIGHT,
        "bm25_index_status": "loaded" if bm25_index is not None else "not loaded",
        "corpus_size": len(corpus_chunks)
    }

@app.post("/reload-bm25")
async def reload_bm25():
    """Reload BM25 index"""
    try:
        load_bm25_index()
        return {
            "status": "success",
            "message": "BM25 index reloaded successfully",
            "corpus_size": len(corpus_chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading BM25 index: {str(e)}")

def main():
    import uvicorn
    uvicorn.run(
        "rag_hybrid_filter:app",      # import string
        host="10.10.1.117",
        port=1337,
        reload=True
    )

if __name__ == "__main__":
    main()
