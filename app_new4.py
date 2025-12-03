from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import ollama
import numpy as np
from typing import List, Dict, Optional
import json
import os
from dotenv import load_dotenv
import requests
import certifi
import urllib3
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

# API Configuration
API_KEY = os.getenv("API_KEY", "").strip()
BASE_URL = os.getenv("BASE_URL", "https://api.studio.nebius.com/v1/")
API_MODEL = os.getenv("model", "moonshotai/Kimi-K2-Instruct")

# Jina Reranker Configuration
JINA_API_KEY = os.getenv("JINA", "").strip()
USE_RERANKER = bool(JINA_API_KEY)

# Local Ollama Models (fallback)
EMBED_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "gemma3:4b"

# Hybrid Search Configuration
VECTOR_WEIGHT = 0.5  # Weight for vector search (0-1)
BM25_WEIGHT = 0.5    # Weight for BM25 search (0-1)

# Determine which mode to use
USE_API = bool(API_KEY)

print(f"🚀 Starting in {'API' if USE_API else 'LOCAL OLLAMA'} mode")
if USE_API:
    print(f"   Using model: {API_MODEL}")
else:
    print(f"   Using local models: {CHAT_MODEL} (chat), {EMBED_MODEL} (embeddings)")
print(f"🔄 Reranker: {'ENABLED' if USE_RERANKER else 'DISABLED'}")
print(f"🔍 Hybrid Search: Vector Weight={VECTOR_WEIGHT}, BM25 Weight={BM25_WEIGHT}")

# Global BM25 index (will be loaded on startup)
bm25_index = None
corpus_chunks = []

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str
    main_sources: List[Dict]
    other_sources: List[Dict]

def get_db_connection():
    """Create database connection"""
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    return conn

def tokenize_text(text: str) -> List[str]:
    """Tokenize text for BM25"""
    try:
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        return tokens
    except:
        # Fallback to simple tokenization if NLTK fails
        return text.lower().split()

def load_bm25_index():
    """Load all documents and create BM25 index"""
    global bm25_index, corpus_chunks
    
    print("📚 Loading BM25 index...")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id,
                    content,
                    source,
                    parent_id,
                    chunk_index,
                    total_chunks
                FROM document_embeddings
                ORDER BY id
            """)
            
            results = cur.fetchall()
            corpus_chunks = [dict(row) for row in results]
            
            # Tokenize all documents
            tokenized_corpus = [tokenize_text(chunk['content']) for chunk in corpus_chunks]
            
            # Create BM25 index
            bm25_index = BM25Okapi(tokenized_corpus)
            
            print(f"✅ BM25 index loaded with {len(corpus_chunks)} documents")
    finally:
        conn.close()

def get_embedding(text: str) -> List[float]:
    """Generate embedding using Ollama (always local for embeddings)"""
    try:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Error generating embedding")

def search_vector(query_embedding: List[float], top_k: int = 50) -> List[Dict]:
    """Search for similar chunks using cosine similarity (vector search)"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            vector_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
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

def search_bm25(query: str, top_k: int = 50) -> List[Dict]:
    """Search using BM25 keyword matching"""
    global bm25_index, corpus_chunks
    
    if bm25_index is None:
        load_bm25_index()
    
    # Tokenize query
    tokenized_query = tokenize_text(query)
    
    # Get BM25 scores
    scores = bm25_index.get_scores(tokenized_query)
    
    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # Return chunks with scores
    results = []
    for idx in top_indices:
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

def hybrid_search(query: str, query_embedding: List[float], top_k: int = 50) -> List[Dict]:
    """
    Perform hybrid search combining vector and BM25 results
    
    Args:
        query: The search query text
        query_embedding: The embedding vector for the query
        top_k: Number of candidates to retrieve from each method
    
    Returns:
        Combined and ranked list of chunks
    """
    print(f"🔍 Performing hybrid search (Vector: {VECTOR_WEIGHT}, BM25: {BM25_WEIGHT})...")
    
    # Get results from both methods
    vector_results = search_vector(query_embedding, top_k=top_k)
    bm25_results = search_bm25(query, top_k=top_k)
    
    print(f"   Vector results: {len(vector_results)}")
    print(f"   BM25 results: {len(bm25_results)}")
    
    # Normalize scores
    vector_results = normalize_scores(vector_results, 'vector_score')
    bm25_results = normalize_scores(bm25_results, 'bm25_score')
    
    # Combine results using a dictionary keyed by chunk ID
    combined_results = {}
    
    # Add vector results
    for chunk in vector_results:
        chunk_id = chunk['id']
        combined_results[chunk_id] = chunk.copy()
        combined_results[chunk_id]['vector_score_normalized'] = chunk['vector_score_normalized']
        combined_results[chunk_id]['has_vector'] = True
        combined_results[chunk_id]['has_bm25'] = False
    
    # Add or update with BM25 results
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
        
        # Weighted combination
        chunk['hybrid_score'] = (VECTOR_WEIGHT * vector_score) + (BM25_WEIGHT * bm25_score)
        chunk['similarity'] = chunk['hybrid_score']  # For compatibility with existing code
    
    # Sort by hybrid score
    sorted_chunks = sorted(combined_results.values(), key=lambda x: x['hybrid_score'], reverse=True)
    
    print(f"   Combined results: {len(sorted_chunks)}")
    
    return sorted_chunks

def rerank_chunks(query: str, chunks: List[Dict], top_n: int = 6) -> List[Dict]:
    """
    Rerank chunks using Jina AI reranker API
    
    Args:
        query: The user's query
        chunks: List of retrieved chunks
        top_n: Number of top results to return after reranking
    
    Returns:
        Reranked list of chunks
    """
    if not USE_RERANKER or not chunks:
        return chunks[:top_n]
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}"
        }
        
        # Prepare documents for reranking
        documents = [chunk['content'] for chunk in chunks]
        
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": min(top_n, len(documents)),
            "documents": documents,
            "return_documents": True
        }
        
        response = requests.post(
            'https://api.jina.ai/v1/rerank',
            headers=headers,
            json=data,
            timeout=30,
            verify=certifi.where()
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Map reranked results back to original chunks
        reranked_chunks = []
        for item in result.get('results', []):
            original_index = item['index']
            chunk = chunks[original_index].copy()
            # Update similarity score with reranker score
            chunk['rerank_score'] = item['relevance_score']
            chunk['original_similarity'] = chunk['similarity']
            chunk['similarity'] = item['relevance_score']
            reranked_chunks.append(chunk)
        
        print(f"✅ Reranked {len(chunks)} chunks to top {len(reranked_chunks)}")
        return reranked_chunks
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Reranker API Error: {e}. Falling back to original ranking.")
        return chunks[:top_n]
    except Exception as e:
        print(f"⚠️ Reranking Error: {e}. Falling back to original ranking.")
        return chunks[:top_n]

def build_context_with_citations(chunks: List[Dict]) -> str:
    """
    Build context string with source citations for each chunk
    Similar to gazette system approach
    """
    context_parts = []
    for chunk in chunks:
        source_label = f"[Source: {chunk['source']}, Chunk {chunk['chunk_index']}/{chunk['total_chunks']}]"
        context_parts.append(f"{source_label}:\n{chunk['content']}")
    
    return "\n\n".join(context_parts)

def generate_response_api(query: str, context_chunks: List[Dict], conversation_history: List[Dict]) -> str:
    """Generate response using external API with citation-aware context"""
    
    # Build context with source citations
    context_text = build_context_with_citations(context_chunks)
    
    # Build messages with improved citation instructions
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that answers questions based on the provided context.

CITATION RULES:
1. Use EXACTLY this format when citing: [Source: filename, Chunk X/Y]
2. Place citations immediately after the information they support
3. Use multiple citations if information comes from multiple sources
4. Answer ONLY based on the provided context

Example: "The new policy states X [Source: policy.pdf, Chunk 2/5] and applies from January [Source: policy.pdf, Chunk 3/5]."

Context:
{context_text}"""
        }
    ]
    
    # Add conversation history (last 6 messages)
    for msg in conversation_history[-6:]:
        messages.append(msg)
    
    # Add current query
    messages.append({
        "role": "user",
        "content": query
    })
    
    # Call API
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": API_MODEL,
            "messages": messages,
            "temperature": 0.2,  # Lower temperature for more factual responses
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

def generate_response_ollama(query: str, context_chunks: List[Dict], conversation_history: List[Dict]) -> str:
    """Generate response using local Ollama with citation-aware context"""
    
    # Build context with source citations
    context_text = build_context_with_citations(context_chunks)
    
    # Build conversation messages with improved citation instructions
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that answers questions based on the provided context.

CITATION RULES:
1. Use EXACTLY this format when citing: [Source: filename, Chunk X/Y]
2. Place citations immediately after the information they support
3. Use multiple citations if information comes from multiple sources
4. Answer ONLY based on the provided context

Example: "The new policy states X [Source: policy.pdf, Chunk 2/5] and applies from January [Source: policy.pdf, Chunk 3/5]."

Context:
{context_text}"""
        }
    ]
    
    # Add conversation history
    for msg in conversation_history[-6:]:
        messages.append(msg)
    
    # Add current query
    messages.append({
        "role": "user",
        "content": query
    })
    
    # Generate response
    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=messages
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")

def generate_response(query: str, context_chunks: List[Dict], conversation_history: List[Dict]) -> str:
    """Generate response using either API or Ollama based on configuration"""
    if USE_API:
        return generate_response_api(query, context_chunks, conversation_history)
    else:
        return generate_response_ollama(query, context_chunks, conversation_history)

@app.on_event("startup")
async def startup_event():
    """Load BM25 index on startup"""
    load_bm25_index()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with hybrid search and gazette-style citations"""
    try:
        # Generate embedding for the query
        query_embedding = get_embedding(request.message)
        
        # Perform hybrid search (retrieve more candidates for reranking)
        initial_k = 50 if USE_RERANKER else 20
        hybrid_results = hybrid_search(request.message, query_embedding, top_k=initial_k)
        
        if not hybrid_results:
            return ChatResponse(
                response="I couldn't find any relevant information to answer your question.",
                main_sources=[],
                other_sources=[]
            )
        
        # Rerank chunks if reranker is enabled (get top 6 for main sources)
        if USE_RERANKER:
            reranked_chunks = rerank_chunks(request.message, hybrid_results, top_n=6)
        else:
            reranked_chunks = hybrid_results[:6]
        # Generate response with citation-aware context
        response_text = generate_response(
            request.message,
            reranked_chunks,
            request.conversation_history
        )
        
        # Build main_sources (top 6 reranked chunks with full context)
        main_sources = []
        for chunk in reranked_chunks:
            source_info = {
                "id": chunk['id'],
                "source": chunk['source'],
                "chunk_index": chunk['chunk_index'],
                "total_chunks": chunk['total_chunks'],
                "content": chunk['content'],  # Full chunk text for verification
                "similarity": round(chunk['similarity'], 3),
                "hybrid_score": round(chunk.get('hybrid_score', chunk['similarity']), 3),
                "vector_score": round(chunk.get('vector_score', 0), 3) if chunk.get('has_vector') else None,
                "bm25_score": round(chunk.get('bm25_score', 0), 3) if chunk.get('has_bm25') else None,
                "reranked": USE_RERANKER,
                "original_similarity": round(chunk.get('original_similarity', chunk['similarity']), 3) if USE_RERANKER else None
            }
            main_sources.append(source_info)
        
        # Build other_sources (remaining candidates not in top 6)
        main_source_ids = {chunk['id'] for chunk in reranked_chunks}
        other_sources = []
        
        # Get up to 10 additional sources from hybrid results
        for chunk in hybrid_results[:16]:  # Check more candidates
            if chunk['id'] not in main_source_ids:
                source_info = {
                    "id": chunk['id'],
                    "source": chunk['source'],
                    "chunk_index": chunk['chunk_index'],
                    "total_chunks": chunk['total_chunks'],
                    "content": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],  # Preview only
                    "similarity": round(chunk['similarity'], 3),
                    "hybrid_score": round(chunk.get('hybrid_score', chunk['similarity']), 3),
                    "vector_score": round(chunk.get('vector_score', 0), 3) if chunk.get('has_vector') else None,
                    "bm25_score": round(chunk.get('bm25_score', 0), 3) if chunk.get('has_bm25') else None
                }
                other_sources.append(source_info)
                if len(other_sources) >= 10:
                    break
        
        return ChatResponse(
            response=response_text,
            main_sources=main_sources,
            other_sources=other_sources
        )
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the frontend with improved source display"""
    mode_badge = "API Mode" if USE_API else "Local Mode"
    mode_color = "#10b981" if USE_API else "#f59e0b"
    reranker_badge = "🔄 Reranker ON" if USE_RERANKER else "Reranker OFF"
    reranker_color = "#8b5cf6" if USE_RERANKER else "#6b7280"
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Chat App - Hybrid Search with Citations</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }}
            
            .container {{
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                width: 100%;
                max-width: 1200px;
                height: 90vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px 30px;
                border-radius: 20px 20px 0 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .header-content h1 {{
                font-size: 24px;
                font-weight: 600;
            }}
            
            .header-content p {{
                font-size: 14px;
                opacity: 0.9;
                margin-top: 5px;
            }}
            
            .badges {{
                display: flex;
                gap: 10px;
                flex-direction: column;
                align-items: flex-end;
            }}
            
            .mode-badge {{
                background: {mode_color};
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 600;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }}
            
            .reranker-badge {{
                background: {reranker_color};
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 11px;
                font-weight: 600;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }}
            
            .hybrid-badge {{
                background: #ec4899;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 11px;
                font-weight: 600;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }}
            
            .chat-container {{
                flex: 1;
                overflow-y: auto;
                padding: 30px;
                background: #f7f9fc;
            }}
            
            .message {{
                margin-bottom: 20px;
                animation: fadeIn 0.3s ease-in;
            }}
            
            @keyframes fadeIn {{
                from {{
                    opacity: 0;
                    transform: translateY(10px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            
            .message.user {{
                text-align: right;
            }}
            
            .message-content {{
                display: inline-block;
                padding: 15px 20px;
                border-radius: 18px;
                max-width: 70%;
                word-wrap: break-word;
                line-height: 1.5;
            }}
            
            .message.user .message-content {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-bottom-right-radius: 4px;
            }}
            
            .message.assistant .message-content {{
                background: white;
                color: #333;
                border: 1px solid #e1e8ed;
                border-bottom-left-radius: 4px;
                text-align: left;
            }}
            
            .sources-section {{
                margin-top: 20px;
            }}
            
            .sources {{
                margin-top: 15px;
                padding: 15px;
                background: #f0f4f8;
                border-radius: 12px;
                border-left: 4px solid #667eea;
            }}
            
            .sources-title {{
                font-weight: 600;
                font-size: 13px;
                color: #667eea;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .source-count {{
                font-size: 11px;
                background: #667eea;
                color: white;
                padding: 2px 8px;
                border-radius: 10px;
            }}
            
            .source-item {{
                background: white;
                padding: 12px;
                margin: 8px 0;
                border-radius: 8px;
                font-size: 13px;
                border: 1px solid #e1e8ed;
                transition: box-shadow 0.2s;
            }}
            
            .source-item:hover {{
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            
            .source-header {{
                display: flex;
                justify-content: space-between;
                align-items: start;
                margin-bottom: 8px;
            }}
            
            .source-name {{
                font-weight: 600;
                color: #667eea;
                flex: 1;
            }}
            
            .source-link {{
                margin-top: 5px;
            }}
            
            .source-link a {{
                color: #667eea;
                text-decoration: none;
                font-size: 12px;
                display: inline-flex;
                align-items: center;
                gap: 5px;
            }}
            
            .source-link a:hover {{
                text-decoration: underline;
            }}
            
            .source-meta {{
                color: #666;
                font-size: 12px;
                margin: 3px 0;
            }}
            
            .source-scores {{
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                margin-top: 8px;
            }}
            
            .score-badge {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
            }}
            
            .score-badge.hybrid {{
                background: #fce7f3;
                color: #ec4899;
            }}
            
            .score-badge.vector {{
                background: #dbeafe;
                color: #3b82f6;
            }}
            
            .score-badge.bm25 {{
                background: #fef3c7;
                color: #f59e0b;
            }}
            
            .score-badge.rerank {{
                background: #ede9fe;
                color: #8b5cf6;
            }}
            
            .source-content {{
                margin-top: 8px;
                padding: 10px;
                background: #f9fafb;
                border-radius: 6px;
                font-size: 12px;
                color: #555;
                line-height: 1.6;
                max-height: 150px;
                overflow-y: auto;
                border-left: 2px solid #667eea;
            }}
            
            .source-preview {{
                margin-top: 8px;
                color: #888;
                font-size: 12px;
                font-style: italic;
                padding-top: 8px;
                border-top: 1px solid #e1e8ed;
            }}
            
            .toggle-content {{
                color: #667eea;
                cursor: pointer;
                font-size: 11px;
                margin-top: 5px;
                display: inline-block;
            }}
            
            .toggle-content:hover {{
                text-decoration: underline;
            }}
            
            .other-sources {{
                margin-top: 15px;
                padding: 15px;
                background: #fef3c7;
                border-radius: 12px;
                border-left: 4px solid #f59e0b;
            }}
            
            .other-sources .sources-title {{
                color: #f59e0b;
            }}
            
            .other-sources .source-count {{
                background: #f59e0b;
            }}
            
            .collapsible-header {{
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .collapse-icon {{
                transition: transform 0.3s;
            }}
            
            .collapse-icon.collapsed {{
                transform: rotate(-90deg);
            }}
            
            .collapsible-content {{
                max-height: 500px;
                overflow: hidden;
                transition: max-height 0.3s ease-out;
            }}
            
            .collapsible-content.collapsed {{
                max-height: 0;
            }}
            
            .input-container {{
                padding: 20px 30px;
                background: white;
                border-top: 1px solid #e1e8ed;
                display: flex;
                gap: 15px;
            }}
            
            #messageInput {{
                flex: 1;
                padding: 15px 20px;
                border: 2px solid #e1e8ed;
                border-radius: 25px;
                font-size: 15px;
                outline: none;
                transition: border-color 0.3s;
            }}
            
            #messageInput:focus {{
                border-color: #667eea;
            }}
            
            #sendButton {{
                padding: 15px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            
            #sendButton:hover:not(:disabled) {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }}
            
            #sendButton:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
            }}
            
            .typing-indicator {{
                display: inline-flex;
                gap: 5px;
                padding: 15px 20px;
                background: white;
                border: 1px solid #e1e8ed;
                border-radius: 18px;
                border-bottom-left-radius: 4px;
            }}
            
            .typing-indicator span {{
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #667eea;
                animation: typing 1.4s infinite;
            }}
            
            .typing-indicator span:nth-child(2) {{
                animation-delay: 0.2s;
            }}
            
            .typing-indicator span:nth-child(3) {{
                animation-delay: 0.4s;
            }}
            
            @keyframes typing {{
                0%, 60%, 100% {{
                    transform: translateY(0);
                }}
                30% {{
                    transform: translateY(-10px);
                }}
            }}
            
            /* Scrollbar styling */
            .chat-container::-webkit-scrollbar,
            .source-content::-webkit-scrollbar {{
                width: 8px;
            }}
            
            .chat-container::-webkit-scrollbar-track,
            .source-content::-webkit-scrollbar-track {{
                background: #f1f1f1;
                border-radius: 10px;
            }}
            
            .chat-container::-webkit-scrollbar-thumb,
            .source-content::-webkit-scrollbar-thumb {{
                background: #888;
                border-radius: 10px;
            }}
            
            .chat-container::-webkit-scrollbar-thumb:hover,
            .source-content::-webkit-scrollbar-thumb:hover {{
                background: #555;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="header-content">
                    <h1>🤖 RAG Chat Assistant</h1>
                    <p>Hybrid Search with Vector + BM25 + Reranking + Citations</p>
                </div>
                <div class="badges">
                    <div class="mode-badge">{mode_badge}</div>
                    <div class="hybrid-badge">🔍 Hybrid Search</div>
                    <div class="reranker-badge">{reranker_badge}</div>
                </div>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="message assistant">
                    <div class="message-content">
                        Hello! I'm your RAG-powered assistant with <strong>Hybrid Search</strong> (Vector + BM25){' and ' + reranker_badge if USE_RERANKER else ''}. 
                        I combine semantic understanding with keyword matching to find the most relevant information. 
                        All my answers include citations in the format <code>[Source: filename, Chunk X/Y]</code>. Ask me anything!
                    </div>
                </div>
            </div>
            
            <div class="input-container">
                <input 
                    type="text" 
                    id="messageInput" 
                    placeholder="Type your question here..."
                    onkeypress="if(event.key === 'Enter') sendMessage()"
                >
                <button id="sendButton" onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <script>
            let conversationHistory = [];
            let sourceContentVisible = {{}};
            
            async function sendMessage() {{
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                addMessage(message, 'user');
                input.value = '';
                
                const sendButton = document.getElementById('sendButton');
                sendButton.disabled = true;
                input.disabled = true;
                
                const typingId = addTypingIndicator();
                
                try {{
                    const response = await fetch('/chat', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            message: message,
                            conversation_history: conversationHistory
                        }})
                    }});
                    
                    if (!response.ok) {{
                        throw new Error('Network response was not ok');
                    }}
                    
                    const data = await response.json();
                    
                    removeTypingIndicator(typingId);
                    addMessage(data.response, 'assistant', data.main_sources, data.other_sources);
                    
                    conversationHistory.push(
                        {{ role: 'user', content: message }},
                        {{ role: 'assistant', content: data.response }}
                    );
                    
                    if (conversationHistory.length > 10) {{
                        conversationHistory = conversationHistory.slice(-10);
                    }}
                    
                }} catch (error) {{
                    removeTypingIndicator(typingId);
                    addMessage('Sorry, there was an error processing your request. Please try again.', 'assistant');
                    console.error('Error:', error);
                }} finally {{
                    sendButton.disabled = false;
                    input.disabled = false;
                    input.focus();
                }}
            }}
            
            function toggleSourceContent(sourceId) {{
                const contentDiv = document.getElementById(`content-${{sourceId}}`);
                const toggleLink = document.getElementById(`toggle-${{sourceId}}`);
                
                if (contentDiv.style.display === 'none') {{
                    contentDiv.style.display = 'block';
                    toggleLink.textContent = '▼ Hide full content';
                }} else {{
                    contentDiv.style.display = 'none';
                    toggleLink.textContent = '▶ Show full content';
                }}
            }}
            
            function toggleOtherSources(messageId) {{
                const contentDiv = document.getElementById(`other-sources-${{messageId}}`);
                const icon = document.getElementById(`other-sources-icon-${{messageId}}`);
                
                if (contentDiv.classList.contains('collapsed')) {{
                    contentDiv.classList.remove('collapsed');
                    icon.classList.remove('collapsed');
                }} else {{
                    contentDiv.classList.add('collapsed');
                    icon.classList.add('collapsed');
                }}
            }}
            
            function addMessage(text, sender, mainSources = null, otherSources = null) {{
                const container = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${{sender}}`;
                const messageId = Date.now();
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.textContent = text;
                
                messageDiv.appendChild(contentDiv);
                
                // Add main sources
                if (mainSources && mainSources.length > 0) {{
                    const sourcesSection = document.createElement('div');
                    sourcesSection.className = 'sources-section';
                    
                    const mainSourcesDiv = document.createElement('div');
                    mainSourcesDiv.className = 'sources';
                    mainSourcesDiv.innerHTML = `
                        <div class="sources-title">
                            📚 Main Sources (Used in Answer)
                            <span class="source-count">${{mainSources.length}}</span>
                        </div>
                    `;
                                        mainSources.forEach((source, index) => {{
                        const sourceItem = document.createElement('div');
                        sourceItem.className = 'source-item';
                        const sourceId = `${{messageId}}-${{index}}`;
                        
                        const pdfFileName = source.source.replace('.md', '.pdf');
                        const pdfPath = `http://10.10.1.117:8008/${{pdfFileName}}`;
                        
                        let scoresHtml = '<div class="source-scores">';
                        
                        // Hybrid score (always present)
                        scoresHtml += `<span class="score-badge hybrid">Hybrid: ${{(source.hybrid_score * 100).toFixed(1)}}%</span>`;
                        
                        // Vector score (if available)
                        if (source.vector_score !== null) {{
                            scoresHtml += `<span class="score-badge vector">Vector: ${{(source.vector_score * 100).toFixed(1)}}%</span>`;
                        }}
                        
                        // BM25 score (if available)
                        if (source.bm25_score !== null) {{
                            scoresHtml += `<span class="score-badge bm25">BM25: ${{source.bm25_score.toFixed(2)}}</span>`;
                        }}
                        
                        // Rerank score (if reranking was used)
                        if (source.reranked && source.original_similarity !== null) {{
                            scoresHtml += `<span class="score-badge rerank">Rerank: ${{(source.similarity * 100).toFixed(1)}}%</span>`;
                        }}
                        
                        scoresHtml += '</div>';
                        
                        sourceItem.innerHTML = `
                            <div class="source-header">
                                <div class="source-name">${{index + 1}}. ${{source.source}}</div>
                            </div>
                            <div class="source-link">
                                <a href="${{pdfPath}}" target="_blank">📄 View PDF: ${{pdfFileName}}</a>
                            </div>
                            <div class="source-meta">Chunk ${{source.chunk_index}}/${{source.total_chunks}}</div>
                            ${{scoresHtml}}
                            <div class="toggle-content" id="toggle-${{sourceId}}" onclick="toggleSourceContent('${{sourceId}}')">
                                ▶ Show full content
                            </div>
                            <div class="source-content" id="content-${{sourceId}}" style="display: none;">
                                ${{source.content}}
                            </div>
                        `;
                        mainSourcesDiv.appendChild(sourceItem);
                    }});
                    
                    sourcesSection.appendChild(mainSourcesDiv);
                    contentDiv.appendChild(sourcesSection);
                }}
                
                // Add other sources (collapsible)
                if (otherSources && otherSources.length > 0) {{
                    const otherSourcesDiv = document.createElement('div');
                    otherSourcesDiv.className = 'other-sources';
                    
                    const headerHtml = `
                        <div class="sources-title collapsible-header" onclick="toggleOtherSources(${{messageId}})">
                            <div>
                                <span class="collapse-icon" id="other-sources-icon-${{messageId}}">▼</span>
                                🔖 Other Relevant Sources
                            </div>
                            <span class="source-count">${{otherSources.length}}</span>
                        </div>
                    `;
                    otherSourcesDiv.innerHTML = headerHtml;
                    
                    const collapsibleContent = document.createElement('div');
                    collapsibleContent.className = 'collapsible-content';
                    collapsibleContent.id = `other-sources-${{messageId}}`;
                    
                    otherSources.forEach((source, index) => {{
                        const sourceItem = document.createElement('div');
                        sourceItem.className = 'source-item';
                        
                        const pdfFileName = source.source.replace('.json', '.pdf');
                        const pdfPath = `http://10.10.1.117:8008/${{pdfFileName}}`;
                        
                        let scoresHtml = '<div class="source-scores">';
                        scoresHtml += `<span class="score-badge hybrid">Hybrid: ${{(source.hybrid_score * 100).toFixed(1)}}%</span>`;
                        
                        if (source.vector_score !== null) {{
                            scoresHtml += `<span class="score-badge vector">Vector: ${{(source.vector_score * 100).toFixed(1)}}%</span>`;
                        }}
                        
                        if (source.bm25_score !== null) {{
                            scoresHtml += `<span class="score-badge bm25">BM25: ${{source.bm25_score.toFixed(2)}}</span>`;
                        }}
                        
                        scoresHtml += '</div>';
                        
                        sourceItem.innerHTML = `
                            <div class="source-header">
                                <div class="source-name">${{index + 1}}. ${{source.source}}</div>
                            </div>
                            <div class="source-link">
                                <a href="${{pdfPath}}" target="_blank">📄 View PDF: ${{pdfFileName}}</a>
                            </div>
                            <div class="source-meta">Chunk ${{source.chunk_index}}/${{source.total_chunks}}</div>
                            ${{scoresHtml}}
                            <div class="source-preview">${{source.content}}</div>
                        `;
                        collapsibleContent.appendChild(sourceItem);
                    }});
                    
                    otherSourcesDiv.appendChild(collapsibleContent);
                    contentDiv.appendChild(otherSourcesDiv);
                }}
                
                container.appendChild(messageDiv);
                container.scrollTop = container.scrollHeight;
            }}
            
            function addTypingIndicator() {{
                const container = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message assistant';
                messageDiv.id = 'typing-indicator-' + Date.now();
                
                const typingDiv = document.createElement('div');
                typingDiv.className = 'typing-indicator';
                typingDiv.innerHTML = '<span></span><span></span><span></span>';
                
                messageDiv.appendChild(typingDiv);
                container.appendChild(messageDiv);
                container.scrollTop = container.scrollHeight;
                
                return messageDiv.id;
            }}
            
            function removeTypingIndicator(id) {{
                const element = document.getElementById(id);
                if (element) {{
                    element.remove();
                }}
            }}
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": "api" if USE_API else "local",
        "chat_model": API_MODEL if USE_API else CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "reranker_enabled": USE_RERANKER,
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
        "reranker_enabled": USE_RERANKER,
        "reranker_model": "jina-reranker-v2-base-multilingual" if USE_RERANKER else None,
        "hybrid_search": True,
        "vector_weight": VECTOR_WEIGHT,
        "bm25_weight": BM25_WEIGHT,
        "bm25_index_status": "loaded" if bm25_index is not None else "not loaded",
        "corpus_size": len(corpus_chunks)
    }

@app.post("/reload-bm25")
async def reload_bm25():
    """Reload BM25 index (useful after adding new documents)"""
    try:
        load_bm25_index()
        return {
            "status": "success",
            "message": "BM25 index reloaded successfully",
            "corpus_size": len(corpus_chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading BM25 index: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.10.1.117", port=1337)
                    