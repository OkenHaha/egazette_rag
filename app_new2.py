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
    "database": "json_embed",
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
EMBED_MODEL = "qwen3-embedding:8b"
CHAT_MODEL = "gemma3:4b"

# Determine which mode to use
USE_API = bool(API_KEY)

print(f"🚀 Starting in {'API' if USE_API else 'LOCAL OLLAMA'} mode")
if USE_API:
    print(f"   Using model: {API_MODEL}")
else:
    print(f"   Using local models: {CHAT_MODEL} (chat), {EMBED_MODEL} (embeddings)")
print(f"🔄 Reranker: {'ENABLED' if USE_RERANKER else 'DISABLED'}")

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict]

def get_db_connection():
    """Create database connection"""
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    return conn

def get_embedding(text: str) -> List[float]:
    """Generate embedding using Ollama (always local for embeddings)"""
    try:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Error generating embedding")

def search_similar_chunks(query_embedding: List[float], top_k: int = 3) -> List[Dict]:
    """Search for similar chunks using cosine similarity"""
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
                chunk['similarity'] = 1 - chunk['distance']
                chunks.append(chunk)
            
            return chunks
    finally:
        conn.close()

def rerank_chunks(query: str, chunks: List[Dict], top_n: int = 3) -> List[Dict]:
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

def generate_response_api(query: str, context_chunks: List[Dict], conversation_history: List[Dict]) -> str:
    """Generate response using external API"""
    
    # Prepare context from retrieved chunks
    context_text = "\n\n".join([
        f"[Source: {chunk['source']}, Chunk {chunk['chunk_index']}/{chunk['total_chunks']}]\n{chunk['content']}"
        for chunk in context_chunks
    ])
    
    # Build messages
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that answers questions based on the provided context. 
Always cite your sources using the format [Source: filename] when referencing information.
If the context doesn't contain relevant information, say so clearly.

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
            "temperature": 0.7,
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
    """Generate response using local Ollama"""
    
    # Prepare context from retrieved chunks
    context_text = "\n\n".join([
        f"[Source: {chunk['source']}, Chunk {chunk['chunk_index']}/{chunk['total_chunks']}]\n{chunk['content']}"
        for chunk in context_chunks
    ])
    
    # Build conversation messages
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that answers questions based on the provided context. 
Always cite your sources using the format [Source: filename] when referencing information.
If the context doesn't contain relevant information, say so clearly.

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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        # Generate embedding for the query
        query_embedding = get_embedding(request.message)
        
        # Search for similar chunks (retrieve more candidates for reranking)
        initial_k = 20 if USE_RERANKER else 3
        similar_chunks = search_similar_chunks(query_embedding, top_k=initial_k)
        
        if not similar_chunks:
            return ChatResponse(
                response="I couldn't find any relevant information to answer your question.",
                sources=[]
            )
        
        # Rerank chunks if reranker is enabled
        if USE_RERANKER:
            reranked_chunks = rerank_chunks(request.message, similar_chunks, top_n=3)
        else:
            reranked_chunks = similar_chunks[:3]
        
        # Generate response
        response_text = generate_response(
            request.message,
            reranked_chunks,
            request.conversation_history
        )
        
        # Prepare sources for citation
        sources = [
            {
                "id": chunk['id'],
                "source": chunk['source'],
                "chunk_index": chunk['chunk_index'],
                "total_chunks": chunk['total_chunks'],
                "similarity": round(chunk['similarity'], 3),
                "reranked": USE_RERANKER,
                "original_similarity": round(chunk.get('original_similarity', chunk['similarity']), 3) if USE_RERANKER else None,
                "preview": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            }
            for chunk in reranked_chunks
        ]
        
        return ChatResponse(response=response_text, sources=sources)
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the frontend"""
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
        <title>RAG Chat App</title>
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
                max-width: 1000px;
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
            }}
            
            .source-item {{
                background: white;
                padding: 10px;
                margin: 8px 0;
                border-radius: 8px;
                font-size: 13px;
                border: 1px solid #e1e8ed;
            }}
            
            .source-name {{
                font-weight: 600;
                color: #667eea;
                margin-bottom: 5px;
            }}
            
            .source-meta {{
                color: #666;
                font-size: 12px;
            }}
            
            .rerank-badge {{
                display: inline-block;
                background: #8b5cf6;
                color: white;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 10px;
                margin-left: 5px;
            }}
            
            .source-preview {{
                margin-top: 5px;
                color: #888;
                font-size: 12px;
                font-style: italic;
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
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="header-content">
                    <h1>🤖 RAG Chat Assistant</h1>
                    <p>Ask questions and get answers with cited sources</p>
                </div>
                <div class="badges">
                    <div class="mode-badge">{mode_badge}</div>
                    <div class="reranker-badge">{reranker_badge}</div>
                </div>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="message assistant">
                    <div class="message-content">
                        Hello! I'm your RAG-powered assistant running in <strong>{mode_badge}</strong>{' with ' + ("{reranker_badge}" if USE_RERANKER else '') if USE_RERANKER else ''}. Ask me anything and I'll search through the knowledge base to provide accurate answers with citations.
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
                    addMessage(data.response, 'assistant', data.sources);
                    
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
            
            function addMessage(text, sender, sources = null) {{
                const container = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${{sender}}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.textContent = text;
                
                messageDiv.appendChild(contentDiv);
                
                if (sources && sources.length > 0) {{
                    const sourcesDiv = document.createElement('div');
                    sourcesDiv.className = 'sources';
                    sourcesDiv.innerHTML = '<div class="sources-title">📚 Sources</div>';
                    
                    sources.forEach((source, index) => {{
                        const sourceItem = document.createElement('div');
                        sourceItem.className = 'source-item';

                        const pdfFileName = source.source.replace('.md', '.pdf');
                        const pdfPath = `http://10.10.1.117:8008/${{ pdfFileName}}`;
                        sourceItem.innerHTML = `
                            <div class="source-name">${{index + 1}}. ${{source.source}}</div>
                            <div class="source-link">
                                <a href="${{pdfPath}}" target="_blank">📄 View PDF: ${{pdfFileName}}</a>
                            </div>
                            <div class="source-meta">Chunk ${{source.chunk_index}}/${{source.total_chunks}} • Similarity: ${{(source.similarity * 100).toFixed(1)}}%</div>
                            <div class="source-preview">${{source.preview}}</div>
                        `;
                        sourcesDiv.appendChild(sourceItem);
                    }});
                    
                    contentDiv.appendChild(sourcesDiv);
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
        "reranker_enabled": USE_RERANKER
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
        "reranker_model": "jina-reranker-v2-base-multilingual" if USE_RERANKER else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.10.1.117", port=1137)