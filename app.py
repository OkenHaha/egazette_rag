from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import ollama
import numpy as np
from typing import List, Dict
import json
#from pgvector.psycopg2 import register_vector
#import psycopg2.extensions

# Register the vector type with psycopg2
#register_vector(psycopg2.extensions)

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

# Models
EMBED_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "gemma3:4b"

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict]

def get_db_connection():
    """Create database connection and register vector type"""
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    #register_vector(conn)  # ✅ Register on the connection
    return conn

def get_embedding(text: str) -> List[float]:
    """Generate embedding using Ollama"""
    try:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Error generating embedding")

def format_vector(embedding: List[float]) -> str:
    """Convert list of floats to PostgreSQL vector literal: [0.1,0.2,-0.3]"""
    return "[" + ",".join(str(x) for x in embedding) + "]"

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_similar_chunks(query_embedding: List[float], top_k: int = 3) -> List[Dict]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Format as proper vector literal string
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

# def search_similar_chunks(query_embedding: List[float], top_k: int = 3) -> List[Dict]:
#     """Search for similar chunks using cosine similarity"""
#     conn = get_db_connection()
#     try:
#         with conn.cursor() as cur:
#             # Using pgvector's cosine distance operator
#             # Note: <=> is cosine distance (1 - cosine similarity)
#             cur.execute("""
#                 SELECT 
#                     id,
#                     content,
#                     source,
#                     parent_id,
#                     chunk_index,
#                     total_chunks,
#                     embedding <=> %s::vector as distance
#                 FROM document_embeddings
#                 ORDER BY embedding <=> %s::vector
#                 LIMIT %s
#             """, (str(query_embedding), str(query_embedding), top_k))
            
#             results = cur.fetchall()
            
#             # Convert to list of dicts and calculate similarity
#             chunks = []
#             for row in results:
#                 chunk = dict(row)
#                 chunk['similarity'] = 1 - chunk['distance']  # Convert distance to similarity
#                 del chunk['distance']
#                 del chunk['embedding']  # Remove embedding from response
#                 chunks.append(chunk)
            
#             return chunks
#     finally:
#         conn.close()

# def generate_response(query: str, context_chunks: List[Dict], conversation_history: List[Dict]) -> str:
#     """Generate response using Ollama with context"""
    
#     # Prepare context from retrieved chunks
#     context_text = "\n\n".join([
#         f"[Source: {chunk['source']}, Chunk {chunk['chunk_index']}/{chunk['total_chunks']}]\n{chunk['content']}"
#         for chunk in similar_chunks  # ← 'content', not 'context'
#     ])
    
#     # Build conversation messages
#     messages = [
#             {
#                 "role": "system",
#                 "content": f"""You are a helpful assistant that answers questions based on the provided context. 
#     Always cite your sources using the format [Source: filename] when referencing information.
#     If the context doesn't contain relevant information, say so clearly.

#     Context:
#     {context_text}"""
#             }
#         ]
    
#     # Add conversation history
#     for msg in conversation_history[-6:]:  # Keep last 6 messages for context
#         messages.append(msg)
    
#     # Add current query
#     messages.append({
#         "role": "user",
#         "content": query
#     })
    
#     # Generate response
#     try:
#         response = ollama.chat(
#             model=CHAT_MODEL,
#             messages=messages
#         )
#         return response['message']['content']
#     except Exception as e:
#         print(f"Error generating response: {e}")
#         raise HTTPException(status_code=500, detail="Error generating response")

def generate_response(query: str, context_chunks: List[Dict], conversation_history: List[Dict]) -> str:
    """Generate response using Ollama with context"""
    
    # Prepare context from retrieved chunks
    context_text = "\n\n".join([
        f"[Source: {chunk['source']}, Chunk {chunk['chunk_index']}/{chunk['total_chunks']}]\n{chunk['content']}"
        for chunk in context_chunks  # ✅ Fixed: was 'similar_chunks'
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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        # Generate embedding for the query
        query_embedding = get_embedding(request.message)
        
        # Search for similar chunks
        similar_chunks = search_similar_chunks(query_embedding, top_k=3)
        
        if not similar_chunks:
            return ChatResponse(
                response="I couldn't find any relevant information to answer your question.",
                sources=[]
            )
        
        # Generate response
        response_text = generate_response(
            request.message,
            similar_chunks,
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
                "preview": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            }
            for chunk in similar_chunks
        ]
        
        return ChatResponse(response=response_text, sources=sources)
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the frontend"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Chat App</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                width: 100%;
                max-width: 1000px;
                height: 90vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px 30px;
                border-radius: 20px 20px 0 0;
            }
            
            .header h1 {
                font-size: 24px;
                font-weight: 600;
            }
            
            .header p {
                font-size: 14px;
                opacity: 0.9;
                margin-top: 5px;
            }
            
            .chat-container {
                flex: 1;
                overflow-y: auto;
                padding: 30px;
                background: #f7f9fc;
            }
            
            .message {
                margin-bottom: 20px;
                animation: fadeIn 0.3s ease-in;
            }
            
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .message.user {
                text-align: right;
            }
            
            .message-content {
                display: inline-block;
                padding: 15px 20px;
                border-radius: 18px;
                max-width: 70%;
                word-wrap: break-word;
                line-height: 1.5;
            }
            
            .message.user .message-content {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-bottom-right-radius: 4px;
            }
            
            .message.assistant .message-content {
                background: white;
                color: #333;
                border: 1px solid #e1e8ed;
                border-bottom-left-radius: 4px;
                text-align: left;
            }
            
            .sources {
                margin-top: 15px;
                padding: 15px;
                background: #f0f4f8;
                border-radius: 12px;
                border-left: 4px solid #667eea;
            }
            
            .sources-title {
                font-weight: 600;
                font-size: 13px;
                color: #667eea;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .source-item {
                background: white;
                padding: 10px;
                margin: 8px 0;
                border-radius: 8px;
                font-size: 13px;
                border: 1px solid #e1e8ed;
            }
            
            .source-name {
                font-weight: 600;
                color: #667eea;
                margin-bottom: 5px;
            }
            
            .source-meta {
                color: #666;
                font-size: 12px;
            }
            
            .source-preview {
                margin-top: 5px;
                color: #888;
                font-size: 12px;
                font-style: italic;
            }
            
            .input-container {
                padding: 20px 30px;
                background: white;
                border-top: 1px solid #e1e8ed;
                display: flex;
                gap: 15px;
            }
            
            #messageInput {
                flex: 1;
                padding: 15px 20px;
                border: 2px solid #e1e8ed;
                border-radius: 25px;
                font-size: 15px;
                outline: none;
                transition: border-color 0.3s;
            }
            
            #messageInput:focus {
                border-color: #667eea;
            }
            
            #sendButton {
                padding: 15px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            #sendButton:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            
            #sendButton:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .loading {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #667eea;
                animation: pulse 1.5s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% {
                    opacity: 1;
                }
                50% {
                    opacity: 0.3;
                }
            }
            
            .typing-indicator {
                display: inline-flex;
                gap: 5px;
                padding: 15px 20px;
                background: white;
                border: 1px solid #e1e8ed;
                border-radius: 18px;
                border-bottom-left-radius: 4px;
            }
            
            .typing-indicator span {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #667eea;
                animation: typing 1.4s infinite;
            }
            
            .typing-indicator span:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .typing-indicator span:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes typing {
                0%, 60%, 100% {
                    transform: translateY(0);
                }
                30% {
                    transform: translateY(-10px);
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 RAG Chat Assistant</h1>
                <p>Ask questions and get answers with cited sources</p>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="message assistant">
                    <div class="message-content">
                        Hello! I'm your RAG-powered assistant. Ask me anything and I'll search through the knowledge base to provide accurate answers with citations.
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
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Add user message to chat
                addMessage(message, 'user');
                input.value = '';
                
                // Disable input
                const sendButton = document.getElementById('sendButton');
                sendButton.disabled = true;
                input.disabled = true;
                
                // Show typing indicator
                const typingId = addTypingIndicator();
                
                try {
                    // Send to backend
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            conversation_history: conversationHistory
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    
                    const data = await response.json();
                    
                    // Remove typing indicator
                    removeTypingIndicator(typingId);
                    
                    // Add assistant response
                    addMessage(data.response, 'assistant', data.sources);
                    
                    // Update conversation history
                    conversationHistory.push(
                        { role: 'user', content: message },
                        { role: 'assistant', content: data.response }
                    );
                    
                    // Keep only last 10 messages in history
                    if (conversationHistory.length > 10) {
                        conversationHistory = conversationHistory.slice(-10);
                    }
                    
                } catch (error) {
                    removeTypingIndicator(typingId);
                    addMessage('Sorry, there was an error processing your request. Please try again.', 'assistant');
                    console.error('Error:', error);
                } finally {
                    sendButton.disabled = false;
                    input.disabled = false;
                    input.focus();
                }
            }
            
            function addMessage(text, sender, sources = null) {
                const container = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.textContent = text;
                
                messageDiv.appendChild(contentDiv);
                
                // Add sources if available
                if (sources && sources.length > 0) {
                    const sourcesDiv = document.createElement('div');
                    sourcesDiv.className = 'sources';
                    sourcesDiv.innerHTML = '<div class="sources-title">📚 Sources</div>';
                    
                    sources.forEach((source, index) => {
                        const sourceItem = document.createElement('div');
                        sourceItem.className = 'source-item';
                        sourceItem.innerHTML = `
                            <div class="source-name">${index + 1}. ${source.source}</div>
                            <div class="source-meta">Chunk ${source.chunk_index}/${source.total_chunks} • Similarity: ${(source.similarity * 100).toFixed(1)}%</div>
                            <div class="source-preview">${source.preview}</div>
                        `;
                        sourcesDiv.appendChild(sourceItem);
                    });
                    
                    contentDiv.appendChild(sourcesDiv);
                }
                
                container.appendChild(messageDiv);
                container.scrollTop = container.scrollHeight;
            }
            
            function addTypingIndicator() {
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
            }
            
            function removeTypingIndicator(id) {
                const element = document.getElementById(id);
                if (element) {
                    element.remove();
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)