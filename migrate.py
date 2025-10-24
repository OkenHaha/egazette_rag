# migrate_chroma_to_postgres.py
import chromadb
import psycopg2
from pgvector.psycopg2 import register_vector
import json
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "document_embeddings"

# PostgreSQL connection settings
PG_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "json_embed",
    "user": "postgres",
    "password": "1234"
}

BATCH_SIZE = 1000  # Insert in batches for better performance

# ----------------------------
# MIGRATION FUNCTION
# ----------------------------
def migrate_chroma_to_postgres():
    """Migrate data from ChromaDB to PostgreSQL with pgvector"""
    
    print("🔄 Starting migration from ChromaDB to PostgreSQL...")
    
    # Connect to ChromaDB
    print(f"📖 Reading from ChromaDB: {CHROMA_DB_PATH}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Error accessing collection '{CHROMA_COLLECTION_NAME}': {e}")
        return
    
    total_count = collection.count()
    print(f"📊 Found {total_count} documents in ChromaDB")
    
    if total_count == 0:
        print("❌ No documents to migrate.")
        return
    
    # Connect to PostgreSQL
    print("🔌 Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        register_vector(conn)
        cursor = conn.cursor()
        print("✅ Connected to PostgreSQL")
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        return
    
    # Check if table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'document_embeddings'
        );
    """)
    
    if not cursor.fetchone()[0]:
        print("❌ Table 'document_embeddings' does not exist. Please run the setup SQL first.")
        conn.close()
        return
    
    # Get all data from ChromaDB in batches
    print("\n🔄 Migrating data...")
    offset = 0
    migrated_count = 0
    
    with tqdm(total=total_count, desc="Migrating") as pbar:
        while offset < total_count:
            # Fetch batch from ChromaDB
            batch_size = min(BATCH_SIZE, total_count - offset)
            results = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas", "embeddings"]
            )
            
            # Prepare batch insert
            insert_data = []
            for doc_id, doc, meta, emb in zip(
                results['ids'],
                results['documents'],
                results['metadatas'],
                results['embeddings']
            ):
                insert_data.append((
                    doc_id,
                    doc,
                    emb,  # pgvector handles list conversion
                    meta.get('source'),
                    meta.get('parent_id'),
                    meta.get('chunk_index'),
                    meta.get('total_chunks')
                ))
            
            # Batch insert into PostgreSQL
            try:
                cursor.executemany("""
                    INSERT INTO document_embeddings 
                    (id, content, embedding, source, parent_id, chunk_index, total_chunks)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        source = EXCLUDED.source,
                        parent_id = EXCLUDED.parent_id,
                        chunk_index = EXCLUDED.chunk_index,
                        total_chunks = EXCLUDED.total_chunks;
                """, insert_data)
                
                conn.commit()
                migrated_count += len(insert_data)
                pbar.update(len(insert_data))
                
            except Exception as e:
                print(f"\n❌ Error inserting batch at offset {offset}: {e}")
                conn.rollback()
                break
            
            offset += batch_size
    
    # Verify migration
    cursor.execute("SELECT COUNT(*) FROM document_embeddings;")
    pg_count = cursor.fetchone()[0]
    
    print(f"\n✅ Migration complete!")
    print(f"   ChromaDB documents: {total_count}")
    print(f"   PostgreSQL rows: {pg_count}")
    print(f"   Successfully migrated: {migrated_count}")
    
    # Show sample statistics
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT parent_id) as unique_parents,
            COUNT(DISTINCT source) as unique_sources,
            AVG(array_length(embedding::real[], 1)) as avg_embedding_dim
        FROM document_embeddings;
    """)
    
    stats = cursor.fetchone()
    print(f"\n📊 Statistics:")
    print(f"   Unique Parent IDs: {stats[0]}")
    print(f"   Unique Sources: {stats[1]}")
    print(f"   Embedding Dimension: {int(stats[2]) if stats[2] else 'N/A'}")
    
    cursor.close()
    conn.close()
    print("\n🎉 PostgreSQL database is ready!")

# ----------------------------
# QUERY FUNCTION FOR POSTGRES
# ----------------------------
def query_postgres(query_text: str, n_results: int = 5):
    """Perform similarity search in PostgreSQL"""
    import requests
    
    OLLAMA_API_URL = "http://localhost:11434/api/embed"
    OLLAMA_EMBED_MODEL = "nomic-embed-text"
    
    print(f"\n🔍 Searching PostgreSQL for: '{query_text}'")
    
    # Get embedding
    payload = {"model": OLLAMA_EMBED_MODEL, "input": [query_text]}
    response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
    
    if response.status_code != 200:
        print(f"❌ Failed to get embedding")
        return
    
    query_embedding = response.json()["embeddings"][0]
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(**PG_CONFIG)
    register_vector(conn)
    cursor = conn.cursor()
    
    # Perform similarity search using cosine distance
    cursor.execute("""
        SELECT 
            id,
            content,
            source,
            parent_id,
            chunk_index,
            total_chunks,
            1 - (embedding <=> %s::vector) as similarity
        FROM document_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, n_results))
    
    results = cursor.fetchall()
    
    print(f"\n🔎 Top {n_results} Results:\n")
    for i, (doc_id, content, source, parent_id, chunk_idx, total_chunks, similarity) in enumerate(results, 1):
        print(f"[{i}] Similarity: {similarity:.4f}")
        print(f"    ID: {doc_id}")
        print(f"    Source: {source}")
        print(f"    Parent ID: {parent_id}")
        print(f"    Chunk: {chunk_idx}/{total_chunks}")
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"    Content: {preview}\n")
    
    cursor.close()
    conn.close()

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What is this about?"
        query_postgres(query)
    else:
        migrate_chroma_to_postgres()