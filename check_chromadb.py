import chromadb

# Initialize the ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Get your specific collection
collection = client.get_collection(name="document_embeddings")

# Get all data from the collection
results = collection.get(include=['documents', 'metadatas', 'embeddings'])

# Open a file to write the data
with open("chromadb_data.txt", "w", encoding="utf-8") as file:
    file.write(f"Total documents: {len(results['ids'])}\n")
    file.write("="*50 + "\n\n")
    
    for i in range(len(results['ids'])):
        file.write(f"Document {i+1}:\n")
        file.write(f"  ID: {results['ids'][i]}\n")
        file.write(f"  Document: {results['documents'][i] if results['documents'] else 'N/A'}\n")
        file.write(f"  Metadata: {results['metadatas'][i] if results['metadatas'] else 'N/A'}\n")
        
        # Fixed embedding line - check if embeddings exist and has items
        if results['embeddings'] is not None and len(results['embeddings']) > i:
            file.write(f"  Embedding: {results['embeddings'][i][:5]}... (truncated)\n")
        else:
            file.write(f"  Embedding: N/A\n")
        
        file.write("-"*50 + "\n")

print("Data saved to chromadb_data.txt")