import psycopg2
import os
from pathlib import Path

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "migrated",
    "user": "postgres",
    "password": "1234",
    "port": 5432
}


# Folder path
RESULTS_FOLDER = './results'

def get_sources_from_db():
    """Fetch all source values from the document_embeddings table."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT source FROM document_embeddings")
        sources = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Extract just the source names and remove .md extension
        return {source[0].replace('.md', '') for source in sources if source[0]}
    
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return set()

def get_folders_from_directory():
    """Get all subdirectories in the results folder."""
    try:
        results_path = Path(RESULTS_FOLDER)
        if not results_path.exists():
            print(f"Error: Results folder '{RESULTS_FOLDER}' does not exist!")
            return set()
        
        # Get all subdirectories (not files)
        folders = {item.name for item in results_path.iterdir() if item.is_dir()}
        return folders
    
    except Exception as e:
        print(f"Error reading directory: {e}")
        return set()

def compare_sources():
    """Compare database sources with folder structure."""
    print("Fetching data from database...")
    db_sources = get_sources_from_db()
    
    print("Scanning results folder...")
    folder_names = get_folders_from_directory()
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    # Find sources in DB but not in folders
    missing_in_folders = db_sources - folder_names
    
    # Find folders that don't have corresponding DB entries
    missing_in_db = folder_names - db_sources
    
    # Count matches
    matches = db_sources & folder_names
    
    print(f"\nTotal entries in database: {len(db_sources)}")
    print(f"Total folders in results: {len(folder_names)}")
    print(f"Matching entries: {len(matches)}")
    
    if missing_in_folders:
        print(f"\n❌ MISSING IN FOLDERS ({len(missing_in_folders)} items):")
        print("These exist in DB but corresponding folders are missing:")
        for item in sorted(missing_in_folders):
            print(f"  - {item} (DB source: {item}.md → Expected folder: ./results/{item})")
    else:
        print("\n✅ All database entries have corresponding folders")
    
    if missing_in_db:
        print(f"\n❌ MISSING IN DATABASE ({len(missing_in_db)} items):")
        print("These folders exist but have no corresponding DB entries:")
        for item in sorted(missing_in_db):
            print(f"  - ./results/{item} → Expected DB source: {item}.md")
    else:
        print("\n✅ All folders have corresponding database entries")
    
    if not missing_in_folders and not missing_in_db:
        print("\n🎉 Perfect match! All entries are consistent.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    compare_sources()