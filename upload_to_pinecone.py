import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def load_vectors():
    """Load vectors from the saved JSON file."""
    with open('vectors.json', 'r') as f:
        return json.load(f)

def upload_vectors(vectors, batch_size=100):
    """Upload vectors to Pinecone in batches with improved error handling."""
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    print("Connecting to index...")
    index = pc.Index("formance")
    
    # Upload vectors in batches
    total_batches = (len(vectors) + batch_size - 1) // batch_size
    
    print(f"\nUploading {len(vectors)} vectors in {total_batches} batches...")
    
    successful_uploads = 0
    failed_batches = []
    
    for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading batches"):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
            successful_uploads += len(batch)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"\nError uploading batch {i//batch_size + 1}: {str(e)}")
            failed_batches.append((i, i + batch_size))
            time.sleep(5)  # Longer delay after error
    
    print(f"\nUpload completed:")
    print(f"Successfully uploaded: {successful_uploads} vectors")
    
    if failed_batches:
        print("\nFailed batches (index ranges):")
        for start, end in failed_batches:
            print(f"- {start} to {end}")
        
        # Save failed batches for retry
        with open('failed_uploads.json', 'w') as f:
            json.dump({
                'failed_ranges': failed_batches,
                'vectors': vectors
            }, f)
        print("\nFailed uploads have been saved to 'failed_uploads.json' for retry")

def main():
    try:
        print("Loading vectors from file...")
        vectors = load_vectors()
        print(f"Loaded {len(vectors)} vectors")
        
        upload_vectors(vectors)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Verify your Pinecone API key")
        print("3. Verify your index name exists in Pinecone")
        print("4. Check if Pinecone service is available")

if __name__ == "__main__":
    main()
