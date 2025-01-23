import os
import json
import tiktoken
from tqdm import tqdm
import openai
import pinecone
from pathlib import Path
from dotenv import load_dotenv
import time
import hashlib

# Load environment variables
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")  # This is the encoding used by text-embedding-ada-002

def chunk_text(text, max_tokens=500, overlap=50):
    """Split text into chunks of approximately max_tokens with overlap."""
    tokens = tokenizer.encode(text)
    chunks = []
    
    i = 0
    while i < len(tokens):
        # Get chunk of tokens
        chunk_end = min(i + max_tokens, len(tokens))
        chunk = tokens[i:chunk_end]
        
        # Decode chunk back to text
        chunk_text = tokenizer.decode(chunk)
        
        # Add to chunks
        chunks.append(chunk_text)
        
        # Move to next chunk, accounting for overlap
        i += (max_tokens - overlap)
    
    return chunks

def extract_metadata(content):
    """Extract metadata from markdown content."""
    lines = content.split('\n')
    metadata = {}
    content_start = 0
    
    # Extract YAML frontmatter
    if lines[0].strip() == '---':
        i = 1
        while i < len(lines) and lines[i].strip() != '---':
            line = lines[i].strip()
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
            i += 1
        content_start = i + 1
    
    # Join remaining lines as content
    content = '\n'.join(lines[content_start:])
    
    return metadata, content

def get_embedding(text, retries=3):
    """Get embedding for text using OpenAI's API with retry logic."""
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"Error getting embedding, retrying... ({str(e)})")
            time.sleep(5)

def create_document_id(text):
    """Create a unique document ID based on content."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def process_markdown_files():
    """Process markdown files and prepare them for Pinecone."""
    input_dir = "ai_processed_content"
    vectors = []
    
    # Get list of markdown files
    md_files = [f for f in os.listdir(input_dir) if f.endswith('.md')]
    print(f"Found {len(md_files)} markdown files to process")
    
    for filename in tqdm(md_files, desc="Processing files"):
        filepath = os.path.join(input_dir, filename)
        
        try:
            # Read markdown file
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata and content
            metadata, main_content = extract_metadata(content)
            
            # Chunk the content
            chunks = chunk_text(main_content)
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Create unique ID for chunk
                doc_id = f"{create_document_id(chunk)}_{i}"
                
                # Get embedding
                embedding = get_embedding(chunk)
                
                # Prepare metadata for this chunk
                chunk_metadata = {
                    **metadata,
                    'filename': filename,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'content': chunk  # Store the actual text for retrieval
                }
                
                # Add to vectors list
                vectors.append({
                    'id': doc_id,
                    'values': embedding,
                    'metadata': chunk_metadata
                })
            
            # Add small delay to respect rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return vectors

def save_vectors(vectors, filename='vectors.json'):
    """Save vectors to a JSON file."""
    print(f"\nSaving {len(vectors)} vectors to {filename}...")
    with open(filename, 'w') as f:
        json.dump(vectors, f)
    print("Vectors saved successfully")

def main():
    print("Starting embedding creation process...")
    
    try:
        # Process markdown files and create embeddings
        vectors = process_markdown_files()
        print(f"\nCreated {len(vectors)} vectors")
        
        # Save vectors to file
        save_vectors(vectors)
        
        print("\nProcess completed successfully!")
        print("You can now run upload_to_pinecone.py to upload the vectors to Pinecone")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
