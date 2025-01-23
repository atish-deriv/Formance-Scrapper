import os
import json
from pathlib import Path

def check_file_content(md_path):
    """Check if the markdown file has proper content and wasn't just created with error content."""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check if file contains error messages or is too short
            if "Error in AI processing" in content or len(content.strip()) < 100:
                return False, content
            return True, content
    except Exception as e:
        return False, str(e)

def check_processing_status():
    scraped_dir = "scraped_data"
    processed_dir = "ai_processed_content"
    
    # Get all JSON files
    json_files = {Path(f).stem for f in os.listdir(scraped_dir) if f.endswith('.json')}
    
    # Check processed markdown files
    processed_files = set()
    failed_files = {}  # Changed to dict to store error content
    
    for f in os.listdir(processed_dir):
        if f.endswith('.md'):
            file_stem = Path(f).stem
            success, content = check_file_content(os.path.join(processed_dir, f))
            if success:
                processed_files.add(file_stem)
            else:
                failed_files[file_stem] = content
    
    # Find files that need processing
    remaining_files = json_files - processed_files - set(failed_files.keys())
    
    print(f"Total JSON files: {len(json_files)}")
    print(f"Successfully processed: {len(processed_files)}")
    print(f"Failed processing: {len(failed_files)}")
    print(f"Files remaining to process: {len(remaining_files)}")
    
    if failed_files:
        print("\nFiles that failed processing (need retry):")
        for file, content in sorted(failed_files.items()):
            print(f"\n- {file}.json")
            print("Error content preview:")
            # Print first 200 characters of content to see the error
            print(content[:200] + "..." if len(content) > 200 else content)
    
    if remaining_files:
        print("\nFiles that still need processing:")
        for file in sorted(remaining_files):
            print(f"- {file}.json")

    # Print some successfully processed files as reference
    if processed_files:
        print("\nSample of successfully processed files:")
        for file in sorted(list(processed_files))[:5]:
            print(f"- {file}.json")

if __name__ == "__main__":
    check_processing_status()
