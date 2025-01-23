import os
import json
from pathlib import Path
from dotenv import load_dotenv
import openai
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

class AIContentProcessor:
    def __init__(self, input_dir="scraped_data", output_dir="ai_processed_content"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _format_headings_for_ai(self, headings):
        """Format headings for AI processing."""
        return "\n".join([f"{'#' * h['level']} {h['text']}" for h in headings])

    def _format_paragraphs_for_ai(self, paragraphs):
        """Format paragraphs for AI processing."""
        return "\n\n".join(paragraphs)

    def _format_code_blocks_for_ai(self, code_blocks):
        """Format code blocks for AI processing."""
        if not code_blocks:
            return ""
        return "\n\n".join([f"```\n{block}\n```" for block in code_blocks])

    def enhance_content(self, content):
        """Use OpenAI to enhance and rewrite the content."""
        # Prepare the input text
        input_text = f"""
# {content['title']}

## Headings
{self._format_headings_for_ai(content['headings'])}

## Content
{self._format_paragraphs_for_ai(content['paragraphs'])}

## Code Examples
{self._format_code_blocks_for_ai(content['code_blocks'])}
"""

        system_prompt = """You are a technical documentation expert. Your task is to:
1. Rewrite the provided technical documentation in a clear, detailed, and comprehensive way
2. Maintain all technical accuracy and details
3. Expand on concepts to make them more understandable
4. Use proper markdown formatting
5. Structure the content logically
6. Keep all code examples intact
7. Add clear section transitions and explanations
8. Ensure the content is optimized for both readability and technical accuracy

Format the output as a proper markdown document with:
- Clear headings and subheadings
- Well-organized sections
- Proper code block formatting
- Consistent styling
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please enhance and rewrite this technical documentation:\n\n{input_text}"}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error in AI processing: {str(e)}")
            return input_text

    def create_markdown(self, original_url, title, enhanced_content):
        """Create the final markdown document."""
        return f"""---
url: {original_url}
title: {title}
---

{enhanced_content}

---
*Source: [{original_url}]({original_url})*
"""

    def process_file(self, json_path):
        """Process a single JSON file with AI enhancement."""
        try:
            # Read JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\nProcessing: {json_path}")
            
            # Enhance content with AI
            enhanced_content = self.enhance_content(data['content'])
            
            # Create markdown content
            markdown_content = self.create_markdown(
                data['url'],
                data['content']['title'],
                enhanced_content
            )
            
            # Create output filename
            output_filename = Path(json_path).stem + '.md'
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Save markdown file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"✓ Created: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {json_path}: {str(e)}")

    def process_all_files(self):
        """Process all JSON files in the input directory."""
        json_files = [f for f in os.listdir(self.input_dir) if f.endswith('.json')]
        total_files = len(json_files)
        
        print(f"Found {total_files} files to process")
        print("Starting AI-enhanced documentation processing...")
        
        for i, filename in enumerate(json_files, 1):
            print(f"\nFile {i}/{total_files}")
            self.process_file(os.path.join(self.input_dir, filename))
            # Small delay to avoid rate limits
            time.sleep(1)
        
        print("\nProcessing completed!")
        print(f"Enhanced markdown files have been saved in: {os.path.abspath(self.output_dir)}")

if __name__ == "__main__":
    processor = AIContentProcessor()
    processor.process_all_files()
