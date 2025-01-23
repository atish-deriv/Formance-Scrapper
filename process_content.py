import os
import json
from pathlib import Path

class ContentProcessor:
    def __init__(self, input_dir="scraped_data", output_dir="processed_content"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def process_headings(self, headings):
        """Process and format headings into a hierarchical structure."""
        formatted_headings = []
        for heading in headings:
            # Create markdown heading with proper level
            heading_md = '#' * heading['level']
            formatted_headings.append(f"{heading_md} {heading['text']}\n")
        return '\n'.join(formatted_headings)
    
    def process_paragraphs(self, paragraphs):
        """Process and format paragraphs with additional context."""
        formatted_paragraphs = []
        for paragraph in paragraphs:
            # Add paragraph with proper spacing
            formatted_paragraphs.append(f"{paragraph}\n")
        return '\n'.join(formatted_paragraphs)
    
    def process_code_blocks(self, code_blocks):
        """Process and format code blocks."""
        if not code_blocks:
            return ""
        
        formatted_blocks = []
        for block in code_blocks:
            # Format as markdown code block
            formatted_blocks.append(f"```\n{block}\n```\n")
        return '\n'.join(formatted_blocks)
    
    def create_markdown(self, content, url):
        """Create a detailed markdown document from the content."""
        sections = []
        
        # Add metadata section
        sections.append("---")
        sections.append(f"url: {url}")
        sections.append(f"title: {content['title']}")
        sections.append("---\n")
        
        # Add title as main heading
        sections.append(f"# {content['title']}\n")
        
        # Add source URL reference
        sections.append(f"*Source: [{url}]({url})*\n")
        
        # Add table of contents if there are headings
        if content['headings']:
            sections.append("## Table of Contents\n")
            for heading in content['headings']:
                indent = "  " * (heading['level'] - 1)
                sections.append(f"{indent}- {heading['text']}\n")
            sections.append("\n")
        
        # Add content sections
        sections.append("## Content\n")
        
        # Add headings and their content
        sections.append(self.process_headings(content['headings']))
        
        # Add paragraphs
        if content['paragraphs']:
            sections.append("\n### Detailed Information\n")
            sections.append(self.process_paragraphs(content['paragraphs']))
        
        # Add code blocks if any
        if content['code_blocks']:
            sections.append("\n### Code Examples\n")
            sections.append(self.process_code_blocks(content['code_blocks']))
        
        return '\n'.join(sections)
    
    def process_file(self, json_path):
        """Process a single JSON file and create corresponding markdown."""
        try:
            # Read JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create markdown content
            markdown_content = self.create_markdown(data['content'], data['url'])
            
            # Create output filename
            output_filename = Path(json_path).stem + '.md'
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Save markdown file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"Processed: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {json_path}: {str(e)}")
    
    def process_all_files(self):
        """Process all JSON files in the input directory."""
        json_files = [f for f in os.listdir(self.input_dir) if f.endswith('.json')]
        total_files = len(json_files)
        
        print(f"Found {total_files} files to process")
        
        for i, filename in enumerate(json_files, 1):
            print(f"Processing file {i}/{total_files}: {filename}")
            self.process_file(os.path.join(self.input_dir, filename))
        
        print("\nProcessing completed!")
        print(f"Markdown files have been saved in: {os.path.abspath(self.output_dir)}")

if __name__ == "__main__":
    processor = ContentProcessor()
    processor.process_all_files()
