import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import json

class FormanceScraper:
    def __init__(self, base_url, output_dir="scraped_data"):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.visited_urls = set()
        self.output_dir = output_dir
        self.rate_limit = 1  # seconds between requests
        self.last_request_time = 0
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def is_valid_url(self, url):
        """Check if URL is valid and belongs to the same domain."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and parsed.netloc == self.domain
    
    def normalize_url(self, url):
        """Normalize URL by removing fragments and normalizing slashes."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    
    def respect_rate_limit(self):
        """Ensure we respect rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def get_page_content(self, url):
        """Fetch and parse page content."""
        self.respect_rate_limit()
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {str(e)}")
            return None
    
    def extract_content(self, soup):
        """Extract relevant content from the page."""
        content = {
            'title': '',
            'headings': [],
            'paragraphs': [],
            'code_blocks': []
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            content['title'] = title_tag.get_text(strip=True)
        
        # Extract headings
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            content['headings'].append({
                'level': int(heading.name[1]),
                'text': heading.get_text(strip=True)
            })
        
        # Extract paragraphs
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text:  # Only add non-empty paragraphs
                content['paragraphs'].append(text)
        
        # Extract code blocks
        for code in soup.find_all(['pre', 'code']):
            text = code.get_text(strip=True)
            if text:
                content['code_blocks'].append(text)
        
        return content
    
    def extract_links(self, soup, base_url):
        """Extract all valid links from the page."""
        links = set()
        for a in soup.find_all('a', href=True):
            url = urljoin(base_url, a['href'])
            normalized_url = self.normalize_url(url)
            if self.is_valid_url(normalized_url):
                links.add(normalized_url)
        return links
    
    def save_content(self, url, content):
        """Save scraped content to a JSON file."""
        filename = urlparse(url).path.strip('/')
        if not filename:
            filename = 'index'
        filename = filename.replace('/', '_') + '.json'
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'url': url,
                'content': content
            }, f, indent=2, ensure_ascii=False)
    
    def scrape(self):
        """Main scraping method."""
        urls_to_visit = {self.base_url}
        
        with tqdm(desc="Scraping pages") as pbar:
            while urls_to_visit:
                url = urls_to_visit.pop()
                if url in self.visited_urls:
                    continue
                
                print(f"\nScraping: {url}")
                content = self.get_page_content(url)
                
                if content:
                    soup = BeautifulSoup(content, 'html.parser')
                    extracted_content = self.extract_content(soup)
                    self.save_content(url, extracted_content)
                    
                    # Add new links to visit
                    new_links = self.extract_links(soup, url)
                    urls_to_visit.update(new_links - self.visited_urls)
                
                self.visited_urls.add(url)
                pbar.update(1)
        
        print(f"\nScraping completed. Scraped {len(self.visited_urls)} pages.")
        print(f"Data saved in: {os.path.abspath(self.output_dir)}")

if __name__ == "__main__":
    BASE_URL = "https://docs.formance.com/"
    scraper = FormanceScraper(BASE_URL)
    scraper.scrape()
