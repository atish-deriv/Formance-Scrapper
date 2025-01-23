import sys
from scraper import FormanceScraper

def main():
    try:
        base_url = "https://docs.formance.com/"
        print(f"Starting to scrape {base_url}")
        print("Data will be saved in the 'scraped_data' directory")
        
        scraper = FormanceScraper(base_url)
        scraper.scrape()
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
