import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import os
import logging
from typing import Set, List, Optional
from aiolimiter import AsyncLimiter
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncCrawler:
    """
    Asynchronous web crawler that fetches and saves content from a specified domain.
    
    Attributes:
        start_url (str): The initial URL to start crawling from.
        domain (str): The domain to restrict crawling to.
        output_dir (str): Directory to save crawled content.
        seen_urls (Set[str]): Set of URLs already processed.
        rate_limiter (AsyncLimiter): Limits the rate of requests to the server.
    """

    def __init__(self, start_url: str, output_dir: str, rate_limit: int = 5):
        """
        Initialize the AsyncCrawler.

        Args:
            start_url (str): The URL to start crawling from.
            output_dir (str): Directory to save crawled content.
            rate_limit (int): Maximum number of requests per second.
        """
        self.start_url = start_url
        self.domain = urlparse(start_url).netloc
        self.output_dir = output_dir
        self.seen_urls: Set[str] = set()
        self.rate_limiter = AsyncLimiter(rate_limit, 1)  # 5 requests per second

    async def crawl(self):
        """
        Start the crawling process.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        async with aiohttp.ClientSession() as session:
            await self.process_url(session, self.start_url)

    async def process_url(self, session: aiohttp.ClientSession, url: str):
        """
        Process a single URL: fetch content, save text, and extract links.

        Args:
            session (aiohttp.ClientSession): The active client session.
            url (str): The URL to process.
        """
        if url in self.seen_urls:
            return
        self.seen_urls.add(url)

        async with self.rate_limiter:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                        return
                    content_type = response.headers.get('Content-Type', '')
                    if not content_type.startswith('text/html'):
                        logger.info(f"Skipping non-HTML content at {url}")
                        return
                    text = await response.text()
            except aiohttp.ClientError as e:
                logger.error(f"Error fetching {url}: {e}")
                return

        soup = BeautifulSoup(text, 'html.parser')
        self.save_text(url, soup.get_text())

        tasks = []
        for link in self.extract_links(soup, url):
            if link not in self.seen_urls:
                tasks.append(asyncio.create_task(self.process_url(session, link)))
        await asyncio.gather(*tasks)

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract all links from a BeautifulSoup object that belong to the same domain.

        Args:
            soup (BeautifulSoup): Parsed HTML content.
            base_url (str): The base URL for resolving relative links.

        Returns:
            List[str]: List of extracted full URLs.
        """
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == self.domain:
                links.append(full_url)
        return links

    def save_text(self, url: str, text: str):
        """
        Save the extracted text content to a file.

        Args:
            url (str): The URL of the page (used for filename).
            text (str): The text content to save.
        """
        filename = url.replace('https://', '').replace('http://', '').replace('/', '_') + '.txt'
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved content from {url} to {filepath}")

def main():
    """
    Main function to set up and run the crawler.
    """
    parser = argparse.ArgumentParser(description="Web Crawler")
    parser.add_argument("start_url", help="The URL to start crawling from")
    parser.add_argument("--output", default="crawled_text", help="Output directory for crawled text")
    parser.add_argument("--rate-limit", type=int, default=5, help="Number of requests per second")
    args = parser.parse_args()

    crawler = AsyncCrawler(args.start_url, args.output, args.rate_limit)
    asyncio.run(crawler.crawl())

if __name__ == "__main__":
    main()


"""

Usage:
pip install aiohttp beautifulsoup4 aiolimiter
python async_crawler.py https:URL --output crawled_data --rate-limit 3

"""