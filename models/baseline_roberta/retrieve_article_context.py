# Standard libraries
import json
import time
import threading
from pathlib import Path
from urllib.parse import urlparse
from urllib import robotparser
from concurrent.futures import ThreadPoolExecutor, as_completed

# Custom libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'src' / 'Sarcasm_Headlines_Dataset_v2.json'
OUTPUT_PATH = PROJECT_ROOT / 'models' / 'baseline_roberta' / 'data' /  'sarcasm_with_context.csv'
CACHE_PATH = PROJECT_ROOT / 'models' / 'baseline_roberta' / 'data' /  'scraped_context.json'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (NUS research project; contact: student)'
}

REQUEST_DELAY = 0.7
MAX_WORKERS = 6
SAVE_EVERY = 50

robot_parsers = {}
robot_lock = threading.Lock()
cache_lock = threading.Lock()
last_request_time = {}
domain_lock = threading.Lock()
domain_permissions = {}

def is_valid_url(url):
    if not isinstance(url, str):
        return False

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return False

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False

    if "http:" in parsed.netloc or "https:" in parsed.netloc:
        return False

    return True

def get_robot_parser(url):
    parsed = urlparse(url)
    domain = parsed.scheme + '://' + parsed.netloc

    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f'Invalid URL for robots check: {url}')

    with robot_lock:
        if domain not in robot_parsers:
            rp = robotparser.RobotFileParser()
            rp.set_url(domain + '/robots.txt')
            try:
                rp.read()
            except Exception as e:
                print(f'Could not read robots.txt for {domain}: {e}')
            robot_parsers[domain] = rp

    return robot_parsers[domain]

def is_allowed(url):
    parsed = urlparse(url)
    domain = parsed.scheme + '://' + parsed.netloc

    if domain in domain_permissions:
        return domain_permissions[domain]

    rp = get_robot_parser(url)
    try:
        allowed = rp.can_fetch(HEADERS['User-Agent'], url)
    except:
        allowed = False

    domain_permissions[domain] = allowed
    return allowed

def extract_metadata(html):
    soup = BeautifulSoup(html, "html.parser")

    def get_meta(candidates):
        for attr_name, attr_value in candidates:
            tag = soup.find("meta", attrs={attr_name: attr_value})
            if tag and tag.get("content"):
                content = tag.get("content").strip()
                if content:
                    return content
        return ""

    description = get_meta([
        ("name", "description"),
        ("property", "og:description"),
        ("name", "twitter:description"),
    ])

    author = get_meta([
        ("name", "author"),
        ("property", "article:author"),
        ("name", "parsely-author"),
    ])

    section = get_meta([
        ("property", "article:section"),
        ("name", "parsely-section"),
        ("name", "section"),
    ])

    if not description or not author or not section:
        for script in soup.find_all("script", type="application/ld+json"):
            raw = script.string
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue

                if not description and isinstance(item.get("description"), str):
                    description = item["description"].strip()

                if not section and isinstance(item.get("articleSection"), str):
                    section = item["articleSection"].strip()

                if not author:
                    author_data = item.get("author")
                    if isinstance(author_data, dict):
                        author = str(author_data.get("name", "")).strip()
                    elif isinstance(author_data, list) and author_data:
                        first = author_data[0]
                        if isinstance(first, dict):
                            author = str(first.get("name", "")).strip()
                    elif isinstance(author_data, str):
                        author = author_data.strip()

    return {
        "description": description,
        "section": section,
        "author": author,
    }

def wait_for_domain_delay(url):
    domain = urlparse(url).scheme + '://' + urlparse(url).netloc
    with domain_lock:
        now = time.time()
        last_time = last_request_time.get(domain, 0)
        elapsed = now - last_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        last_request_time[domain] = time.time()

def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session

def scrape_url(session, url):
    try:
        if not is_valid_url(url):
            print(f'Invalid URL: {url}')
            return {
                'description': '',
                'section': '',
                'author': '',
                'blocked_by_robots': False
            }

        if not isinstance(url, str) or not url.strip():
            print(f'{url} not instance.')
            return {
                'description': '',
                'section': '',
                'author': '',
                'blocked_by_robots': False
            }

        if not is_allowed(url):
            print(f'{url} is not allowed to be scraped.')
            return {
                'description': '',
                'section': '',
                'author': '',
                'blocked_by_robots': True
            }

        wait_for_domain_delay(url)

        response = session.get(url, timeout=10)

        if response.status_code != 200:
            return {
                'description': '',
                'section': '',
                'author': '',
                'blocked_by_robots': False
            }

        metadata = extract_metadata(response.text)
        metadata['blocked_by_robots'] = False
        return metadata

    except Exception as e:
        print(f'Error scraping {url}: {e}')
        return {
            'description': '',
            'section': '',
            'author': '',
            'blocked_by_robots': False
        }
    
def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CACHE_PATH.with_suffix('.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)
    temp_path.replace(CACHE_PATH)

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        thread_local.session.headers.update(HEADERS)
    return thread_local.session

def worker(url):
    session = get_session()
    result = scrape_url(session, url)
    return url, result

def main():
    print('Loading dataset...')
    df = pd.read_json(DATA_PATH, lines=True)

    if 'article_link' not in df.columns:
        raise ValueError("Dataset does not contain 'article_link' column")

    cache = load_cache()

    # Deduplicate URLs first
    unique_urls = df['article_link'].dropna().astype(str).unique().tolist()
    urls_to_scrape = [url for url in unique_urls if url not in cache]

    print(f'Total rows: {len(df)}')
    print(f'Unique URLs: {len(unique_urls)}')
    print(f'Already cached: {len(cache)}')
    print(f'Remaining URLs to scrape: {len(urls_to_scrape)}')

    if urls_to_scrape:
        print('Scraping remaining URLs...')
        completed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(worker, url): url for url in urls_to_scrape}
            results_buffer = []
            for future in tqdm(as_completed(futures), total=len(futures)):
                url, result = future.result()

                results_buffer.append((url, result))

                if len(results_buffer) >= SAVE_EVERY:
                    with cache_lock:
                        for u, r in results_buffer:
                            cache[u] = r
                        save_cache(cache)
                    results_buffer = []


    print('Building dataframe from cache...')
    df['description'] = df['article_link'].map(lambda x: cache.get(str(x), {}).get('description', ''))
    df['section'] = df['article_link'].map(lambda x: cache.get(str(x), {}).get('section', ''))
    df['author'] = df['article_link'].map(lambda x: cache.get(str(x), {}).get('author', ''))
    df['blocked_by_robots'] = df['article_link'].map(lambda x: cache.get(str(x), {}).get('blocked_by_robots', False))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f'Saved enriched dataset to {OUTPUT_PATH}')

if __name__ == '__main__':
    main()