# Scraper.py
import requests
from bs4 import BeautifulSoup, Comment
from transformers import pipeline
from keybert import KeyBERT
from pymongo import MongoClient, errors
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from bson import ObjectId
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
import os
import uvicorn
import time
import random
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, SessionNotCreatedException
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Changed to INFO for clearer production-like logs. Use DEBUG for detailed troubleshooting.
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize AI models
try:
    # Use device="cpu" if you don't have a GPU or encounter issues
    # device=-1 for CPU, 0 for first GPU
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1) 
    kw_model = KeyBERT()
    logger.info("✅ AI models initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize AI models: {e}")
    # Consider whether to exit or continue with limited functionality
    # For critical components like this, exiting is often safer if they are truly required
    exit()

# MongoDB setup
try:
    client = MongoClient(
        os.getenv("MONGODB_URL"),
        serverSelectionTimeoutMS=10000, # Increased timeout
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
    )
    client.admin.command('ping')
    logger.info("✅ Successfully connected to MongoDB")
    db = client["novapress_db"]
    base_collection = db["articles"] # This will store all articles
except errors.ConnectionFailure as err:
    logger.error(f"❌ MongoDB connection failed: {err}")
    exit()

# FastAPI app setup
app = FastAPI(
    title="NovaPress Scraper API",
    description="API for scraping and serving news articles",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models - Updated to match output of transform_article
class Article(BaseModel):
    id: str
    title: str
    summary: str
    imageUrl: str
    date: str
    category: str
    readTime: str
    source: str
    content: str
    tags: List[str]
    likes_count: Optional[int] = 0 # Added for consistency with interaction backend
    comments_count: Optional[int] = 0 # Added for consistency with interaction backend

class ArticleListResponse(BaseModel):
    articles: List[Article]

# News sources configuration
NEWS_SOURCES = {
    "Indian Express": {
        "India": "https://indianexpress.com/section/india/",
        "World": "https://indianexpress.com/section/world/",
        "Sports": "https://indianexpress.com/section/sports/",
        "Entertainment": "https://indianexpress.com/section/entertainment/",
        "Technology": "https://indianexpress.com/section/technology/",
        "Health": "https://indianexpress.com/section/lifestyle/health/",
        "Science": "https://indianexpress.com/section/explained/science/"
    },
    "Times of India": {
        "India": "https://timesofindia.indiatimes.com/india",
        "World": "https://timesofindia.indiatimes.com/world",
        "Sports": "https://timesofindia.indiatimes.com/sports",
        "Entertainment": "https://timesofindia.indiatimes.com/entertainment",
        "Technology": "https://timesofindia.indiatimes.com/business/tech",
        "Health": "https://timesofindia.indiatimes.com/life-style/health-fitness",
        "Science": "https://timesofindia.indiatimes.com/home/science"
    },
    "Hindustan Times": {
        "India": "https://www.hindustantimes.com/india-news",
        "World": "https://www.hindustantimes.com/world-news",
        "Sports": "https://www.hindustantimes.com/sports",
        "Entertainment": "https://www.hindustantimes.com/entertainment",
        "Technology": "https://www.hindustantimes.com/tech",
        "Health": "https://www.hindustantimes.com/lifestyle/health",
        "Science": "https://www.hindustantimes.com/it-s-viral" # HT doesn't have a direct "Science" section. This might need refinement.
    }
}

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1"
]

# Thread-safe queue and control variables
article_queue = queue.Queue()
stop_event = threading.Event()
NUM_WORKERS = 5 # Increased worker count for processing scraped articles

# Utility functions
def get_random_user_agent():
    return random.choice(USER_AGENTS)

def generate_ai_title(text):
    try:
        # If content is very short, just take the first sentence
        if len(text.split()) < 30: 
            return text.split('.')[0][:100].strip() + ("..." if len(text.split('.')[0]) > 100 else "")
        # Use summarizer for title generation
        prompt = f"Generate a concise news title under 15 words: {text[:1000]}" # Limit input for speed
        title = summarizer(prompt, max_length=40, min_length=10, do_sample=False)[0]["summary_text"]
        return title.strip().capitalize()
    except Exception as e:
        logger.warning(f"⚠️ Title generation failed for text length {len(text.split())}: {e}. Returning first sentence.")
        return text.split('.')[0][:100].strip() + ("..." if len(text.split('.')[0]) > 100 else "")

def calculate_read_time(content):
    word_count = len(content.split())
    if word_count < 100:
        return "1 min read"
    return f"{word_count // 200 + 1} min read"

def get_image_url(category):
    # Fallback to Unsplash with a relevant category
    return f"https://source.unsplash.com/800x600/?{category.lower().replace(' ', '')},news,headlines"

def transform_article(article):
    """Transforms a MongoDB document into the Article Pydantic model format."""
    try:
        article_id_str = str(article["_id"])
    except KeyError:
        logger.error(f"❌ Article document missing '_id' field: {article}")
        # If _id is missing, it's a critical data issue, raise error or assign a temporary ID
        return None # Or raise ValueError("Article document is missing '_id' field.")
    
    category = article.get("category", "General")
    
    # Robust date transformation
    published_at = article.get("published_at")
    date_str = "Unknown Date"
    if isinstance(published_at, datetime):
        date_str = published_at.strftime("%B %d, %Y")
    elif isinstance(published_at, str):
        try:
            # Handle various ISO formats and simpler date strings
            parsed_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            date_str = parsed_date.strftime("%B %d, %Y")
        except ValueError:
            # Try parsing common date formats if isoformat fails
            for fmt in ["%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d %b %Y"]:
                try:
                    parsed_date = datetime.strptime(published_at, fmt)
                    date_str = parsed_date.strftime("%B %d, %Y")
                    break
                except ValueError:
                    pass
            if date_str == "Unknown Date":
                logger.warning(f"⚠️ Could not parse 'published_at' string for article {article_id_str}: '{published_at}'. Using 'Unknown Date'.")
    else:
        logger.warning(f"⚠️ 'published_at' field is missing or invalid type for article {article_id_str}: {published_at}. Using 'Unknown Date'.")
        
    content = article.get("content", "")
    summary = article.get("summary", "No summary available")
    title = article.get("title", "No title")

    # Add fallback image URL if not found or invalid
    final_image_url = article.get("imageUrl")
    if not final_image_url or not final_image_url.startswith('http'):
        final_image_url = get_image_url(category)

    return {
        "id": article_id_str,
        "title": title,
        "summary": summary,
        "content": content,
        "imageUrl": final_image_url,
        "date": date_str,
        "category": category,
        "readTime": calculate_read_time(content),
        "source": article.get("source", "Unknown"),
        "tags": article.get("tags", []),
        "likes_count": article.get("likes_count", 0), # Ensure these are present
        "comments_count": article.get("comments_count", 0) # Ensure these are present
    }

def get_domain_specific_selectors(domain):
    """
    Returns specific CSS selectors for different news domains.
    **IMPORTANT**: These selectors might become outdated as websites change.
    You might need to update these based on live website HTML.
    """
    selectors = {
        "indianexpress.com": {
            "article_blocks": [
                "div.nation-sec > ul > li", "div.top-news > ul > li", "div.articles > div",
                "div.story-grid > div", "div.story-card", "article.story"
            ],
            "content": [
                'div.story-details', 'div.full-details', 'div[itemprop="articleBody"]',
                'div.ie-article-content', '#pcl-content'
            ],
            "title": ['h1.title', 'h2.story-title', 'h1', 'meta[property="og:title"]', 'meta[name="twitter:title"]'],
            "summary": ['h2.description', 'meta[property="og:description"]', 'meta[name="description"]'],
            "image": ['meta[property="og:image"]', 'meta[name="twitter:image"]', 'figure.wp-block-image img', 'img.ie-img-top']
        },
        "timesofindia.indiatimes.com": {
            "article_blocks": [
                "div.list5 > ul > li", "div.news-list > div", "div.content > div",
                "span.w_tle", "div.col_l_8 > div", "div.top-story-section li"
            ],
            "content": [
                'div.Normal', 'div.ga-headlines', 'arttextxml', 'div[itemprop="articleBody"]',
                'div._3Mkg-read', 'div._3-cQ_'
            ],
            "title": ['h1.headline', 'h1', 'span.w_tle a', 'meta[property="og:title"]', 'meta[name="twitter:title"]'],
            "summary": ['h2.news_descp', 'meta[property="og:description"]', 'meta[name="description"]'],
            "image": ['meta[property="og:image"]', 'img.artImg', 'img.picture_placeholder']
        },
        "hindustantimes.com": {
            "article_blocks": [
                "div.cartHolder", "div.bigCart", "div.secElement", "div.media-box",
                "h3.hdg3 > a", "div.sortable-item"
            ],
            "content": [
                'div.fullStory', 'div.storyDetails', 'div[itemprop="articleBody"]',
                'div.story-data'
            ],
            "title": ['h1.hdg1', 'h1', 'h3.hdg3 a', '.headline', 'meta[property="og:title"]', 'meta[name="twitter:title"]'],
            "summary": ['h2.hdg4', 'meta[property="og:description"]', 'meta[name="description"]'],
            "image": ['meta[property="og:image"]', 'img.lazy', 'figure.storyPicture img']
        }
    }
    
    # Generic fallback selectors if domain-specific ones are not found or fail
    default_selectors = {
        "article_blocks": [
            "article", "div.article", "div.story", "div.news-item", "div.post",
            "section.story-card", "div[class*='article-card']"
        ],
        "content": [
            'div.content-body', 'div.article-content', 'div[itemprop="articleBody"]',
            'div.entry-content', 'div.story-content', '.article__content'
        ],
        "title": ['h1.story-headline', 'h1.article-title', 'h1', 'h2', '.title', 'meta[property="og:title"]', 'meta[name="twitter:title"]'],
        "summary": ['meta[property="og:description"]', 'meta[name="description"]', 'p.summary', '.article__summary'],
        "image": ['meta[property="og:image"]', 'meta[name="twitter:image"]', 'img.main-image', 'figure img']
    }
    
    return selectors.get(domain, default_selectors)


def get_rendered_content(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument(f"user-agent={get_random_user_agent()}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--no-sandbox") # Required for running in some environments like Docker
    options.add_argument("--disable-dev-shm-usage") # Overcomes limited resource problems
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--incognito")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-features=IsolateOrigins,site-per-process") # Helps with some sites

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(random.uniform(5, 10)) # Increased sleep for more reliable rendering
        logger.debug(f"Successfully rendered content for {url} with Selenium.")
        return driver.page_source
    except SessionNotCreatedException as e:
        logger.error(f"❌ Selenium session creation failed. Ensure ChromeDriver is correctly installed and matches Chrome version. Error: {e}")
        return None
    except WebDriverException as e:
        logger.error(f"❌ Selenium WebDriver error for {url}: {str(e)}")
        # Check if it's a timeout or navigation error
        if "timeout" in str(e).lower() or "net::err" in str(e).lower():
            logger.warning(f"Selenium navigation or timeout error for {url}. Trying direct request.")
            return None # Indicate to try requests.get
        return None
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred in Selenium for {url}: {str(e)}")
        return None
    finally:
        if driver:
            driver.quit()

def extract_content(soup, selectors):
    """Extracts and cleans main article content."""
    content_block = None
    for selector in selectors.get('content', []):
        content_block = soup.select_one(selector)
        if content_block:
            logger.debug(f"Content block found with selector: {selector}")
            break

    if not content_block:
        logger.warning(f"⚠️ No primary content block found.")
        return None

    # Remove unwanted elements (ads, navigation, scripts, etc.)
    for element in content_block.select('''
        .ad-container, .read-more, script, style, iframe, 
        .hidden, .related-news, .social-share, .excerpt,
        .article-meta, .author-info, .comments-section,
        .newsletter-signup, .tags-container, .recommended,
        [class*="ad-"], [id*="ad-"], .inarticle-ad, .mobile-ad,
        .byline, .dateline, .caption, figcaption, figure.image-caption
    '''):
        element.decompose()

    # Extract text, combining various block elements
    paragraphs = []
    for element in content_block.find_all(['p', 'h2', 'h3', 'ul', 'ol', 'li']):
        text = ' '.join(element.get_text(separator=' ', strip=True).split())
        if text and len(text.split()) > 5: # Basic filter for short/empty lines
            # Filter out common boilerplate phrases
            if not any(phrase in text.lower() for phrase in [
                'read more', 'follow us on', 'subscribe to our', 'share this article',
                'comment here', 'also read', 'click here', 'watch video', 'by clicking here',
                'download our app', 'our whatsapp channel', 'latest news', 'e-paper',
                'disclaimer:', 'privacy policy', 'terms of use', 'copyright',
                'all rights reserved'
            ]):
                paragraphs.append(text)

    full_text = "\n\n".join(paragraphs)
    
    if not full_text or len(full_text.split()) < 50: # Ensure content is substantial
        logger.warning(f"⚠️ Extracted content too short (length: {len(full_text.split())} words).")
        return None
    return full_text

def extract_title(soup, selectors):
    """Extracts article title using multiple selectors and meta tags."""
    for selector in selectors.get('title', []):
        if selector.startswith('meta'):
            tag = soup.select_one(selector)
            if tag and tag.get('content'):
                return tag.get('content').strip()
        else:
            tag = soup.select_one(selector)
            if tag:
                return tag.get_text(strip=True)
    logger.warning("⚠️ No title found with provided selectors.")
    return None

def extract_summary(soup, selectors):
    """Extracts article summary using multiple selectors and meta tags."""
    for selector in selectors.get('summary', []):
        if selector.startswith('meta'):
            tag = soup.select_one(selector)
            if tag and tag.get('content'):
                return tag.get('content').strip()
        else:
            tag = soup.select_one(selector)
            if tag:
                return tag.get_text(strip=True)
    logger.warning("⚠️ No summary found with provided selectors.")
    return None

def extract_image_url(soup, article_url, selectors):
    """Extracts article image URL."""
    image_url = None
    # Try meta tags first
    for meta_selector in selectors.get('image', []):
        if meta_selector.startswith('meta'):
            meta_tag = soup.select_one(meta_selector)
            if meta_tag and meta_tag.get('content'):
                image_url = meta_tag.get('content')
                break
    
    # If not found in meta, try img tags
    if not image_url:
        for img_selector in selectors.get('image', []):
            if not img_selector.startswith('meta'): # Only look for actual img tags now
                img_tag = soup.select_one(img_selector)
                if img_tag:
                    image_url = img_tag.get('src') or img_tag.get('data-src') or img_tag.get('content')
                    if image_url:
                        break
    
    if image_url and not image_url.startswith('http'):
        image_url = urljoin(article_url, image_url) # Resolve relative URLs
    
    if not image_url:
        logger.warning("⚠️ No image URL found with provided selectors.")
    return image_url

def extract_published_date(soup, article_url):
    """
    Attempts to extract the publication date from common HTML patterns.
    This is highly site-specific and might need custom rules per domain.
    """
    # Common meta tags
    for meta_tag in soup.find_all('meta', attrs={'property': ['article:published_time', 'og:pubdate', 'date']}):
        date_str = meta_tag.get('content')
        if date_str:
            try:
                # Attempt to parse various ISO formats
                return datetime.fromisoformat(date_str.replace('Z', '+00:00')).isoformat()
            except ValueError:
                pass # Try next format/selector

    # Common time/span/div tags
    date_selectors = [
        'time[datetime]', 'span.date', '.article-date', '.publish-date', '.timestamp',
        'div.dateline', 'div.byline span', 'div.story-meta span.date'
    ]
    for selector in date_selectors:
        tag = soup.select_one(selector)
        if tag:
            date_str = tag.get('datetime') or tag.get_text(strip=True)
            if date_str:
                # Try common formats
                for fmt in ["%Y-%m-%d %H:%M:%S", "%B %d, %Y %I:%M%p", "%B %d, %Y", "%d %b %Y", "%Y/%m/%d"]:
                    try:
                        return datetime.strptime(date_str, fmt).isoformat()
                    except ValueError:
                        pass
    logger.warning(f"⚠️ Could not extract specific published date for {article_url}. Will use fallback.")
    return None

def fetch_full_article(article_url):
    """Fetches and parses the full content of an article."""
    domain = urlparse(article_url).netloc
    selectors = get_domain_specific_selectors(domain)
    
    soup = None
    page_source = get_rendered_content(article_url) # Try Selenium first
    if page_source:
        soup = BeautifulSoup(page_source, 'html.parser')
        logger.info(f"✅ Fetched with Selenium for {article_url}")
    else:
        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": f"https://{domain}/",
            "DNT": "1"
        }
        time.sleep(random.uniform(2, 5))
        try:
            response = requests.get(article_url, headers=headers, timeout=30) # Increased timeout
            response.raise_for_status()
            if 'text/html' not in response.headers.get('Content-Type', ''):
                logger.warning(f"⚠️ Non-HTML content for {article_url}")
                return None, None, None, None, None # Return None for all
            soup = BeautifulSoup(response.text, 'html.parser')
            logger.info(f"✅ Fetched with requests for {article_url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Requests failed for {article_url}: {e}")
            return None, None, None, None, None

    if not soup:
        logger.error(f"❌ Could not get soup object for {article_url}")
        return None, None, None, None, None

    # Extract all components
    extracted_title = extract_title(soup, selectors)
    extracted_summary = extract_summary(soup, selectors)
    extracted_content = extract_content(soup, selectors)
    extracted_image_url = extract_image_url(soup, article_url, selectors)
    extracted_published_date = extract_published_date(soup, article_url)

    return extracted_content, extracted_image_url, extracted_title, extracted_summary, extracted_published_date

def fetch_with_retry(url, max_retries=3):
    """Retries fetching an article with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fetch_full_article(url)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"❌ All retries failed for {url}: {e}")
                raise # Re-raise if all retries fail
            wait_time = (attempt + 1) * 5
            logger.warning(f"⚠️ Retry {attempt + 1}/{max_retries} for {url} due to error: {e}. Waiting {wait_time}s.")
            time.sleep(wait_time)
    return None, None, None, None, None # Should not be reached if exception is re-raised

def process_article(link, category_name, list_page_title, list_page_summary, source_name):
    """
    Processes a single article link: fetches full content, applies AI, and prepares data for saving.
    Prioritizes scraped data over AI generated data, then uses provided list page data, then fallbacks.
    """
    logger.info(f"Processing article: '{list_page_title[:50]}...' from {source_name}")
    
    full_content, scraped_image_url, scraped_title, scraped_summary, scraped_published_date = None, None, None, None, None
    try:
        full_content, scraped_image_url, scraped_title, scraped_summary, scraped_published_date = fetch_with_retry(link)
    except Exception as e:
        logger.error(f"❌ Failed to fetch full article data for {link}: {e}")
        # Continue with best available data, but log failure.
        pass 
                
    if not full_content or len(full_content.split()) < 100: # Minimum length for meaningful AI processing
        logger.warning(f"⚠️ Full content extraction failed or too short for {link}. Skipping AI summarization/tagging.")
        # Fallback content if extraction failed, to ensure article isn't entirely empty
        full_content = scraped_summary or list_page_summary or "No content available."

    # Determine final title
    final_title = scraped_title or list_page_title
    if full_content and len(full_content.split()) > 30 and (not final_title or "no title" in final_title.lower()):
        ai_title = generate_ai_title(full_content)
        if ai_title and len(ai_title) > 5:
            final_title = ai_title
    if not final_title: # Last resort
        final_title = "No title"

    # Determine final summary
    final_summary = scraped_summary or list_page_summary
    if full_content and len(full_content.split()) > 100: # Only summarize if content is substantial
        try:
            generated_summary = summarizer(full_content, max_length=350, min_length=150, do_sample=False)[0]["summary_text"]
            if generated_summary and len(generated_summary.split()) > 20:
                final_summary = generated_summary
        except Exception as e:
            logger.warning(f"⚠️ AI summary generation failed for {link}: {e}. Using best available.")
    if not final_summary: # Fallback if no summary available
        final_summary = (full_content[:250] + "...") if len(full_content) > 250 else full_content or "No summary available."

    # Generate tags
    tags = []
    if full_content and len(full_content.split()) > 50: # Only extract tags if content is reasonable
        try:
            tags = [k[0] for k in kw_model.extract_keywords(
                full_content, 
                keyphrase_ngram_range=(1, 2), 
                top_n=5,
                stop_words='english'
            )]
            tags = [tag for tag in tags if len(tag) > 2 and tag.lower() not in ['india', 'express', 'news', 'times', 'read']]
        except Exception as e:
            logger.warning(f"⚠️ Tag extraction failed for {link}: {e}")

    # Determine final image URL
    final_image_url = scraped_image_url or get_image_url(category_name)

    # Determine final published_at date
    final_published_at = None
    if scraped_published_date:
        try:
            final_published_at = datetime.fromisoformat(scraped_published_date)
        except ValueError:
            logger.warning(f"Could not parse scraped date '{scraped_published_date}' for {link}. Using current time.")
    
    if not final_published_at: # Fallback to a plausible recent date if scraping failed
        final_published_at = datetime.now() - timedelta(minutes=random.randint(60, 480)) # Last 1-8 hours

    return {
        "title": final_title,
        "original_title": list_page_title, # Keep original title for reference
        "url": link,
        "summary": final_summary,
        "content": full_content,
        "tags": tags,
        "category": category_name,
        "imageUrl": final_image_url,
        "source": source_name,
        "scraped_at": datetime.now(),
        "published_at": final_published_at, # This will always be a datetime object now
        "likes_count": 0, # Initialize these fields for new articles
        "comments_count": 0
    }

def save_article(article_data, category_collection):
    """Saves article data to both base and category-specific collections."""
    try:
        # Check for existing article by URL to avoid duplicates
        existing_article = base_collection.find_one({"url": article_data["url"]})

        # Set creation timestamp if new, otherwise update 'scraped_at'
        if existing_article:
            article_data["_id"] = existing_article["_id"] # Ensure we use existing ID for update
            update_result_base = base_collection.update_one(
                {"_id": existing_article["_id"]},
                {"$set": {
                    **article_data, 
                    "likes_count": existing_article.get("likes_count", 0), # Preserve existing counts
                    "comments_count": existing_article.get("comments_count", 0)
                }},
                upsert=False
            )
            update_result_category = category_collection.update_one(
                {"_id": existing_article["_id"]}, # Use _id to update in category collection as well
                {"$set": {
                    **article_data, 
                    "likes_count": existing_article.get("likes_count", 0),
                    "comments_count": existing_article.get("comments_count", 0)
                }},
                upsert=False
            )
            logger.info(f"🔄 Updated existing article: {article_data['title'][:50]}... ({update_result_base.modified_count} docs modified)")
        else:
            # Insert new article
            insert_result_base = base_collection.insert_one(article_data)
            article_data["_id"] = insert_result_base.inserted_id # Use the new _id for category collection
            category_collection.insert_one(article_data)
            logger.info(f"✅ Saved new article: {article_data['title'][:50]}... with ID {article_data['_id']}")
            
            # Limit collection size to 100 docs per category after insertion
            # This logic should ideally be applied if a new article was added
            if category_collection.count_documents({}) > 100:
                oldest = category_collection.find_one(sort=[("published_at", 1)])
                if oldest and oldest["_id"] != article_data["_id"]: # Don't delete the newly added article
                    category_collection.delete_one({"_id": oldest["_id"]})
                    base_collection.delete_one({"_id": oldest["_id"]}) # Also remove from base
                    logger.info(f"🗑️ Deleted oldest article from {category_collection.name} and base collection.")

    except Exception as e:
        logger.error(f"❌ Failed to save article {article_data.get('url', 'N/A')}: {e}")
        # Optionally re-raise if you want to stop the worker on save failures
        # raise

def extract_list_page_data(block, selectors):
    """
    Extracts link, title, and summary from an article block on a listing page.
    This is for initial article discovery, not full content.
    """
    link_tag = block.find('a', href=True)
    if not link_tag:
        return None, None, None

    link = link_tag['href']
    
    # Try to find a title within the block or from the link text itself
    title_tag = block.select_one('h2') or block.select_one('h3') or block.select_one('.title') or link_tag
    title = title_tag.get_text(strip=True) if title_tag else "No Title"

    # Try to find a summary/description within the block
    summary_tag = block.select_one('p.description') or block.select_one('.summary') or block.select_one('.excerpt')
    summary = summary_tag.get_text(strip=True) if summary_tag else ""

    return link, title, summary

def worker():
    """Worker thread that processes articles from the queue and saves them."""
    while not stop_event.is_set():
        try:
            item = article_queue.get(timeout=10) # Increased timeout for queue wait
            if item is None: # Sentinel value to signal stop
                break
                
            article_data, category_collection = item
            try:
                save_article(article_data, category_collection)
            except Exception as e:
                logger.error(f"Worker failed to save article: {e}")
            finally:
                article_queue.task_done()
        except queue.Empty:
            # If queue is empty and stop event is not set, keep trying
            continue
        except Exception as e:
            logger.error(f"Error in worker thread: {e}")
            break # Exit worker on unexpected error

def scrape_news_source_thread(source_name, section_url, category_name):
    """
    Scrapes a single news section (e.g., Indian Express - Technology).
    Puts article data into a queue for workers to process.
    """
    logger.info(f"Initiating scrape for {source_name} - {category_name}: {section_url}")
    try:
        domain = urlparse(section_url).netloc
        selectors = get_domain_specific_selectors(domain)
        
        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"https://{domain}/"
        }
        
        collection_name = f"articles_{category_name.lower().replace(' ', '_')}"
        category_collection = db[collection_name]

        try:
            time.sleep(random.uniform(3, 10))
            response = requests.get(section_url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            logger.error(f"❌ Failed to fetch {source_name} {category_name} page {section_url}: {e}")
            return

        article_blocks = []
        for selector in selectors.get('article_blocks', []):
            found_blocks = soup.select(selector)
            if found_blocks:
                article_blocks.extend(found_blocks)
                if len(article_blocks) >= 20: # Get enough blocks to find recent ones
                    break

        if not article_blocks:
            logger.warning(f"⚠️ No article blocks found for {source_name} - {category_name} on {section_url}")
            return

        logger.info(f"Found {len(article_blocks)} article candidates on list page for {source_name} - {category_name}")

        base_url = f"https://{domain}"
        
        for block in article_blocks[:15]: # Process more blocks to ensure enough unique articles
            try:
                link, title, summary = extract_list_page_data(block, selectors)
                if not link:
                    continue

                # Resolve relative link
                link = urljoin(base_url, link)

                # Skip if already scraped very recently (within 6 hours)
                # This check prevents re-scraping articles that are already in queue or just saved
                existing_article = base_collection.find_one({"url": link, "scraped_at": {"$gt": datetime.now() - timedelta(hours=6)}})
                if existing_article:
                    logger.debug(f"Article already scraped recently: {link}")
                    continue

                # Process the article and add to queue
                article_data = process_article(link, category_name, title, summary, source_name)
                if article_data:
                    article_queue.put((article_data, category_collection))
                    time.sleep(random.uniform(0.5, 2)) # Shorter delay between putting items in queue
            except Exception as e:
                logger.warning(f"⚠️ Error processing article block from list page for {source_name} {category_name}: {str(e)[:200]}")

    except Exception as e:
        logger.error(f"❌ Critical error in scrape_news_source_thread for {source_name} {category_name}: {e}")
        raise

def scrape_all_news_sources():
    """Main function to orchestrate the scraping process."""
    logger.info("🕒 Starting comprehensive scraping cycle...")
    stop_event.clear() # Clear stop event at the beginning of each cycle
    
    # Start worker threads
    worker_threads = []
    for _ in range(NUM_WORKERS):
        t = threading.Thread(target=worker, daemon=True) # Set as daemon so they exit with main program
        t.start()
        worker_threads.append(t)

    try:
        # Using ThreadPoolExecutor for concurrent scraping of news sections
        with ThreadPoolExecutor(max_workers=len(NEWS_SOURCES) * 2) as executor: # More threads for section scraping
            futures = []
            for source_name, categories in NEWS_SOURCES.items():
                logger.info(f"📰 Submitting tasks for {source_name}")
                for category, url in categories.items():
                    futures.append(
                        executor.submit(
                            scrape_news_source_thread,
                            source_name,
                            url,
                            category
                        )
                    )
                    time.sleep(0.5) # Small delay between submitting section scrapes
            
            # Wait for all section scraping tasks to complete
            for future in futures:
                try:
                    future.result() # This will re-raise exceptions from the threads
                except Exception as e:
                    logger.error(f"Section scraping failed for one source: {e}")
                    
    finally:
        logger.info("Scraping finished. Signalling workers to stop and waiting for queue to empty...")
        # Signal workers to stop by putting None for each worker
        for _ in range(NUM_WORKERS):
            article_queue.put(None)
        
        # Wait for all articles in the queue to be processed
        article_queue.join()
        stop_event.set() # Set stop event to ensure workers truly exit
        
        # Wait for worker threads to actually finish
        for t in worker_threads:
            t.join(timeout=10) # Give workers a bit more time to finish
            if t.is_alive():
                logger.warning(f"Worker thread {t.name} did not terminate gracefully.")

        logger.info("✅ Scraping cycle completed and workers stopped.")


# FastAPI endpoints
@app.get("/")
async def root():
    return {"message": "NovaPress Scraper API is running"}

@app.get("/articles", response_model=ArticleListResponse)
async def get_articles(limit: int = 100):
    """Fetches a list of all articles, sorted by publication date."""
    try:
        articles = list(base_collection.find().sort("published_at", -1).limit(limit))
        # Filter out None values that might result from transform_article if _id is missing
        transformed_articles = [transform_article(article) for article in articles if transform_article(article) is not None]
        return {"articles": transformed_articles}
    except Exception as e:
        logger.error(f"Error fetching all articles: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error fetching articles")

@app.get("/articles/category/{category_name}", response_model=ArticleListResponse)
async def get_articles_by_category(category_name: str, limit: int = 100):
    """Fetches articles for a specific category, sorted by publication date."""
    try:
        collection_name = f"articles_{category_name.lower().replace(' ', '_')}"
        category_collection = db[collection_name]
        articles = list(
            category_collection.find().sort("published_at", -1).limit(limit)
        )
        transformed_articles = [transform_article(article) for article in articles if transform_article(article) is not None]
        return {"articles": transformed_articles}
    except Exception as e:
        logger.error(f"Error fetching {category_name} articles: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error fetching {category_name} articles")

@app.get("/articles/id/{article_id}", response_model=Article)
async def get_article(article_id: str):
    """Fetches a single article by its MongoDB ObjectId."""
    try:
        if not ObjectId.is_valid(article_id):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid article ID format")

        obj_id = ObjectId(article_id)
        article = None
        
        # Search in base articles collection first (most efficient)
        article = base_collection.find_one({"_id": obj_id})
        
        # If not found, search in all category collections (fallback for older or missed articles)
        if not article:
            category_collections_names = [
                f"articles_{cat.lower().replace(' ', '_')}" for source_dict in NEWS_SOURCES.values() for cat in source_dict.keys()
            ]
            # Add 'articles' collection if not already included (it should be base_collection)
            if 'articles' not in category_collections_names:
                category_collections_names.append('articles') # Ensure the main collection is searched

            for collection_name in set(category_collections_names): # Use set to avoid duplicates
                try:
                    current_collection = db[collection_name]
                    article = current_collection.find_one({"_id": obj_id})
                    if article:
                        logger.info(f"✅ Found article in {collection_name}")
                        break
                except Exception as e:
                    logger.debug(f"Collection {collection_name} not accessible or error during find: {e}")
                    continue

        if not article:
            logger.warning(f"❌ Article {article_id} not found in any collection")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

        transformed_article = transform_article(article)
        if transformed_article is None:
            logger.error(f"Failed to transform article {article_id} after fetching from DB. Data integrity issue.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Article data malformed in database.")
            
        return transformed_article
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching article {article_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected server error occurred")

# Debug endpoints
@app.get("/debug/collections")
async def list_collections_and_counts():
    """Lists all MongoDB collections and their document counts, focusing on article collections."""
    try:
        collections = db.list_collection_names()
        result = {}
        for collection_name in collections:
            # Only include collections related to articles
            if collection_name.startswith('articles') or collection_name in ['likes', 'comments']:
                count = db[collection_name].count_documents({})
                result[collection_name] = count
        return result
    except Exception as e:
        logger.error(f"Error in /debug/collections: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/debug/recent-articles")
async def get_recent_articles_debug():
    """Fetches the 5 most recent articles from each known article collection for debugging."""
    result = {}
    
    # Dynamically build list of category collections
    category_collections_names = [
        f"articles_{cat.lower().replace(' ', '_')}" for source_dict in NEWS_SOURCES.values() for cat in source_dict.keys()
    ]
    # Add the main 'articles' collection if it's not already covered
    if 'articles' not in category_collections_names:
        category_collections_names.append('articles') 
    
    for collection_name in set(category_collections_names): # Use set to avoid duplicates
        try:
            collection = db[collection_name]
            articles = list(collection.find({}, {"_id": 1, "title": 1, "category": 1, "url": 1}).sort("published_at", -1).limit(5))
            result[collection_name] = []
            for article in articles:
                result[collection_name].append({
                    "id": str(article["_id"]),
                    "title": article.get("title", "NO TITLE FOUND"), # More explicit debug message
                    "category": article.get("category", "UNKNOWN CATEGORY"),
                    "url": article.get("url", "NO URL")
                })
        except Exception as e:
            result[f"error_{collection_name}"] = str(e)
    
    return result

# This block ensures the scraper runs in a separate thread when the API starts
if __name__ == "__main__":
    logger.info("Starting NovaPress Scraper API...")
    # Start the scraper in a separate thread
    scraper_thread = threading.Thread(target=scrape_all_news_sources, daemon=True)
    scraper_thread.start()
    
    # Start the FastAPI app (blocking call)
    uvicorn.run(app, host="0.0.0.0", port=8000)