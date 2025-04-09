#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Any, Union, List, Tuple
import sys
import os
import time
import requests
import re
import numpy as np
from datetime import datetime, timedelta
import anthropic
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys
import random
import statistics
import threading
import queue
import json
from llm_provider import LLMProvider

from utils.logger import logger
from utils.browser import browser
from config import config
from coingecko_handler import CoinGeckoHandler
from mood_config import MoodIndicators, determine_advanced_mood, Mood, MemePhraseGenerator
from meme_phrases import MEME_PHRASES
from prediction_engine import PredictionEngine

# Import new modules for reply functionality
from timeline_scraper import TimelineScraper
from reply_handler import ReplyHandler
from content_analyzer import ContentAnalyzer

class CryptoAnalysisBot:
    def __init__(self) -> None:
       self.browser = browser
       self.config = config
       self.llm_provider = LLMProvider(self.config)  
       self.past_predictions = []
       self.meme_phrases = MEME_PHRASES
       self.last_check_time = datetime.now()
       self.last_market_data = {}
       
       # Multi-timeframe prediction tracking
       self.timeframes = ["1h", "24h", "7d"]
       self.timeframe_predictions = {tf: {} for tf in self.timeframes}
       self.timeframe_last_post = {tf: datetime.now() - timedelta(hours=3) for tf in self.timeframes}
       
       # Timeframe posting frequency controls (in hours)
       self.timeframe_posting_frequency = {
           "1h": 1,    # Every hour
           "24h": 6,   # Every 6 hours
           "7d": 24    # Once per day
       }
       
       # Prediction accuracy tracking by timeframe
       self.prediction_accuracy = {tf: {'correct': 0, 'total': 0} for tf in self.timeframes}
       
       # Initialize prediction engine with database and Claude API key
       self.prediction_engine = PredictionEngine(
           database=self.config.db,
           llm_provider=self.llm_provider  # Pass provider instead of API key
       )
       
       # Create a queue for predictions to process
       self.prediction_queue = queue.Queue()
       
       # Initialize thread for async prediction generation
       self.prediction_thread = None
       self.prediction_thread_running = False
       
       # Initialize CoinGecko handler with 60s cache duration
       self.coingecko = CoinGeckoHandler(
           base_url=self.config.COINGECKO_BASE_URL,
           cache_duration=60
       )

       # Target chains to analyze
       self.target_chains = {
           'BTC': 'bitcoin',
           'ETH': 'ethereum',
           'SOL': 'solana',
           'XRP': 'ripple',
           'BNB': 'binancecoin',
           'AVAX': 'avalanche-2',
           'DOT': 'polkadot',
           'UNI': 'uniswap',
           'NEAR': 'near',
           'AAVE': 'aave',
           'FIL': 'filecoin',
           'POL': 'matic-network',
           'KAITO': 'kaito'  # Kept in the list but not given special treatment
       }

       # All tokens for reference and comparison
       self.reference_tokens = list(self.target_chains.keys())
       
       # Chain name mapping for display
       self.chain_name_mapping = self.target_chains.copy()
       
       self.CORRELATION_THRESHOLD = 0.75  
       self.VOLUME_THRESHOLD = 0.60  
       self.TIME_WINDOW = 24
       
       # Smart money thresholds
       self.SMART_MONEY_VOLUME_THRESHOLD = 1.5  # 50% above average
       self.SMART_MONEY_ZSCORE_THRESHOLD = 2.0  # 2 standard deviations
       
       # Timeframe-specific triggers and thresholds
       self.timeframe_thresholds = {
           "1h": {
               "price_change": 3.0,    # 3% price change for 1h predictions
               "volume_change": 8.0,   # 8% volume change
               "confidence": 70,       # Minimum confidence percentage
               "fomo_factor": 1.0      # FOMO enhancement factor
           },
           "24h": {
               "price_change": 5.0,    # 5% price change for 24h predictions
               "volume_change": 12.0,  # 12% volume change
               "confidence": 65,       # Slightly lower confidence for longer timeframe
               "fomo_factor": 1.2      # Higher FOMO factor
           },
           "7d": {
               "price_change": 8.0,    # 8% price change for 7d predictions
               "volume_change": 15.0,  # 15% volume change
               "confidence": 60,       # Even lower confidence for weekly predictions
               "fomo_factor": 1.5      # Highest FOMO factor
           }
       }
       
       # Initialize scheduled timeframe posts
       self.next_scheduled_posts = {
           "1h": datetime.now() + timedelta(minutes=random.randint(10, 30)),
           "24h": datetime.now() + timedelta(hours=random.randint(1, 3)),
           "7d": datetime.now() + timedelta(hours=random.randint(4, 8))
       }
       
       # Initialize reply functionality components
       self.timeline_scraper = TimelineScraper(self.browser, self.config, self.config.db)
       self.reply_handler = ReplyHandler(self.browser, self.config, self.llm_provider.client, self.coingecko, self.config.db)
       self.content_analyzer = ContentAnalyzer(self.config, self.config.db)
       
       # Reply tracking and control
       self.last_reply_check = datetime.now() - timedelta(minutes=30)  # Start checking soon
       self.reply_check_interval = 5  # Check for posts to reply to every 60 minutes
       self.max_replies_per_cycle = 10  # Maximum 10 replies per cycle
       self.reply_cooldown = 5  # Minutes between reply cycles
       self.last_reply_time = datetime.now() - timedelta(minutes=self.reply_cooldown)  # Allow immediate first run
       
       logger.log_startup()

    def _get_historical_volume_data(self, chain: str, minutes: int = None, timeframe: str = "1h") -> List[Dict[str, Any]]:
       """
       Get historical volume data for the specified window period
       Adjusted based on timeframe for appropriate historical context
       """
       try:
           # Adjust window size based on timeframe if not specifically provided
           if minutes is None:
               if timeframe == "1h":
                   minutes = self.config.VOLUME_WINDOW_MINUTES  # Default (typically 60)
               elif timeframe == "24h":
                   minutes = 24 * 60  # Last 24 hours
               elif timeframe == "7d":
                   minutes = 7 * 24 * 60  # Last 7 days
               else:
                   minutes = self.config.VOLUME_WINDOW_MINUTES
               
           window_start = datetime.now() - timedelta(minutes=minutes)
           query = """
               SELECT timestamp, volume
               FROM market_data
               WHERE chain = ? AND timestamp >= ?
               ORDER BY timestamp DESC
           """
           
           conn = self.config.db.conn
           cursor = conn.cursor()
           cursor.execute(query, (chain, window_start))
           results = cursor.fetchall()
           
           volume_data = [
               {
                   'timestamp': datetime.fromisoformat(row[0]),
                   'volume': float(row[1])
               }
               for row in results
           ]
           
           logger.logger.debug(
               f"Retrieved {len(volume_data)} volume data points for {chain} "
               f"over last {minutes} minutes (timeframe: {timeframe})"
           )
           
           return volume_data
           
       except Exception as e:
           logger.log_error(f"Historical Volume Data - {chain} ({timeframe})", str(e))
           return []
       
    def _is_duplicate_analysis(self, new_tweet: str, last_posts: List[str], timeframe: str = "1h") -> bool:
       """
       Enhanced duplicate detection with time-based thresholds and timeframe awareness.
       Applies different checks based on how recently similar content was posted:
       - Very recent posts (< 15 min): Check for exact matches
       - Recent posts (15-30 min): Check for high similarity
       - Older posts (> 30 min): Allow similar content
       """
       try:
           # Log that we're using enhanced duplicate detection
           logger.logger.info(f"Using enhanced time-based duplicate detection for {timeframe} timeframe")
           
           # Define time windows for different levels of duplicate checking
           # Adjust windows based on timeframe
           if timeframe == "1h":
               VERY_RECENT_WINDOW_MINUTES = 15
               RECENT_WINDOW_MINUTES = 30
               HIGH_SIMILARITY_THRESHOLD = 0.85  # 85% similar for recent posts
           elif timeframe == "24h":
               VERY_RECENT_WINDOW_MINUTES = 120  # 2 hours
               RECENT_WINDOW_MINUTES = 240       # 4 hours
               HIGH_SIMILARITY_THRESHOLD = 0.80  # Slightly lower threshold for daily predictions
           else:  # 7d
               VERY_RECENT_WINDOW_MINUTES = 720  # 12 hours
               RECENT_WINDOW_MINUTES = 1440      # 24 hours
               HIGH_SIMILARITY_THRESHOLD = 0.75  # Even lower threshold for weekly predictions
           
           # 1. Check for exact matches in very recent database entries
           conn = self.config.db.conn
           cursor = conn.cursor()
           
           # Very recent exact duplicates check
           cursor.execute("""
               SELECT content FROM posted_content 
               WHERE timestamp >= datetime('now', '-' || ? || ' minutes')
               AND timeframe = ?
           """, (VERY_RECENT_WINDOW_MINUTES, timeframe))
           
           very_recent_posts = [row[0] for row in cursor.fetchall()]
           
           # Check for exact matches in very recent posts
           for post in very_recent_posts:
               if post.strip() == new_tweet.strip():
                   logger.logger.info(f"Exact duplicate detected within last {VERY_RECENT_WINDOW_MINUTES} minutes for {timeframe}")
                   return True
           
           # 2. Check for high similarity in recent posts
           cursor.execute("""
               SELECT content FROM posted_content 
               WHERE timestamp >= datetime('now', '-' || ? || ' minutes')
               AND timestamp < datetime('now', '-' || ? || ' minutes')
               AND timeframe = ?
           """, (RECENT_WINDOW_MINUTES, VERY_RECENT_WINDOW_MINUTES, timeframe))
           
           recent_posts = [row[0] for row in cursor.fetchall()]
           
           # Calculate similarity for recent posts
           new_content = new_tweet.lower()
           
           for post in recent_posts:
               post_content = post.lower()
               
               # Calculate a simple similarity score based on word overlap
               new_words = set(new_content.split())
               post_words = set(post_content.split())
               
               if new_words and post_words:
                   overlap = len(new_words.intersection(post_words))
                   similarity = overlap / max(len(new_words), len(post_words))
                   
                   # Apply high similarity threshold for recent posts
                   if similarity > HIGH_SIMILARITY_THRESHOLD:
                       logger.logger.info(f"High similarity ({similarity:.2f}) detected within last {RECENT_WINDOW_MINUTES} minutes for {timeframe}")
                       return True
           
           # 3. Also check exact duplicates in last posts from Twitter
           # This prevents double-posting in case of database issues
           for post in last_posts:
               if post.strip() == new_tweet.strip():
                   logger.logger.info(f"Exact duplicate detected in recent Twitter posts for {timeframe}")
                   return True
           
           # If we get here, it's not a duplicate according to our criteria
           logger.logger.info(f"No duplicates detected with enhanced time-based criteria for {timeframe}")
           return False
           
       except Exception as e:
           logger.log_error(f"Duplicate Check - {timeframe}", str(e))
           # If the duplicate check fails, allow the post to be safe
           logger.logger.warning("Duplicate check failed, allowing post to proceed")
           return False

    def _start_prediction_thread(self) -> None:
       """Start background thread for asynchronous prediction generation"""
       if self.prediction_thread is None or not self.prediction_thread.is_alive():
           self.prediction_thread_running = True
           self.prediction_thread = threading.Thread(target=self._process_prediction_queue)
           self.prediction_thread.daemon = True
           self.prediction_thread.start()
           logger.logger.info("Started prediction processing thread")
           
    def _process_prediction_queue(self) -> None:
       """Process predictions from the queue in the background"""
       while self.prediction_thread_running:
           try:
               # Get a prediction task from the queue with timeout
               try:
                   task = self.prediction_queue.get(timeout=10)
               except queue.Empty:
                   # No tasks, just continue the loop
                   continue
                   
               # Process the prediction task
               token, timeframe, market_data = task
               
               logger.logger.debug(f"Processing queued prediction for {token} ({timeframe})")
               
               # Generate the prediction
               prediction = self.prediction_engine.generate_prediction(
                   token=token, 
                   market_data=market_data,
                   timeframe=timeframe
               )
               
               # Store in memory for quick access
               self.timeframe_predictions[timeframe][token] = prediction
               
               # Mark task as done
               self.prediction_queue.task_done()
               
               # Short sleep to prevent CPU overuse
               time.sleep(0.5)
               
           except Exception as e:
               logger.log_error("Prediction Thread Error", str(e))
               time.sleep(5)  # Sleep longer on error
               
       logger.logger.info("Prediction processing thread stopped")

    def _login_to_twitter(self) -> bool:
       """Log into Twitter with enhanced verification"""
       try:
           logger.logger.info("Starting Twitter login")
           self.browser.driver.set_page_load_timeout(45)
           self.browser.driver.get('https://twitter.com/login')
           time.sleep(5)

           username_field = WebDriverWait(self.browser.driver, 20).until(
               EC.element_to_be_clickable((By.CSS_SELECTOR, "input[autocomplete='username']"))
           )
           username_field.click()
           time.sleep(1)
           username_field.send_keys(self.config.TWITTER_USERNAME)
           time.sleep(2)

           next_button = WebDriverWait(self.browser.driver, 10).until(
               EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']"))
           )
           next_button.click()
           time.sleep(3)

           password_field = WebDriverWait(self.browser.driver, 20).until(
               EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='password']"))
           )
           password_field.click()
           time.sleep(1)
           password_field.send_keys(self.config.TWITTER_PASSWORD)
           time.sleep(2)

           login_button = WebDriverWait(self.browser.driver, 10).until(
               EC.element_to_be_clickable((By.XPATH, "//span[text()='Log in']"))
           )
           login_button.click()
           time.sleep(10) 

           return self._verify_login()

       except Exception as e:
           logger.log_error("Twitter Login", str(e))
           return False

    def _verify_login(self) -> bool:
       """Verify Twitter login success"""
       try:
           verification_methods = [
               lambda: WebDriverWait(self.browser.driver, 30).until(
                   EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="SideNav_NewTweet_Button"]'))
               ),
               lambda: WebDriverWait(self.browser.driver, 30).until(
                   EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="AppTabBar_Profile_Link"]'))
               ),
               lambda: any(path in self.browser.driver.current_url 
                         for path in ['home', 'twitter.com/home'])
           ]
           
           for method in verification_methods:
               try:
                   if method():
                       return True
               except:
                   continue
           
           return False
           
       except Exception as e:
           logger.log_error("Login Verification", str(e))
           return False

    def _queue_predictions_for_all_timeframes(self, token: str, market_data: Dict[str, Any]) -> None:
       """Queue predictions for all timeframes for a specific token"""
       for timeframe in self.timeframes:
           # Skip if we already have a recent prediction
           if (token in self.timeframe_predictions.get(timeframe, {}) and 
               datetime.now() - self.timeframe_predictions[timeframe].get(token, {}).get('timestamp', 
                                                                                        datetime.now() - timedelta(hours=3)) 
               < timedelta(hours=1)):
               logger.logger.debug(f"Skipping {timeframe} prediction for {token} - already have recent prediction")
               continue
               
           # Add prediction task to queue
           self.prediction_queue.put((token, timeframe, market_data))
           logger.logger.debug(f"Queued {timeframe} prediction for {token}")

    def _post_analysis(self, tweet_text: str, timeframe: str = "1h") -> bool:
       """
       Post analysis to Twitter with robust button handling
       Tracks post by timeframe
       """
       max_retries = 3
       retry_count = 0
       
       while retry_count < max_retries:
           try:
               self.browser.driver.get('https://twitter.com/compose/tweet')
               time.sleep(3)
               
               text_area = WebDriverWait(self.browser.driver, 10).until(
                   EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetTextarea_0"]'))
               )
               text_area.click()
               time.sleep(1)
               
               # Ensure tweet text only contains BMP characters
               safe_tweet_text = ''.join(char for char in tweet_text if ord(char) < 0x10000)
               
               # Simply send the tweet text directly - no handling of hashtags needed
               text_area.send_keys(safe_tweet_text)
               time.sleep(2)

               post_button = None
               button_locators = [
                   (By.CSS_SELECTOR, '[data-testid="tweetButton"]'),
                   (By.XPATH, "//div[@role='button'][contains(., 'Post')]"),
                   (By.XPATH, "//span[text()='Post']")
               ]

               for locator in button_locators:
                   try:
                       post_button = WebDriverWait(self.browser.driver, 5).until(
                           EC.element_to_be_clickable(locator)
                       )
                       if post_button:
                           break
                   except:
                       continue

               if post_button:
                   self.browser.driver.execute_script("arguments[0].scrollIntoView(true);", post_button)
                   time.sleep(1)
                   self.browser.driver.execute_script("arguments[0].click();", post_button)
                   time.sleep(5)
                   
                   # Update last post time for this timeframe
                   self.timeframe_last_post[timeframe] = datetime.now()
                   
                   # Update next scheduled post time
                   hours_to_add = self.timeframe_posting_frequency.get(timeframe, 1)
                   # Add some randomness to prevent predictable patterns
                   jitter = random.uniform(0.8, 1.2)
                   self.next_scheduled_posts[timeframe] = datetime.now() + timedelta(hours=hours_to_add * jitter)
                   
                   logger.logger.info(f"{timeframe} tweet posted successfully")
                   logger.logger.debug(f"Next {timeframe} post scheduled for {self.next_scheduled_posts[timeframe]}")
                   return True
               else:
                   logger.logger.error(f"Could not find post button for {timeframe} tweet")
                   retry_count += 1
                   time.sleep(2)
                   
           except Exception as e:
               logger.logger.error(f"{timeframe} tweet posting error, attempt {retry_count + 1}: {str(e)}")
               retry_count += 1
               wait_time = retry_count * 10
               logger.logger.warning(f"Waiting {wait_time}s before retry...")
               time.sleep(wait_time)
               continue
       
       logger.log_error(f"Tweet Creation - {timeframe}", "Maximum retries reached")
       return False
   
    def _get_last_posts(self, count: int = 10) -> List[Dict[str, Any]]:
       """
       Get last N posts from timeline with timeframe detection
       Returns list of post information including detected timeframe
       """
       try:
           self.browser.driver.get(f'https://twitter.com/{self.config.TWITTER_USERNAME}')
           time.sleep(3)
           
           posts = WebDriverWait(self.browser.driver, 10).until(
               EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-testid="tweetText"]'))
           )
           
           timestamps = self.browser.driver.find_elements(By.CSS_SELECTOR, 'time')
           
           # Get only the first count posts
           posts = posts[:count]
           timestamps = timestamps[:count]
           
           result = []
           for i in range(min(len(posts), len(timestamps))):
               post_text = posts[i].text
               timestamp_str = timestamps[i].get_attribute('datetime') if timestamps[i].get_attribute('datetime') else None
               
               # Detect timeframe from post content
               detected_timeframe = "1h"  # Default
               
               # Look for timeframe indicators in the post
               if "7D PREDICTION" in post_text.upper() or "7-DAY" in post_text.upper() or "WEEKLY" in post_text.upper():
                   detected_timeframe = "7d"
               elif "24H PREDICTION" in post_text.upper() or "24-HOUR" in post_text.upper() or "DAILY" in post_text.upper():
                   detected_timeframe = "24h"
               elif "1H PREDICTION" in post_text.upper() or "1-HOUR" in post_text.upper() or "HOURLY" in post_text.upper():
                   detected_timeframe = "1h"
               
               post_info = {
                   'text': post_text,
                   'timestamp': datetime.fromisoformat(timestamp_str) if timestamp_str else None,
                   'timeframe': detected_timeframe
               }
               
               result.append(post_info)
           
           return result
       except Exception as e:
           logger.log_error("Get Last Posts", str(e))
           return []

    def _get_last_posts_by_timeframe(self, timeframe: str = "1h", count: int = 5) -> List[str]:
       """
       Get last N posts for a specific timeframe
       Returns just the text content
       """
       all_posts = self._get_last_posts(count=20)  # Get more posts to filter from
       
       # Filter posts by the requested timeframe
       filtered_posts = [post['text'] for post in all_posts if post['timeframe'] == timeframe]
       
       # Return the requested number of posts
       return filtered_posts[:count]
   
    def _schedule_timeframe_post(self, timeframe: str, delay_hours: float = None) -> None:
       """
       Schedule the next post for a specific timeframe
       """
       if delay_hours is None:
           # Use default frequency with some randomness
           base_hours = self.timeframe_posting_frequency.get(timeframe, 1)
           delay_hours = base_hours * random.uniform(0.9, 1.1)
           
       self.next_scheduled_posts[timeframe] = datetime.now() + timedelta(hours=delay_hours)
       logger.logger.debug(f"Scheduled next {timeframe} post for {self.next_scheduled_posts[timeframe]}")
   
    def _should_post_timeframe_now(self, timeframe: str) -> bool:
       """
       Check if it's time to post for a specific timeframe
       """
       # Check if enough time has passed since last post
       min_interval = timedelta(hours=self.timeframe_posting_frequency.get(timeframe, 1) * 0.8)
       if datetime.now() - self.timeframe_last_post.get(timeframe, datetime.min) < min_interval:
           return False
           
       # Check if scheduled time has been reached
       return datetime.now() >= self.next_scheduled_posts.get(timeframe, datetime.now())
   
    def _post_prediction_for_timeframe(self, token: str, market_data: Dict[str, Any], timeframe: str) -> bool:
       """
       Post a prediction for a specific timeframe
       """
       try:
           # Check if we have a prediction
           prediction = self.timeframe_predictions.get(timeframe, {}).get(token)
           
           # If no prediction exists, generate one
           if not prediction:
               prediction = self.prediction_engine.generate_prediction(
                   token=token,
                   market_data=market_data,
                   timeframe=timeframe
               )
               
               # Store for future use
               if timeframe not in self.timeframe_predictions:
                   self.timeframe_predictions[timeframe] = {}
               self.timeframe_predictions[timeframe][token] = prediction
           
           # Format the prediction for posting
           tweet_text = self._format_prediction_tweet(token, prediction, market_data, timeframe)
           
           # Check for duplicates
           last_posts = self._get_last_posts_by_timeframe(timeframe=timeframe)
           if self._is_duplicate_analysis(tweet_text, last_posts, timeframe):
               logger.logger.warning(f"Skipping duplicate {timeframe} prediction for {token}")
               return False
               
           # Post the prediction
           if self._post_analysis(tweet_text, timeframe):
               # Store in database
               sentiment = prediction.get("sentiment", "NEUTRAL")
               price_data = {token: {'price': market_data[token]['current_price'], 
                                    'volume': market_data[token]['volume']}}
               
               # Create storage data
               storage_data = {
                   'content': tweet_text,
                   'sentiment': {token: sentiment},
                   'trigger_type': f"scheduled_{timeframe}_post",
                   'price_data': price_data,
                   'meme_phrases': {token: ""},  # No meme phrases for predictions
                   'is_prediction': True,
                   'prediction_data': prediction,
                   'timeframe': timeframe
               }
               
               # Store in database
               self.config.db.store_posted_content(**storage_data)
               
               logger.logger.info(f"Successfully posted {timeframe} prediction for {token}")
               return True
           else:
               logger.logger.error(f"Failed to post {timeframe} prediction for {token}")
               return False
               
       except Exception as e:
           logger.log_error(f"Post Prediction For Timeframe - {token} ({timeframe})", str(e))
           return False
   
    def _post_timeframe_rotation(self, market_data: Dict[str, Any]) -> bool:
       """
       Post predictions in a rotation across timeframes
       Returns True if any prediction was posted
       """
       # First check if any timeframe is due for posting
       due_timeframes = [tf for tf in self.timeframes if self._should_post_timeframe_now(tf)]
       
       if not due_timeframes:
           logger.logger.debug("No timeframes due for posting")
           return False
           
       # Pick the most overdue timeframe
       chosen_timeframe = max(due_timeframes, 
                             key=lambda tf: (datetime.now() - self._ensure_datetime(self.next_scheduled_posts.get(tf, datetime.min))).total_seconds())
       
       logger.logger.info(f"Selected {chosen_timeframe} for timeframe rotation posting")

       # Choose best token for this timeframe
       token_to_post = self._select_best_token_for_timeframe(market_data, chosen_timeframe)
       
       if not token_to_post:
           logger.logger.warning(f"No suitable token found for {chosen_timeframe} timeframe")
           # Reschedule this timeframe for later
           self._schedule_timeframe_post(chosen_timeframe, delay_hours=1)
           return False
           
       # Post the prediction
       success = self._post_prediction_for_timeframe(token_to_post, market_data, chosen_timeframe)
       
       # If post failed, reschedule for later
       if not success:
           self._schedule_timeframe_post(chosen_timeframe, delay_hours=1)
           
       return success

    def _select_best_token_for_timeframe(self, market_data: Dict[str, Any], timeframe: str) -> Optional[str]:
        """
        Select the best token to use for a specific timeframe post
        Uses momentum scoring, prediction accuracy, and market activity
        """
        candidates = []
        
        # Get tokens with data
        available_tokens = [t for t in self.reference_tokens if t in market_data]
        
        # Score each token
        for token in available_tokens:
            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(token, market_data)
            
            # Calculate activity score based on recent volume and price changes
            token_data = market_data.get(token, {})
            volume = token_data.get('volume', 0)
            price_change = abs(token_data.get('price_change_percentage_24h', 0))
            
            # Get volume trend
            volume_trend, _ = self._analyze_volume_trend(volume, 
                                                     self._get_historical_volume_data(token, timeframe=timeframe),
                                                     timeframe=timeframe)
            
            # Get historical prediction accuracy
            perf_stats = self.config.db.get_prediction_performance(token=token, timeframe=timeframe)
            
            # Calculate accuracy score
            accuracy_score = 0
            if perf_stats:
                accuracy = perf_stats[0].get('accuracy_rate', 0)
                total_preds = perf_stats[0].get('total_predictions', 0)
                
                # Only consider accuracy if we have enough data
                if total_preds >= 5:
                    accuracy_score = accuracy * (min(total_preds, 20) / 20)  # Scale by number of predictions up to 20
            
            # Calculate recency score - prefer tokens we haven't posted about recently
            recency_score = 0
            
            # Check when this token was last posted for this timeframe
            recent_posts = self.config.db.get_recent_posts(hours=48, timeframe=timeframe)
            
            token_posts = [p for p in recent_posts if token.upper() in p.get('content', '')]
            
            if not token_posts:
                # Never posted - maximum recency score
                recency_score = 100
            else:
                # Calculate hours since last post
                last_post_time = max(p.get('timestamp', datetime.min) for p in token_posts)
                hours_since = (datetime.now() - last_post_time).total_seconds() / 3600
                
                # Scale recency score based on timeframe
                if timeframe == "1h":
                    recency_score = min(100, hours_since * 10)  # Max score after 10 hours
                elif timeframe == "24h":
                    recency_score = min(100, hours_since * 2)   # Max score after 50 hours
                else:  # 7d
                    recency_score = min(100, hours_since * 0.5)  # Max score after 200 hours
            
            # Combine scores with timeframe-specific weightings
            if timeframe == "1h":
                # For hourly, momentum and price action matter most
                total_score = (
                    momentum_score * 0.5 +
                    price_change * 3.0 +
                    volume_trend * 0.7 +
                    accuracy_score * 0.3 +
                    recency_score * 0.4
                )
            elif timeframe == "24h":
                # For daily, balance between momentum, accuracy and recency
                total_score = (
                    momentum_score * 0.4 +
                    price_change * 2.0 +
                    volume_trend * 0.8 +
                    accuracy_score * 0.5 +
                    recency_score * 0.6
                )
            else:  # 7d
                # For weekly, accuracy and longer-term views matter more
                total_score = (
                    momentum_score * 0.3 +
                    price_change * 1.0 +
                    volume_trend * 1.0 +
                    accuracy_score * 0.8 +
                    recency_score * 0.8
                )
            
            candidates.append((token, total_score))
        
        # Sort by total score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        logger.logger.debug(f"Token candidates for {timeframe}: {candidates[:3]}")
        
        return candidates[0][0] if candidates else None

    # New method for handling replies
    def _check_for_posts_to_reply(self, market_data: Dict[str, Any]) -> bool:
        """
        Check for posts to reply to and generate replies
        Returns True if any replies were posted
        """
        now = datetime.now()
    
        # Check if it's time to look for posts to reply to
        time_since_last_check = (now - self.last_reply_check).total_seconds() / 60
        if time_since_last_check < self.reply_check_interval:
            logger.logger.debug(f"Skipping reply check, {time_since_last_check:.1f} minutes since last check (interval: {self.reply_check_interval})")
            return False
        
        # Also check cooldown period
        time_since_last_reply = (now - self.last_reply_time).total_seconds() / 60
        if time_since_last_reply < self.reply_cooldown:
            logger.logger.debug(f"In reply cooldown period, {time_since_last_reply:.1f} minutes since last reply (cooldown: {self.reply_cooldown})")
            return False
        
        logger.logger.info("Starting check for posts to reply to")
        self.last_reply_check = now
    
        try:
            # Scrape timeline for posts
            posts = self.timeline_scraper.scrape_timeline(count=self.max_replies_per_cycle * 2)  # Get more to filter
            logger.logger.info(f"Timeline scraping completed - found {len(posts) if posts else 0} posts")
        
            if not posts:
                logger.logger.warning("No posts found during timeline scraping")
                return False

            # Log sample posts for debugging
            for i, post in enumerate(posts[:3]):  # Log first 3 posts
                logger.logger.info(f"Sample post {i}: {post.get('content', '')[:100]}...")

            # Find market-related posts
            logger.logger.info(f"Finding market-related posts among {len(posts)} scraped posts")
            market_posts = self.content_analyzer.find_market_related_posts(posts)
            logger.logger.info(f"Found {len(market_posts)} market-related posts, checking which ones need replies")
            
            # Filter out posts we've already replied to
            unreplied_posts = self.timeline_scraper.filter_already_replied_posts(market_posts)
            logger.logger.info(f"Found {len(unreplied_posts)} unreplied market-related posts")
            if unreplied_posts:
                for i, post in enumerate(unreplied_posts[:3]):
                    logger.logger.info(f"Sample unreplied post {i}: {post.get('content', '')[:100]}...")
            
            if not unreplied_posts:
                return False
                
            # Prioritize posts (engagement, relevance, etc.)
            prioritized_posts = self.timeline_scraper.prioritize_posts(unreplied_posts)
            
            # Limit to max replies per cycle
            posts_to_reply = prioritized_posts[:self.max_replies_per_cycle]
            
            # Generate and post replies
            logger.logger.info(f"Starting to reply to {len(posts_to_reply)} prioritized posts")
            successful_replies = self.reply_handler.reply_to_posts(posts_to_reply, market_data, max_replies=self.max_replies_per_cycle)
            
            if successful_replies > 0:
                logger.logger.info(f"Successfully posted {successful_replies} replies")
                self.last_reply_time = now                    
                return True
            else:
                logger.logger.info("No replies were successfully posted")
                return False
                
        except Exception as e:
            logger.log_error("Check For Posts To Reply", str(e))
            return False

    def _cleanup(self) -> None:
       """Cleanup resources and save state"""
       try:
           # Stop prediction thread if running
           if self.prediction_thread_running:
               self.prediction_thread_running = False
               if self.prediction_thread and self.prediction_thread.is_alive():
                   self.prediction_thread.join(timeout=5)
               logger.logger.info("Stopped prediction thread")
           
           # Close browser
           if self.browser:
               logger.logger.info("Closing browser...")
               try:
                   self.browser.close_browser()
                   time.sleep(1)
               except Exception as e:
                   logger.logger.warning(f"Error during browser close: {str(e)}")
           
           # Save timeframe prediction data to database for persistence
           try:
               timeframe_state = {
                   "predictions": self.timeframe_predictions,
                   "last_post": {tf: ts.isoformat() for tf, ts in self.timeframe_last_post.items()},
                   "next_scheduled": {tf: ts.isoformat() for tf, ts in self.next_scheduled_posts.items()},
                   "accuracy": self.prediction_accuracy
               }
               
               # Store using the generic JSON data storage
               self.config.db._store_json_data(
                   data_type="timeframe_state",
                   data=timeframe_state
               )
               logger.logger.info("Saved timeframe state to database")
           except Exception as e:
               logger.logger.warning(f"Failed to save timeframe state: {str(e)}")
           
           # Close database connection
           if self.config:
               self.config.cleanup()
               
           logger.log_shutdown()
       except Exception as e:
           logger.log_error("Cleanup", str(e))

    def _ensure_datetime(self, value):
        """Convert value to datetime if it's a string"""
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return datetime.min
        return value

    def _get_crypto_data(self) -> Optional[Dict[str, Any]]:
       """Fetch crypto data from CoinGecko with retries"""
       try:
           params = {
               **self.config.get_coingecko_params(),
               'ids': ','.join(self.target_chains.values()), 
               'sparkline': True 
           }
           
           data = self.coingecko.get_market_data(params)
           if not data:
               logger.logger.error("Failed to fetch market data from CoinGecko")
               return None
               
           formatted_data = {
               coin['symbol'].upper(): {
                   'current_price': coin['current_price'],
                   'volume': coin['total_volume'],
                   'price_change_percentage_24h': coin['price_change_percentage_24h'],
                   'sparkline': coin.get('sparkline_in_7d', {}).get('price', []),
                   'market_cap': coin['market_cap'],
                   'market_cap_rank': coin['market_cap_rank'],
                   'total_supply': coin.get('total_supply'),
                   'max_supply': coin.get('max_supply'),
                   'circulating_supply': coin.get('circulating_supply'),
                   'ath': coin.get('ath'),
                   'ath_change_percentage': coin.get('ath_change_percentage')
               } for coin in data
           }
           
           # Map to correct symbol if needed (particularly for POL which might return as MATIC)
           symbol_corrections = {'MATIC': 'POL'}
           for old_sym, new_sym in symbol_corrections.items():
               if old_sym in formatted_data and new_sym not in formatted_data:
                   formatted_data[new_sym] = formatted_data[old_sym]
                   logger.logger.debug(f"Mapped {old_sym} data to {new_sym}")
           
           # Log API usage statistics
           stats = self.coingecko.get_request_stats()
           logger.logger.debug(
               f"CoinGecko API stats - Daily requests: {stats['daily_requests']}, "
               f"Failed: {stats['failed_requests']}, Cache size: {stats['cache_size']}"
           )
           
           # Store market data in database
           for chain, chain_data in formatted_data.items():
               self.config.db.store_market_data(chain, chain_data)
           
           # Check if all data was retrieved
           missing_tokens = [token for token in self.reference_tokens if token not in formatted_data]
           if missing_tokens:
               logger.logger.warning(f"Missing data for tokens: {', '.join(missing_tokens)}")
               
               # Try fallback mechanism for missing tokens
               if 'POL' in missing_tokens and 'MATIC' in formatted_data:
                   formatted_data['POL'] = formatted_data['MATIC']
                   missing_tokens.remove('POL')
                   logger.logger.info("Applied fallback for POL using MATIC data")
               
           logger.logger.info(f"Successfully fetched crypto data for {', '.join(formatted_data.keys())}")
           return formatted_data
               
       except Exception as e:
           logger.log_error("CoinGecko API", str(e))
           return None

    def _load_saved_timeframe_state(self) -> None:
       """Load previously saved timeframe state from database"""
       try:
           # Query the latest timeframe state
           conn, cursor = self.config.db._get_connection()
           
           cursor.execute("""
               SELECT data 
               FROM generic_json_data 
               WHERE data_type = 'timeframe_state'
               ORDER BY timestamp DESC
               LIMIT 1
           """)
           
           result = cursor.fetchone()
           
           if not result:
               logger.logger.info("No saved timeframe state found")
               return
               
           # Parse the saved state
           state_json = result[0]
           state = json.loads(state_json)
           
           # Restore timeframe predictions
           for timeframe, predictions in state.get("predictions", {}).items():
               self.timeframe_predictions[timeframe] = predictions
           
           # Restore last post times
           for timeframe, timestamp in state.get("last_post", {}).items():
               try:
                   self.timeframe_last_post[timeframe] = datetime.fromisoformat(timestamp)
               except (ValueError, TypeError):
                   # If timestamp can't be parsed, use a safe default
                   self.timeframe_last_post[timeframe] = datetime.now() - timedelta(hours=3)
           
           # Restore next scheduled posts
           for timeframe, timestamp in state.get("next_scheduled", {}).items():
               try:
                   self.next_scheduled_posts[timeframe] = datetime.fromisoformat(timestamp)
                   
                   # If scheduled time is in the past, reschedule
                   if self.next_scheduled_posts[timeframe] < datetime.now():
                       delay_hours = self.timeframe_posting_frequency.get(timeframe, 1) * random.uniform(0.1, 0.5)
                       self.next_scheduled_posts[timeframe] = datetime.now() + timedelta(hours=delay_hours)
               except (ValueError, TypeError):
                   # If timestamp can't be parsed, set a default
                   delay_hours = self.timeframe_posting_frequency.get(timeframe, 1) * random.uniform(0.1, 0.5)
                   self.next_scheduled_posts[timeframe] = datetime.now() + timedelta(hours=delay_hours)
           
           # Restore accuracy tracking
           self.prediction_accuracy = state.get("accuracy", {timeframe: {'correct': 0, 'total': 0} for timeframe in self.timeframes})
           
           logger.logger.info("Restored timeframe state from database")
           
       except Exception as e:
           logger.log_error("Load Timeframe State", str(e))
           # Continue with default values

    def _get_historical_price_data(self, chain: str, hours: int = None, timeframe: str = "1h") -> List[Dict[str, Any]]:
       """
       Get historical price data for the specified time period
       Adjusted based on timeframe for appropriate historical context
       """
       try:
           # Adjust time period based on timeframe if not specified
           if hours is None:
               if timeframe == "1h":
                   hours = 24  # Last 24 hours for hourly predictions
               elif timeframe == "24h":
                   hours = 7 * 24  # Last 7 days for daily predictions
               elif timeframe == "7d":
                   hours = 30 * 24  # Last 30 days for weekly predictions
               else:
                   hours = 24
           
           # Query the database
           return self.config.db.get_recent_market_data(chain, hours)
           
       except Exception as e:
           logger.log_error(f"Historical Price Data - {chain} ({timeframe})", str(e))
           return []
   
    def _get_token_timeframe_performance(self, token: str) -> Dict[str, Dict[str, Any]]:
       """
       Get prediction performance statistics for a token across all timeframes
       """
       try:
           result = {}
           
           # Gather performance for each timeframe
           for timeframe in self.timeframes:
               perf_stats = self.config.db.get_prediction_performance(token=token, timeframe=timeframe)
               
               if perf_stats:
                   result[timeframe] = {
                       "accuracy": perf_stats[0].get("accuracy_rate", 0),
                       "total": perf_stats[0].get("total_predictions", 0),
                       "correct": perf_stats[0].get("correct_predictions", 0),
                       "avg_deviation": perf_stats[0].get("avg_deviation", 0)
                   }
               else:
                   result[timeframe] = {
                       "accuracy": 0,
                       "total": 0,
                       "correct": 0,
                       "avg_deviation": 0
                   }
           
           # Get cross-timeframe comparison
           cross_comparison = self.config.db.get_prediction_comparison_across_timeframes(token)
           
           if cross_comparison:
               result["best_timeframe"] = cross_comparison.get("best_timeframe", {}).get("timeframe", "1h")
               result["overall"] = cross_comparison.get("overall", {})
           
           return result
           
       except Exception as e:
           logger.log_error(f"Get Token Timeframe Performance - {token}", str(e))
           return {tf: {"accuracy": 0, "total": 0, "correct": 0, "avg_deviation": 0} for tf in self.timeframes}
   
    def _get_all_active_predictions(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
       """
       Get all active predictions organized by timeframe and token
       """
       try:
           result = {tf: {} for tf in self.timeframes}
           
           # Get active predictions from the database
           active_predictions = self.config.db.get_active_predictions()
           
           for prediction in active_predictions:
               timeframe = prediction.get("timeframe", "1h")
               token = prediction.get("token", "")
               
               if timeframe in result and token:
                   result[timeframe][token] = prediction
           
           # Merge with in-memory predictions which might be more recent
           for timeframe, predictions in self.timeframe_predictions.items():
               for token, prediction in predictions.items():
                   result.setdefault(timeframe, {})[token] = prediction
           
           return result
           
       except Exception as e:
           logger.log_error("Get All Active Predictions", str(e))
           return {tf: {} for tf in self.timeframes}
   
    def _evaluate_expired_timeframe_predictions(self) -> Dict[str, int]:
       """
       Find and evaluate expired predictions across all timeframes
       Returns count of evaluated predictions by timeframe
       """
       try:
           # Get expired unevaluated predictions
           all_expired = self.config.db.get_expired_unevaluated_predictions()
           
           if not all_expired:
               logger.logger.debug("No expired predictions to evaluate")
               return {tf: 0 for tf in self.timeframes}
               
           # Group by timeframe
           expired_by_timeframe = {tf: [] for tf in self.timeframes}
           
           for prediction in all_expired:
               timeframe = prediction.get("timeframe", "1h")
               if timeframe in expired_by_timeframe:
                   expired_by_timeframe[timeframe].append(prediction)
           
           # Get current market data for evaluation
           market_data = self._get_crypto_data()
           if not market_data:
               logger.logger.error("Failed to fetch market data for prediction evaluation")
               return {tf: 0 for tf in self.timeframes}
           
           # Track evaluated counts
           evaluated_counts = {tf: 0 for tf in self.timeframes}
           
           # Evaluate each prediction by timeframe
           for timeframe, predictions in expired_by_timeframe.items():
               for prediction in predictions:
                   token = prediction["token"]
                   prediction_id = prediction["id"]
                   
                   # Get current price for the token
                   token_data = market_data.get(token, {})
                   if not token_data:
                       logger.logger.warning(f"No current price data for {token}, skipping evaluation")
                       continue
                       
                   current_price = token_data.get("current_price", 0)
                   if current_price == 0:
                       logger.logger.warning(f"Zero price for {token}, skipping evaluation")
                       continue
                       
                   # Record the outcome
                   result = self.config.db.record_prediction_outcome(prediction_id, current_price)
                   
                   if result:
                       logger.logger.debug(f"Evaluated {timeframe} prediction {prediction_id} for {token}")
                       evaluated_counts[timeframe] += 1
                   else:
                       logger.logger.error(f"Failed to evaluate {timeframe} prediction {prediction_id} for {token}")
           
           # Log evaluation summaries
           for timeframe, count in evaluated_counts.items():
               if count > 0:
                   logger.logger.info(f"Evaluated {count} expired {timeframe} predictions")
           
           # Update prediction performance metrics
           self._update_prediction_performance_metrics()
           
           return evaluated_counts
           
       except Exception as e:
           logger.log_error("Evaluate Expired Timeframe Predictions", str(e))
           return {tf: 0 for tf in self.timeframes}

    def _update_prediction_performance_metrics(self) -> None:
        """Update in-memory prediction performance metrics from database"""
        try:
            # Get overall performance by timeframe
            for timeframe in self.timeframes:
                performance = self.config.db.get_prediction_performance(timeframe=timeframe)
                
                total_correct = sum(p.get("correct_predictions", 0) for p in performance)
                total_predictions = sum(p.get("total_predictions", 0) for p in performance)
                
                # Update in-memory tracking
                self.prediction_accuracy[timeframe] = {
                    'correct': total_correct,
                    'total': total_predictions
                }
            
            # Log overall performance
            for timeframe, stats in self.prediction_accuracy.items():
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    logger.logger.info(f"{timeframe} prediction accuracy: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
                    
        except Exception as e:
            logger.log_error("Update Prediction Performance Metrics", str(e))

    def _analyze_volume_trend(self, current_volume: float, historical_data: List[Dict[str, Any]], 
                             timeframe: str = "1h") -> Tuple[float, str]:
        """
        Analyze volume trend over the window period, adjusted for timeframe
        Returns (percentage_change, trend_description)
        """
        if not historical_data:
            return 0.0, "insufficient_data"
            
        try:
            # Adjust trend thresholds based on timeframe
            if timeframe == "1h":
                SIGNIFICANT_THRESHOLD = self.config.VOLUME_TREND_THRESHOLD  # Default (usually 15%)
                MODERATE_THRESHOLD = 5.0
            elif timeframe == "24h":
                SIGNIFICANT_THRESHOLD = 20.0  # Higher threshold for daily predictions
                MODERATE_THRESHOLD = 10.0
            else:  # 7d
                SIGNIFICANT_THRESHOLD = 30.0  # Even higher for weekly predictions
                MODERATE_THRESHOLD = 15.0
            
            # Calculate average volume excluding the current volume
            historical_volumes = [entry['volume'] for entry in historical_data]
            avg_volume = statistics.mean(historical_volumes) if historical_volumes else current_volume
            
            # Calculate percentage change
            volume_change = ((current_volume - avg_volume) / avg_volume) * 100 if avg_volume > 0 else 0
            
            # Determine trend based on timeframe-specific thresholds
            if volume_change >= SIGNIFICANT_THRESHOLD:
                trend = "significant_increase"
            elif volume_change <= -SIGNIFICANT_THRESHOLD:
                trend = "significant_decrease"
            elif volume_change >= MODERATE_THRESHOLD:
                trend = "moderate_increase"
            elif volume_change <= -MODERATE_THRESHOLD:
                trend = "moderate_decrease"
            else:
                trend = "stable"
                
            logger.logger.debug(
                f"Volume trend analysis ({timeframe}): {volume_change:.2f}% change from average. "
                f"Current: {current_volume:,.0f}, Avg: {avg_volume:,.0f}, "
                f"Trend: {trend}"
            )
            
            return volume_change, trend
            
        except Exception as e:
            logger.log_error(f"Volume Trend Analysis - {timeframe}", str(e))
            return 0.0, "error"

    def _analyze_smart_money_indicators(self, token: str, token_data: Dict[str, Any], 
                                      timeframe: str = "1h") -> Dict[str, Any]:
        """
        Analyze potential smart money movements in a token
        Adjusted for different timeframes
        """
        try:
            # Get historical data over multiple timeframes - adjusted based on prediction timeframe
            if timeframe == "1h":
                hourly_data = self._get_historical_volume_data(token, minutes=60, timeframe=timeframe)
                daily_data = self._get_historical_volume_data(token, minutes=1440, timeframe=timeframe)
                # For 1h predictions, we care about recent volume patterns
                short_term_focus = True
            elif timeframe == "24h":
                # For 24h predictions, we want more data
                hourly_data = self._get_historical_volume_data(token, minutes=240, timeframe=timeframe)  # 4 hours
                daily_data = self._get_historical_volume_data(token, minutes=7*1440, timeframe=timeframe)  # 7 days
                short_term_focus = False
            else:  # 7d
                # For weekly predictions, we need even more historical context
                hourly_data = self._get_historical_volume_data(token, minutes=24*60, timeframe=timeframe)  # 24 hours
                daily_data = self._get_historical_volume_data(token, minutes=30*1440, timeframe=timeframe)  # 30 days
                short_term_focus = False
            
            current_volume = token_data['volume']
            current_price = token_data['current_price']
            
            # Volume anomaly detection
            hourly_volumes = [entry['volume'] for entry in hourly_data]
            daily_volumes = [entry['volume'] for entry in daily_data]
            
            # Calculate baselines
            avg_hourly_volume = statistics.mean(hourly_volumes) if hourly_volumes else current_volume
            avg_daily_volume = statistics.mean(daily_volumes) if daily_volumes else current_volume
            
            # Volume Z-score (how many standard deviations from mean)
            hourly_std = statistics.stdev(hourly_volumes) if len(hourly_volumes) > 1 else 1
            volume_z_score = (current_volume - avg_hourly_volume) / hourly_std if hourly_std != 0 else 0
            
            # Price-volume divergence
            # (Price going down while volume increasing suggests accumulation)
            price_direction = 1 if token_data['price_change_percentage_24h'] > 0 else -1
            volume_direction = 1 if current_volume > avg_daily_volume else -1
            
            # Divergence detected when price and volume move in opposite directions
            divergence = (price_direction != volume_direction)
            
            # Adjust accumulation thresholds based on timeframe
            if timeframe == "1h":
                price_change_threshold = 2.0
                volume_multiplier = 1.5
            elif timeframe == "24h":
                price_change_threshold = 3.0
                volume_multiplier = 1.8
            else:  # 7d
                price_change_threshold = 5.0
                volume_multiplier = 2.0
            
            # Check for abnormal volume with minimal price movement (potential accumulation)
            stealth_accumulation = (abs(token_data['price_change_percentage_24h']) < price_change_threshold and 
                                  (current_volume > avg_daily_volume * volume_multiplier))
            
            # Calculate volume profile - percentage of volume in each hour
            volume_profile = {}
            
            # Adjust volume profiling based on timeframe
            if timeframe == "1h":
                # For 1h predictions, look at hourly volume distribution over the day
                hours_to_analyze = 24
            elif timeframe == "24h":
                # For 24h predictions, look at volume by day over the week 
                hours_to_analyze = 7 * 24
            else:  # 7d
                # For weekly, look at entire month
                hours_to_analyze = 30 * 24
            
            if hourly_data:
                for i in range(min(hours_to_analyze, 24)):  # Cap at 24 hours for profile
                    hour_window = datetime.now() - timedelta(hours=i+1)
                    hour_volume = sum(entry['volume'] for entry in hourly_data 
                                    if hour_window <= entry['timestamp'] <= hour_window + timedelta(hours=1))
                    volume_profile[f"hour_{i+1}"] = hour_volume
            
            # Detect unusual trading hours (potential institutional activity)
            total_volume = sum(volume_profile.values()) if volume_profile else 0
            unusual_hours = []
            
            # Adjust unusual hour threshold based on timeframe
            unusual_hour_threshold = 15 if timeframe == "1h" else 20 if timeframe == "24h" else 25
            
            if total_volume > 0:
                for hour, vol in volume_profile.items():
                    hour_percentage = (vol / total_volume) * 100
                    if hour_percentage > unusual_hour_threshold:  # % threshold varies by timeframe
                        unusual_hours.append(hour)
            
            # Detect volume clusters (potential accumulation zones)
            volume_cluster_detected = False
            min_cluster_size = 3 if timeframe == "1h" else 2 if timeframe == "24h" else 2
            cluster_threshold = 1.3 if timeframe == "1h" else 1.5 if timeframe == "24h" else 1.8
            
            if len(hourly_volumes) >= min_cluster_size:
                for i in range(len(hourly_volumes)-min_cluster_size+1):
                    if all(vol > avg_hourly_volume * cluster_threshold for vol in hourly_volumes[i:i+min_cluster_size]):
                        volume_cluster_detected = True
                        break           
            # Calculate additional metrics for longer timeframes
            pattern_metrics = {}
            
            if timeframe in ["24h", "7d"]:
                # Calculate volume trends over different periods
                if len(daily_volumes) >= 7:
                    week1_avg = statistics.mean(daily_volumes[:7])
                    week2_avg = statistics.mean(daily_volumes[7:14]) if len(daily_volumes) >= 14 else week1_avg
                    week3_avg = statistics.mean(daily_volumes[14:21]) if len(daily_volumes) >= 21 else week1_avg
                    
                    pattern_metrics["volume_trend_week1_to_week2"] = ((week1_avg / week2_avg) - 1) * 100 if week2_avg > 0 else 0
                    pattern_metrics["volume_trend_week2_to_week3"] = ((week2_avg / week3_avg) - 1) * 100 if week3_avg > 0 else 0
                
                # Check for volume breakout patterns
                if len(hourly_volumes) >= 48:
                    recent_max = max(hourly_volumes[:24])
                    previous_max = max(hourly_volumes[24:48])
                    
                    pattern_metrics["volume_breakout"] = recent_max > previous_max * 1.5
                
                # Check for consistent high volume days
                if len(daily_volumes) >= 14:
                    high_volume_days = [vol > avg_daily_volume * 1.3 for vol in daily_volumes[:14]]
                    pattern_metrics["consistent_high_volume"] = sum(high_volume_days) >= 5
            
            # Results
            results = {
                'volume_z_score': volume_z_score,
                'price_volume_divergence': divergence,
                'stealth_accumulation': stealth_accumulation,
                'abnormal_volume': abs(volume_z_score) > self.SMART_MONEY_ZSCORE_THRESHOLD,
                'volume_vs_hourly_avg': (current_volume / avg_hourly_volume) - 1,
                'volume_vs_daily_avg': (current_volume / avg_daily_volume) - 1,
                'unusual_trading_hours': unusual_hours,
                'volume_cluster_detected': volume_cluster_detected,
                'timeframe': timeframe
            }
            
            # Add pattern metrics for longer timeframes
            if pattern_metrics:
                results['pattern_metrics'] = pattern_metrics
            
            # Store in database
            self.config.db.store_smart_money_indicators(token, results)
            
            return results
        except Exception as e:
            logger.log_error(f"Smart Money Analysis - {token} ({timeframe})", str(e))
            return {'timeframe': timeframe}
    
    def _analyze_volume_profile(self, token: str, market_data: Dict[str, Any], 
                              timeframe: str = "1h") -> Dict[str, Any]:
        """
        Analyze volume distribution and patterns for a token
        Returns different volume metrics based on timeframe
        """
        try:
            token_data = market_data.get(token, {})
            if not token_data:
                return {}
            
            current_volume = token_data.get('volume', 0)
            
            # Adjust analysis window based on timeframe
            if timeframe == "1h":
                hours_to_analyze = 24
                days_to_analyze = 1
            elif timeframe == "24h":
                hours_to_analyze = 7 * 24
                days_to_analyze = 7
            else:  # 7d
                hours_to_analyze = 30 * 24
                days_to_analyze = 30
            
            # Get historical data
            historical_data = self._get_historical_volume_data(token, minutes=hours_to_analyze * 60, timeframe=timeframe)
            
            # Create volume profile by hour of day
            hourly_profile = {}
            for hour in range(24):
                hourly_profile[hour] = 0
            
            # Fill the profile
            for entry in historical_data:
                timestamp = entry.get('timestamp')
                if timestamp:
                    hour = timestamp.hour
                    hourly_profile[hour] += entry.get('volume', 0)
            
            # Calculate daily pattern
            total_volume = sum(hourly_profile.values())
            if total_volume > 0:
                hourly_percentage = {hour: (volume / total_volume) * 100 for hour, volume in hourly_profile.items()}
            else:
                hourly_percentage = {hour: 0 for hour in range(24)}
            
            # Find peak volume hours
            peak_hours = sorted(hourly_percentage.items(), key=lambda x: x[1], reverse=True)[:3]
            low_hours = sorted(hourly_percentage.items(), key=lambda x: x[1])[:3]
            
            # Check for consistent daily patterns
            historical_volumes = [entry.get('volume', 0) for entry in historical_data]
            avg_volume = statistics.mean(historical_volumes) if historical_volumes else current_volume
            
            # Create day of week profile for longer timeframes
            day_of_week_profile = {}
            if timeframe in ["24h", "7d"] and len(historical_data) >= 7 * 24:
                for day in range(7):
                    day_of_week_profile[day] = 0
                
                # Fill the profile
                for entry in historical_data:
                    timestamp = entry.get('timestamp')
                    if timestamp:
                        day = timestamp.weekday()
                        day_of_week_profile[day] += entry.get('volume', 0)
                
                # Calculate percentages
                dow_total = sum(day_of_week_profile.values())
                if dow_total > 0:
                    day_of_week_percentage = {day: (volume / dow_total) * 100 
                                           for day, volume in day_of_week_profile.items()}
                else:
                    day_of_week_percentage = {day: 0 for day in range(7)}
                
                # Find peak trading days
                peak_days = sorted(day_of_week_percentage.items(), key=lambda x: x[1], reverse=True)[:2]
                low_days = sorted(day_of_week_percentage.items(), key=lambda x: x[1])[:2]
            else:
                day_of_week_percentage = {}
                peak_days = []
                low_days = []
            
            # Calculate volume consistency
            if len(historical_volumes) > 0:
                volume_std = statistics.stdev(historical_volumes) if len(historical_volumes) > 1 else 0
                volume_variability = (volume_std / avg_volume) * 100 if avg_volume > 0 else 0
                
                # Volume consistency score (0-100)
                volume_consistency = max(0, 100 - volume_variability)
            else:
                volume_consistency = 50  # Default if not enough data
            
            # Calculate volume trend over the period
            if len(historical_volumes) >= 2:
                earliest_volume = historical_volumes[0]
                latest_volume = historical_volumes[-1]
                period_change = ((latest_volume - earliest_volume) / earliest_volume) * 100 if earliest_volume > 0 else 0
            else:
                period_change = 0
            
            # Assemble results
            volume_profile_results = {
                'hourly_profile': hourly_percentage,
                'peak_hours': peak_hours,
                'low_hours': low_hours,
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'current_vs_avg': ((current_volume / avg_volume) - 1) * 100 if avg_volume > 0 else 0,
                'volume_consistency': volume_consistency,
                'period_change': period_change,
                'timeframe': timeframe
            }
            
            # Add day of week profile for longer timeframes
            if day_of_week_percentage:
                volume_profile_results['day_of_week_profile'] = day_of_week_percentage
                volume_profile_results['peak_days'] = peak_days
                volume_profile_results['low_days'] = low_days
            
            return volume_profile_results
            
        except Exception as e:
            logger.log_error(f"Volume Profile Analysis - {token} ({timeframe})", str(e))
            return {'timeframe': timeframe}
    
    def _detect_volume_anomalies(self, token: str, market_data: Dict[str, Any], 
                               timeframe: str = "1h") -> Dict[str, Any]:
        """
        Detect volume anomalies and unusual patterns
        Adjust detection thresholds based on timeframe
        """
        try:
            token_data = market_data.get(token, {})
            if not token_data:
                return {}
            
            # Adjust anomaly detection window and thresholds based on timeframe
            if timeframe == "1h":
                detection_window = 24  # 24 hours for hourly predictions
                z_score_threshold = 2.0
                volume_spike_threshold = 3.0
                volume_drop_threshold = 0.3
            elif timeframe == "24h":
                detection_window = 7 * 24  # 7 days for daily predictions
                z_score_threshold = 2.5
                volume_spike_threshold = 4.0
                volume_drop_threshold = 0.25
            else:  # 7d
                detection_window = 30 * 24  # 30 days for weekly predictions
                z_score_threshold = 3.0
                volume_spike_threshold = 5.0
                volume_drop_threshold = 0.2
            
            # Get historical data
            volume_data = self._get_historical_volume_data(token, minutes=detection_window * 60, timeframe=timeframe)
            
            volumes = [entry.get('volume', 0) for entry in volume_data] 
            if len(volumes) < 5:
                return {'insufficient_data': True, 'timeframe': timeframe}
            
            current_volume = token_data.get('volume', 0)
            
            # Calculate metrics
            avg_volume = statistics.mean(volumes)
            if len(volumes) > 1:
                vol_std = statistics.stdev(volumes)
                # Z-score: how many standard deviations from the mean
                volume_z_score = (current_volume - avg_volume) / vol_std if vol_std > 0 else 0
            else:
                volume_z_score = 0
            
            # Moving average calculation
            if len(volumes) >= 10:
                ma_window = 5 if timeframe == "1h" else 7 if timeframe == "24h" else 10
                moving_avgs = []
                
                for i in range(len(volumes) - ma_window + 1):
                    window = volumes[i:i+ma_window]
                    moving_avgs.append(sum(window) / len(window))
                
                # Calculate rate of change in moving average
                if len(moving_avgs) >= 2:
                    ma_change = ((moving_avgs[-1] / moving_avgs[0]) - 1) * 100
                else:
                    ma_change = 0
            else:
                ma_change = 0
            
            # Volume spike detection
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            has_volume_spike = volume_ratio > volume_spike_threshold
            
            # Volume drop detection
            has_volume_drop = volume_ratio < volume_drop_threshold
            
            # Detect sustained high/low volume
            if len(volumes) >= 5:
                recent_volumes = volumes[-5:]
                avg_recent_volume = sum(recent_volumes) / len(recent_volumes)
                sustained_high_volume = avg_recent_volume > avg_volume * 1.5
                sustained_low_volume = avg_recent_volume < avg_volume * 0.5
            else:
                sustained_high_volume = False
                sustained_low_volume = False
            
            # Detect volume patterns for longer timeframes
            pattern_detection = {}
            
            if timeframe in ["24h", "7d"] and len(volumes) >= 14:
                # Check for "volume climax" pattern (increasing volumes culminating in a spike)
                vol_changes = [volumes[i]/volumes[i-1] if volumes[i-1] > 0 else 1 for i in range(1, len(volumes))]
                
                if len(vol_changes) >= 5:
                    recent_changes = vol_changes[-5:]
                    climax_pattern = (sum(1 for change in recent_changes if change > 1.1) >= 3) and has_volume_spike
                    pattern_detection["volume_climax"] = climax_pattern
                
                # Check for "volume exhaustion" pattern (decreasing volumes after a spike)
                if len(volumes) >= 10:
                    peak_idx = volumes.index(max(volumes[-10:]))
                    if peak_idx < len(volumes) - 3:
                        post_peak = volumes[peak_idx+1:]
                        exhaustion_pattern = all(post_peak[i] < post_peak[i-1] for i in range(1, len(post_peak)))
                        pattern_detection["volume_exhaustion"] = exhaustion_pattern
            
            # Assemble results
            anomaly_results = {
                'volume_z_score': volume_z_score,
                'volume_ratio': volume_ratio,
                'has_volume_spike': has_volume_spike,
                'has_volume_drop': has_volume_drop,
                'ma_change': ma_change,
                'sustained_high_volume': sustained_high_volume,
                'sustained_low_volume': sustained_low_volume,
                'abnormal_volume': abs(volume_z_score) > z_score_threshold,
                'timeframe': timeframe
            }
            
            # Add pattern detection for longer timeframes
            if pattern_detection:
                anomaly_results['patterns'] = pattern_detection
            
            return anomaly_results
            
        except Exception as e:
            logger.log_error(f"Volume Anomaly Detection - {token} ({timeframe})", str(e))
            return {'timeframe': timeframe}   

    def _analyze_token_vs_market(self, token: str, market_data: Dict[str, Any], 
                              timeframe: str = "1h") -> Dict[str, Any]:
        """
        Analyze token performance relative to the overall crypto market
        Adjusted for different timeframes
        """
        try:
            token_data = market_data.get(token, {})
            if not token_data:
                return {'timeframe': timeframe}
                
            # Filter out the token itself from reference tokens to avoid self-comparison
            reference_tokens = [t for t in self.reference_tokens if t != token]
            
            # Select appropriate reference tokens based on timeframe
            if timeframe == "1h":
                # For hourly predictions, focus on major tokens and similar market cap tokens
                reference_tokens = ["BTC", "ETH", "SOL", "BNB", "XRP"]
            elif timeframe == "24h":
                # For daily predictions, use all major tokens
                reference_tokens = ["BTC", "ETH", "SOL", "BNB", "XRP", "AVAX", "DOT", "POL"]
            else:  # 7d
                # For weekly predictions, use all reference tokens
                reference_tokens = [t for t in self.reference_tokens if t != token]
            
            # Compare 24h performance
            market_avg_change = statistics.mean([
                market_data.get(ref_token, {}).get('price_change_percentage_24h', 0) 
                for ref_token in reference_tokens
                if ref_token in market_data
            ])
            
            performance_diff = token_data['price_change_percentage_24h'] - market_avg_change
            
            # Compare volume growth - adjust analysis window based on timeframe
            if timeframe == "1h":
                volume_window_minutes = 60  # 1 hour for hourly predictions
            elif timeframe == "24h":
                volume_window_minutes = 24 * 60  # 24 hours for daily predictions
            else:  # 7d
                volume_window_minutes = 7 * 24 * 60  # 7 days for weekly predictions
            
            market_avg_volume_change = statistics.mean([
                self._analyze_volume_trend(
                    market_data.get(ref_token, {}).get('volume', 0),
                    self._get_historical_volume_data(ref_token, minutes=volume_window_minutes, timeframe=timeframe),
                    timeframe=timeframe
                )[0]
                for ref_token in reference_tokens
                if ref_token in market_data
            ])
            
            token_volume_change = self._analyze_volume_trend(
                token_data['volume'],
                self._get_historical_volume_data(token, minutes=volume_window_minutes, timeframe=timeframe),
                timeframe=timeframe
            )[0]
            
            volume_growth_diff = token_volume_change - market_avg_volume_change
            
            # Calculate correlation with each reference token
            correlations = {}
            
            # Get historical price data for correlation calculation
            # Time window depends on timeframe
            if timeframe == "1h":
                history_hours = 24  # Last 24 hours for hourly
            elif timeframe == "24h":
                history_hours = 7 * 24  # Last 7 days for daily
            else:  # 7d
                history_hours = 30 * 24  # Last 30 days for weekly
            
            token_history = self._get_historical_price_data(token, hours=history_hours, timeframe=timeframe)
            token_prices = [entry.get('price', 0) for entry in token_history]
            
            for ref_token in reference_tokens:
                if ref_token in market_data:
                    # Get historical data for reference token
                    ref_history = self._get_historical_price_data(ref_token, hours=history_hours, timeframe=timeframe)
                    ref_prices = [entry.get('price', 0) for entry in ref_history]
                    
                    # Calculate price correlation if we have enough data
                    price_correlation = 0
                    if len(token_prices) > 5 and len(ref_prices) > 5:
                        # Match data lengths for correlation calculation
                        min_length = min(len(token_prices), len(ref_prices))
                        token_prices_adjusted = token_prices[:min_length]
                        ref_prices_adjusted = ref_prices[:min_length]
                        
                        # Calculate correlation coefficient
                        try:
                            if len(token_prices_adjusted) > 1 and len(ref_prices_adjusted) > 1:
                                price_correlation = np.corrcoef(token_prices_adjusted, ref_prices_adjusted)[0, 1]
                        except Exception as e:
                            logger.logger.debug(f"Correlation calculation error: {e}")
                            price_correlation = 0
                    
                    # Get simple 24h change correlation
                    token_direction = 1 if token_data['price_change_percentage_24h'] > 0 else -1
                    ref_token_direction = 1 if market_data[ref_token]['price_change_percentage_24h'] > 0 else -1
                    direction_match = token_direction == ref_token_direction
                    
                    correlations[ref_token] = {
                        'price_correlation': price_correlation,
                        'direction_match': direction_match,
                        'token_change': token_data['price_change_percentage_24h'],
                        'ref_token_change': market_data[ref_token]['price_change_percentage_24h']
                    }
            
            # Determine if token is outperforming the market
            outperforming = performance_diff > 0
            
            # Calculate BTC correlation specifically
            btc_correlation = correlations.get('BTC', {}).get('price_correlation', 0)
            
            # Calculate additional metrics for longer timeframes
            extended_metrics = {}
            
            if timeframe in ["24h", "7d"]:
                # For daily and weekly, analyze sector performance
                defi_tokens = [t for t in reference_tokens if t in ["UNI", "AAVE"]]
                layer1_tokens = [t for t in reference_tokens if t in ["ETH", "SOL", "AVAX", "NEAR"]]
                
                # Calculate sector averages
                if defi_tokens:
                    defi_avg_change = statistics.mean([
                        market_data.get(t, {}).get('price_change_percentage_24h', 0) 
                        for t in defi_tokens if t in market_data
                    ])
                    extended_metrics['defi_sector_diff'] = token_data['price_change_percentage_24h'] - defi_avg_change
                
                if layer1_tokens:
                    layer1_avg_change = statistics.mean([
                        market_data.get(t, {}).get('price_change_percentage_24h', 0) 
                        for t in layer1_tokens if t in market_data
                    ])
                    extended_metrics['layer1_sector_diff'] = token_data['price_change_percentage_24h'] - layer1_avg_change
                
                # Calculate market dominance trend
                if 'BTC' in market_data:
                    btc_mc = market_data['BTC'].get('market_cap', 0)
                    total_mc = sum([data.get('market_cap', 0) for data in market_data.values()])
                    if total_mc > 0:
                        btc_dominance = (btc_mc / total_mc) * 100
                        extended_metrics['btc_dominance'] = btc_dominance
                
                # Analyze token's relative volatility
                token_volatility = self._calculate_relative_volatility(token, reference_tokens, market_data, timeframe)
                if token_volatility is not None:
                    extended_metrics['relative_volatility'] = token_volatility
            
            # Store for any token using the generic method
            self.config.db.store_token_market_comparison(
                token,
                performance_diff,
                volume_growth_diff,
                outperforming,
                correlations
            )
            
            # Create result dict
            result = {
                'vs_market_avg_change': performance_diff,
                'vs_market_volume_growth': volume_growth_diff,
                'correlations': correlations,
                'outperforming_market': outperforming,
                'btc_correlation': btc_correlation,
                'timeframe': timeframe
            }
            
            # Add extended metrics for longer timeframes
            if extended_metrics:
                result['extended_metrics'] = extended_metrics
            
            return result
            
        except Exception as e:
            logger.log_error(f"Token vs Market Analysis - {token} ({timeframe})", str(e))
            return {'timeframe': timeframe}
        
    def _calculate_relative_volatility(self, token: str, reference_tokens: List[str], 
                                     market_data: Dict[str, Any], timeframe: str) -> Optional[float]:
        """
        Calculate token's volatility relative to market average
        Returns a ratio where >1 means more volatile than market, <1 means less volatile
        """
        try:
            # Get historical data with appropriate window for the timeframe
            if timeframe == "1h":
                hours = 24
            elif timeframe == "24h":
                hours = 7 * 24
            else:  # 7d
                hours = 30 * 24
            
            # Get token history
            token_history = self._get_historical_price_data(token, hours=hours, timeframe=timeframe)
            if len(token_history) < 5:
                return None
            
            token_prices = [entry.get('price', 0) for entry in token_history]
            
            # Calculate token volatility (standard deviation of percent changes)
            token_changes = []
            for i in range(1, len(token_prices)):
                if token_prices[i-1] > 0:
                    pct_change = ((token_prices[i] / token_prices[i-1]) - 1) * 100
                    token_changes.append(pct_change)
                    
            if not token_changes:
                return None
                
            token_volatility = statistics.stdev(token_changes) if len(token_changes) > 1 else 0
            
            # Calculate market average volatility
            market_volatilities = []
            
            for ref_token in reference_tokens:
                if ref_token in market_data:
                    ref_history = self._get_historical_price_data(ref_token, hours=hours, timeframe=timeframe)
                    if len(ref_history) < 5:
                        continue
                        
                    ref_prices = [entry.get('price', 0) for entry in ref_history]
                    
                    ref_changes = []
                    for i in range(1, len(ref_prices)):
                        if ref_prices[i-1] > 0:
                            pct_change = ((ref_prices[i] / ref_prices[i-1]) - 1) * 100
                            ref_changes.append(pct_change)
                            
                    if len(ref_changes) > 1:
                        ref_volatility = statistics.stdev(ref_changes)
                        market_volatilities.append(ref_volatility)
            
            # Calculate relative volatility
            if market_volatilities:
                market_avg_volatility = statistics.mean(market_volatilities)
                if market_avg_volatility > 0:
                    return token_volatility / market_avg_volatility
            
            return None
            
        except Exception as e:
            logger.log_error(f"Calculate Relative Volatility - {token} ({timeframe})", str(e))
            return None

    def _calculate_correlations(self, token: str, market_data: Dict[str, Any], 
                              timeframe: str = "1h") -> Dict[str, float]:
        """
        Calculate token correlations with the market
        Adjust correlation window based on timeframe
        """
        try:
            token_data = market_data.get(token, {})
            if not token_data:
                return {'timeframe': timeframe}
                
            # Filter out the token itself from reference tokens to avoid self-comparison
            reference_tokens = [t for t in self.reference_tokens if t != token]
            
            # Select appropriate reference tokens based on timeframe and relevance
            if timeframe == "1h":
                # For hourly, just use major tokens
                reference_tokens = ["BTC", "ETH", "SOL"]
            elif timeframe == "24h":
                # For daily, use more tokens
                reference_tokens = ["BTC", "ETH", "SOL", "BNB", "XRP"]
            # For weekly, use all tokens (default)
            
            correlations = {}
            
            # Calculate correlation with each reference token
            for ref_token in reference_tokens:
                if ref_token not in market_data:
                    continue
                    
                ref_data = market_data[ref_token]
                
                # Time window for correlation calculation based on timeframe
                if timeframe == "1h":
                    # Use 24h change for hourly predictions (short-term)
                    price_correlation_metric = abs(token_data['price_change_percentage_24h'] - ref_data['price_change_percentage_24h'])
                elif timeframe == "24h":
                    # For daily, check if we have 7d change data available
                    if 'price_change_percentage_7d' in token_data and 'price_change_percentage_7d' in ref_data:
                        price_correlation_metric = abs(token_data['price_change_percentage_7d'] - ref_data['price_change_percentage_7d'])
                    else:
                        # Fall back to 24h change if 7d not available
                        price_correlation_metric = abs(token_data['price_change_percentage_24h'] - ref_data['price_change_percentage_24h'])
                else:  # 7d
                    # For weekly, use historical correlation if available
                    # Get historical data with longer window
                    token_history = self._get_historical_price_data(token, hours=30*24, timeframe=timeframe)
                    ref_history = self._get_historical_price_data(ref_token, hours=30*24, timeframe=timeframe)
                    
                    if len(token_history) >= 14 and len(ref_history) >= 14:
                        # Calculate 14-day rolling correlation
                        token_prices = [entry.get('price', 0) for entry in token_history[:14]]
                        ref_prices = [entry.get('price', 0) for entry in ref_history[:14]]
                        
                        if len(token_prices) == len(ref_prices) and len(token_prices) > 2:
                            try:
                                # Calculate correlation coefficient
                                historical_corr = np.corrcoef(token_prices, ref_prices)[0, 1]
                                price_correlation_metric = abs(1 - historical_corr)
                            except:
                                # Fall back to 24h change if correlation fails
                                price_correlation_metric = abs(token_data['price_change_percentage_24h'] - ref_data['price_change_percentage_24h'])
                        else:
                            price_correlation_metric = abs(token_data['price_change_percentage_24h'] - ref_data['price_change_percentage_24h'])
                    else:
                        price_correlation_metric = abs(token_data['price_change_percentage_24h'] - ref_data['price_change_percentage_24h'])
                
                # Calculate price correlation (convert difference to correlation coefficient)
                # Smaller difference = higher correlation
                max_diff = 15 if timeframe == "1h" else 25 if timeframe == "24h" else 40
                price_correlation = 1 - min(1, price_correlation_metric / max_diff)
                
                # Volume correlation (simplified)
                volume_correlation = abs(
                    (token_data['volume'] - ref_data['volume']) / 
                    max(token_data['volume'], ref_data['volume'])
                )
                volume_correlation = 1 - volume_correlation  # Convert to correlation coefficient
                
                correlations[f'price_correlation_{ref_token}'] = price_correlation
                correlations[f'volume_correlation_{ref_token}'] = volume_correlation
            
            # Calculate average correlations
            price_correlations = [v for k, v in correlations.items() if 'price_correlation_' in k]
            volume_correlations = [v for k, v in correlations.items() if 'volume_correlation_' in k]
            
            correlations['avg_price_correlation'] = statistics.mean(price_correlations) if price_correlations else 0
            correlations['avg_volume_correlation'] = statistics.mean(volume_correlations) if volume_correlations else 0
            
            # Add BTC dominance correlation for longer timeframes
            if timeframe in ["24h", "7d"] and 'BTC' in market_data:
                btc_mc = market_data['BTC'].get('market_cap', 0)
                total_mc = sum([data.get('market_cap', 0) for data in market_data.values()])
                
                if total_mc > 0:
                    btc_dominance = (btc_mc / total_mc) * 100
                    btc_change = market_data['BTC'].get('price_change_percentage_24h', 0)
                    token_change = token_data.get('price_change_percentage_24h', 0)
                    
                    # Simple heuristic: if token moves opposite to BTC and dominance is high,
                    # it might be experiencing a rotation from/to BTC
                    btc_rotation_indicator = (btc_change * token_change < 0) and (btc_dominance > 50)
                    
                    correlations['btc_dominance'] = btc_dominance
                    correlations['btc_rotation_indicator'] = btc_rotation_indicator
            
            # Store correlation data for any token using the generic method
            self.config.db.store_token_correlations(token, correlations)
            
            logger.logger.debug(
                f"{token} correlations calculated ({timeframe}) - Avg Price: {correlations['avg_price_correlation']:.2f}, "
                f"Avg Volume: {correlations['avg_volume_correlation']:.2f}"
            )
            
            return correlations
            
        except Exception as e:
            logger.log_error(f"Correlation Calculation - {token} ({timeframe})", str(e))
            return {
                'avg_price_correlation': 0.0,
                'avg_volume_correlation': 0.0,
                'timeframe': timeframe
            }

    def _generate_correlation_report(self, market_data: Dict[str, Any], timeframe: str = "1h") -> str:
        """
        Generate a report of correlations between top tokens
        Customized based on timeframe
        """
        try:
            # Fix 1: Add check for market_data being None
            if not market_data:
                return f"Failed to generate {timeframe} correlation report: No market data available"
                
            # Select tokens to include based on timeframe
            if timeframe == "1h":
                tokens = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']  # Focus on major tokens for hourly
            elif timeframe == "24h":
                tokens = ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX', 'XRP']  # More tokens for daily
            else:  # 7d
                tokens = ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX', 'DOT', 'XRP']  # Most tokens for weekly
        
            # Create correlation matrix
            correlation_matrix = {}
            for token1 in tokens:
                correlation_matrix[token1] = {}
                # Fix 2: Properly nest the token2 loop inside the token1 loop
                for token2 in tokens:
                    if token1 == token2:
                        correlation_matrix[token1][token2] = 1.0
                        continue
                    
                    if token1 not in market_data or token2 not in market_data:
                        correlation_matrix[token1][token2] = 0.0
                        continue
                    
                    # Adjust correlation calculation based on timeframe
                    if timeframe == "1h":
                        # For hourly, use 24h price change
                        price_change1 = market_data[token1]['price_change_percentage_24h']
                        price_change2 = market_data[token2]['price_change_percentage_24h']
                        
                        # Calculate simple correlation
                        price_direction1 = 1 if price_change1 > 0 else -1
                        price_direction2 = 1 if price_change2 > 0 else -1
                        
                        # Basic correlation (-1.0 to 1.0)
                        correlation = 1.0 if price_direction1 == price_direction2 else -1.0
                    else:
                        # For longer timeframes, try to use more sophisticated correlation
                        # Get historical data
                        token1_history = self._get_historical_price_data(token1, timeframe=timeframe)
                        token2_history = self._get_historical_price_data(token2, timeframe=timeframe)
                        
                        if len(token1_history) >= 5 and len(token2_history) >= 5:
                            # Extract prices for correlation calculation
                            prices1 = [entry.get('price', 0) for entry in token1_history][:min(len(token1_history), len(token2_history))]
                            prices2 = [entry.get('price', 0) for entry in token2_history][:min(len(token1_history), len(token2_history))]
                            
                            try:
                                # Calculate Pearson correlation
                                correlation = np.corrcoef(prices1, prices2)[0, 1]
                                if np.isnan(correlation):
                                    # Fall back to simple method if NaN
                                    price_direction1 = 1 if market_data[token1]['price_change_percentage_24h'] > 0 else -1
                                    price_direction2 = 1 if market_data[token2]['price_change_percentage_24h'] > 0 else -1
                                    correlation = 1.0 if price_direction1 == price_direction2 else -1.0
                            except:
                                # Fall back to simple method if calculation fails
                                price_direction1 = 1 if market_data[token1]['price_change_percentage_24h'] > 0 else -1
                                price_direction2 = 1 if market_data[token2]['price_change_percentage_24h'] > 0 else -1
                                correlation = 1.0 if price_direction1 == price_direction2 else -1.0
                        else:
                            # Not enough historical data, use simple method
                            price_direction1 = 1 if market_data[token1]['price_change_percentage_24h'] > 0 else -1
                            price_direction2 = 1 if market_data[token2]['price_change_percentage_24h'] > 0 else -1
                            correlation = 1.0 if price_direction1 == price_direction2 else -1.0
                            
                    correlation_matrix[token1][token2] = correlation
            
            # Fix 3: Move the report formatting outside the loops
            if timeframe == "1h":
                report = "1H CORRELATION MATRIX:\n\n"
            elif timeframe == "24h":
                report = "24H CORRELATION MATRIX:\n\n"
            else:
                report = "7D CORRELATION MATRIX:\n\n"
    
            # Create ASCII art heatmap
            for token1 in tokens:
                report += f"{token1} "
                for token2 in tokens:
                    corr = correlation_matrix[token1][token2]
                    if token1 == token2:
                        report += "✦ "  # Self correlation
                    elif corr > 0.5:
                        report += "➕ "  # Strong positive
                    elif corr > 0:
                        report += "📈 "  # Positive
                    elif corr < -0.5:
                        report += "➖ "  # Strong negative
                    else:
                        report += "📉 "  # Negative
                report += "\n"
            
            report += "\nKey: ✦=Same ➕=Strong+ 📈=Weak+ ➖=Strong- 📉=Weak-"
            
            # Add timeframe-specific insights
            if timeframe == "24h" or timeframe == "7d":
                # For longer timeframes, add sector analysis
                defi_tokens = [t for t in tokens if t in ["UNI", "AAVE"]]
                layer1_tokens = [t for t in tokens if t in ["ETH", "SOL", "AVAX", "NEAR"]]
                
                # Check if we have enough tokens from each sector
                if len(defi_tokens) >= 2 and len(layer1_tokens) >= 2:
                    # Calculate average intra-sector correlation
                    defi_corrs = []
                    for i in range(len(defi_tokens)):
                        for j in range(i+1, len(defi_tokens)):
                            t1, t2 = defi_tokens[i], defi_tokens[j]
                            if t1 in correlation_matrix and t2 in correlation_matrix[t1]:
                                defi_corrs.append(correlation_matrix[t1][t2])
                    
                    layer1_corrs = []
                    for i in range(len(layer1_tokens)):
                        for j in range(i+1, len(layer1_tokens)):
                            t1, t2 = layer1_tokens[i], layer1_tokens[j]
                            if t1 in correlation_matrix and t2 in correlation_matrix[t1]:
                                layer1_corrs.append(correlation_matrix[t1][t2])
                    
                    # Calculate cross-sector correlation
                    cross_corrs = []
                    for t1 in defi_tokens:
                        for t2 in layer1_tokens:
                            if t1 in correlation_matrix and t2 in correlation_matrix[t1]:
                                cross_corrs.append(correlation_matrix[t1][t2])
                    
                    # Add to report if we have correlation data
                    if defi_corrs and layer1_corrs and cross_corrs:
                        avg_defi_corr = sum(defi_corrs) / len(defi_corrs)
                        avg_layer1_corr = sum(layer1_corrs) / len(layer1_corrs)
                        avg_cross_corr = sum(cross_corrs) / len(cross_corrs)
                        
                        report += f"\n\nSector Analysis:"
                        report += f"\nDeFi internal correlation: {avg_defi_corr:.2f}"
                        report += f"\nLayer1 internal correlation: {avg_layer1_corr:.2f}"
                        report += f"\nCross-sector correlation: {avg_cross_corr:.2f}"
                        
                        # Interpret sector rotation
                        if avg_cross_corr < min(avg_defi_corr, avg_layer1_corr) - 0.3:
                            report += "\nPossible sector rotation detected!"
            
            return report
        except Exception as e:
            logger.log_error(f"Correlation Report - {timeframe}", str(e))
            return f"Failed to generate {timeframe} correlation report."

    def _calculate_momentum_score(self, token: str, market_data: Dict[str, Any], timeframe: str = "1h") -> float:
        """
        Calculate a momentum score (0-100) for a token based on various metrics
        Adjusted for different timeframes
        """
        try:
            token_data = market_data.get(token, {})
            if not token_data:
              return 50.0  # Neutral score
            
            # Get basic metrics
            price_change = token_data.get('price_change_percentage_24h', 0)
            volume = token_data.get('volume', 0)
        
            # Get historical volume for volume change - adjust window based on timeframe
            if timeframe == "1h":
                window_minutes = 60  # Last hour for hourly predictions
            elif timeframe == "24h":
                window_minutes = 24 * 60  # Last day for daily predictions
            else:  # 7d
                window_minutes = 7 * 24 * 60  # Last week for weekly predictions
                
            historical_volume = self._get_historical_volume_data(token, minutes=window_minutes, timeframe=timeframe)
            volume_change, _ = self._analyze_volume_trend(volume, historical_volume, timeframe=timeframe)
        
            # Get smart money indicators
            smart_money = self._analyze_smart_money_indicators(token, token_data, timeframe=timeframe)
        
            # Get market comparison
            vs_market = self._analyze_token_vs_market(token, market_data, timeframe=timeframe)
        
            # Calculate score components (0-20 points each)
            # Adjust price score scaling based on timeframe
            if timeframe == "1h":
                price_range = 5.0  # ±5% for hourly
            elif timeframe == "24h":
                price_range = 10.0  # ±10% for daily
            else:  # 7d
                price_range = 20.0  # ±20% for weekly
                
            price_score = min(20, max(0, (price_change + price_range) * (20 / (2 * price_range))))
        
            # Adjust volume score scaling based on timeframe
            if timeframe == "1h":
                volume_range = 10.0  # ±10% for hourly
            elif timeframe == "24h":
                volume_range = 20.0  # ±20% for daily
            else:  # 7d
                volume_range = 40.0  # ±40% for weekly
                
            volume_score = min(20, max(0, (volume_change + volume_range) * (20 / (2 * volume_range))))
        
            # Smart money score - additional indicators for longer timeframes
            smart_money_score = 0
            if smart_money.get('abnormal_volume', False):
                smart_money_score += 5
            if smart_money.get('stealth_accumulation', False):
                smart_money_score += 5
            if smart_money.get('volume_cluster_detected', False):
                smart_money_score += 5
            if smart_money.get('volume_z_score', 0) > 1.0:
                smart_money_score += 5
                
            # Add pattern metrics for longer timeframes
            if timeframe in ["24h", "7d"] and 'pattern_metrics' in smart_money:
                pattern_metrics = smart_money['pattern_metrics']
                if pattern_metrics.get('volume_breakout', False):
                    smart_money_score += 5
                if pattern_metrics.get('consistent_high_volume', False):
                    smart_money_score += 5
                    
            smart_money_score = min(20, smart_money_score)
        
            # Market comparison score
            market_score = 0
            if vs_market.get('outperforming_market', False):
                market_score += 10
            market_score += min(10, max(0, (vs_market.get('vs_market_avg_change', 0) + 5)))
            market_score = min(20, market_score)
        
            # Trend consistency score - higher standards for longer timeframes
            if timeframe == "1h":
                trend_score = 20 if all([price_score > 10, volume_score > 10, smart_money_score > 5, market_score > 10]) else 0
            elif timeframe == "24h":
                trend_score = 20 if all([price_score > 12, volume_score > 12, smart_money_score > 8, market_score > 12]) else 0
            else:  # 7d
                trend_score = 20 if all([price_score > 15, volume_score > 15, smart_money_score > 10, market_score > 15]) else 0
        
            # Calculate total score (0-100)
            # Adjust component weights based on timeframe
            if timeframe == "1h":
                # For hourly, recent price action and smart money more important
                total_score = (
                    price_score * 0.25 +
                    volume_score * 0.2 +
                    smart_money_score * 0.25 +
                    market_score * 0.15 +
                    trend_score * 0.15
                ) * 1.0
            elif timeframe == "24h":
                # For daily, balance factors with more weight to market comparison
                total_score = (
                    price_score * 0.2 +
                    volume_score * 0.2 +
                    smart_money_score * 0.2 +
                    market_score * 0.25 +
                    trend_score * 0.15
                ) * 1.0
            else:  # 7d
                # For weekly, market factors and trend consistency more important
                total_score = (
                    price_score * 0.15 +
                    volume_score * 0.15 +
                    smart_money_score * 0.2 +
                    market_score * 0.3 +
                    trend_score * 0.2
                ) * 1.0
        
            return total_score
        
        except Exception as e:
            logger.log_error(f"Momentum Score - {token} ({timeframe})", str(e))
            return 50.0  # Neutral score on error

    def _format_prediction_tweet(self, token: str, prediction: Dict[str, Any], market_data: Dict[str, Any], timeframe: str = "1h") -> str:
        """
        Format a prediction into a tweet with FOMO-inducing content
        Supports multiple timeframes (1h, 24h, 7d)
        """
        try:
            # Get prediction details
            pred_data = prediction.get("prediction", {})
            sentiment = prediction.get("sentiment", "NEUTRAL")
            rationale = prediction.get("rationale", "")
        
            # Format prediction values
            price = pred_data.get("price", 0)
            confidence = pred_data.get("confidence", 70)
            lower_bound = pred_data.get("lower_bound", 0)
            upper_bound = pred_data.get("upper_bound", 0)
            percent_change = pred_data.get("percent_change", 0)
        
            # Get current price
            token_data = market_data.get(token, {})
            current_price = token_data.get("current_price", 0)
        
            # Format timeframe for display
            if timeframe == "1h":
                display_timeframe = "1HR"
            elif timeframe == "24h":
                display_timeframe = "24HR"
            else:  # 7d
                display_timeframe = "7DAY"
            
            # Format the tweet
            tweet = f"#{token} {display_timeframe} PREDICTION:\n\n"
        
            # Sentiment-based formatting
            if sentiment == "BULLISH":
                tweet += "BULLISH ALERT\n"
            elif sentiment == "BEARISH":
                tweet += "BEARISH WARNING\n"
            else:
                tweet += "MARKET ANALYSIS\n"
            
            # Add prediction with confidence
            tweet += f"Target: ${price:.4f} ({percent_change:+.2f}%)\n"
            tweet += f"Range: ${lower_bound:.4f} - ${upper_bound:.4f}\n"
            tweet += f"Confidence: {confidence}%\n\n"
        
            # Add rationale - adjust length based on timeframe
            if timeframe == "7d":
                # For weekly predictions, add more detail to rationale
                tweet += f"{rationale}\n\n"
            else:
                # For shorter timeframes, keep it brief
                if len(rationale) > 100:
                    # Truncate at a sensible point
                    last_period = rationale[:100].rfind('. ')
                    if last_period > 50:
                        rationale = rationale[:last_period+1]
                    else:
                        rationale = rationale[:100] + "..."
                tweet += f"{rationale}\n\n"
        
            # Add accuracy tracking if available
            performance = self.config.db.get_prediction_performance(token=token, timeframe=timeframe)
            if performance and performance[0]["total_predictions"] > 0:
                accuracy = performance[0]["accuracy_rate"]
                tweet += f"Accuracy: {accuracy:.1f}% on {performance[0]['total_predictions']} predictions"
            
            # Ensure tweet is within the hard stop length
            max_length = self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']
            if len(tweet) > max_length:
                # Smart truncate to preserve essential info
                last_paragraph = tweet.rfind("\n\n")
                if last_paragraph > max_length * 0.7:
                    # Truncate at the last paragraph break
                    tweet = tweet[:last_paragraph].strip()
                else:
                    # Simply truncate with ellipsis
                    tweet = tweet[:max_length-3] + "..."
            
            return tweet
        
        except Exception as e:
            logger.log_error(f"Format Prediction Tweet - {token} ({timeframe})", str(e))
            return f"#{token} {timeframe.upper()} PREDICTION: ${price:.4f} ({percent_change:+.2f}%) - {sentiment}"

    def _track_prediction(self, token: str, prediction: Dict[str, Any], relevant_tokens: List[str], timeframe: str = "1h") -> None:
        """
        Track predictions for future callbacks and analysis
        Supports multiple timeframes (1h, 24h, 7d)
        """
        MAX_PREDICTIONS = 20  
    
        # Get current prices of relevant tokens from prediction
        current_prices = {chain: prediction.get(f'{chain.upper()}_price', 0) for chain in relevant_tokens if f'{chain.upper()}_price' in prediction}
    
        # Add the prediction to the tracking list with timeframe info
        self.past_predictions.append({
            'timestamp': datetime.now(),
            'token': token,
            'prediction': prediction['analysis'],
            'prices': current_prices,
            'sentiment': prediction['sentiment'],
            'timeframe': timeframe,
            'outcome': None
        })
    
        # Keep only predictions from the last 24 hours, up to MAX_PREDICTIONS
        self.past_predictions = [p for p in self.past_predictions 
                              if (datetime.now() - p['timestamp']).total_seconds() < 86400]
    
        # Trim to max predictions if needed
        if len(self.past_predictions) > MAX_PREDICTIONS:
            self.past_predictions = self.past_predictions[-MAX_PREDICTIONS:]
        
        logger.logger.debug(f"Tracked {timeframe} prediction for {token}")
        
    def _validate_past_prediction(self, prediction: Dict[str, Any], current_prices: Dict[str, float]) -> str:
        """
        Check if a past prediction was accurate
        Returns evaluation outcome: 'right', 'wrong', or 'undetermined'
        """
        sentiment_map = {
            'bullish': 1,
            'bearish': -1,
            'neutral': 0,
            'volatile': 0,
            'recovering': 0.5
        }
    
        # Apply different thresholds based on the timeframe
        timeframe = prediction.get('timeframe', '1h')
        if timeframe == '1h':
            threshold = 2.0  # 2% for 1-hour predictions
        elif timeframe == '24h':
            threshold = 4.0  # 4% for 24-hour predictions
        else:  # 7d
            threshold = 7.0  # 7% for 7-day predictions
    
        wrong_tokens = []
        for token, old_price in prediction['prices'].items():
            if token in current_prices and old_price > 0:
                price_change = ((current_prices[token] - old_price) / old_price) * 100
            
                # Get sentiment for this token
                token_sentiment_key = token.upper() if token.upper() in prediction['sentiment'] else token
                token_sentiment_value = prediction['sentiment'].get(token_sentiment_key)
            
                # Handle nested dictionary structure
                if isinstance(token_sentiment_value, dict) and 'mood' in token_sentiment_value:
                    token_sentiment = sentiment_map.get(token_sentiment_value['mood'], 0)
                else:
                    token_sentiment = sentiment_map.get(token_sentiment_value, 0)
            
                # A prediction is wrong if:
                # 1. Bullish but price dropped more than threshold%
                # 2. Bearish but price rose more than threshold%
                if (token_sentiment > 0 and price_change < -threshold) or (token_sentiment < 0 and price_change > threshold):
                    wrong_tokens.append(token)
    
        return 'wrong' if wrong_tokens else 'right'
    
    def _get_spicy_callback(self, token: str, current_prices: Dict[str, float], timeframe: str = "1h") -> Optional[str]:
        """
        Generate witty callbacks to past terrible predictions
        Supports multiple timeframes
        """
        # Look for the most recent prediction for this token and timeframe
        recent_predictions = [p for p in self.past_predictions 
                           if p['timestamp'] > (datetime.now() - timedelta(hours=24))
                           and p['token'] == token
                           and p.get('timeframe', '1h') == timeframe]
    
        if not recent_predictions:
            return None
        
        # Evaluate any unvalidated predictions
        for pred in recent_predictions:
            if pred['outcome'] is None:
                pred['outcome'] = self._validate_past_prediction(pred, current_prices)
            
        # Find any wrong predictions
        wrong_predictions = [p for p in recent_predictions if p['outcome'] == 'wrong']
        if wrong_predictions:
            worst_pred = wrong_predictions[-1]
            time_ago = int((datetime.now() - worst_pred['timestamp']).total_seconds() / 3600)
        
            # If time_ago is 0, set it to 1 to avoid awkward phrasing
            if time_ago == 0:
                time_ago = 1
        
            # Format timeframe for display
            time_unit = "hr" if timeframe in ["1h", "24h"] else "day"
            time_display = f"{time_ago}{time_unit}"
        
            # Token-specific callbacks
            callbacks = [
                f"(Unlike my galaxy-brain take {time_display} ago about {worst_pred['prediction'].split('.')[0]}... this time I'm sure!)",
                f"(Looks like my {time_display} old prediction about {token} aged like milk. But trust me bro!)",
                f"(That awkward moment when your {time_display} old {token} analysis was completely wrong... but this one's different!)",
                f"(My {token} trading bot would be down bad after that {time_display} old take. Good thing I'm just an analyst!)",
                f"(Excuse the {time_display} old miss on {token}. Even the best crypto analysts are wrong sometimes... just not usually THIS wrong!)"
            ]
        
            # Select a callback deterministically but with variation
            callback_seed = f"{datetime.now().date()}_{token}_{timeframe}"
            callback_index = hash(callback_seed) % len(callbacks)
        
            return callbacks[callback_index]
        
        return None

    def _format_tweet_analysis(self, token: str, analysis: str, crypto_data: Dict[str, Any], timeframe: str = "1h") -> str:
        """
        Format analysis for Twitter with no hashtags to maximize content
        Supports multiple timeframes (1h, 24h, 7d)
        """
        # Check if we need to add timeframe prefix
        if timeframe != "1h" and not any(prefix in analysis.upper() for prefix in [f"{timeframe.upper()} ", f"{timeframe}-"]):
            # Add timeframe prefix if not already present
            if timeframe == "24h":
                prefix = "24H ANALYSIS: "
            else:  # 7d
                prefix = "7DAY OUTLOOK: "
            
            # Only add prefix if not already present in some form
            analysis = prefix + analysis
    
        # Simply use the analysis text with no hashtags
        tweet = analysis
    
        # Sanitize text to remove non-BMP characters that ChromeDriver can't handle
        tweet = ''.join(char for char in tweet if ord(char) < 0x10000)
    
        # Check for minimum length
        min_length = self.config.TWEET_CONSTRAINTS['MIN_LENGTH']
        if len(tweet) < min_length:
            logger.logger.warning(f"{timeframe} analysis too short ({len(tweet)} chars). Minimum: {min_length}")
            # Not much we can do here since Claude should have generated the right length
            # We'll log but not try to fix, as Claude should be instructed correctly
    
        # Check for maximum length
        max_length = self.config.TWEET_CONSTRAINTS['MAX_LENGTH']
        if len(tweet) > max_length:
            logger.logger.warning(f"{timeframe} analysis too long ({len(tweet)} chars). Maximum: {max_length}")
    
        # Check for hard stop length
        hard_stop = self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']
        if len(tweet) > hard_stop:
            # Smart truncation - find the last sentence boundary before the limit
            # First try to end on a period, question mark, or exclamation
            last_period = tweet[:hard_stop-3].rfind('. ')
            last_question = tweet[:hard_stop-3].rfind('? ')
            last_exclamation = tweet[:hard_stop-3].rfind('! ')
        
            # Find the last sentence-ending punctuation
            last_sentence_end = max(last_period, last_question, last_exclamation)
        
            if last_sentence_end > hard_stop * 0.7:  # If we can find a good sentence break in the latter 30% of the text
                # Truncate at the end of a sentence and add no ellipsis
                tweet = tweet[:last_sentence_end+1]  # Include the punctuation
            else:
                # Fallback: find the last word boundary
                last_space = tweet[:hard_stop-3].rfind(' ')
                if last_space > 0:
                    tweet = tweet[:last_space] + "..."
                else:
                    # Last resort: hard truncation
                    tweet = tweet[:hard_stop-3] + "..."
            
            logger.logger.warning(f"Trimmed {timeframe} analysis to {len(tweet)} chars using smart truncation")
    
        return tweet

    def _analyze_market_sentiment(self, token: str, market_data: Dict[str, Any], 
                                 trigger_type: str, timeframe: str = "1h") -> Tuple[Optional[str], Optional[Dict]]:
        """
        Generate token-specific market analysis with focus on volume and smart money.
        Supports multiple timeframes (1h, 24h, 7d)
        Returns the formatted tweet and data needed to store it in the database.
        """
        max_retries = 3
        retry_count = 0
    
        # Define rotating focus areas for more varied analyses
        focus_areas = [
            "Focus on volume patterns, smart money movements, and how the token is performing relative to the broader market.",
            "Emphasize technical indicators showing money flow in the market. Pay special attention to volume-to-price divergence.",
            "Analyze accumulation patterns and capital rotation. Look for subtle signs of institutional interest.",
            "Examine volume preceding price action. Note any leading indicators.",
            "Highlight the relationship between price action and significant volume changes.",
            "Investigate potential smart money positioning ahead of market moves. Note any anomalous volume signatures.",
            "Focus on recent volume clusters and their impact on price stability. Look for divergence patterns.",
            "Analyze volatility profile compared to the broader market and what this suggests about sentiment."
        ]
    
        # Define timeframe-specific prompting guidance
        timeframe_guidance = {
            "1h": "Focus on immediate market microstructure and short-term price action for hourly traders.",
            "24h": "Emphasize market momentum over the full day and key levels for short-term traders.",
            "7d": "Analyze macro market structure, key support/resistance zones, and medium-term trend direction."
        }
    
        # Define character count limits based on timeframe
        char_limits = {
            "1h": "260-275",
            "24h": "265-280", 
            "7d": "270-285"
        }
    
        # Define target character counts
        target_chars = {
            "1h": 270,
            "24h": 275,
            "7d": 280
        }
    
        while retry_count < max_retries:
            try:
                logger.logger.debug(f"Starting {token} {timeframe} market sentiment analysis (attempt {retry_count + 1})")
            
                # Get token data
                token_data = market_data.get(token, {})
                if not token_data:
                    logger.log_error("Market Analysis", f"Missing {token} data")
                    return None, None
            
                # Calculate correlations with market
                correlations = self._calculate_correlations(token, market_data, timeframe=timeframe)
            
                # Get smart money indicators
                smart_money = self._analyze_smart_money_indicators(token, token_data, timeframe=timeframe)
            
                # Get token vs market performance
                vs_market = self._analyze_token_vs_market(token, market_data, timeframe=timeframe)
            
                # Get spicy callback for previous predictions
                callback = self._get_spicy_callback(token, {sym: data['current_price'] 
                                                   for sym, data in market_data.items()}, timeframe=timeframe)
            
                # Analyze mood
                indicators = MoodIndicators(
                    price_change=token_data['price_change_percentage_24h'],
                    trading_volume=token_data['volume'],
                    volatility=abs(token_data['price_change_percentage_24h']) / 100,
                    social_sentiment=None,
                    funding_rates=None,
                    liquidation_volume=None
                )
            
                mood = determine_advanced_mood(indicators)
                token_mood = {
                    'mood': mood.value,
                    'change': token_data['price_change_percentage_24h'],
                    'ath_distance': token_data['ath_change_percentage']
                }
            
                # Store mood data
                self.config.db.store_mood(token, mood.value, indicators)
            
                # Generate meme phrase - use the generic method for all tokens
                meme_context = MemePhraseGenerator.generate_meme_phrase(
                    chain=token,
                    mood=Mood(mood.value)
                )
            
                # Get volume trend for additional context
                historical_volume = self._get_historical_volume_data(token, timeframe=timeframe)
                if historical_volume:
                    volume_change_pct, trend = self._analyze_volume_trend(
                        token_data['volume'],
                        historical_volume,
                        timeframe=timeframe
                    )
                    volume_trend = {
                        'change_pct': volume_change_pct,
                        'trend': trend
                    }
                else:
                    volume_trend = {'change_pct': 0, 'trend': 'stable'}

                # Get historical context from database - adjust time window based on timeframe
                hours_window = 24
                if timeframe == "24h":
                    hours_window = 7 * 24  # 7 days of context
                elif timeframe == "7d":
                    hours_window = 30 * 24  # 30 days of context
                
                stats = self.config.db.get_chain_stats(token, hours=hours_window)
            
                # Format the historical context based on timeframe
                if stats:
                    if timeframe == "1h":
                        historical_context = f"24h Avg: ${stats['avg_price']:,.2f}, "
                        historical_context += f"High: ${stats['max_price']:,.2f}, "
                        historical_context += f"Low: ${stats['min_price']:,.2f}"
                    elif timeframe == "24h":
                        historical_context = f"7d Avg: ${stats['avg_price']:,.2f}, "
                        historical_context += f"7d High: ${stats['max_price']:,.2f}, "
                        historical_context += f"7d Low: ${stats['min_price']:,.2f}"
                    else:  # 7d
                        historical_context = f"30d Avg: ${stats['avg_price']:,.2f}, "
                        historical_context += f"30d High: ${stats['max_price']:,.2f}, "
                        historical_context += f"30d Low: ${stats['min_price']:,.2f}"
                else:
                    historical_context = "No historical data"
            
                # Check if this is a volume trend trigger
                volume_context = ""
                if "volume_trend" in trigger_type:
                    change = volume_trend['change_pct']
                    direction = "increase" if change > 0 else "decrease"
                    time_period = "hour" if timeframe == "1h" else "day" if timeframe == "24h" else "week"
                    volume_context = f"\nVolume Analysis:\n{token} showing {abs(change):.1f}% {direction} in volume over last {time_period}. This is a significant {volume_trend['trend']}."

                # Smart money context - adjust based on timeframe
                smart_money_context = ""
                if smart_money.get('abnormal_volume'):
                    smart_money_context += f"\nAbnormal volume detected: {smart_money['volume_z_score']:.1f} standard deviations from mean."
                if smart_money.get('stealth_accumulation'):
                    smart_money_context += f"\nPotential stealth accumulation detected with minimal price movement and elevated volume."
                if smart_money.get('volume_cluster_detected'):
                    smart_money_context += f"\nVolume clustering detected, suggesting possible institutional activity."
                if smart_money.get('unusual_trading_hours'):
                    smart_money_context += f"\nUnusual trading hours detected: {', '.join(smart_money['unusual_trading_hours'])}."
                
                # Add pattern metrics for longer timeframes
                if timeframe in ["24h", "7d"] and 'pattern_metrics' in smart_money:
                    pattern_metrics = smart_money['pattern_metrics']
                    if pattern_metrics.get('volume_breakout', False):
                        smart_money_context += f"\nVolume breakout pattern detected, suggesting potential trend continuation."
                    if pattern_metrics.get('consistent_high_volume', False):
                        smart_money_context += f"\nConsistent high volume days detected, indicating sustained interest."

                # Market comparison context
                market_context = ""
                if vs_market.get('outperforming_market'):
                    market_context += f"\n{token} outperforming market average by {vs_market['vs_market_avg_change']:.1f}%"
                else:
                    market_context += f"\n{token} underperforming market average by {abs(vs_market['vs_market_avg_change']):.1f}%"
                
                # Add extended metrics for longer timeframes
                if timeframe in ["24h", "7d"] and 'extended_metrics' in vs_market:
                    extended = vs_market['extended_metrics']
                    if 'btc_dominance' in extended:
                        market_context += f"\nBTC Dominance: {extended['btc_dominance']:.1f}%"
                    if 'relative_volatility' in extended:
                        rel_vol = extended['relative_volatility']
                        vol_desc = "more" if rel_vol > 1 else "less"
                        market_context += f"\n{token} is {rel_vol:.1f}x {vol_desc} volatile than market average"
            
                # Market volume flow technical analysis
                reference_tokens = [t for t in self.reference_tokens if t != token and t in market_data]
                market_total_volume = sum([data['volume'] for sym, data in market_data.items() if sym in reference_tokens])
                market_volume_ratio = (token_data['volume'] / market_total_volume * 100) if market_total_volume > 0 else 0
            
                capital_rotation = "Yes" if vs_market.get('outperforming_market', False) and smart_money.get('volume_vs_daily_avg', 0) > 0.2 else "No"
            
                selling_pattern = "Detected" if vs_market.get('vs_market_volume_growth', 0) < 0 and volume_trend['change_pct'] > 5 else "Not detected"
            
                # Find top 2 correlated tokens
                price_correlations = {k.replace('price_correlation_', ''): v 
                                     for k, v in correlations.items() 
                                     if k.startswith('price_correlation_')}
                top_correlated = sorted(price_correlations.items(), key=lambda x: x[1], reverse=True)[:2]
            
                technical_context = f"""
Market Flow Analysis:
- {token}/Market volume ratio: {market_volume_ratio:.2f}%
- Potential capital rotation: {capital_rotation}
- Market selling {token} buying patterns: {selling_pattern}
"""
                if top_correlated:
                    technical_context += "- Highest correlations: "
                    for corr_token, corr_value in top_correlated:
                        technical_context += f"{corr_token}: {corr_value:.2f}, "
                    technical_context = technical_context.rstrip(", ")

                # Select a focus area using a deterministic but varied approach
                # Use a combination of date, hour, token, timeframe and trigger type to ensure variety
                focus_seed = f"{datetime.now().date()}_{datetime.now().hour}_{token}_{timeframe}_{trigger_type}"
                focus_index = hash(focus_seed) % len(focus_areas)
                selected_focus = focus_areas[focus_index]

                # Get timeframe-specific guidance
                timeframe_guide = timeframe_guidance.get(timeframe, "Focus on immediate market conditions and opportunities.")
            
                # Set character limits based on timeframe
                char_limit = char_limits.get(timeframe, "260-275")
                target_char = target_chars.get(timeframe, 270)

                # Add timeframe prefix to prompt if needed
                timeframe_prefix = ""
                if timeframe == "24h":
                    timeframe_prefix = "24H ANALYSIS: "
                elif timeframe == "7d":
                    timeframe_prefix = "7DAY OUTLOOK: "

                prompt = f"""Write a witty {timeframe} market analysis focusing on {token} token with attention to volume changes and smart money movements. Format as a single paragraph.

IMPORTANT: 
1. The analysis MUST be between {char_limit} characters long. Target exactly {target_char} characters. This is a STRICT requirement.
2. Always use #{token} instead of {token} when referring to the token in your analysis. This is critical!
3. Do NOT use any emojis or special Unicode characters. Stick to basic ASCII and standard punctuation only!
4. End with a complete sentence and a proper punctuation mark (., !, or ?). Make sure your final sentence is complete.
5. Count your characters carefully before submitting!
6. {timeframe_guide}
7. {timeframe_prefix}If creating a {timeframe} analysis, you may begin with "{timeframe_prefix}" but this is optional.

Market data:
                
{token} Performance:
- Price: ${token_data['current_price']:,.4f}
- 24h Change: {token_mood['change']:.1f}% ({token_mood['mood']})
- Volume: ${token_data['volume']:,.0f}
                
Historical Context:
- {token}: {historical_context}
                
Volume Analysis:
- {timeframe} trend: {volume_trend['change_pct']:.1f}% ({volume_trend['trend']})
- vs hourly avg: {smart_money.get('volume_vs_hourly_avg', 0)*100:.1f}%
- vs daily avg: {smart_money.get('volume_vs_daily_avg', 0)*100:.1f}%
{volume_context}
                
Smart Money Indicators:
- Volume Z-score: {smart_money.get('volume_z_score', 0):.2f}
- Price-Volume Divergence: {smart_money.get('price_volume_divergence', False)}
- Stealth Accumulation: {smart_money.get('stealth_accumulation', False)}
- Abnormal Volume: {smart_money.get('abnormal_volume', False)}
- Volume Clustering: {smart_money.get('volume_cluster_detected', False)}
{smart_money_context}
                
Market Comparison:
- vs Market avg change: {vs_market.get('vs_market_avg_change', 0):.1f}%
- vs Market volume growth: {vs_market.get('vs_market_volume_growth', 0):.1f}%
- Outperforming Market: {vs_market.get('outperforming_market', False)}
{market_context}
                
ATH Distance:
- {token}: {token_mood['ath_distance']:.1f}%
                
{technical_context}
                
Token-specific context:
- Meme: {meme_context}
                
Trigger Type: {trigger_type}
                
Past Context: {callback if callback else 'None'}
                
Note: {selected_focus} Keep the analysis fresh and varied. Avoid repetitive phrases."""
            
                logger.logger.debug(f"Sending {timeframe} analysis request to Claude")
                response = self.llm_provider.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
            
                analysis = response.content[0].text
                logger.logger.debug(f"Received {timeframe} analysis from Claude")
            
                # Store prediction data
                prediction_data = {
                    'analysis': analysis,
                    'sentiment': {token: token_mood['mood']},
                    **{f"{sym.upper()}_price": data['current_price'] for sym, data in market_data.items()}
                }
                self._track_prediction(token, prediction_data, [token], timeframe=timeframe)
            
                formatted_tweet = self._format_tweet_analysis(token, analysis, market_data, timeframe=timeframe)
            
                # Create the storage data to be stored later (after duplicate check)
                storage_data = {
                    'content': formatted_tweet,
                    'sentiment': {token: token_mood},
                    'trigger_type': trigger_type,
                    'price_data': {token: {'price': token_data['current_price'], 
                                         'volume': token_data['volume']}},
                    'meme_phrases': {token: meme_context},
                    'timeframe': timeframe
                }
            
                return formatted_tweet, storage_data
            
            except Exception as e:
                retry_count += 1
                wait_time = retry_count * 10
                logger.logger.error(f"{timeframe} analysis error details: {str(e)}", exc_info=True)
                logger.logger.warning(f"{timeframe} analysis error, attempt {retry_count}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
    
        logger.log_error(f"Market Analysis - {timeframe}", "Maximum retries reached")
        return None, None

    def _should_post_update(self, token: str, new_data: Dict[str, Any], timeframe: str = "1h") -> Tuple[bool, str]:
        """
        Determine if we should post an update based on market changes for a specific timeframe
        Returns (should_post, trigger_reason)
        """
        if not self.last_market_data:
            self.last_market_data = new_data
            return True, f"initial_post_{timeframe}"

        trigger_reason = None

        # Check token for significant changes
        if token in new_data and token in self.last_market_data:
            # Get timeframe-specific thresholds
            thresholds = self.timeframe_thresholds.get(timeframe, self.timeframe_thresholds["1h"])
        
            # Calculate immediate price change since last check
            price_change = abs(
                (new_data[token]['current_price'] - self.last_market_data[token]['current_price']) /
                self.last_market_data[token]['current_price'] * 100
            )
        
            # Calculate immediate volume change since last check
            immediate_volume_change = abs(
                (new_data[token]['volume'] - self.last_market_data[token]['volume']) /
                self.last_market_data[token]['volume'] * 100
            )

            logger.logger.debug(
                f"{token} immediate changes ({timeframe}) - "
                f"Price: {price_change:.2f}%, Volume: {immediate_volume_change:.2f}%"
            )

            # Check immediate price change against timeframe threshold
            price_threshold = thresholds["price_change"]
            if price_change >= price_threshold:
                trigger_reason = f"price_change_{token.lower()}_{timeframe}"
                logger.logger.info(
                    f"Significant price change detected for {token} ({timeframe}): "
                    f"{price_change:.2f}% (threshold: {price_threshold}%)"
                )
            # Check immediate volume change against timeframe threshold
            else:
                volume_threshold = thresholds["volume_change"]
                if immediate_volume_change >= volume_threshold:
                    trigger_reason = f"volume_change_{token.lower()}_{timeframe}"
                    logger.logger.info(
                        f"Significant immediate volume change detected for {token} ({timeframe}): "
                        f"{immediate_volume_change:.2f}% (threshold: {volume_threshold}%)"
                )
                # Check rolling window volume trend
                else:
                    historical_volume = self._get_historical_volume_data(token, timeframe=timeframe)
                    if historical_volume:
                        volume_change_pct, trend = self._analyze_volume_trend(
                            new_data[token]['volume'],
                            historical_volume,
                            timeframe=timeframe
                        )
                
                    # Log the volume trend
                    logger.logger.debug(
                        f"{token} rolling window volume trend ({timeframe}): {volume_change_pct:.2f}% ({trend})"
                    )
                
                    # Check if trend is significant enough to trigger
                    if trend in ["significant_increase", "significant_decrease"]:
                        trigger_reason = f"volume_trend_{token.lower()}_{trend}_{timeframe}"
                        logger.logger.info(
                            f"Significant volume trend detected for {token} ({timeframe}): "
                            f"{volume_change_pct:.2f}% - {trend}"
                        )
        
            # Check for smart money indicators
            if not trigger_reason:
                smart_money = self._analyze_smart_money_indicators(token, new_data[token], timeframe=timeframe)
                if smart_money.get('abnormal_volume') or smart_money.get('stealth_accumulation'):
                    trigger_reason = f"smart_money_{token.lower()}_{timeframe}"
                    logger.logger.info(f"Smart money movement detected for {token} ({timeframe})")
                
                # Check for pattern metrics in longer timeframes
                elif timeframe in ["24h", "7d"] and 'pattern_metrics' in smart_money:
                    pattern_metrics = smart_money['pattern_metrics']
                    if pattern_metrics.get('volume_breakout', False) or pattern_metrics.get('consistent_high_volume', False):
                        trigger_reason = f"pattern_metrics_{token.lower()}_{timeframe}"
                        logger.logger.info(f"Advanced pattern metrics detected for {token} ({timeframe})")
        
            # Check for significant outperformance vs market
            if not trigger_reason:
                vs_market = self._analyze_token_vs_market(token, new_data, timeframe=timeframe)
                outperformance_threshold = 3.0 if timeframe == "1h" else 5.0 if timeframe == "24h" else 8.0
            
                if vs_market.get('outperforming_market') and abs(vs_market.get('vs_market_avg_change', 0)) > outperformance_threshold:
                    trigger_reason = f"{token.lower()}_outperforming_market_{timeframe}"
                    logger.logger.info(f"{token} significantly outperforming market ({timeframe})")
                
                # Check if we need to post prediction update
                # Trigger prediction post based on time since last prediction
                if not trigger_reason:
                    # Check when the last prediction was posted
                    last_prediction = self.config.db.get_active_predictions(token=token, timeframe=timeframe)
                    if not last_prediction:
                        # No recent predictions for this timeframe, should post one
                        trigger_reason = f"prediction_needed_{token.lower()}_{timeframe}"
                        logger.logger.info(f"No recent {timeframe} prediction for {token}, triggering prediction post")

        # Check if regular interval has passed (only for 1h timeframe)
        if not trigger_reason and timeframe == "1h":
            time_since_last = (datetime.now() - self.last_check_time).total_seconds()
            if time_since_last >= self.config.BASE_INTERVAL:
                trigger_reason = f"regular_interval_{timeframe}"
                logger.logger.debug(f"Regular interval check triggered for {timeframe}")

        should_post = trigger_reason is not None
        if should_post:
            self.last_market_data = new_data
            logger.logger.info(f"Update triggered by: {trigger_reason}")
        else:
            logger.logger.debug(f"No {timeframe} triggers activated for {token}, skipping update")

        return should_post, trigger_reason

    def _evaluate_expired_predictions(self) -> None:
        """
        Find and evaluate expired predictions across all timeframes
        """
        try:
            # Get expired unevaluated predictions for all timeframes
            expired_predictions = self.config.db.get_expired_unevaluated_predictions()
        
            if not expired_predictions:
                logger.logger.debug("No expired predictions to evaluate")
                return
            
            # Group by timeframe
            expired_by_timeframe = {tf: [] for tf in self.timeframes}
        
            for prediction in expired_predictions:
                timeframe = prediction.get("timeframe", "1h")
                if timeframe in expired_by_timeframe:
                    expired_by_timeframe[timeframe].append(prediction)
        
            # Log count of expired predictions by timeframe
            for timeframe, preds in expired_by_timeframe.items():
                if preds:
                    logger.logger.info(f"Found {len(preds)} expired {timeframe} predictions to evaluate")
            
            # Get current market data for evaluation
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to fetch market data for prediction evaluation")
                return
            
            # Track evaluated counts
            evaluated_counts = {tf: 0 for tf in self.timeframes}
            
            # Evaluate each prediction by timeframe
            for timeframe, predictions in expired_by_timeframe.items():
                for prediction in predictions:
                    token = prediction["token"]
                    prediction_id = prediction["id"]
                    
                    # Get current price for the token
                    token_data = market_data.get(token, {})
                    if not token_data:
                        logger.logger.warning(f"No current price data for {token}, skipping evaluation")
                        continue
                        
                    current_price = token_data.get("current_price", 0)
                    if current_price == 0:
                        logger.logger.warning(f"Zero price for {token}, skipping evaluation")
                        continue
                        
                    # Record the outcome
                    result = self.config.db.record_prediction_outcome(prediction_id, current_price)
                    
                    if result:
                        logger.logger.debug(f"Evaluated {timeframe} prediction {prediction_id} for {token}")
                        evaluated_counts[timeframe] += 1
                    else:
                        logger.logger.error(f"Failed to evaluate {timeframe} prediction {prediction_id} for {token}")
            
            # Log evaluation summaries
            for timeframe, count in evaluated_counts.items():
                if count > 0:
                    logger.logger.info(f"Evaluated {count} expired {timeframe} predictions")
            
            # Update prediction performance metrics
            self._update_prediction_performance_metrics()
            
        except Exception as e:
            logger.log_error("Evaluate Expired Predictions", str(e))

    def _generate_weekly_summary(self) -> bool:
        """Generate and post a weekly summary of predictions and performance across all timeframes"""
        try:
            # Check if it's Sunday (weekday 6) and around midnight
            now = datetime.now()
            if now.weekday() != 6 or now.hour != 0:
                return False
                
            # Get performance stats for all timeframes
            overall_stats = {}
            for timeframe in self.timeframes:
                performance_stats = self.config.db.get_prediction_performance(timeframe=timeframe)
                
                if not performance_stats:
                    continue
                    
                # Calculate overall stats for this timeframe
                total_correct = sum(p["correct_predictions"] for p in performance_stats)
                total_predictions = sum(p["total_predictions"] for p in performance_stats)
                
                if total_predictions > 0:
                    overall_accuracy = (total_correct / total_predictions) * 100
                    overall_stats[timeframe] = {
                        "accuracy": overall_accuracy,
                        "total": total_predictions,
                        "correct": total_correct
                    }
                    
                    # Get token-specific stats
                    token_stats = {}
                    for stat in performance_stats:
                        token = stat["token"]
                        if stat["total_predictions"] > 0:
                            token_stats[token] = {
                                "accuracy": stat["accuracy_rate"],
                                "total": stat["total_predictions"]
                            }
                    
                    # Sort tokens by accuracy
                    sorted_tokens = sorted(token_stats.items(), key=lambda x: x[1]["accuracy"], reverse=True)
                    overall_stats[timeframe]["top_tokens"] = sorted_tokens[:3]
                    overall_stats[timeframe]["bottom_tokens"] = sorted_tokens[-3:] if len(sorted_tokens) >= 3 else []
            
            if not overall_stats:
                return False
                
            # Generate report
            report = "📊 WEEKLY PREDICTION SUMMARY 📊\n\n"
            
            # Add summary for each timeframe
            for timeframe, stats in overall_stats.items():
                if timeframe == "1h":
                    display_tf = "1 HOUR"
                elif timeframe == "24h":
                    display_tf = "24 HOUR"
                else:  # 7d
                    display_tf = "7 DAY"
                    
                report += f"== {display_tf} PREDICTIONS ==\n"
                report += f"Overall Accuracy: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})\n\n"
                
                if stats.get("top_tokens"):
                    report += "Top Performers:\n"
                    for token, token_stats in stats["top_tokens"]:
                        report += f"#{token}: {token_stats['accuracy']:.1f}% ({token_stats['total']} predictions)\n"
                        
                if stats.get("bottom_tokens"):
                    report += "\nBottom Performers:\n"
                    for token, token_stats in stats["bottom_tokens"]:
                        report += f"#{token}: {token_stats['accuracy']:.1f}% ({token_stats['total']} predictions)\n"
                        
                report += "\n"
                
            # Ensure report isn't too long
            max_length = self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']
            if len(report) > max_length:
                # Truncate report intelligently
                sections = report.split("==")
                shortened_report = sections[0]  # Keep header
                
                # Add as many sections as will fit
                for section in sections[1:]:
                    if len(shortened_report + "==" + section) <= max_length:
                        shortened_report += "==" + section
                    else:
                        break
                        
                report = shortened_report
            
            # Post the weekly summary
            return self._post_analysis(report, timeframe="summary")
            
        except Exception as e:
            logger.log_error("Weekly Summary", str(e))
            return False

    def _prioritize_tokens(self, available_tokens: List[str], market_data: Dict[str, Any]) -> List[str]:
        """Prioritize tokens across all timeframes based on momentum score and other factors"""
        try:
            token_priorities = []
        
            for token in available_tokens:
                # Calculate token-specific priority scores for each timeframe
                priority_scores = {}
                for timeframe in self.timeframes:
                    # Calculate momentum score for this timeframe
                    momentum_score = self._calculate_momentum_score(token, market_data, timeframe=timeframe)
                
                    # Get latest prediction time for this token and timeframe
                    last_prediction = self.config.db.get_active_predictions(token=token, timeframe=timeframe)
                    hours_since_prediction = 24  # Default high value
                
                    if last_prediction:
                        last_time = datetime.fromisoformat(last_prediction[0]["timestamp"])
                        hours_since_prediction = (datetime.now() - last_time).total_seconds() / 3600
                
                    # Scale time factor based on timeframe
                    if timeframe == "1h":
                        time_factor = 2.0  # Regular weight for 1h
                    elif timeframe == "24h":
                        time_factor = 0.5  # Lower weight for 24h
                    else:  # 7d
                        time_factor = 0.1  # Lowest weight for 7d
                        
                    # Priority score combines momentum and time since last prediction
                    priority_scores[timeframe] = momentum_score + (hours_since_prediction * time_factor)
                
                # Combined score is weighted average across all timeframes with focus on shorter timeframes
                combined_score = (
                    priority_scores.get("1h", 0) * 0.6 +
                    priority_scores.get("24h", 0) * 0.3 +
                    priority_scores.get("7d", 0) * 0.1
                )
                
                token_priorities.append((token, combined_score))
        
            # Sort by priority score (highest first)
            sorted_tokens = [t[0] for t in sorted(token_priorities, key=lambda x: x[1], reverse=True)]
        
            return sorted_tokens
        
        except Exception as e:
            logger.log_error("Token Prioritization", str(e))
            return available_tokens  # Return original list on error

    def _generate_predictions(self, token: str, market_data: Dict[str, Any], timeframe: str = "1h") -> Dict[str, Any]:
        """
        Generate market predictions for a specific token at a specific timeframe
        """
        try:
            logger.logger.info(f"Generating {timeframe} predictions for {token}")
        
            # Fix: Add try/except to handle the max() arg is an empty sequence error
            try:
                # Generate prediction for the specified timeframe
                prediction = self.prediction_engine.generate_prediction(
                    token=token,
                    market_data=market_data,
                    timeframe=timeframe
                )
            except ValueError as ve:
                # Handle the empty sequence error specifically
                if "max() arg is an empty sequence" in str(ve):
                    logger.logger.warning(f"Empty sequence error for {token} ({timeframe}), using fallback prediction")
                    # Create a basic fallback prediction
                    token_data = market_data.get(token, {})
                    current_price = token_data.get('current_price', 0)
                
                    # Adjust fallback values based on timeframe
                    if timeframe == "1h":
                        change_pct = 0.5
                        confidence = 60
                        range_factor = 0.01
                    elif timeframe == "24h":
                        change_pct = 1.2
                        confidence = 55
                        range_factor = 0.025
                    else:  # 7d
                        change_pct = 2.5
                        confidence = 50
                        range_factor = 0.05
                
                    prediction = {
                        "prediction": {
                            "price": current_price * (1 + change_pct/100),
                            "confidence": confidence,
                            "lower_bound": current_price * (1 - range_factor),
                            "upper_bound": current_price * (1 + range_factor),
                            "percent_change": change_pct,
                            "timeframe": timeframe
                        },
                        "rationale": f"Technical analysis based on recent price action for {token} over the {timeframe} timeframe.",
                        "sentiment": "NEUTRAL",
                        "key_factors": ["Technical analysis", "Recent price action", "Market conditions"]
                    }
                else:
                    # Re-raise other ValueError exceptions
                    raise
        
            # Store prediction in database
            prediction_id = self.config.db.store_prediction(token, prediction, timeframe=timeframe)
            logger.logger.info(f"Stored {token} {timeframe} prediction with ID {prediction_id}")
        
            return prediction
        
        except Exception as e:
            logger.log_error(f"Generate Predictions - {token} ({timeframe})", str(e))
            return {}

    # NEW METHODS FOR REPLY FUNCTIONALITY

    def _check_for_reply_opportunities(self, market_data: Dict[str, Any]) -> bool:
        """
        Check for opportunities to reply to other users' posts
        Returns True if any replies were made
        """
        now = datetime.now()
        
        # Check if it's time to look for posts to reply to
        time_since_last_check = (now - self.last_reply_check).total_seconds() / 60
        if time_since_last_check < self.reply_check_interval:
            logger.logger.debug(f"Skipping reply check, {time_since_last_check:.1f} minutes since last check (interval: {self.reply_check_interval})")
            return False
            
        # Also check cooldown period
        time_since_last_reply = (now - self.last_reply_time).total_seconds() / 60
        if time_since_last_reply < self.reply_cooldown:
            logger.logger.debug(f"In reply cooldown period, {time_since_last_reply:.1f} minutes since last reply (cooldown: {self.reply_cooldown})")
            return False
            
        logger.logger.info("Starting check for posts to reply to")
        self.last_reply_check = now
        
        try:
            # Call the reply handler to handle the process
            return self._check_for_posts_to_reply(market_data)
            
        except Exception as e:
            logger.log_error("Reply Opportunity Check", str(e))
            return False

    def _run_analysis_cycle(self) -> None:
        """Run analysis and posting cycle for all tokens with multi-timeframe prediction integration"""
        try:
            # First, evaluate any expired predictions
            self._evaluate_expired_predictions()
        
            # Get market data
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to fetch market data")
                return
            
            # Get available tokens
            available_tokens = [token for token in self.reference_tokens if token in market_data]
            if not available_tokens:
                logger.logger.error("No token data available")
                return
            
            # Initialize trigger_type with a default value to prevent NoneType errors
            trigger_type = "regular_interval"
            
            # Try scheduled timeframe rotation first - this handles 24h and 7d predictions
            if self._post_timeframe_rotation(market_data):
                logger.logger.info("Posted scheduled timeframe prediction in rotation")
                return
            
            # Check if we should check for reply opportunities first
            # We'll do this with approximately 30% probability to balance original posts and replies
            should_check_replies = True  # Always check replies for testing
            if should_check_replies:
                logger.logger.info("Checking for reply opportunities first")
                if self._check_for_reply_opportunities(market_data):
                    logger.logger.info("Successfully posted replies, skipping regular analysis cycle")
                    return
            
            # Prioritize tokens instead of just shuffling
            available_tokens = self._prioritize_tokens(available_tokens, market_data)
        
            # For 1h predictions and regular updates, try each token until we find one that's suitable
            for token_to_analyze in available_tokens:
                should_post, token_trigger_type = self._should_post_update(token_to_analyze, market_data, timeframe="1h")
            
                if should_post:
                    # Update the main trigger_type variable
                    trigger_type = token_trigger_type
                    logger.logger.info(f"Starting {token_to_analyze} analysis cycle - Trigger: {trigger_type}")
                
                    # Generate prediction for this token with 1h timeframe
                    prediction = self._generate_predictions(token_to_analyze, market_data, timeframe="1h")
                
                    if not prediction:
                        logger.logger.error(f"Failed to generate 1h prediction for {token_to_analyze}")
                        continue

                    # Get both standard analysis and prediction-focused content 
                    standard_analysis, storage_data = self._analyze_market_sentiment(
                        token_to_analyze, market_data, trigger_type, timeframe="1h"
                    )
                    prediction_tweet = self._format_prediction_tweet(token_to_analyze, prediction, market_data, timeframe="1h")
                
                    # Choose which type of content to post based on trigger and past posts
                    # For prediction-specific triggers or every third post, post prediction
                    should_post_prediction = (
                        "prediction" in trigger_type or 
                        random.random() < 0.35  # 35% chance of posting prediction instead of analysis
                    )
                
                    if should_post_prediction:
                        analysis_to_post = prediction_tweet
                        # Add prediction data to storage
                        if storage_data:
                            storage_data['is_prediction'] = True
                            storage_data['prediction_data'] = prediction
                    else:
                        analysis_to_post = standard_analysis
                        if storage_data:
                            storage_data['is_prediction'] = False
                
                    if not analysis_to_post:
                        logger.logger.error(f"Failed to generate content for {token_to_analyze}")
                        continue
                    
                    # Check for duplicates
                    last_posts = self._get_last_posts_by_timeframe(timeframe="1h")
                    if not self._is_duplicate_analysis(analysis_to_post, last_posts, timeframe="1h"):
                        if self._post_analysis(analysis_to_post, timeframe="1h"):
                            # Only store in database after successful posting
                            if storage_data:
                                self.config.db.store_posted_content(**storage_data)
                            
                            logger.logger.info(
                                f"Successfully posted {token_to_analyze} "
                                f"{'prediction' if should_post_prediction else 'analysis'} - "
                                f"Trigger: {trigger_type}"
                            )
                        
                            # Store additional smart money metrics
                            if token_to_analyze in market_data:
                                smart_money = self._analyze_smart_money_indicators(
                                    token_to_analyze, market_data[token_to_analyze], timeframe="1h"
                                )
                                self.config.db.store_smart_money_indicators(token_to_analyze, smart_money)
                            
                                # Store market comparison data
                                vs_market = self._analyze_token_vs_market(token_to_analyze, market_data, timeframe="1h")
                                if vs_market:
                                    self.config.db.store_token_market_comparison(
                                        token_to_analyze,
                                        vs_market.get('vs_market_avg_change', 0),
                                        vs_market.get('vs_market_volume_growth', 0),
                                        vs_market.get('outperforming_market', False),
                                        vs_market.get('correlations', {})
                                    )
                        
                            # Successfully posted, so we're done with this cycle
                            return
                        else:
                            logger.logger.error(f"Failed to post {token_to_analyze} {'prediction' if should_post_prediction else 'analysis'}")
                            continue  # Try next token
                    else:
                        logger.logger.info(f"Skipping duplicate {token_to_analyze} content - trying another token")
                        continue  # Try next token
                else:
                    logger.logger.debug(f"No significant {token_to_analyze} changes detected, trying another token")
        
            # If we couldn't find any token-specific update to post, 
            # try posting a correlation report on regular intervals
            if "regular_interval" in trigger_type:
                # Alternate between different timeframe correlation reports
                current_hour = datetime.now().hour
                report_timeframe = self.timeframes[current_hour % len(self.timeframes)]
                
                correlation_report = self._generate_correlation_report(market_data, timeframe=report_timeframe)
                if correlation_report and self._post_analysis(correlation_report, timeframe=report_timeframe):
                    logger.logger.info(f"Posted {report_timeframe} correlation matrix report")
                    return      

            # If still no post, try reply opportunities as a last resort
            if not should_check_replies:  # Only if we haven't checked replies already
                logger.logger.info("Checking for reply opportunities as fallback")
                if self._check_for_reply_opportunities(market_data):
                    logger.logger.info("Successfully posted replies as fallback")
                    return

            # If we get here, we tried all tokens but couldn't post anything
            logger.logger.warning("Tried all available tokens but couldn't post any analysis or replies")
            
        except Exception as e:
            logger.log_error("Token Analysis Cycle", str(e))

    def start(self) -> None:
        """Main bot execution loop with multi-timeframe support and reply functionality"""
        try:
            retry_count = 0
            max_setup_retries = 3
            
            # Start the prediction thread early
            self._start_prediction_thread()
            
            # Load saved timeframe state
            self._load_saved_timeframe_state()
            
            # Initialize the browser and login
            while retry_count < max_setup_retries:
                if not self.browser.initialize_driver():
                    retry_count += 1
                    logger.logger.warning(f"Browser initialization attempt {retry_count} failed, retrying...")
                    time.sleep(10)
                    continue
                    
                if not self._login_to_twitter():
                    retry_count += 1
                    logger.logger.warning(f"Twitter login attempt {retry_count} failed, retrying...")
                    time.sleep(15)
                    continue
                    
                break
            
            if retry_count >= max_setup_retries:
                raise Exception("Failed to initialize bot after maximum retries")

            logger.logger.info("Bot initialized successfully")
            
            # Log the timeframes that will be used
            logger.logger.info(f"Bot configured with timeframes: {', '.join(self.timeframes)}")
            logger.logger.info(f"Timeframe posting frequencies: {self.timeframe_posting_frequency}")
            logger.logger.info(f"Reply checking interval: {self.reply_check_interval} minutes")

            # Pre-queue predictions for all tokens and timeframes
            market_data = self._get_crypto_data()
            if market_data:
                available_tokens = [token for token in self.reference_tokens if token in market_data]
                
                # Only queue predictions for the most important tokens to avoid overloading
                top_tokens = self._prioritize_tokens(available_tokens, market_data)[:5]
                
                logger.logger.info(f"Pre-queueing predictions for top tokens: {', '.join(top_tokens)}")
                for token in top_tokens:
                    self._queue_predictions_for_all_timeframes(token, market_data)

            while True:
                try:
                    self._run_analysis_cycle()
                    
                    # Calculate sleep time until next regular check
                    time_since_last = (datetime.now() - self.last_check_time).total_seconds()
                    sleep_time = max(0, self.config.BASE_INTERVAL - time_since_last)
                    
                    # Check if we should post a weekly summary
                    if self._generate_weekly_summary():
                        logger.logger.info("Posted weekly performance summary")   

                    logger.logger.debug(f"Sleeping for {sleep_time:.1f}s until next check")
                    time.sleep(sleep_time)
                    
                    self.last_check_time = datetime.now()
                    
                except Exception as e:
                    logger.log_error("Analysis Cycle", str(e), exc_info=True)
                    time.sleep(60)  # Shorter sleep on error
                    continue

        except KeyboardInterrupt:
            logger.logger.info("Bot stopped by user")
        except Exception as e:
            logger.log_error("Bot Execution", str(e))
        finally:
            self._cleanup()

if __name__ == "__main__":
    try:
        bot = CryptoAnalysisBot()
        bot.start()
    except Exception as e:
        logger.log_error("Bot Startup", str(e))                     
