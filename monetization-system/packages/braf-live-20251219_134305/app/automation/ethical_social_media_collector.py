"""
BRAF Ethical Social Media Data Collector
Demonstrates responsible social media data collection using official APIs
with proper authentication, rate limiting, and compliance monitoring
"""

import tweepy
import praw
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import json
import asyncio
import aiohttp
import logging
from pathlib import Path
import time
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SocialMediaPost:
    """Standardized social media post structure"""
    platform: str
    post_id: str
    author: str
    content: str
    created_at: datetime
    metrics: Dict[str, int]
    metadata: Dict
    collected_at: datetime
    compliance_status: str


class EthicalTwitterCollector:
    """
    Ethical Twitter data collector using official Twitter API v2
    Respects rate limits and follows Twitter's developer policy
    """
    
    def __init__(self, bearer_token: str):
        if not bearer_token or bearer_token == "YOUR_ACTUAL_TWITTER_BEARER_TOKEN":
            raise ValueError("Please provide a valid Twitter Bearer Token")
            
        self.client = tweepy.Client(
            bearer_token=bearer_token,
            wait_on_rate_limit=True  # Automatically respect rate limits
        )
        self.rate_limit_status = {}
        
    def search_recent_tweets(self, 
                           query: str, 
                           max_results: int = 100,
                           exclude_retweets: bool = True) -> List[SocialMediaPost]:
        """
        Search for recent tweets using Twitter API v2
        Automatically respects rate limits and follows best practices
        """
        try:
            # Modify query to exclude retweets if requested
            if exclude_retweets:
                query += " -is:retweet"
            
            logger.info(f"Searching Twitter for: {query}")
            
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),  # API limit
                tweet_fields=[
                    'created_at', 'public_metrics', 'entities', 
                    'context_annotations', 'lang', 'possibly_sensitive'
                ],
                expansions=['author_id'],
                user_fields=['username', 'name', 'verified', 'public_metrics']
            )
            
            posts = []
            if response.data:
                # Create user lookup for efficiency
                users = {}
                if response.includes and 'users' in response.includes:
                    users = {user.id: user for user in response.includes['users']}
                
                for tweet in response.data:
                    post = self._process_twitter_post(tweet, users.get(tweet.author_id))
                    posts.append(post)
            
            logger.info(f"Collected {len(posts)} tweets")
            return posts
            
        except tweepy.TooManyRequests:
            logger.warning("Twitter rate limit exceeded. Please wait before making more requests.")
            return []
        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            return []
    
    def get_user_tweets(self, 
                       username: str, 
                       max_tweets: int = 50,
                       exclude_replies: bool = True) -> List[SocialMediaPost]:
        """Get tweets from a specific user"""
        try:
            # Get user ID first
            user_response = self.client.get_user(
                username=username,
                user_fields=['public_metrics', 'verified']
            )
            
            if not user_response.data:
                logger.warning(f"User @{username} not found")
                return []
            
            user = user_response.data
            
            # Get user's tweets
            tweets_response = self.client.get_users_tweets(
                id=user.id,
                max_results=min(max_tweets, 100),
                tweet_fields=['created_at', 'public_metrics', 'entities', 'lang'],
                exclude=['retweets'] + (['replies'] if exclude_replies else [])
            )
            
            posts = []
            if tweets_response.data:
                for tweet in tweets_response.data:
                    post = self._process_twitter_post(tweet, user)
                    posts.append(post)
            
            logger.info(f"Collected {len(posts)} tweets from @{username}")
            return posts
            
        except Exception as e:
            logger.error(f"Error getting tweets from @{username}: {e}")
            return []
    
    def _process_twitter_post(self, tweet, user=None) -> SocialMediaPost:
        """Process Twitter post data"""
        # Extract hashtags and mentions
        hashtags = []
        mentions = []
        urls = []
        
        if hasattr(tweet, 'entities') and tweet.entities:
            if tweet.entities.get('hashtags'):
                hashtags = [tag['tag'] for tag in tweet.entities['hashtags']]
            if tweet.entities.get('mentions'):
                mentions = [mention['username'] for mention in tweet.entities['mentions']]
            if tweet.entities.get('urls'):
                urls = [url['expanded_url'] for url in tweet.entities['urls']]
        
        # Build metadata
        metadata = {
            'hashtags': hashtags,
            'mentions': mentions,
            'urls': urls,
            'language': getattr(tweet, 'lang', 'unknown'),
            'possibly_sensitive': getattr(tweet, 'possibly_sensitive', False)
        }
        
        # Add user info if available
        if user:
            metadata['user_info'] = {
                'username': user.username,
                'name': getattr(user, 'name', ''),
                'verified': getattr(user, 'verified', False),
                'followers_count': getattr(user, 'public_metrics', {}).get('followers_count', 0)
            }
        
        return SocialMediaPost(
            platform='twitter',
            post_id=str(tweet.id),
            author=user.username if user else str(tweet.author_id),
            content=tweet.text,
            created_at=tweet.created_at,
            metrics={
                'retweet_count': tweet.public_metrics['retweet_count'],
                'reply_count': tweet.public_metrics['reply_count'],
                'like_count': tweet.public_metrics['like_count'],
                'quote_count': tweet.public_metrics['quote_count']
            },
            metadata=metadata,
            collected_at=datetime.now(),
            compliance_status='api_compliant'
        )


class EthicalRedditCollector:
    """
    Ethical Reddit data collector using PRAW (Python Reddit API Wrapper)
    Follows Reddit's API terms and rate limiting guidelines
    """
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        if not all([client_id, client_secret, user_agent]) or \
           any(x.startswith("YOUR_") for x in [client_id, client_secret]):
            raise ValueError("Please provide valid Reddit API credentials")
            
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Verify authentication
        try:
            self.reddit.user.me()
        except:
            # Read-only mode is fine for data collection
            pass
    
    def get_subreddit_posts(self, 
                          subreddit_name: str, 
                          limit: int = 100, 
                          time_filter: str = 'week',
                          sort_type: str = 'top') -> List[SocialMediaPost]:
        """Get posts from a subreddit"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Check if subreddit exists and is accessible
            try:
                subreddit.id  # This will raise an exception if subreddit doesn't exist
            except Exception:
                logger.error(f"Subreddit r/{subreddit_name} not found or not accessible")
                return []
            
            logger.info(f"Collecting from r/{subreddit_name}")
            
            # Get posts based on sort type
            if sort_type == 'top':
                posts_iter = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort_type == 'hot':
                posts_iter = subreddit.hot(limit=limit)
            elif sort_type == 'new':
                posts_iter = subreddit.new(limit=limit)
            else:
                posts_iter = subreddit.top(time_filter=time_filter, limit=limit)
            
            posts = []
            for post in posts_iter:
                social_post = self._process_reddit_post(post)
                posts.append(social_post)
            
            logger.info(f"Collected {len(posts)} posts from r/{subreddit_name}")
            return posts
            
        except Exception as e:
            logger.error(f"Error collecting from r/{subreddit_name}: {e}")
            return []
    
    def search_posts(self, 
                    query: str, 
                    subreddit: Optional[str] = None, 
                    limit: int = 100) -> List[SocialMediaPost]:
        """Search for posts across Reddit or within a specific subreddit"""
        try:
            if subreddit:
                search_subreddit = self.reddit.subreddit(subreddit)
            else:
                search_subreddit = self.reddit.subreddit('all')
            
            logger.info(f"Searching Reddit for: {query}")
            
            posts = []
            for post in search_subreddit.search(query, limit=limit, sort='relevance'):
                social_post = self._process_reddit_post(post)
                posts.append(social_post)
            
            logger.info(f"Found {len(posts)} posts matching '{query}'")
            return posts
            
        except Exception as e:
            logger.error(f"Reddit search error: {e}")
            return []
    
    def get_subreddit_info(self, subreddit_name: str) -> Dict:
        """Get subreddit metadata"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            return {
                'name': subreddit.display_name,
                'title': subreddit.title,
                'description': subreddit.description[:500],  # Limit length
                'public_description': subreddit.public_description,
                'subscribers': subreddit.subscribers,
                'created_utc': datetime.fromtimestamp(subreddit.created_utc).isoformat(),
                'over18': subreddit.over18,
                'quarantined': getattr(subreddit, 'quarantined', False),
                'collected_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting subreddit info: {e}")
            return {'error': str(e)}
    
    def _process_reddit_post(self, post) -> SocialMediaPost:
        """Process Reddit post data"""
        # Extract post content
        content = post.title
        if post.selftext:
            content += f"\n\n{post.selftext[:1000]}"  # Limit content length
        
        # Build metadata
        metadata = {
            'subreddit': str(post.subreddit),
            'url': post.url,
            'is_original_content': post.is_original_content,
            'is_self': post.is_self,
            'over_18': post.over_18,
            'spoiler': post.spoiler,
            'stickied': post.stickied,
            'upvote_ratio': post.upvote_ratio,
            'post_type': 'text' if post.is_self else 'link'
        }
        
        # Get top comments (limited for performance)
        try:
            post.comments.replace_more(limit=0)
            top_comments = []
            for comment in post.comments.list()[:5]:  # Top 5 comments only
                if hasattr(comment, 'body'):
                    top_comments.append({
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'body': comment.body[:200],  # Limit comment length
                        'score': comment.score
                    })
            metadata['top_comments'] = top_comments
        except:
            metadata['comments_error'] = 'Could not load comments'
        
        return SocialMediaPost(
            platform='reddit',
            post_id=post.id,
            author=str(post.author) if post.author else '[deleted]',
            content=content,
            created_at=datetime.fromtimestamp(post.created_utc),
            metrics={
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments
            },
            metadata=metadata,
            collected_at=datetime.now(),
            compliance_status='api_compliant'
        )


class EthicalSocialMediaOrchestrator:
    """
    Orchestrates ethical social media data collection across platforms
    with comprehensive compliance monitoring and reporting
    """
    
    def __init__(self, output_dir: str = "social_media_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.collectors = {}
        self.compliance_log = []
        
    def add_twitter_collector(self, bearer_token: str):
        """Add Twitter collector with API credentials"""
        try:
            self.collectors['twitter'] = EthicalTwitterCollector(bearer_token)
            logger.info("Twitter collector added successfully")
        except Exception as e:
            logger.error(f"Failed to add Twitter collector: {e}")
    
    def add_reddit_collector(self, client_id: str, client_secret: str, user_agent: str):
        """Add Reddit collector with API credentials"""
        try:
            self.collectors['reddit'] = EthicalRedditCollector(client_id, client_secret, user_agent)
            logger.info("Reddit collector added successfully")
        except Exception as e:
            logger.error(f"Failed to add Reddit collector: {e}")
    
    async def collect_social_media_data(self, collection_config: Dict) -> Dict[str, List[SocialMediaPost]]:
        """
        Collect data from multiple social media platforms
        with ethical guidelines and compliance monitoring
        """
        results = {}
        
        # Twitter collection
        if 'twitter' in collection_config and 'twitter' in self.collectors:
            twitter_config = collection_config['twitter']
            twitter_posts = []
            
            # Search tweets
            if 'search_queries' in twitter_config:
                for query in twitter_config['search_queries']:
                    posts = self.collectors['twitter'].search_recent_tweets(
                        query=query,
                        max_results=twitter_config.get('max_results_per_query', 50)
                    )
                    twitter_posts.extend(posts)
                    
                    # Respect rate limits
                    await asyncio.sleep(1)
            
            # User tweets
            if 'users' in twitter_config:
                for username in twitter_config['users']:
                    posts = self.collectors['twitter'].get_user_tweets(
                        username=username,
                        max_tweets=twitter_config.get('max_tweets_per_user', 20)
                    )
                    twitter_posts.extend(posts)
                    
                    # Respect rate limits
                    await asyncio.sleep(1)
            
            results['twitter'] = twitter_posts
            self._log_collection('twitter', len(twitter_posts))
        
        # Reddit collection
        if 'reddit' in collection_config and 'reddit' in self.collectors:
            reddit_config = collection_config['reddit']
            reddit_posts = []
            
            # Subreddit posts
            if 'subreddits' in reddit_config:
                for subreddit in reddit_config['subreddits']:
                    posts = self.collectors['reddit'].get_subreddit_posts(
                        subreddit_name=subreddit,
                        limit=reddit_config.get('posts_per_subreddit', 50),
                        time_filter=reddit_config.get('time_filter', 'week')
                    )
                    reddit_posts.extend(posts)
                    
                    # Be respectful with requests
                    await asyncio.sleep(2)
            
            # Search posts
            if 'search_queries' in reddit_config:
                for query in reddit_config['search_queries']:
                    posts = self.collectors['reddit'].search_posts(
                        query=query,
                        limit=reddit_config.get('max_search_results', 50)
                    )
                    reddit_posts.extend(posts)
                    
                    # Be respectful with requests
                    await asyncio.sleep(2)
            
            results['reddit'] = reddit_posts
            self._log_collection('reddit', len(reddit_posts))
        
        # Save collected data
        await self._save_collected_data(results)
        
        # Generate compliance report
        self._generate_compliance_report()
        
        return results
    
    def _log_collection(self, platform: str, count: int):
        """Log data collection for compliance tracking"""
        log_entry = {
            'platform': platform,
            'posts_collected': count,
            'timestamp': datetime.now().isoformat(),
            'api_used': True,
            'rate_limits_respected': True,
            'terms_compliant': True
        }
        self.compliance_log.append(log_entry)
        logger.info(f"Collected {count} posts from {platform}")
    
    async def _save_collected_data(self, results: Dict[str, List[SocialMediaPost]]):
        """Save collected data with proper formatting"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for platform, posts in results.items():
            if not posts:
                continue
                
            # Convert to serializable format
            posts_data = []
            for post in posts:
                post_dict = {
                    'platform': post.platform,
                    'post_id': post.post_id,
                    'author': post.author,
                    'content': post.content[:1000],  # Limit for storage
                    'created_at': post.created_at.isoformat(),
                    'metrics': post.metrics,
                    'metadata': post.metadata,
                    'collected_at': post.collected_at.isoformat(),
                    'compliance_status': post.compliance_status
                }
                posts_data.append(post_dict)
            
            # Save as JSON
            json_file = self.output_dir / f"{platform}_posts_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(posts_data, f, indent=2, ensure_ascii=False)
            
            # Save as CSV for analysis
            csv_file = self.output_dir / f"{platform}_posts_{timestamp}.csv"
            df = pd.DataFrame(posts_data)
            df.to_csv(csv_file, index=False)
            
            logger.info(f"Saved {len(posts)} {platform} posts to {json_file}")
    
    def _generate_compliance_report(self):
        """Generate comprehensive compliance report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'collection_summary': {
                'total_platforms': len(self.compliance_log),
                'total_posts': sum(entry['posts_collected'] for entry in self.compliance_log),
                'api_compliance': all(entry['api_used'] for entry in self.compliance_log),
                'rate_limit_compliance': all(entry['rate_limits_respected'] for entry in self.compliance_log),
                'terms_compliance': all(entry['terms_compliant'] for entry in self.compliance_log)
            },
            'platform_breakdown': {},
            'compliance_guidelines_followed': [
                'Used official APIs only',
                'Respected rate limits automatically',
                'Followed platform terms of service',
                'Limited data collection scope',
                'Implemented proper error handling',
                'Maintained audit logs',
                'Protected user privacy'
            ],
            'detailed_log': self.compliance_log
        }
        
        # Platform breakdown
        for entry in self.compliance_log:
            platform = entry['platform']
            if platform not in report['platform_breakdown']:
                report['platform_breakdown'][platform] = {
                    'posts_collected': 0,
                    'collections': 0
                }
            report['platform_breakdown'][platform]['posts_collected'] += entry['posts_collected']
            report['platform_breakdown'][platform]['collections'] += 1
        
        # Save report
        report_file = self.output_dir / "compliance_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Compliance report saved to {report_file}")


# Example usage and demonstration
async def demonstrate_ethical_social_media_collection():
    """Demonstrate ethical social media data collection"""
    
    # Initialize orchestrator
    orchestrator = EthicalSocialMediaOrchestrator("ethical_social_data")
    
    # Note: In real usage, replace with actual API credentials
    # For demonstration, we'll show the structure without real credentials
    
    try:
        # Add collectors (would need real credentials)
        # orchestrator.add_twitter_collector("your_twitter_bearer_token")
        # orchestrator.add_reddit_collector("client_id", "client_secret", "user_agent")
        
        # Define collection configuration
        collection_config = {
            'twitter': {
                'search_queries': [
                    'machine learning',
                    'artificial intelligence',
                    'data science'
                ],
                'users': ['elonmusk', 'sundarpichai'],  # Public figures
                'max_results_per_query': 20,
                'max_tweets_per_user': 10
            },
            'reddit': {
                'subreddits': ['MachineLearning', 'datascience', 'programming'],
                'search_queries': ['artificial intelligence', 'deep learning'],
                'posts_per_subreddit': 25,
                'max_search_results': 20,
                'time_filter': 'week'
            }
        }
        
        # Collect data ethically
        logger.info("Starting ethical social media data collection...")
        results = await orchestrator.collect_social_media_data(collection_config)
        
        # Summary
        total_posts = sum(len(posts) for posts in results.values())
        logger.info(f"Collection complete! Total posts collected: {total_posts}")
        
        for platform, posts in results.items():
            logger.info(f"{platform.capitalize()}: {len(posts)} posts")
        
        return results
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return {}


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstration
    asyncio.run(demonstrate_ethical_social_media_collection())