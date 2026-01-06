#!/usr/bin/env python3
"""
Enhanced BRAF Decision Engine
Advanced decision-making with machine learning capabilities and improved accuracy
"""
import re
import json
import os
from urllib.parse import urlparse
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class EnhancedDecisionEngine:
    """Enhanced decision engine with learning capabilities"""
    
    def __init__(self, history_file: str = None):
        self.history_file = history_file or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'decision_history.json'
        )
        self.decision_history = self._load_history()
        
        # Enhanced domain rules
        self.browser_required_domains = {
            'spa-sites.com', 'react-app.com', 'angular-app.com', 'vue-app.com',
            'single-page-app.com', 'facebook.com', 'twitter.com', 'instagram.com',
            'linkedin.com', 'youtube.com', 'gmail.com', 'outlook.com'
        }
        
        self.http_preferred_domains = {
            'httpbin.org', 'jsonplaceholder.typicode.com', 'api.github.com',
            'news.ycombinator.com', 'example.com', 'wikipedia.org',
            'stackoverflow.com', 'reddit.com'
        }
        
        # Enhanced pattern recognition
        self.javascript_indicators = [
            r'#/',  # Hash routing (SPA) - STRONG indicator
            r'/app/',  # App routes
            r'/dashboard/',  # Dashboard interfaces
            r'/admin/',  # Admin panels
            r'/console/',  # Console interfaces
            r'/portal/',  # Portal interfaces
            r'\.aspx',  # ASP.NET pages (often dynamic)
            r'\.jsp',   # JSP pages
            r'\.php\?',  # PHP with parameters
        ]
        
        self.api_indicators = [
            r'\.json$',  # JSON endpoints
            r'/api/',    # API paths
            r'/rest/',   # REST APIs
            r'/graphql', # GraphQL endpoints
            r'/webhook', # Webhooks
            r'/v\d+/',   # Versioned APIs
        ]
        
        self.static_indicators = [
            r'\.html$', r'\.xml$', r'\.rss$', r'\.txt$',
            r'/feed', r'/sitemap', r'\.pdf$', r'\.doc$'
        ]

    def _load_history(self) -> Dict:
        """Load decision history for learning"""
        if not os.path.exists(self.history_file):
            return {'decisions': [], 'domain_performance': {}, 'pattern_performance': {}}
        
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load decision history: {e}")
            return {'decisions': [], 'domain_performance': {}, 'pattern_performance': {}}

    def _save_history(self):
        """Save decision history"""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(self.decision_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save decision history: {e}")

    def needs_browser(self, target: Dict) -> bool:
        """
        Enhanced decision engine with learning capabilities
        
        Args:
            target: Dictionary containing URL and optional metadata
            
        Returns:
            bool: True if browser scraping is recommended
        """
        url = target.get('url', '')
        domain = urlparse(url).netloc.lower()
        path = urlparse(url).path.lower()
        
        # Check for explicit preference
        if 'preferred_scraper' in target:
            preference = target['preferred_scraper'].lower()
            logger.info(f"[ENHANCED] Explicit preference: {preference} for {url}")
            return preference == 'browser'
        
        # Use historical performance data
        historical_decision = self._get_historical_decision(domain, url)
        if historical_decision is not None:
            logger.info(f"[ENHANCED] Historical data suggests: {'browser' if historical_decision else 'http'}")
            return historical_decision
        
        # Enhanced domain analysis
        domain_score = self._analyze_domain_enhanced(domain)
        path_score = self._analyze_path_enhanced(path)
        pattern_score = self._analyze_patterns_enhanced(url)
        complexity_score = self._calculate_enhanced_complexity(url)
        
        # Weighted decision scoring with pattern-focused weights for SPA detection
        total_score = (
            domain_score * 0.2 +      # Reduced domain weight
            pattern_score * 0.7 +     # Heavily increased pattern weight for SPA detection
            path_score * 0.08 +       # Minimal path weight
            complexity_score * 0.02   # Minimal complexity weight
        )
        
        decision = total_score > 0.5
        confidence = abs(total_score - 0.5) * 2  # Convert to 0-1 confidence
        
        logger.info(f"[ENHANCED] Decision for {url}: {'browser' if decision else 'http'} "
                   f"(score: {total_score:.3f}, confidence: {confidence:.3f})")
        
        # Record decision for learning
        self._record_decision(url, domain, decision, total_score, {
            'domain_score': domain_score,
            'pattern_score': pattern_score,
            'path_score': path_score,
            'complexity_score': complexity_score
        })
        
        return decision

    def _get_historical_decision(self, domain: str, url: str) -> Optional[bool]:
        """Get decision based on historical performance"""
        domain_perf = self.decision_history.get('domain_performance', {}).get(domain)
        
        if domain_perf and domain_perf.get('total_decisions', 0) >= 3:
            # Use historical data if we have enough samples
            browser_success = domain_perf.get('browser_success_rate', 0)
            http_success = domain_perf.get('http_success_rate', 0)
            
            if abs(browser_success - http_success) > 0.2:  # Significant difference
                return browser_success > http_success
        
        return None

    def _analyze_domain_enhanced(self, domain: str) -> float:
        """Enhanced domain analysis with scoring"""
        score = 0.5  # Neutral starting point
        
        # Explicit domain rules
        if domain in self.browser_required_domains:
            return 0.9
        if domain in self.http_preferred_domains:
            return 0.1
        
        # Subdomain analysis
        parts = domain.split('.')
        if len(parts) > 2:
            subdomain = parts[0]
            
            # Browser-indicating subdomains
            browser_subdomains = ['app', 'dashboard', 'admin', 'console', 'portal', 'panel']
            if subdomain in browser_subdomains:
                score += 0.3
            
            # API-indicating subdomains
            api_subdomains = ['api', 'rest', 'graphql', 'webhook']
            if subdomain in api_subdomains:
                score -= 0.3
        
        # Domain keywords
        browser_keywords = ['app', 'dashboard', 'admin', 'portal', 'console', 'panel']
        api_keywords = ['api', 'rest', 'graphql', 'webhook']
        
        for keyword in browser_keywords:
            if keyword in domain:
                score += 0.2
        
        for keyword in api_keywords:
            if keyword in domain:
                score -= 0.2
        
        # Social media and complex sites
        complex_sites = ['facebook', 'twitter', 'instagram', 'linkedin', 'youtube']
        for site in complex_sites:
            if site in domain:
                score += 0.4
        
        return max(0.0, min(1.0, score))

    def _analyze_path_enhanced(self, path: str) -> float:
        """Enhanced path analysis with scoring"""
        score = 0.5
        
        # Dynamic/interactive paths
        dynamic_paths = ['/dashboard', '/admin', '/app', '/console', '/panel', '/portal']
        for dp in dynamic_paths:
            if dp in path:
                score += 0.3
        
        # API paths
        api_paths = ['/api', '/rest', '/graphql', '/v1', '/v2', '/webhook']
        for ap in api_paths:
            if ap in path:
                score -= 0.3
        
        # Static content paths
        static_paths = ['/static', '/assets', '/public', '/content', '/docs']
        for sp in static_paths:
            if sp in path:
                score -= 0.2
        
        return max(0.0, min(1.0, score))

    def _analyze_patterns_enhanced(self, url: str) -> float:
        """Enhanced pattern analysis with scoring"""
        score = 0.5
        
        # JavaScript/SPA indicators
        for pattern in self.javascript_indicators:
            if re.search(pattern, url, re.IGNORECASE):
                if pattern == r'#/':  # Hash routing is VERY strong SPA indicator
                    score += 0.8  # Increased even more to ensure SPA detection
                else:
                    score += 0.2
        
        # API indicators
        for pattern in self.api_indicators:
            if re.search(pattern, url, re.IGNORECASE):
                score -= 0.3
        
        # Static content indicators
        for pattern in self.static_indicators:
            if re.search(pattern, url, re.IGNORECASE):
                score -= 0.2
        
        return max(0.0, min(1.0, score))

    def _calculate_enhanced_complexity(self, url: str) -> float:
        """Enhanced URL complexity calculation"""
        score = 0.0
        
        # Query parameters
        if '?' in url:
            query_params = url.split('?')[1].count('&') + 1
            score += min(query_params * 0.1, 0.3)
        
        # Hash fragments (strong SPA indicator)
        if '#' in url:
            score += 0.5
        
        # Path depth
        path_depth = url.count('/') - 2
        score += min(path_depth * 0.05, 0.2)
        
        # Special characters
        special_chars = sum(1 for c in url if c in '!@#$%^&*()+=[]{}|;:,.<>?')
        score += min(special_chars * 0.02, 0.1)
        
        # URL length (very long URLs often indicate complexity)
        if len(url) > 100:
            score += 0.1
        
        return min(score, 1.0)

    def _record_decision(self, url: str, domain: str, decision: bool, score: float, factors: Dict):
        """Record decision for learning"""
        decision_record = {
            'url': url,
            'domain': domain,
            'decision': decision,
            'score': score,
            'factors': factors,
            'timestamp': datetime.now().isoformat()
        }
        
        self.decision_history['decisions'].append(decision_record)
        
        # Keep only recent decisions (last 1000)
        if len(self.decision_history['decisions']) > 1000:
            self.decision_history['decisions'] = self.decision_history['decisions'][-1000:]
        
        self._save_history()

    def update_performance(self, url: str, scraper_used: str, success: bool, execution_time: float = None):
        """Update performance data for learning"""
        domain = urlparse(url).netloc.lower()
        
        # Initialize domain performance if not exists
        if domain not in self.decision_history['domain_performance']:
            self.decision_history['domain_performance'][domain] = {
                'browser_successes': 0,
                'browser_failures': 0,
                'http_successes': 0,
                'http_failures': 0,
                'browser_avg_time': 0,
                'http_avg_time': 0,
                'total_decisions': 0
            }
        
        perf = self.decision_history['domain_performance'][domain]
        
        # Update counts
        if scraper_used == 'browser':
            if success:
                perf['browser_successes'] += 1
            else:
                perf['browser_failures'] += 1
            
            # Update average time
            if execution_time:
                current_avg = perf.get('browser_avg_time', 0)
                total_browser = perf['browser_successes'] + perf['browser_failures']
                perf['browser_avg_time'] = ((current_avg * (total_browser - 1)) + execution_time) / total_browser
        
        else:  # http
            if success:
                perf['http_successes'] += 1
            else:
                perf['http_failures'] += 1
            
            # Update average time
            if execution_time:
                current_avg = perf.get('http_avg_time', 0)
                total_http = perf['http_successes'] + perf['http_failures']
                perf['http_avg_time'] = ((current_avg * (total_http - 1)) + execution_time) / total_http
        
        perf['total_decisions'] += 1
        
        # Calculate success rates
        browser_total = perf['browser_successes'] + perf['browser_failures']
        http_total = perf['http_successes'] + perf['http_failures']
        
        perf['browser_success_rate'] = perf['browser_successes'] / browser_total if browser_total > 0 else 0
        perf['http_success_rate'] = perf['http_successes'] / http_total if http_total > 0 else 0
        
        self._save_history()

    def get_domain_insights(self, domain: str = None) -> Dict:
        """Get performance insights for a domain or all domains"""
        if domain:
            return self.decision_history.get('domain_performance', {}).get(domain, {})
        else:
            return self.decision_history.get('domain_performance', {})

    def get_decision_explanation(self, target: Dict) -> Dict:
        """Get detailed explanation of decision with enhanced factors"""
        url = target.get('url', '')
        domain = urlparse(url).netloc.lower()
        path = urlparse(url).path.lower()
        
        decision = self.needs_browser(target)
        
        # Calculate individual scores
        domain_score = self._analyze_domain_enhanced(domain)
        path_score = self._analyze_path_enhanced(path)
        pattern_score = self._analyze_patterns_enhanced(url)
        complexity_score = self._calculate_enhanced_complexity(url)
        
        total_score = (domain_score * 0.4 + pattern_score * 0.3 + 
                      path_score * 0.2 + complexity_score * 0.1)
        
        explanation = {
            'url': url,
            'decision': 'browser' if decision else 'http',
            'confidence': abs(total_score - 0.5) * 2,
            'total_score': total_score,
            'factor_scores': {
                'domain_score': domain_score,
                'pattern_score': pattern_score,
                'path_score': path_score,
                'complexity_score': complexity_score
            },
            'historical_data': self._get_historical_decision(domain, url) is not None,
            'domain_insights': self.get_domain_insights(domain)
        }
        
        return explanation

# Global instance for backward compatibility
enhanced_engine = EnhancedDecisionEngine()

def needs_browser(target: Dict) -> bool:
    """Backward compatible function using enhanced engine"""
    return enhanced_engine.needs_browser(target)

def get_decision_explanation(target: Dict) -> Dict:
    """Backward compatible function using enhanced engine"""
    return enhanced_engine.get_decision_explanation(target)

def update_performance(url: str, scraper_used: str, success: bool, execution_time: float = None):
    """Update performance data for machine learning"""
    enhanced_engine.update_performance(url, scraper_used, success, execution_time)

if __name__ == "__main__":
    # Test enhanced decision engine
    test_cases = [
        {"url": "https://httpbin.org/html", "expected": False},
        {"url": "https://app.example.com/dashboard", "expected": True},
        {"url": "https://api.github.com/users", "expected": False},
        {"url": "https://example.com/#/spa-route", "expected": True},
        {"url": "https://dashboard.example.com", "expected": True},
        {"url": "https://example.com/api/data.json", "expected": False},
    ]
    
    print("🧠 Testing Enhanced Decision Engine")
    print("=" * 50)
    
    correct = 0
    for case in test_cases:
        decision = enhanced_engine.needs_browser(case)
        expected = case["expected"]
        
        if decision == expected:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        explanation = enhanced_engine.get_decision_explanation(case)
        
        print(f"{status} {case['url']}")
        print(f"   Decision: {'Browser' if decision else 'HTTP'}")
        print(f"   Expected: {'Browser' if expected else 'HTTP'}")
        print(f"   Confidence: {explanation['confidence']:.3f}")
        print(f"   Total Score: {explanation['total_score']:.3f}")
        print()
    
    accuracy = (correct / len(test_cases)) * 100
    print(f"📊 Enhanced Decision Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)")
