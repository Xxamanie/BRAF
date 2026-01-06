#!/usr/bin/env python3
"""
BRAF Decision Engine
Intelligent decision-making for scraper selection based on target analysis
"""
import re
from urllib.parse import urlparse
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# JavaScript-heavy domains that typically need browser scraping
BROWSER_REQUIRED_DOMAINS = {
    'spa-sites.com',
    'react-app.com',
    'angular-app.com',
    'vue-app.com',
    'single-page-app.com'
}

# Domains known to work well with HTTP scraping
HTTP_PREFERRED_DOMAINS = {
    'httpbin.org',
    'jsonplaceholder.typicode.com',
    'api.github.com',
    'news.ycombinator.com',
    'example.com'
}

# URL patterns that indicate JavaScript-heavy content
JAVASCRIPT_INDICATORS = [
    r'#/',  # Hash routing (SPA)
    r'/app/',  # App routes
    r'/dashboard/',  # Dashboard interfaces
    r'/admin/',  # Admin panels
    r'\.json$',  # JSON endpoints (better with HTTP)
    r'/api/',  # API endpoints
]

# Content type indicators
STATIC_CONTENT_PATTERNS = [
    r'\.html$',
    r'\.xml$',
    r'\.rss$',
    r'\.txt$',
    r'/feed',
    r'/sitemap'
]

def needs_browser(target: Dict) -> bool:
    """
    Intelligent decision engine to determine if browser scraping is needed
    
    Args:
        target: Dictionary containing URL and optional metadata
        
    Returns:
        bool: True if browser scraping is recommended, False for HTTP
    """
    url = target.get('url', '')
    domain = urlparse(url).netloc.lower()
    path = urlparse(url).path.lower()
    
    # Check for explicit preference in target
    if 'preferred_scraper' in target:
        preference = target['preferred_scraper'].lower()
        logger.info(f"[DECISION] Explicit preference: {preference} for {url}")
        return preference == 'browser'
    
    # Domain-based decisions (highest priority)
    if domain in BROWSER_REQUIRED_DOMAINS:
        logger.info(f"[DECISION] Browser required for domain: {domain}")
        return True
    
    if domain in HTTP_PREFERRED_DOMAINS:
        logger.info(f"[DECISION] HTTP preferred for domain: {domain}")
        return False
    
    # URL pattern analysis
    for pattern in JAVASCRIPT_INDICATORS:
        if re.search(pattern, url, re.IGNORECASE):
            if pattern in [r'\.json$', r'/api/']:
                logger.info(f"[DECISION] API/JSON endpoint detected: HTTP preferred")
                return False
            else:
                logger.info(f"[DECISION] JavaScript pattern detected ({pattern}): Browser needed")
                return True
    
    # Static content patterns
    for pattern in STATIC_CONTENT_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            logger.info(f"[DECISION] Static content pattern detected: HTTP preferred")
            return False
    
    # Domain-specific heuristics
    domain_indicators = analyze_domain(domain)
    if domain_indicators['needs_browser']:
        logger.info(f"[DECISION] Domain analysis suggests browser: {domain}")
        return True
    
    # Path-based analysis
    path_indicators = analyze_path(path)
    if path_indicators['needs_browser']:
        logger.info(f"[DECISION] Path analysis suggests browser: {path}")
        return True
    
    # Default decision based on URL complexity
    complexity_score = calculate_url_complexity(url)
    if complexity_score > 0.7:
        logger.info(f"[DECISION] High complexity score ({complexity_score:.2f}): Browser recommended")
        return True
    
    # Default to HTTP for better performance
    logger.info(f"[DECISION] Default to HTTP for: {url}")
    return False

def analyze_domain(domain: str) -> Dict:
    """Analyze domain characteristics"""
    indicators = {
        'needs_browser': False,
        'confidence': 0.5,
        'reasons': []
    }
    
    # Check for common SPA/JavaScript framework indicators
    js_keywords = ['app', 'dashboard', 'admin', 'portal', 'console', 'panel']
    for keyword in js_keywords:
        if keyword in domain:
            indicators['needs_browser'] = True
            indicators['confidence'] += 0.2
            indicators['reasons'].append(f"Domain contains '{keyword}'")
    
    # Check for API indicators
    api_keywords = ['api', 'rest', 'graphql', 'webhook']
    for keyword in api_keywords:
        if keyword in domain:
            indicators['needs_browser'] = False
            indicators['confidence'] += 0.3
            indicators['reasons'].append(f"API domain detected: '{keyword}'")
    
    # Social media and complex sites
    complex_sites = ['facebook', 'twitter', 'instagram', 'linkedin', 'youtube']
    for site in complex_sites:
        if site in domain:
            indicators['needs_browser'] = True
            indicators['confidence'] += 0.4
            indicators['reasons'].append(f"Complex social media site: '{site}'")
    
    return indicators

def analyze_path(path: str) -> Dict:
    """Analyze URL path characteristics"""
    indicators = {
        'needs_browser': False,
        'confidence': 0.5,
        'reasons': []
    }
    
    # Interactive/dynamic paths
    dynamic_paths = ['/dashboard', '/admin', '/app', '/console', '/panel', '/portal']
    for dp in dynamic_paths:
        if dp in path:
            indicators['needs_browser'] = True
            indicators['confidence'] += 0.3
            indicators['reasons'].append(f"Dynamic path detected: '{dp}'")
    
    # Static content paths
    static_paths = ['/static', '/assets', '/public', '/content', '/docs']
    for sp in static_paths:
        if sp in path:
            indicators['needs_browser'] = False
            indicators['confidence'] += 0.2
            indicators['reasons'].append(f"Static path detected: '{sp}'")
    
    # API paths
    api_paths = ['/api', '/rest', '/graphql', '/v1', '/v2', '/webhook']
    for ap in api_paths:
        if ap in path:
            indicators['needs_browser'] = False
            indicators['confidence'] += 0.4
            indicators['reasons'].append(f"API path detected: '{ap}'")
    
    return indicators

def calculate_url_complexity(url: str) -> float:
    """Calculate URL complexity score (0.0 to 1.0)"""
    score = 0.0
    
    # Query parameters increase complexity
    if '?' in url:
        query_params = url.split('?')[1].count('&') + 1
        score += min(query_params * 0.1, 0.3)
    
    # Hash fragments (SPA routing)
    if '#' in url:
        score += 0.4
    
    # Deep paths
    path_depth = url.count('/') - 2  # Subtract protocol slashes
    score += min(path_depth * 0.05, 0.2)
    
    # Special characters
    special_chars = sum(1 for c in url if c in '!@#$%^&*()+=[]{}|;:,.<>?')
    score += min(special_chars * 0.02, 0.1)
    
    return min(score, 1.0)

def get_decision_explanation(target: Dict) -> Dict:
    """Get detailed explanation of scraper decision"""
    url = target.get('url', '')
    domain = urlparse(url).netloc.lower()
    path = urlparse(url).path.lower()
    
    decision = needs_browser(target)
    
    explanation = {
        'url': url,
        'decision': 'browser' if decision else 'http',
        'confidence': 0.5,
        'factors': [],
        'domain_analysis': analyze_domain(domain),
        'path_analysis': analyze_path(path),
        'complexity_score': calculate_url_complexity(url)
    }
    
    # Add decision factors
    if domain in BROWSER_REQUIRED_DOMAINS:
        explanation['factors'].append('Domain requires browser')
        explanation['confidence'] = 0.9
    elif domain in HTTP_PREFERRED_DOMAINS:
        explanation['factors'].append('Domain works well with HTTP')
        explanation['confidence'] = 0.8
    
    # Pattern matching results
    for pattern in JAVASCRIPT_INDICATORS:
        if re.search(pattern, url, re.IGNORECASE):
            if pattern in [r'\.json$', r'/api/']:
                explanation['factors'].append(f'API/JSON pattern: {pattern}')
            else:
                explanation['factors'].append(f'JavaScript pattern: {pattern}')
    
    return explanation

def test_decision_engine():
    """Test the decision engine with various URLs"""
    test_cases = [
        # HTTP preferred cases
        {"url": "https://httpbin.org/html", "expected": False},
        {"url": "https://example.com", "expected": False},
        {"url": "https://api.github.com/users", "expected": False},
        {"url": "https://jsonplaceholder.typicode.com/posts/1", "expected": False},
        {"url": "https://news.ycombinator.com", "expected": False},
        
        # Browser required cases
        {"url": "https://app.example.com/dashboard", "expected": True},
        {"url": "https://example.com/admin/panel", "expected": True},
        {"url": "https://spa-sites.com/app", "expected": True},
        {"url": "https://example.com/#/dashboard", "expected": True},
        
        # Edge cases
        {"url": "https://example.com/api/data.json", "expected": False},
        {"url": "https://dashboard.example.com", "expected": True},
    ]
    
    print("🧪 Testing Decision Engine")
    print("=" * 40)
    
    correct = 0
    total = len(test_cases)
    
    for case in test_cases:
        decision = needs_browser(case)
        expected = case["expected"]
        status = "✅" if decision == expected else "❌"
        
        print(f"{status} {case['url']}")
        print(f"   Decision: {'Browser' if decision else 'HTTP'}")
        print(f"   Expected: {'Browser' if expected else 'HTTP'}")
        
        if decision == expected:
            correct += 1
        
        # Show explanation for failed cases
        if decision != expected:
            explanation = get_decision_explanation(case)
            print(f"   Factors: {', '.join(explanation['factors'])}")
        
        print()
    
    accuracy = (correct / total) * 100
    print(f"📊 Accuracy: {correct}/{total} ({accuracy:.1f}%)")

if __name__ == "__main__":
    test_decision_engine()
