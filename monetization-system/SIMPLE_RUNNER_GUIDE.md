# BRAF Simple Runner Guide

## Quick Start

Your exact code works perfectly with the enhanced BRAF system:

```python
from core.runner import run_targets

TARGETS = [
    {"url": "https://example.com", "requires_js": False}
]

run_targets(TARGETS)
```

## Features

### ✅ **Automatic Enhancement Detection**
- Uses Enhanced BRAF Runner with machine learning when available
- Falls back to basic runner if enhanced version not available
- Seamless upgrade path without code changes

### ✅ **Legacy Compatibility**
- Supports `requires_js` field (converts to `preferred_scraper`)
- Handles both dictionary and string URL formats
- Backward compatible with existing code

### ✅ **Intelligent Processing**
- Machine learning-based scraper selection
- Parallel processing for multiple targets
- Real-time progress tracking
- Comprehensive analytics and logging

## Usage Examples

### Basic Usage (Your Code)
```python
from core.runner import run_targets

TARGETS = [
    {"url": "https://example.com", "requires_js": False}
]

results = run_targets(TARGETS)
```

### Multiple Targets
```python
from core.runner import run_targets

TARGETS = [
    {"url": "https://example.com", "requires_js": False},
    {"url": "https://httpbin.org/json"},
    {"url": "https://news.ycombinator.com"},
    {"url": "https://app.example.com/dashboard", "requires_js": True}
]

results = run_targets(TARGETS)
```

### Modern Format (Recommended)
```python
from core.runner import run_targets

TARGETS = [
    {"url": "https://example.com", "preferred_scraper": "http"},
    {"url": "https://app.example.com", "preferred_scraper": "browser"},
    {"url": "https://api.github.com/users"},  # Auto-detected
]

results = run_targets(TARGETS)
```

### String URLs (Simple)
```python
from core.runner import run_targets

TARGETS = [
    "https://example.com",
    "https://httpbin.org/html",
    "https://jsonplaceholder.typicode.com/posts/1"
]

results = run_targets(TARGETS)
```

## Target Format Options

### Legacy Format (Your Code)
```python
{
    "url": "https://example.com",
    "requires_js": False  # Converts to preferred_scraper: "http"
}
```

### Modern Format
```python
{
    "url": "https://example.com",
    "preferred_scraper": "http",  # or "browser"
    "description": "Static website"
}
```

### Auto-Detection Format
```python
{
    "url": "https://example.com"
    # System automatically chooses best scraper
}
```

## Result Format

Each result contains:

```python
{
    "url": "https://example.com",
    "success": True,
    "scraper_used": "http",
    "execution_time": 0.9,
    "title": "Example Domain",
    "content": "This domain is for use in illustrative examples...",
    "word_count": 100,
    "decision_explanation": {
        "decision": "http",
        "confidence": 0.412,
        "total_score": 0.294,
        "factor_scores": {
            "domain_score": 0.1,
            "pattern_score": 0.5,
            "path_score": 0.5,
            "complexity_score": 0.09
        }
    }
}
```

## Enhanced Features (Automatic)

When using the enhanced runner, you get:

### 🧠 **Machine Learning**
- Learns from execution history
- Adapts scraper selection based on success rates
- Domain-specific optimization

### ⚡ **Parallel Processing**
- Automatic parallel execution for multiple targets
- Intelligent load balancing
- 3x+ performance improvement

### 📊 **Analytics**
- Performance tracking and optimization
- Success rate monitoring
- AI-powered suggestions

### 🔄 **Progress Tracking**
```
🚀 BRAF Runner - Processing 4 targets
✨ Using Enhanced BRAF Runner with machine learning
   📈 25.0% - ✅ HTTP - https://example.com
   📈 50.0% - ✅ HTTP - https://httpbin.org/json
   📈 75.0% - ✅ HTTP - https://news.ycombinator.com
   📈 100.0% - ❌ BROWSER - https://app.example.com
📊 Results: 3/4 successful
⚡ Parallel speedup: 2.5x
```

## Configuration Options

### Default Configuration
The runner uses optimal defaults:
- 4 parallel workers
- 2 browser workers (resource intensive)
- Analytics enabled
- Machine learning enabled

### Custom Configuration
For advanced usage, use the enhanced runner directly:

```python
from enhanced_braf_runner_fixed import EnhancedBRAFRunner

runner = EnhancedBRAFRunner(
    max_workers=6,
    max_browser_workers=3,
    enable_analytics=True
)

results = runner.run_enhanced(targets, parallel=True)
```

## Migration Guide

### From Basic BRAF
```python
# Old way
from braf_runner import run_targets
results = run_targets(targets)

# New way (drop-in replacement)
from core.runner import run_targets
results = run_targets(targets)
```

### From Custom Implementations
```python
# Old way
for target in targets:
    if target.get('requires_js'):
        result = browser_scraper.run(target)
    else:
        result = http_scraper.run(target)

# New way (automatic optimization)
from core.runner import run_targets
results = run_targets(targets)
```

## Error Handling

The runner handles errors gracefully:

```python
results = run_targets(TARGETS)

for result in results:
    if result['success']:
        print(f"✅ {result['url']}: {result['title']}")
    else:
        print(f"❌ {result['url']}: {result.get('error', 'Unknown error')}")
```

## Performance Tips

### For Best Performance
1. **Use multiple targets**: Parallel processing kicks in automatically
2. **Let the system decide**: Don't force scraper types unless necessary
3. **Use modern format**: Avoid legacy `requires_js` field when possible

### Example: High-Performance Batch Processing
```python
from core.runner import run_targets

# Large batch of URLs
TARGETS = [
    {"url": f"https://httpbin.org/delay/{i}"} 
    for i in range(10)
]

# Automatically uses parallel processing
results = run_targets(TARGETS)
print(f"Processed {len(results)} URLs with parallel speedup")
```

## Integration Examples

### With Existing Code
```python
# Your existing function
def scrape_websites(urls):
    targets = [{"url": url, "requires_js": False} for url in urls]
    return run_targets(targets)

# Usage remains the same
urls = ["https://example.com", "https://httpbin.org/html"]
results = scrape_websites(urls)
```

### With Data Processing
```python
import pandas as pd
from core.runner import run_targets

# Load URLs from CSV
df = pd.read_csv('urls.csv')
targets = [{"url": url} for url in df['url']]

# Scrape with BRAF
results = run_targets(targets)

# Convert back to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('scraped_results.csv', index=False)
```

## Summary

The simple runner interface provides:

✅ **Drop-in Compatibility**: Your existing code works without changes
✅ **Automatic Enhancement**: Uses advanced features when available
✅ **Intelligent Processing**: Machine learning and parallel execution
✅ **Legacy Support**: Handles old and new target formats
✅ **Production Ready**: Comprehensive error handling and logging

Your code `run_targets(TARGETS)` now benefits from all the enhanced BRAF features automatically! 🚀