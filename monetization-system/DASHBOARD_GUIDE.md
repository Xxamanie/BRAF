# BRAF Dashboard Guide

## Overview
Beautiful, interactive dashboard for visualizing BRAF scraping results with real-time updates and comprehensive analytics.

## 🚀 **Quick Start**

### 1. Generate Results
```bash
cd monetization-system
python scraper.py
```

### 2. Start Dashboard Server
```bash
python start_dashboard.py
```

### 3. View Dashboard
- **Enhanced Dashboard**: http://localhost:8080/dashboard/
- **Simple Dashboard**: http://localhost:8080/dashboard/simple.html
- **Raw JSON API**: http://localhost:8080/data/enhanced_results.json

## 📊 **Dashboard Features**

### ✅ **Enhanced Dashboard (`/dashboard/`)**
- **Beautiful UI**: Modern, responsive design with gradient backgrounds
- **Real-time Stats**: Success rates, execution times, parallel speedup
- **Interactive Results**: Hover effects and detailed result cards
- **Auto-refresh**: Updates every 5 minutes automatically
- **JSON Toggle**: View raw JSON data with syntax highlighting
- **Mobile Responsive**: Works perfectly on all devices

### ✅ **Simple Dashboard (`/dashboard/simple.html`)**
- **Minimal Design**: Clean, focused interface (matches your original HTML)
- **Quick Stats**: Total targets, success count, success rate
- **Raw JSON View**: Formatted JSON output with syntax highlighting
- **Auto-refresh**: Updates every 30 seconds
- **Lightweight**: Fast loading and minimal resource usage

## 📈 **Dashboard Screenshots**

### Enhanced Dashboard Features
```
🚀 BRAF Dashboard - Browser Automation Framework [Enhanced ML]

┌─────────────────┬─────────────────┬─────────────────┐
│ Total Targets   │ Success Rate    │ HTTP Scraper    │
│      6          │     100.0%      │       6         │
│   Processed     │   Successful    │ Fast & Efficient│
├─────────────────┼─────────────────┼─────────────────┤
│ Browser Scraper │ Execution Time  │ Parallel Speedup│
│       0         │     4.71s       │     1.91x       │
│JavaScript Capable│ Total Duration │ Performance Gain│
└─────────────────┴─────────────────┴─────────────────┘

📋 Scraping Results
✅ https://httpbin.org/html
   Scraper: HTTP | Time: 2.12s | Words: 147 | Confidence: 41.2%

✅ https://example.com  
   Scraper: HTTP | Time: 1.63s | Words: 26 | Confidence: 41.2%

✅ https://jsonplaceholder.typicode.com/posts/1
   Scraper: HTTP | Time: 0.26s | Words: 30 | Confidence: 39.8%
```

## 🔧 **Server Configuration**

### Basic Usage
```bash
# Start with default settings (port 8080, auto-open browser)
python start_dashboard.py

# Custom port
python start_dashboard.py --port 3000

# Don't open browser automatically
python start_dashboard.py --no-browser

# Custom port without browser
python start_dashboard.py --port 3000 --no-browser
```

### Server Features
- **CORS Enabled**: Works with local development
- **Auto-browser**: Opens dashboard automatically
- **Custom Logging**: Shows request details
- **Error Handling**: Graceful error messages
- **Port Detection**: Suggests alternative ports if busy

## 📁 **File Structure**

```
monetization-system/
├── dashboard/
│   ├── index.html              # Enhanced dashboard
│   └── simple.html             # Simple dashboard (your style)
├── data/
│   ├── results.json            # GitHub Actions results
│   └── enhanced_results.json   # Enhanced BRAF results
├── start_dashboard.py          # Dashboard server
└── scraper.py                  # Generate results
```

## 🎨 **Dashboard Customization**

### Enhanced Dashboard Styling
Edit `dashboard/index.html`:

```css
/* Custom color scheme */
body {
    background: linear-gradient(135deg, #your-color1, #your-color2);
}

.stat-card {
    background: your-background-color;
    border: your-border-style;
}
```

### Simple Dashboard Styling  
Edit `dashboard/simple.html`:

```css
/* Matches your original simple style */
body {
    font-family: monospace; /* or your preferred font */
    background: #your-bg-color;
}

#out {
    background: #1e1e1e; /* Dark code background */
    color: #d4d4d4;       /* Light text */
}
```

## 🔄 **Auto-refresh Configuration**

### Enhanced Dashboard
```javascript
// Change refresh interval (default: 5 minutes)
setInterval(loadResults, 2 * 60 * 1000); // 2 minutes
```

### Simple Dashboard
```javascript
// Change refresh interval (default: 30 seconds)
setInterval(loadResults, 10000); // 10 seconds
```

## 📊 **Data Sources**

### Enhanced Results Format
```json
{
  "enhanced_braf_execution": {
    "version": "2.0",
    "execution_id": "20251220_170838",
    "statistics": {
      "total_targets": 6,
      "successful": 6,
      "success_rate": 100.0,
      "parallel_speedup": 1.91
    }
  },
  "results": [
    {
      "url": "https://example.com",
      "success": true,
      "scraper_used": "http",
      "execution_time": 1.63,
      "decision_explanation": {
        "decision": "http",
        "confidence": 0.412
      }
    }
  ]
}
```

### GitHub Actions Format
```json
{
  "github_actions_execution": {
    "timestamp": "2025-12-20T17:08:38.382",
    "targets_processed": 6,
    "workflow_run": "123"
  },
  "results": [...]
}
```

## 🌐 **API Endpoints**

### Available Endpoints
- **`/dashboard/`** - Enhanced dashboard UI
- **`/dashboard/simple.html`** - Simple dashboard UI
- **`/data/results.json`** - GitHub Actions results
- **`/data/enhanced_results.json`** - Enhanced BRAF results
- **`/data/`** - Browse data directory
- **`/logs/`** - Browse log files (if available)

### API Usage Examples
```javascript
// Fetch results programmatically
fetch('/data/enhanced_results.json')
  .then(r => r.json())
  .then(data => {
    console.log('Success rate:', data.enhanced_braf_execution.statistics.success_rate);
  });
```

## 🔍 **Troubleshooting**

### Common Issues

#### 1. Dashboard Shows "Loading..."
- **Check Results**: Ensure `data/results.json` or `data/enhanced_results.json` exists
- **Run Scraper**: Execute `python scraper.py` to generate results
- **Check Console**: Open browser dev tools for error messages

#### 2. Server Won't Start
- **Port in Use**: Try different port with `--port 8081`
- **Permission Error**: Run with appropriate permissions
- **Python Path**: Ensure you're in the `monetization-system` directory

#### 3. Results Not Updating
- **Manual Refresh**: Press F5 or Ctrl+R
- **Check Auto-refresh**: Verify JavaScript console for errors
- **Server Logs**: Check server output for request errors

#### 4. Styling Issues
- **Browser Cache**: Hard refresh with Ctrl+Shift+R
- **CSS Conflicts**: Check browser dev tools for CSS errors
- **Mobile View**: Test responsive design on different screen sizes

### Debug Mode
```bash
# Start server with verbose logging
python start_dashboard.py --port 8080 --no-browser
# Watch server logs for request details
```

## 🚀 **Advanced Usage**

### Integration with GitHub Actions
The dashboard automatically detects and displays GitHub Actions results:

```yaml
# In your GitHub Actions workflow
- name: Generate results
  run: python scraper.py

- name: Deploy dashboard
  run: |
    python start_dashboard.py --no-browser &
    echo "Dashboard available at: ${{ github.server_url }}/${{ github.repository }}/actions"
```

### Custom Result Processing
```javascript
// Add custom processing in dashboard
function processCustomResults(data) {
    // Add your custom logic here
    const customMetrics = {
        avgWordsPerPage: data.results.reduce((sum, r) => sum + (r.word_count || 0), 0) / data.results.length,
        uniqueDomains: new Set(data.results.map(r => new URL(r.url).hostname)).size
    };
    
    return customMetrics;
}
```

### Embedding in Other Applications
```html
<!-- Embed dashboard in iframe -->
<iframe src="http://localhost:8080/dashboard/" 
        width="100%" height="600px" 
        frameborder="0">
</iframe>
```

## 📱 **Mobile Optimization**

### Responsive Features
- **Adaptive Grid**: Statistics cards adjust to screen size
- **Touch Friendly**: Large tap targets and smooth scrolling
- **Readable Text**: Optimized font sizes for mobile
- **Fast Loading**: Minimal resource usage

### Mobile Testing
```bash
# Test on mobile devices
python start_dashboard.py --port 8080
# Access from mobile: http://your-ip:8080/dashboard/
```

## 🎯 **Best Practices**

### Performance Optimization
1. **Regular Cleanup**: Remove old result files periodically
2. **Efficient Refresh**: Use appropriate auto-refresh intervals
3. **Browser Caching**: Leverage browser cache for static assets
4. **Minimal Data**: Only load necessary result data

### Security Considerations
1. **Local Use Only**: Dashboard is designed for local development
2. **No Authentication**: Don't expose to public networks
3. **CORS Headers**: Only enabled for local development
4. **File Access**: Server only serves intended directories

### Monitoring
1. **Server Logs**: Monitor server output for errors
2. **Browser Console**: Check for JavaScript errors
3. **Network Tab**: Monitor API request performance
4. **Result Validation**: Verify result data integrity

## 🎉 **Summary**

The BRAF Dashboard provides:

✅ **Two Dashboard Options**: Enhanced UI and simple HTML (your style)
✅ **Real-time Updates**: Auto-refresh with live data
✅ **Comprehensive Stats**: Success rates, timing, parallel speedup
✅ **Interactive Design**: Hover effects and responsive layout
✅ **Easy Setup**: One command to start (`python start_dashboard.py`)
✅ **API Access**: Direct JSON access for custom integrations
✅ **Mobile Friendly**: Works perfectly on all devices

Perfect for monitoring BRAF scraping operations with beautiful, real-time visualization! 🚀

### Quick Commands
```bash
# Generate results and start dashboard
python scraper.py && python start_dashboard.py

# View in browser
# Enhanced: http://localhost:8080/dashboard/
# Simple:   http://localhost:8080/dashboard/simple.html
```