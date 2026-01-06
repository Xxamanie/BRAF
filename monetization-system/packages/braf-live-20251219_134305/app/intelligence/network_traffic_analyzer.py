"""
Network Traffic Analysis System
Captures and analyzes platform network traffic for API discovery and optimization
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import re
import hashlib
from urllib.parse import urlparse, parse_qs
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class NetworkRequest:
    """Captured network request data"""
    url: str
    method: str
    headers: Dict[str, str]
    body: Optional[str]
    timestamp: datetime
    resource_type: str
    response_status: Optional[int] = None
    response_headers: Optional[Dict[str, str]] = None
    response_body: Optional[str] = None
    duration_ms: Optional[int] = None

@dataclass
class APIEndpoint:
    """Discovered API endpoint"""
    url: str
    method: str
    required_params: List[str]
    optional_params: List[str]
    required_headers: Dict[str, str]
    response_format: str
    rate_limit: Optional[int]
    authentication_required: bool
    success_indicators: List[str]
    error_patterns: List[str]

class NetworkTrafficAnalyzer:
    """Advanced network traffic capture and analysis"""
    
    def __init__(self):
        self.captured_requests: List[NetworkRequest] = []
        self.discovered_endpoints: Dict[str, APIEndpoint] = {}
        self.traffic_patterns: Dict[str, List[Dict]] = {}
        self.api_clients: Dict[str, str] = {}
        self.monitoring_active = False
        
    async def start_capture(self, page, platform: str):
        """Start comprehensive network traffic capture"""
        self.monitoring_active = True
        self.current_platform = platform
        
        # Set up request/response interception
        await page.route("**/*", self._handle_route)
        
        # Monitor network events
        page.on("request", self._log_request)
        page.on("response", self._log_response)
        page.on("requestfailed", self._log_failed_request)
        
        # Inject network monitoring JavaScript
        await self._inject_network_monitor(page)
        
        logger.info(f"Network traffic capture started for {platform}")
    
    async def stop_capture(self):
        """Stop network traffic capture and analyze results"""
        self.monitoring_active = False
        
        # Analyze captured traffic
        await self._analyze_traffic_patterns()
        await self._discover_api_endpoints()
        await self._generate_api_clients()
        
        logger.info(f"Network capture stopped. Analyzed {len(self.captured_requests)} requests")
        
        return {
            "total_requests": len(self.captured_requests),
            "discovered_endpoints": len(self.discovered_endpoints),
            "api_clients_generated": len(self.api_clients),
            "traffic_patterns": self.traffic_patterns
        }
    
    async def _handle_route(self, route):
        """Intercept and analyze all network requests"""
        request = route.request
        
        # Capture request details
        captured_request = NetworkRequest(
            url=request.url,
            method=request.method,
            headers=dict(request.headers),
            body=request.post_data,
            timestamp=datetime.now(),
            resource_type=request.resource_type
        )
        
        # Store request
        self.captured_requests.append(captured_request)
        
        # Continue request
        await route.continue_()
    
    async def _log_request(self, request):
        """Log request details"""
        if not self.monitoring_active:
            return
            
        logger.debug(f"Request: {request.method} {request.url}")
    
    async def _log_response(self, response):
        """Log response details and update captured request"""
        if not self.monitoring_active:
            return
            
        # Find corresponding request
        for req in reversed(self.captured_requests):
            if req.url == response.url and req.response_status is None:
                req.response_status = response.status
                req.response_headers = dict(response.headers)
                
                # Capture response body for API endpoints
                if self._is_api_endpoint(response.url):
                    try:
                        req.response_body = await response.text()
                    except:
                        pass
                
                break
        
        logger.debug(f"Response: {response.status} {response.url}")
    
    async def _log_failed_request(self, request):
        """Log failed requests"""
        logger.warning(f"Failed request: {request.method} {request.url}")
    
    async def _inject_network_monitor(self, page):
        """Inject JavaScript to monitor fetch/XHR calls"""
        
        monitor_script = """
        (function() {
            window._networkMonitor = {
                requests: [],
                
                logFetch: function(details) {
                    this.requests.push({
                        type: 'fetch',
                        ...details,
                        timestamp: Date.now()
                    });
                },
                
                logXHR: function(details) {
                    this.requests.push({
                        type: 'xhr',
                        ...details,
                        timestamp: Date.now()
                    });
                },
                
                getRequests: function() {
                    return this.requests;
                },
                
                clearRequests: function() {
                    this.requests = [];
                }
            };
            
            // Hook fetch
            const originalFetch = window.fetch;
            window.fetch = async function(...args) {
                const startTime = Date.now();
                
                try {
                    const response = await originalFetch.apply(this, args);
                    const endTime = Date.now();
                    
                    window._networkMonitor.logFetch({
                        url: args[0],
                        method: args[1]?.method || 'GET',
                        headers: args[1]?.headers || {},
                        body: args[1]?.body,
                        duration: endTime - startTime,
                        status: response.status,
                        success: response.ok
                    });
                    
                    return response;
                } catch (error) {
                    const endTime = Date.now();
                    
                    window._networkMonitor.logFetch({
                        url: args[0],
                        method: args[1]?.method || 'GET',
                        duration: endTime - startTime,
                        error: error.message,
                        success: false
                    });
                    
                    throw error;
                }
            };
            
            // Hook XMLHttpRequest
            const originalXHROpen = XMLHttpRequest.prototype.open;
            const originalXHRSend = XMLHttpRequest.prototype.send;
            
            XMLHttpRequest.prototype.open = function(method, url, async, user, password) {
                this._requestDetails = {
                    method: method,
                    url: url,
                    startTime: Date.now()
                };
                
                return originalXHROpen.apply(this, arguments);
            };
            
            XMLHttpRequest.prototype.send = function(body) {
                const details = this._requestDetails;
                
                const onLoadEnd = () => {
                    const duration = Date.now() - details.startTime;
                    
                    window._networkMonitor.logXHR({
                        ...details,
                        body: body,
                        duration: duration,
                        status: this.status,
                        response: this.responseText,
                        success: this.status >= 200 && this.status < 300
                    });
                };
                
                this.addEventListener('loadend', onLoadEnd);
                return originalXHRSend.apply(this, arguments);
            };
            
            console.log('Network monitor injected successfully');
        })();
        """
        
        await page.evaluate(monitor_script)
    
    def _is_api_endpoint(self, url: str) -> bool:
        """Determine if URL is an API endpoint"""
        api_indicators = [
            '/api/', '/v1/', '/v2/', '/v3/',
            'api.', 'rest.', 'service.',
            '.json', '/json/', '/ajax/',
            '/graphql', '/rpc/'
        ]
        
        url_lower = url.lower()
        return any(indicator in url_lower for indicator in api_indicators)
    
    async def _analyze_traffic_patterns(self):
        """Analyze captured traffic for patterns"""
        
        # Group requests by domain
        domain_requests = {}
        for req in self.captured_requests:
            domain = urlparse(req.url).netloc
            if domain not in domain_requests:
                domain_requests[domain] = []
            domain_requests[domain].append(req)
        
        # Analyze patterns for each domain
        for domain, requests in domain_requests.items():
            patterns = {
                "request_frequency": len(requests),
                "common_endpoints": self._find_common_endpoints(requests),
                "authentication_patterns": self._analyze_auth_patterns(requests),
                "parameter_patterns": self._analyze_parameter_patterns(requests),
                "timing_patterns": self._analyze_timing_patterns(requests)
            }
            
            self.traffic_patterns[domain] = patterns
    
    def _find_common_endpoints(self, requests: List[NetworkRequest]) -> List[Dict]:
        """Find commonly accessed endpoints"""
        endpoint_counts = {}
        
        for req in requests:
            # Extract endpoint path (without query params)
            parsed = urlparse(req.url)
            endpoint = parsed.path
            
            if endpoint not in endpoint_counts:
                endpoint_counts[endpoint] = {
                    "count": 0,
                    "methods": set(),
                    "example_url": req.url
                }
            
            endpoint_counts[endpoint]["count"] += 1
            endpoint_counts[endpoint]["methods"].add(req.method)
        
        # Sort by frequency
        common_endpoints = []
        for endpoint, data in sorted(endpoint_counts.items(), 
                                   key=lambda x: x[1]["count"], reverse=True):
            if data["count"] > 1:  # Only include endpoints hit multiple times
                common_endpoints.append({
                    "endpoint": endpoint,
                    "frequency": data["count"],
                    "methods": list(data["methods"]),
                    "example_url": data["example_url"]
                })
        
        return common_endpoints[:10]  # Top 10
    
    def _analyze_auth_patterns(self, requests: List[NetworkRequest]) -> Dict:
        """Analyze authentication patterns"""
        auth_headers = set()
        auth_cookies = set()
        auth_params = set()
        
        for req in requests:
            # Check headers for auth patterns
            for header, value in req.headers.items():
                if any(auth_term in header.lower() for auth_term in 
                      ['auth', 'token', 'key', 'session', 'bearer']):
                    auth_headers.add(header)
            
            # Check for auth-related cookies
            cookie_header = req.headers.get('Cookie', '')
            if cookie_header:
                cookies = cookie_header.split(';')
                for cookie in cookies:
                    if '=' in cookie:
                        name = cookie.split('=')[0].strip()
                        if any(auth_term in name.lower() for auth_term in 
                              ['auth', 'token', 'session', 'login']):
                            auth_cookies.add(name)
            
            # Check URL parameters for auth
            parsed = urlparse(req.url)
            params = parse_qs(parsed.query)
            for param in params.keys():
                if any(auth_term in param.lower() for auth_term in 
                      ['auth', 'token', 'key', 'session']):
                    auth_params.add(param)
        
        return {
            "auth_headers": list(auth_headers),
            "auth_cookies": list(auth_cookies),
            "auth_params": list(auth_params)
        }
    
    def _analyze_parameter_patterns(self, requests: List[NetworkRequest]) -> Dict:
        """Analyze common parameters"""
        all_params = {}
        
        for req in requests:
            parsed = urlparse(req.url)
            params = parse_qs(parsed.query)
            
            for param, values in params.items():
                if param not in all_params:
                    all_params[param] = {
                        "frequency": 0,
                        "example_values": set()
                    }
                
                all_params[param]["frequency"] += 1
                all_params[param]["example_values"].update(values[:3])  # Keep first 3 examples
        
        # Convert to list and sort by frequency
        common_params = []
        for param, data in sorted(all_params.items(), 
                                key=lambda x: x[1]["frequency"], reverse=True):
            if data["frequency"] > 1:
                common_params.append({
                    "parameter": param,
                    "frequency": data["frequency"],
                    "example_values": list(data["example_values"])
                })
        
        return {"common_parameters": common_params[:20]}  # Top 20
    
    def _analyze_timing_patterns(self, requests: List[NetworkRequest]) -> Dict:
        """Analyze request timing patterns"""
        if len(requests) < 2:
            return {"intervals": [], "patterns": []}
        
        # Calculate intervals between requests
        intervals = []
        for i in range(1, len(requests)):
            interval = (requests[i].timestamp - requests[i-1].timestamp).total_seconds()
            intervals.append(interval)
        
        # Find patterns
        avg_interval = sum(intervals) / len(intervals) if intervals else 0
        min_interval = min(intervals) if intervals else 0
        max_interval = max(intervals) if intervals else 0
        
        return {
            "average_interval": avg_interval,
            "min_interval": min_interval,
            "max_interval": max_interval,
            "total_requests": len(requests),
            "request_rate_per_minute": len(requests) / max(1, max_interval / 60) if max_interval > 0 else 0
        }
    
    async def _discover_api_endpoints(self):
        """Discover and analyze API endpoints"""
        
        api_requests = [req for req in self.captured_requests if self._is_api_endpoint(req.url)]
        
        # Group by endpoint pattern
        endpoint_groups = {}
        for req in api_requests:
            parsed = urlparse(req.url)
            endpoint_key = f"{req.method}:{parsed.path}"
            
            if endpoint_key not in endpoint_groups:
                endpoint_groups[endpoint_key] = []
            endpoint_groups[endpoint_key].append(req)
        
        # Analyze each endpoint group
        for endpoint_key, requests in endpoint_groups.items():
            if len(requests) < 2:  # Skip endpoints with only one request
                continue
                
            method, path = endpoint_key.split(':', 1)
            
            # Analyze endpoint
            endpoint_analysis = self._analyze_endpoint_group(requests)
            
            # Create API endpoint object
            api_endpoint = APIEndpoint(
                url=f"{urlparse(requests[0].url).scheme}://{urlparse(requests[0].url).netloc}{path}",
                method=method,
                required_params=endpoint_analysis["required_params"],
                optional_params=endpoint_analysis["optional_params"],
                required_headers=endpoint_analysis["required_headers"],
                response_format=endpoint_analysis["response_format"],
                rate_limit=endpoint_analysis["rate_limit"],
                authentication_required=endpoint_analysis["auth_required"],
                success_indicators=endpoint_analysis["success_indicators"],
                error_patterns=endpoint_analysis["error_patterns"]
            )
            
            self.discovered_endpoints[endpoint_key] = api_endpoint
    
    def _analyze_endpoint_group(self, requests: List[NetworkRequest]) -> Dict:
        """Analyze a group of requests to the same endpoint"""
        
        # Analyze parameters
        all_params = set()
        required_params = set()
        
        for req in requests:
            parsed = urlparse(req.url)
            params = set(parse_qs(parsed.query).keys())
            all_params.update(params)
            
            if not required_params:
                required_params = params.copy()
            else:
                required_params &= params  # Intersection - params present in all requests
        
        optional_params = all_params - required_params
        
        # Analyze headers
        common_headers = {}
        for req in requests:
            for header, value in req.headers.items():
                if header not in common_headers:
                    common_headers[header] = []
                common_headers[header].append(value)
        
        # Keep headers that appear in all requests with same value
        required_headers = {}
        for header, values in common_headers.items():
            if len(set(values)) == 1 and len(values) == len(requests):
                required_headers[header] = values[0]
        
        # Analyze responses
        response_formats = set()
        success_statuses = set()
        error_statuses = set()
        
        for req in requests:
            if req.response_status:
                if 200 <= req.response_status < 300:
                    success_statuses.add(req.response_status)
                else:
                    error_statuses.add(req.response_status)
                
                # Determine response format
                content_type = req.response_headers.get('content-type', '') if req.response_headers else ''
                if 'json' in content_type:
                    response_formats.add('json')
                elif 'xml' in content_type:
                    response_formats.add('xml')
                elif 'html' in content_type:
                    response_formats.add('html')
                else:
                    response_formats.add('text')
        
        # Calculate rate limit
        if len(requests) > 1:
            time_span = (requests[-1].timestamp - requests[0].timestamp).total_seconds()
            rate_limit = int(len(requests) / max(time_span / 60, 1))  # requests per minute
        else:
            rate_limit = None
        
        return {
            "required_params": list(required_params),
            "optional_params": list(optional_params),
            "required_headers": required_headers,
            "response_format": list(response_formats)[0] if response_formats else "unknown",
            "rate_limit": rate_limit,
            "auth_required": any(auth_header in required_headers for auth_header in 
                               ['Authorization', 'X-Auth-Token', 'X-API-Key']),
            "success_indicators": list(success_statuses),
            "error_patterns": list(error_statuses)
        }
    
    async def _generate_api_clients(self):
        """Generate Python API client code for discovered endpoints"""
        
        for endpoint_key, endpoint in self.discovered_endpoints.items():
            client_code = self._generate_endpoint_client(endpoint_key, endpoint)
            self.api_clients[endpoint_key] = client_code
    
    def _generate_endpoint_client(self, endpoint_key: str, endpoint: APIEndpoint) -> str:
        """Generate Python client code for an API endpoint"""
        
        method, path = endpoint_key.split(':', 1)
        class_name = self._endpoint_to_class_name(endpoint_key)
        
        # Generate method parameters
        params = []
        if endpoint.required_params:
            params.extend(endpoint.required_params)
        if endpoint.optional_params:
            params.extend([f"{param}=None" for param in endpoint.optional_params])
        
        param_str = ", ".join(params)
        
        # Generate client code
        client_code = f'''
class {class_name}:
    """Auto-generated API client for {endpoint.url}"""
    
    def __init__(self, session, base_url=None):
        self.session = session
        self.base_url = base_url or "{urlparse(endpoint.url).scheme}://{urlparse(endpoint.url).netloc}"
        self.endpoint = "{path}"
        self.method = "{method.upper()}"
        self.required_headers = {json.dumps(endpoint.required_headers, indent=8)}
    
    async def call(self, {param_str}, **kwargs):
        """Call the {endpoint_key} endpoint"""
        url = f"{{self.base_url}}{{self.endpoint}}"
        
        # Prepare parameters
        params = {{}}
        {self._generate_param_handling(endpoint)}
        
        # Prepare headers
        headers = {{**self.required_headers}}
        if 'headers' in kwargs:
            headers.update(kwargs['headers'])
        
        # Make request
        async with self.session.request(
            method=self.method,
            url=url,
            params=params if self.method == 'GET' else None,
            json=kwargs.get('json') if self.method != 'GET' else None,
            data=kwargs.get('data') if self.method != 'GET' else None,
            headers=headers
        ) as response:
            response.raise_for_status()
            
            if "{endpoint.response_format}" == "json":
                return await response.json()
            else:
                return await response.text()
'''
        
        return client_code
    
    def _generate_param_handling(self, endpoint: APIEndpoint) -> str:
        """Generate parameter handling code"""
        lines = []
        
        for param in endpoint.required_params:
            lines.append(f'        params["{param}"] = {param}')
        
        for param in endpoint.optional_params:
            lines.append(f'        if {param} is not None:')
            lines.append(f'            params["{param}"] = {param}')
        
        return '\n'.join(lines)
    
    def _endpoint_to_class_name(self, endpoint_key: str) -> str:
        """Convert endpoint key to class name"""
        method, path = endpoint_key.split(':', 1)
        
        # Clean up path and convert to class name
        path_parts = [part for part in path.split('/') if part and not part.startswith('{')]
        class_name = ''.join(word.capitalize() for word in path_parts)
        class_name += method.capitalize() + 'Client'
        
        return class_name or 'APIClient'
    
    def get_discovered_endpoints(self) -> Dict[str, APIEndpoint]:
        """Get all discovered API endpoints"""
        return self.discovered_endpoints
    
    def get_api_clients(self) -> Dict[str, str]:
        """Get generated API client code"""
        return self.api_clients
    
    def get_traffic_summary(self) -> Dict[str, Any]:
        """Get traffic analysis summary"""
        return {
            "total_requests": len(self.captured_requests),
            "api_requests": len([req for req in self.captured_requests if self._is_api_endpoint(req.url)]),
            "discovered_endpoints": len(self.discovered_endpoints),
            "traffic_patterns": self.traffic_patterns,
            "monitoring_duration": (
                self.captured_requests[-1].timestamp - self.captured_requests[0].timestamp
            ).total_seconds() if len(self.captured_requests) > 1 else 0
        }

# Global instance
network_analyzer = NetworkTrafficAnalyzer()
