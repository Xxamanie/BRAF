"""
CAPTCHA Solver with multiple fallback methods for BRAF.

This module provides CAPTCHA solving capabilities using paid services
with OCR fallbacks and test mode support for development.
"""

import asyncio
import base64
import io
import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import aiohttp
from PIL import Image
from playwright.async_api import Page

logger = logging.getLogger(__name__)


class CaptchaService:
    """Base class for CAPTCHA solving services."""
    
    def __init__(self, api_key: str, timeout: int = 120):
        """
        Initialize CAPTCHA service.
        
        Args:
            api_key: API key for the service
            timeout: Timeout in seconds for solving
        """
        self.api_key = api_key
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def solve_recaptcha_v2(self, site_key: str, page_url: str) -> Optional[str]:
        """
        Solve reCAPTCHA v2 challenge.
        
        Args:
            site_key: reCAPTCHA site key
            page_url: URL of the page with CAPTCHA
            
        Returns:
            Solution token or None if failed
        """
        raise NotImplementedError("Subclasses must implement solve_recaptcha_v2")
    
    async def solve_image_captcha(self, image_data: bytes) -> Optional[str]:
        """
        Solve image-based CAPTCHA.
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Solution text or None if failed
        """
        raise NotImplementedError("Subclasses must implement solve_image_captcha")


class TwoCaptchaService(CaptchaService):
    """2Captcha service implementation."""
    
    def __init__(self, api_key: str, timeout: int = 120):
        """Initialize 2Captcha service."""
        super().__init__(api_key, timeout)
        self.base_url = "http://2captcha.com"
    
    async def solve_recaptcha_v2(self, site_key: str, page_url: str) -> Optional[str]:
        """Solve reCAPTCHA v2 using 2Captcha."""
        if not self.session:
            raise RuntimeError("Service not initialized. Use async context manager.")
        
        try:
            # Submit CAPTCHA
            submit_data = {
                "key": self.api_key,
                "method": "userrecaptcha",
                "googlekey": site_key,
                "pageurl": page_url,
                "json": 1
            }
            
            async with self.session.post(f"{self.base_url}/in.php", data=submit_data) as response:
                result = await response.json()
                
                if result.get("status") != 1:
                    logger.error(f"2Captcha submission failed: {result.get('error_text')}")
                    return None
                
                captcha_id = result["request"]
            
            # Poll for solution
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                await asyncio.sleep(5)  # Wait before polling
                
                async with self.session.get(
                    f"{self.base_url}/res.php",
                    params={"key": self.api_key, "action": "get", "id": captcha_id, "json": 1}
                ) as response:
                    result = await response.json()
                    
                    if result.get("status") == 1:
                        logger.info(f"2Captcha solved reCAPTCHA v2 in {time.time() - start_time:.1f}s")
                        return result["request"]
                    elif result.get("error_text") == "CAPCHA_NOT_READY":
                        continue
                    else:
                        logger.error(f"2Captcha error: {result.get('error_text')}")
                        return None
            
            logger.error("2Captcha timeout")
            return None
            
        except Exception as e:
            logger.error(f"2Captcha service error: {e}")
            return None
    
    async def solve_image_captcha(self, image_data: bytes) -> Optional[str]:
        """Solve image CAPTCHA using 2Captcha."""
        if not self.session:
            raise RuntimeError("Service not initialized. Use async context manager.")
        
        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Submit CAPTCHA
            submit_data = {
                "key": self.api_key,
                "method": "base64",
                "body": image_b64,
                "json": 1
            }
            
            async with self.session.post(f"{self.base_url}/in.php", data=submit_data) as response:
                result = await response.json()
                
                if result.get("status") != 1:
                    logger.error(f"2Captcha image submission failed: {result.get('error_text')}")
                    return None
                
                captcha_id = result["request"]
            
            # Poll for solution
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                await asyncio.sleep(3)  # Shorter wait for image CAPTCHAs
                
                async with self.session.get(
                    f"{self.base_url}/res.php",
                    params={"key": self.api_key, "action": "get", "id": captcha_id, "json": 1}
                ) as response:
                    result = await response.json()
                    
                    if result.get("status") == 1:
                        logger.info(f"2Captcha solved image CAPTCHA in {time.time() - start_time:.1f}s")
                        return result["request"]
                    elif result.get("error_text") == "CAPCHA_NOT_READY":
                        continue
                    else:
                        logger.error(f"2Captcha error: {result.get('error_text')}")
                        return None
            
            logger.error("2Captcha timeout")
            return None
            
        except Exception as e:
            logger.error(f"2Captcha image service error: {e}")
            return None


class AntiCaptchaService(CaptchaService):
    """Anti-Captcha service implementation."""
    
    def __init__(self, api_key: str, timeout: int = 120):
        """Initialize Anti-Captcha service."""
        super().__init__(api_key, timeout)
        self.base_url = "https://api.anti-captcha.com"
    
    async def solve_recaptcha_v2(self, site_key: str, page_url: str) -> Optional[str]:
        """Solve reCAPTCHA v2 using Anti-Captcha."""
        if not self.session:
            raise RuntimeError("Service not initialized. Use async context manager.")
        
        try:
            # Submit CAPTCHA
            submit_data = {
                "clientKey": self.api_key,
                "task": {
                    "type": "NoCaptchaTaskProxyless",
                    "websiteURL": page_url,
                    "websiteKey": site_key
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/createTask",
                json=submit_data
            ) as response:
                result = await response.json()
                
                if result.get("errorId") != 0:
                    logger.error(f"Anti-Captcha submission failed: {result.get('errorDescription')}")
                    return None
                
                task_id = result["taskId"]
            
            # Poll for solution
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                await asyncio.sleep(5)
                
                poll_data = {
                    "clientKey": self.api_key,
                    "taskId": task_id
                }
                
                async with self.session.post(
                    f"{self.base_url}/getTaskResult",
                    json=poll_data
                ) as response:
                    result = await response.json()
                    
                    if result.get("status") == "ready":
                        solution = result["solution"]["gRecaptchaResponse"]
                        logger.info(f"Anti-Captcha solved reCAPTCHA v2 in {time.time() - start_time:.1f}s")
                        return solution
                    elif result.get("status") == "processing":
                        continue
                    else:
                        logger.error(f"Anti-Captcha error: {result.get('errorDescription')}")
                        return None
            
            logger.error("Anti-Captcha timeout")
            return None
            
        except Exception as e:
            logger.error(f"Anti-Captcha service error: {e}")
            return None
    
    async def solve_image_captcha(self, image_data: bytes) -> Optional[str]:
        """Solve image CAPTCHA using Anti-Captcha."""
        if not self.session:
            raise RuntimeError("Service not initialized. Use async context manager.")
        
        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Submit CAPTCHA
            submit_data = {
                "clientKey": self.api_key,
                "task": {
                    "type": "ImageToTextTask",
                    "body": image_b64
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/createTask",
                json=submit_data
            ) as response:
                result = await response.json()
                
                if result.get("errorId") != 0:
                    logger.error(f"Anti-Captcha image submission failed: {result.get('errorDescription')}")
                    return None
                
                task_id = result["taskId"]
            
            # Poll for solution
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                await asyncio.sleep(3)
                
                poll_data = {
                    "clientKey": self.api_key,
                    "taskId": task_id
                }
                
                async with self.session.post(
                    f"{self.base_url}/getTaskResult",
                    json=poll_data
                ) as response:
                    result = await response.json()
                    
                    if result.get("status") == "ready":
                        solution = result["solution"]["text"]
                        logger.info(f"Anti-Captcha solved image CAPTCHA in {time.time() - start_time:.1f}s")
                        return solution
                    elif result.get("status") == "processing":
                        continue
                    else:
                        logger.error(f"Anti-Captcha error: {result.get('errorDescription')}")
                        return None
            
            logger.error("Anti-Captcha timeout")
            return None
            
        except Exception as e:
            logger.error(f"Anti-Captcha image service error: {e}")
            return None


class OCRFallback:
    """OCR fallback using Tesseract for image CAPTCHAs."""
    
    def __init__(self):
        """Initialize OCR fallback."""
        self.available = self._check_tesseract_availability()
    
    def _check_tesseract_availability(self) -> bool:
        """Check if Tesseract is available."""
        try:
            import pytesseract
            # Try a simple OCR operation
            test_image = Image.new('RGB', (100, 30), color='white')
            pytesseract.image_to_string(test_image)
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            return False
    
    async def solve_image_captcha(self, image_data: bytes) -> Optional[str]:
        """
        Solve image CAPTCHA using OCR.
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            OCR text or None if failed
        """
        if not self.available:
            logger.error("OCR fallback not available (Tesseract not installed)")
            return None
        
        try:
            import pytesseract
            
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess image for better OCR
            image = self._preprocess_image(image)
            
            # Perform OCR
            text = pytesseract.image_to_string(
                image,
                config='--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            ).strip()
            
            if text:
                logger.info(f"OCR fallback extracted text: '{text}'")
                return text
            else:
                logger.warning("OCR fallback extracted no text")
                return None
                
        except Exception as e:
            logger.error(f"OCR fallback error: {e}")
            return None
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize if too small
        width, height = image.size
        if width < 100 or height < 30:
            scale_factor = max(100 / width, 30 / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Apply threshold to make text clearer
        import numpy as np
        img_array = np.array(image)
        threshold = np.mean(img_array)
        img_array = np.where(img_array > threshold, 255, 0)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        return image


class TestModeCaptchaSolver:
    """Test mode CAPTCHA solver for development and testing."""
    
    def __init__(self):
        """Initialize test mode solver."""
        self.test_tokens = {
            "recaptcha_v2": "03AGdBq25SiXT-pmSeBXjzScW-EiocHwwpwqJRCAC7g...",
            "image_captcha": "TEST123"
        }
    
    async def solve_recaptcha_v2(self, site_key: str, page_url: str) -> Optional[str]:
        """
        Solve reCAPTCHA v2 in test mode.
        
        Args:
            site_key: reCAPTCHA site key
            page_url: URL of the page
            
        Returns:
            Test token if in test environment
        """
        # Check if this is a test environment
        if self._is_test_environment(page_url, site_key):
            # Simulate solving delay
            await asyncio.sleep(2)
            logger.info("Test mode: Using reCAPTCHA v2 test token")
            return self.test_tokens["recaptcha_v2"]
        
        return None
    
    async def solve_image_captcha(self, image_data: bytes) -> Optional[str]:
        """
        Solve image CAPTCHA in test mode.
        
        Args:
            image_data: Image data
            
        Returns:
            Test solution
        """
        # Always return test solution in test mode
        await asyncio.sleep(1)
        logger.info("Test mode: Using image CAPTCHA test solution")
        return self.test_tokens["image_captcha"]
    
    def _is_test_environment(self, page_url: str, site_key: str) -> bool:
        """
        Check if this is a test environment.
        
        Args:
            page_url: Page URL
            site_key: Site key
            
        Returns:
            True if test environment detected
        """
        test_indicators = [
            "localhost", "127.0.0.1", "test", "dev", "staging",
            "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI"  # Google's test site key
        ]
        
        return any(indicator in page_url.lower() or indicator in site_key for indicator in test_indicators)


class CaptchaSolver:
    """Main CAPTCHA solver with multiple fallback methods."""
    
    def __init__(
        self,
        primary_service: str = "2captcha",
        api_key: Optional[str] = None,
        fallback_ocr: bool = True,
        test_mode: bool = False,
        timeout: int = 120
    ):
        """
        Initialize CAPTCHA solver.
        
        Args:
            primary_service: Primary service ('2captcha', 'anticaptcha')
            api_key: API key for primary service
            fallback_ocr: Enable OCR fallback
            test_mode: Enable test mode for development
            timeout: Timeout in seconds
        """
        self.primary_service = primary_service
        self.api_key = api_key
        self.fallback_ocr = fallback_ocr
        self.test_mode = test_mode
        self.timeout = timeout
        
        # Initialize components
        self.ocr_fallback = OCRFallback() if fallback_ocr else None
        self.test_solver = TestModeCaptchaSolver() if test_mode else None
        
        # Service mapping
        self.service_classes = {
            "2captcha": TwoCaptchaService,
            "anticaptcha": AntiCaptchaService
        }
    
    async def solve_recaptcha_v2(self, site_key: str, page_url: str) -> Optional[str]:
        """
        Solve reCAPTCHA v2 with fallback methods.
        
        Args:
            site_key: reCAPTCHA site key
            page_url: URL of the page with CAPTCHA
            
        Returns:
            Solution token or None if all methods failed
        """
        logger.info(f"Solving reCAPTCHA v2 for {page_url}")
        
        # Try test mode first if enabled
        if self.test_mode and self.test_solver:
            solution = await self.test_solver.solve_recaptcha_v2(site_key, page_url)
            if solution:
                return solution
        
        # Try primary service if API key is available
        if self.api_key and self.primary_service in self.service_classes:
            try:
                service_class = self.service_classes[self.primary_service]
                async with service_class(self.api_key, self.timeout) as service:
                    solution = await service.solve_recaptcha_v2(site_key, page_url)
                    if solution:
                        return solution
            except Exception as e:
                logger.error(f"Primary service {self.primary_service} failed: {e}")
        
        logger.error("All reCAPTCHA v2 solving methods failed")
        return None
    
    async def solve_image_captcha(self, image_data: bytes) -> Optional[str]:
        """
        Solve image CAPTCHA with fallback methods.
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Solution text or None if all methods failed
        """
        logger.info("Solving image CAPTCHA")
        
        # Try test mode first if enabled
        if self.test_mode and self.test_solver:
            solution = await self.test_solver.solve_image_captcha(image_data)
            if solution:
                return solution
        
        # Try primary service if API key is available
        if self.api_key and self.primary_service in self.service_classes:
            try:
                service_class = self.service_classes[self.primary_service]
                async with service_class(self.api_key, self.timeout) as service:
                    solution = await service.solve_image_captcha(image_data)
                    if solution:
                        return solution
            except Exception as e:
                logger.error(f"Primary service {self.primary_service} failed: {e}")
        
        # Try OCR fallback
        if self.ocr_fallback and self.ocr_fallback.available:
            logger.info("Trying OCR fallback")
            solution = await self.ocr_fallback.solve_image_captcha(image_data)
            if solution:
                return solution
        
        logger.error("All image CAPTCHA solving methods failed")
        return None
    
    async def inject_recaptcha_solution(self, page: Page, solution: str) -> bool:
        """
        Inject reCAPTCHA solution into page.
        
        Args:
            page: Playwright page
            solution: Solution token
            
        Returns:
            True if injection successful
        """
        try:
            # Inject solution into reCAPTCHA response element
            await page.evaluate(f"""
                () => {{
                    const responseElement = document.getElementById('g-recaptcha-response');
                    if (responseElement) {{
                        responseElement.innerHTML = '{solution}';
                        responseElement.style.display = 'block';
                        
                        // Trigger callback if available
                        if (window.grecaptcha && window.grecaptcha.getResponse) {{
                            const callback = window.grecaptcha.getResponse();
                            if (callback) {{
                                callback('{solution}');
                            }}
                        }}
                        
                        return true;
                    }}
                    return false;
                }}
            """)
            
            logger.info("reCAPTCHA solution injected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to inject reCAPTCHA solution: {e}")
            return False
    
    async def detect_captcha_type(self, page: Page) -> Optional[str]:
        """
        Detect CAPTCHA type on page.
        
        Args:
            page: Playwright page
            
        Returns:
            CAPTCHA type or None if not detected
        """
        try:
            captcha_info = await page.evaluate("""
                () => {
                    // Check for reCAPTCHA
                    if (document.querySelector('.g-recaptcha') || 
                        document.querySelector('[data-sitekey]') ||
                        window.grecaptcha) {
                        return {type: 'recaptcha_v2'};
                    }
                    
                    // Check for hCaptcha
                    if (document.querySelector('.h-captcha') ||
                        window.hcaptcha) {
                        return {type: 'hcaptcha'};
                    }
                    
                    // Check for image CAPTCHAs
                    const images = document.querySelectorAll('img');
                    for (let img of images) {
                        if (img.src && (
                            img.src.includes('captcha') ||
                            img.src.includes('challenge') ||
                            img.alt && img.alt.toLowerCase().includes('captcha')
                        )) {
                            return {type: 'image_captcha', element: img.src};
                        }
                    }
                    
                    return null;
                }
            """)
            
            if captcha_info:
                logger.info(f"Detected CAPTCHA type: {captcha_info['type']}")
                return captcha_info['type']
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting CAPTCHA type: {e}")
            return None
    
    def get_solver_stats(self) -> Dict[str, Any]:
        """
        Get CAPTCHA solver statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "primary_service": self.primary_service,
            "api_key_configured": bool(self.api_key),
            "fallback_ocr_available": self.ocr_fallback.available if self.ocr_fallback else False,
            "test_mode_enabled": self.test_mode,
            "timeout": self.timeout,
            "supported_services": list(self.service_classes.keys())
        }


# Global CAPTCHA solver instance
_captcha_solver: Optional[CaptchaSolver] = None


def get_captcha_solver() -> Optional[CaptchaSolver]:
    """
    Get global CAPTCHA solver instance.
    
    Returns:
        CAPTCHA solver instance or None if not initialized
    """
    return _captcha_solver


def init_captcha_solver(
    primary_service: str = "2captcha",
    api_key: Optional[str] = None,
    fallback_ocr: bool = True,
    test_mode: bool = False,
    timeout: int = 120
) -> CaptchaSolver:
    """
    Initialize global CAPTCHA solver.
    
    Args:
        primary_service: Primary service name
        api_key: API key for primary service
        fallback_ocr: Enable OCR fallback
        test_mode: Enable test mode
        timeout: Timeout in seconds
        
    Returns:
        Initialized CAPTCHA solver
    """
    global _captcha_solver
    
    _captcha_solver = CaptchaSolver(
        primary_service=primary_service,
        api_key=api_key,
        fallback_ocr=fallback_ocr,
        test_mode=test_mode,
        timeout=timeout
    )
    
    return _captcha_solver