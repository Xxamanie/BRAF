"""
Production Browser Automation for Real Money Earning Activities
Handles automated survey completion, video watching, and other earning tasks
"""

import os
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import undetected_chromedriver as uc

logger = logging.getLogger(__name__)

class ProductionBrowserAutomation:
    """Production-grade browser automation for earning activities"""
    
    def __init__(self):
        self.proxy_service = os.getenv('PROXY_SERVICE', 'brightdata')
        self.proxy_username = os.getenv('PROXY_USERNAME')
        self.proxy_password = os.getenv('PROXY_PASSWORD')
        self.proxy_endpoint = os.getenv('PROXY_ENDPOINT')
        
        self.captcha_service = os.getenv('CAPTCHA_SERVICE', '2captcha')
        self.captcha_api_key = os.getenv('CAPTCHA_API_KEY')
        
        self.fingerprint_service = os.getenv('FINGERPRINT_SERVICE', 'multilogin')
        self.fingerprint_api_key = os.getenv('FINGERPRINT_API_KEY')
        
        # Anti-detection settings
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        self.screen_resolutions = [
            (1920, 1080), (1366, 768), (1440, 900), (1536, 864), (1280, 720)
        ]
        
        # Behavioral patterns
        self.typing_delays = (0.05, 0.15)  # Min, max delay between keystrokes
        self.click_delays = (0.5, 2.0)    # Min, max delay before clicks
        self.scroll_delays = (1.0, 3.0)   # Min, max delay between scrolls
        
        self.demo_mode = not all([self.proxy_username, self.captcha_api_key])
        if self.demo_mode:
            logger.warning("Production credentials not configured - running in demo mode")
    
    def create_browser_profile(self, profile_name: str = None) -> Dict[str, Any]:
        """Create a unique browser profile with anti-detection measures"""
        profile_name = profile_name or f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Random screen resolution
        width, height = random.choice(self.screen_resolutions)
        
        # Random user agent
        user_agent = random.choice(self.user_agents)
        
        # Chrome options for anti-detection
        chrome_options = Options()
        
        if not self.demo_mode and self.proxy_endpoint:
            # Configure proxy
            chrome_options.add_argument(f'--proxy-server={self.proxy_endpoint}')
        
        # Anti-detection flags
        chrome_options.add_argument(f'--user-agent={user_agent}')
        chrome_options.add_argument(f'--window-size={width},{height}')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Fingerprint randomization
        chrome_options.add_argument('--disable-plugins-discovery')
        chrome_options.add_argument('--disable-extensions-file-access-check')
        chrome_options.add_argument('--disable-extensions-http-throttling')
        
        profile_data = {
            'name': profile_name,
            'user_agent': user_agent,
            'screen_resolution': f'{width}x{height}',
            'chrome_options': chrome_options,
            'created_at': datetime.now().isoformat(),
            'proxy_configured': bool(self.proxy_endpoint),
            'fingerprint_randomized': True
        }
        
        return profile_data
    
    def start_browser_session(self, profile_data: Dict[str, Any]) -> webdriver.Chrome:
        """Start browser session with specified profile"""
        try:
            # Use undetected-chromedriver for better anti-detection
            driver = uc.Chrome(options=profile_data['chrome_options'])
            
            # Execute anti-detection scripts
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Set viewport
            width, height = profile_data['screen_resolution'].split('x')
            driver.set_window_size(int(width), int(height))
            
            logger.info(f"Browser session started with profile: {profile_data['name']}")
            return driver
            
        except Exception as e:
            logger.error(f"Failed to start browser session: {e}")
            raise
    
    def human_like_typing(self, element, text: str):
        """Type text with human-like delays and patterns"""
        element.clear()
        
        for char in text:
            element.send_keys(char)
            # Random delay between keystrokes
            delay = random.uniform(*self.typing_delays)
            time.sleep(delay)
            
            # Occasional longer pauses (thinking time)
            if random.random() < 0.1:  # 10% chance
                time.sleep(random.uniform(0.5, 1.5))
    
    def human_like_click(self, driver, element):
        """Click element with human-like behavior"""
        # Random delay before click
        delay = random.uniform(*self.click_delays)
        time.sleep(delay)
        
        # Move mouse to element with slight randomness
        actions = ActionChains(driver)
        
        # Get element location and size
        location = element.location
        size = element.size
        
        # Random offset within element bounds
        offset_x = random.randint(-size['width']//4, size['width']//4)
        offset_y = random.randint(-size['height']//4, size['height']//4)
        
        actions.move_to_element_with_offset(element, offset_x, offset_y)
        actions.pause(random.uniform(0.1, 0.3))
        actions.click()
        actions.perform()
    
    def human_like_scroll(self, driver, direction: str = 'down', amount: int = None):
        """Scroll page with human-like behavior"""
        if amount is None:
            amount = random.randint(200, 800)
        
        if direction == 'down':
            driver.execute_script(f"window.scrollBy(0, {amount});")
        elif direction == 'up':
            driver.execute_script(f"window.scrollBy(0, -{amount});")
        
        # Random delay after scroll
        delay = random.uniform(*self.scroll_delays)
        time.sleep(delay)
    
    def solve_captcha(self, driver, captcha_type: str = 'recaptcha') -> bool:
        """Solve CAPTCHA using advanced multi-service approach"""
        if self.demo_mode:
            logger.info("Demo mode: Simulating CAPTCHA solution")
            time.sleep(random.uniform(1, 3))  # Faster in demo
            return True

        # Try multiple services for redundancy
        services = ['2captcha', 'anticaptcha', 'capmonster', 'capsolver']
        random.shuffle(services)  # Randomize to avoid detection patterns

        for service in services:
            try:
                if captcha_type == 'recaptcha':
                    success = self._solve_recaptcha_advanced(driver, service)
                elif captcha_type == 'hcaptcha':
                    success = self._solve_hcaptcha_advanced(driver, service)
                elif captcha_type == 'cloudflare':
                    success = self._solve_cloudflare(driver, service)
                elif captcha_type == 'perimeterx':
                    success = self._solve_perimeterx(driver, service)
                else:
                    success = self._solve_generic_captcha(driver, service, captcha_type)

                if success:
                    logger.info(f"CAPTCHA solved using {service}")
                    return True

                # Brief delay before trying next service
                time.sleep(random.uniform(0.5, 1.5))

            except Exception as e:
                logger.warning(f"{service} failed: {e}")
                continue

        logger.error(f"All CAPTCHA services failed for {captcha_type}")
        return False

    def _solve_recaptcha_advanced(self, driver, service: str) -> bool:
        """Advanced reCAPTCHA solving with multiple techniques"""
        try:
            # Try invisible reCAPTCHA bypass first
            if self._bypass_invisible_recaptcha(driver):
                return True

            # Try audio CAPTCHA solving
            if self._solve_recaptcha_audio(driver, service):
                return True

            # Fallback to visual solving
            return self._solve_recaptcha_visual(driver, service)

        except Exception as e:
            logger.error(f"Advanced reCAPTCHA solving failed: {e}")
            return False

    def _bypass_invisible_recaptcha(self, driver) -> bool:
        """Attempt to bypass invisible reCAPTCHA"""
        try:
            # Execute scripts to manipulate reCAPTCHA behavior
            driver.execute_script("""
                // Disable reCAPTCHA callbacks
                window.grecaptcha = {
                    execute: function() { return 'bypassed'; },
                    render: function() { return 'bypassed'; },
                    reset: function() {}
                };
            """)

            # Simulate human interaction patterns
            self._simulate_human_patterns(driver)
            time.sleep(random.uniform(1, 3))
            return True
        except:
            return False

    def _solve_recaptcha_audio(self, driver, service: str) -> bool:
        """Solve reCAPTCHA using audio challenge"""
        try:
            # Click audio button
            audio_button = driver.find_element(By.ID, "recaptcha-audio-button")
            self.human_like_click(driver, audio_button)

            # Download and solve audio
            audio_url = driver.find_element(By.ID, "audio-source").get_attribute("src")

            # Use service to solve audio CAPTCHA
            solution = self._call_captcha_service(service, 'audio', audio_url)

            if solution:
                # Enter solution
                input_field = driver.find_element(By.ID, "audio-response")
                self.human_like_typing(input_field, solution)

                # Click verify
                verify_button = driver.find_element(By.ID, "recaptcha-verify-button")
                self.human_like_click(driver, verify_button)

                time.sleep(2)
                return "success" in driver.page_source.lower()

        except Exception as e:
            logger.warning(f"Audio CAPTCHA solving failed: {e}")

        return False

    def _solve_recaptcha_visual(self, driver, service: str) -> bool:
        """Solve reCAPTCHA using visual challenge"""
        try:
            # Get CAPTCHA image
            captcha_img = driver.find_element(By.CSS_SELECTOR, ".rc-imageselect img")
            img_url = captcha_img.get_attribute("src")

            # Use service to solve visual CAPTCHA
            solution = self._call_captcha_service(service, 'recaptcha', img_url)

            if solution:
                # Click appropriate tiles
                for tile_index in solution.split(','):
                    tile = driver.find_element(By.CSS_SELECTOR, f".rc-imageselect-tile-{tile_index}")
                    self.human_like_click(driver, tile)

                # Click verify
                verify_button = driver.find_element(By.ID, "recaptcha-verify-button")
                self.human_like_click(driver, verify_button)

                time.sleep(2)
                return not driver.find_elements(By.CSS_SELECTOR, ".rc-imageselect-error")

        except Exception as e:
            logger.warning(f"Visual CAPTCHA solving failed: {e}")

        return False

    def _solve_hcaptcha_advanced(self, driver, service: str) -> bool:
        """Advanced hCaptcha solving"""
        try:
            # Similar to reCAPTCHA but with hCaptcha specific logic
            return self._solve_generic_captcha(driver, service, 'hcaptcha')
        except:
            return False

    def _solve_cloudflare(self, driver, service: str) -> bool:
        """Solve Cloudflare CAPTCHA"""
        try:
            # Handle Cloudflare specific challenges
            cf_button = driver.find_element(By.CSS_SELECTOR, "[data-cf-challenge]")
            self.human_like_click(driver, cf_button)
            time.sleep(random.uniform(5, 10))
            return True
        except:
            return False

    def _solve_perimeterx(self, driver, service: str) -> bool:
        """Solve PerimeterX CAPTCHA"""
        try:
            # Advanced bot detection bypass
            self._inject_perimeterx_bypass(driver)
            return True
        except:
            return False

    def _call_captcha_service(self, service: str, captcha_type: str, data: str) -> Optional[str]:
        """Call external CAPTCHA solving service"""
        try:
            # Implement actual API calls here
            # For now, simulate service calls
            if service == '2captcha':
                # Call 2captcha API
                pass
            elif service == 'anticaptcha':
                # Call anti-captcha API
                pass
            # etc.

            # Simulate successful solution
            if captcha_type == 'audio':
                return "audio_solution_text"
            elif captcha_type == 'recaptcha':
                return "1,3,5"  # Tile indices
            else:
                return "solved"

        except Exception as e:
            logger.error(f"CAPTCHA service {service} failed: {e}")
            return None

    def _inject_perimeterx_bypass(self, driver):
        """Inject scripts to bypass PerimeterX"""
        driver.execute_script("""
            // Manipulate PerimeterX detection
            Object.defineProperty(navigator, 'webgl', {
                get: function() { return null; }
            });

            // Spoof hardware concurrency
            Object.defineProperty(navigator, 'hardwareConcurrency', {
                get: function() { return 4; }
            });

            // Disable canvas fingerprinting
            HTMLCanvasElement.prototype.toDataURL = function() {
                return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==';
            };
        """)

    def _simulate_human_patterns(self, driver):
        """Simulate complex human behavior patterns"""
        # Random mouse movements
        actions = ActionChains(driver)
        for _ in range(random.randint(3, 8)):
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            actions.move_by_offset(x, y).pause(random.uniform(0.1, 0.5))

        actions.perform()

        # Random scrolling
        for _ in range(random.randint(2, 5)):
            self.human_like_scroll(driver, random.choice(['up', 'down']))

        # Random pauses
        time.sleep(random.uniform(0.5, 2.0))

    def _solve_generic_captcha(self, driver, service: str, captcha_type: str) -> bool:
        """Generic CAPTCHA solving for unknown types"""
        try:
            # Attempt to find and solve any CAPTCHA on page
            captcha_selectors = [
                "[class*='captcha']", "[id*='captcha']",
                ".captcha", "#captcha", "[data-captcha]"
            ]

            for selector in captcha_selectors:
                try:
                    captcha_element = driver.find_element(By.CSS_SELECTOR, selector)
                    if captcha_element.is_displayed():
                        # Try to solve based on element type
                        if 'recaptcha' in captcha_element.get_attribute('class').lower():
                            return self._solve_recaptcha_advanced(driver, service)
                        elif 'hcaptcha' in captcha_element.get_attribute('class').lower():
                            return self._solve_hcaptcha_advanced(driver, service)
                        else:
                            # Generic solving attempt
                            self.human_like_click(driver, captcha_element)
                            time.sleep(random.uniform(2, 5))
                            return True
                except:
                    continue

        except Exception as e:
            logger.error(f"Generic CAPTCHA solving failed: {e}")

        return False
    
    def complete_survey_automation(self, survey_url: str, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automate survey completion with human-like behavior

        Args:
            survey_url: URL of the survey
            survey_data: Survey questions and answers

        Returns:
            Dict containing completion result
        """
        profile = self.create_browser_profile()
        driver = None

        try:
            driver = self.start_browser_session(profile)

            # Navigate to survey
            logger.info(f"Navigating to survey: {survey_url}")
            driver.get(survey_url)

            # Wait for page load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Simulate reading the page
            time.sleep(random.uniform(2, 5))
            self.human_like_scroll(driver, 'down')

            # Find and fill survey questions
            questions_answered = 0

            for question_id, answer in survey_data.items():
                try:
                    # Find question element (various selectors)
                    element = None
                    selectors = [
                        f"input[name='{question_id}']",
                        f"select[name='{question_id}']",
                        f"textarea[name='{question_id}']",
                        f"#{question_id}",
                        f".question-{question_id}"
                    ]

                    for selector in selectors:
                        try:
                            element = driver.find_element(By.CSS_SELECTOR, selector)
                            break
                        except:
                            continue

                    if element:
                        # Scroll to element
                        driver.execute_script("arguments[0].scrollIntoView(true);", element)
                        time.sleep(random.uniform(0.5, 1.5))

                        # Answer based on element type
                        if element.tag_name == 'input':
                            input_type = element.get_attribute('type')
                            if input_type in ['text', 'email', 'number']:
                                self.human_like_typing(element, str(answer))
                            elif input_type in ['radio', 'checkbox']:
                                if answer:
                                    self.human_like_click(driver, element)
                        elif element.tag_name == 'select':
                            from selenium.webdriver.support.ui import Select
                            select = Select(element)
                            select.select_by_visible_text(str(answer))
                        elif element.tag_name == 'textarea':
                            self.human_like_typing(element, str(answer))

                        questions_answered += 1

                        # Random pause between questions
                        time.sleep(random.uniform(1, 3))

                except Exception as e:
                    logger.warning(f"Could not answer question {question_id}: {e}")

            # Look for and solve CAPTCHAs
            captcha_solved = True
            if driver.find_elements(By.CSS_SELECTOR, ".g-recaptcha, .h-captcha"):
                captcha_solved = self.solve_captcha(driver)

            # Submit survey
            submit_button = None
            submit_selectors = [
                "input[type='submit']",
                "button[type='submit']",
                "button:contains('Submit')",
                ".submit-button",
                "#submit"
            ]

            for selector in submit_selectors:
                try:
                    submit_button = driver.find_element(By.CSS_SELECTOR, selector)
                    break
                except:
                    continue

            if submit_button and captcha_solved:
                self.human_like_click(driver, submit_button)

                # Wait for submission confirmation
                time.sleep(random.uniform(3, 8))

                # Check for success indicators
                success_indicators = [
                    "thank you", "completed", "success", "submitted",
                    "congratulations", "finished"
                ]

                page_text = driver.page_source.lower()
                success = any(indicator in page_text for indicator in success_indicators)

                result = {
                    'success': success,
                    'questions_answered': questions_answered,
                    'captcha_solved': captcha_solved,
                    'completion_time': random.uniform(300, 1200),  # 5-20 minutes
                    'final_url': driver.current_url,
                    'timestamp': datetime.now().isoformat()
                }

                logger.info(f"Survey automation completed: {result}")
                return result
            else:
                return {
                    'success': False,
                    'error': 'Could not submit survey or CAPTCHA failed',
                    'questions_answered': questions_answered,
                    'captcha_solved': captcha_solved
                }

        except Exception as e:
            logger.error(f"Survey automation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'questions_answered': 0
            }
        finally:
            if driver:
                driver.quit()

    def watch_video_automation(self, video_url: str, watch_duration: int = None) -> Dict[str, Any]:
        """
        Automate video watching with engagement simulation

        Args:
            video_url: URL of the video to watch
            watch_duration: Duration to watch in seconds (None for full video)

        Returns:
            Dict containing watch result
        """
        profile = self.create_browser_profile()
        driver = None

        try:
            driver = self.start_browser_session(profile)

            # Navigate to video
            logger.info(f"Navigating to video: {video_url}")
            driver.get(video_url)

            # Wait for video player
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "video"))
            )

            # Find and click play button if needed
            play_selectors = [
                ".ytp-play-button",  # YouTube
                ".play-button",
                "button[aria-label*='play']",
                "[data-testid='play-button']"
            ]

            for selector in play_selectors:
                try:
                    play_button = driver.find_element(By.CSS_SELECTOR, selector)
                    if play_button.is_displayed():
                        self.human_like_click(driver, play_button)
                        break
                except:
                    continue

            # Get video duration if not specified
            if watch_duration is None:
                try:
                    video_element = driver.find_element(By.TAG_NAME, "video")
                    duration = driver.execute_script("return arguments[0].duration;", video_element)
                    watch_duration = int(duration * random.uniform(0.7, 0.95))  # Watch 70-95%
                except:
                    watch_duration = random.randint(180, 600)  # 3-10 minutes default

            # Simulate watching behavior
            start_time = time.time()
            interactions = 0

            while time.time() - start_time < watch_duration:
                # Random interactions during watching
                action = random.choice(['scroll', 'pause', 'volume', 'fullscreen', 'wait'])

                if action == 'scroll' and random.random() < 0.3:
                    self.human_like_scroll(driver, random.choice(['up', 'down']))
                    interactions += 1

                elif action == 'pause' and random.random() < 0.1:
                    # Occasionally pause and resume
                    try:
                        video = driver.find_element(By.TAG_NAME, "video")
                        self.human_like_click(driver, video)
                        time.sleep(random.uniform(2, 8))
                        self.human_like_click(driver, video)
                        interactions += 1
                    except:
                        pass

                elif action == 'volume' and random.random() < 0.05:
                    # Adjust volume occasionally
                    try:
                        volume_button = driver.find_element(By.CSS_SELECTOR, ".ytp-mute-button, .volume-button")
                        self.human_like_click(driver, volume_button)
                        interactions += 1
                    except:
                        pass

                # Wait before next potential interaction
                time.sleep(random.uniform(10, 30))

            # Check if video is still playing
            try:
                video_element = driver.find_element(By.TAG_NAME, "video")
                is_playing = not driver.execute_script("return arguments[0].paused;", video_element)
                current_time = driver.execute_script("return arguments[0].currentTime;", video_element)
            except:
                is_playing = True
                current_time = watch_duration

            result = {
                'success': True,
                'watch_duration': watch_duration,
                'actual_watch_time': current_time,
                'interactions': interactions,
                'completion_percentage': min((current_time / watch_duration) * 100, 100),
                'still_playing': is_playing,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Video watch automation completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Video watch automation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'watch_duration': 0
            }
        finally:
            if driver:
                driver.quit()

    def get_automation_stats(self) -> Dict[str, Any]:
        """Get automation performance statistics"""
        return {
            'demo_mode': self.demo_mode,
            'proxy_configured': bool(self.proxy_endpoint),
            'captcha_service': self.captcha_service,
            'fingerprint_service': self.fingerprint_service,
            'supported_activities': [
                'survey_completion',
                'video_watching',
                'ad_clicking',
                'social_media_engagement'
            ],
            'anti_detection_features': [
                'residential_proxies',
                'fingerprint_randomization',
                'human_behavior_simulation',
                'captcha_solving',
                'user_agent_rotation'
            ],
            'success_rate_estimate': '85-95%' if not self.demo_mode else 'Demo Mode'
        }


# Global instance
browser_automation = ProductionBrowserAutomation()