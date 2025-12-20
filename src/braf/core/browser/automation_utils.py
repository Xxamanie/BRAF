"""
Browser automation utilities for BRAF.

This module provides high-level automation functions that integrate
behavioral patterns with browser interactions.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from playwright.async_api import Page, ElementHandle, Locator

from braf.core.behavioral import get_behavioral_engine
from braf.core.models import AutomationAction, ActionType

logger = logging.getLogger(__name__)


class AutomationExecutor:
    """Executor for automation actions with behavioral integration."""
    
    def __init__(self, page: Page):
        """
        Initialize automation executor.
        
        Args:
            page: Playwright page instance
        """
        self.page = page
        self.behavioral_engine = get_behavioral_engine()
    
    async def execute_action(self, action: AutomationAction) -> Dict[str, Any]:
        """
        Execute automation action with behavioral patterns.
        
        Args:
            action: Automation action to execute
            
        Returns:
            Execution result dictionary
        """
        logger.debug(f"Executing action: {action.type} on {action.selector}")
        
        # Pre-action delay
        await self.behavioral_engine.wait_with_human_delay(
            action.type.value, 
            action.metadata
        )
        
        try:
            if action.type == ActionType.NAVIGATE:
                result = await self._navigate(action)
            elif action.type == ActionType.CLICK:
                result = await self._click(action)
            elif action.type == ActionType.TYPE:
                result = await self._type_text(action)
            elif action.type == ActionType.WAIT:
                result = await self._wait(action)
            elif action.type == ActionType.EXTRACT:
                result = await self._extract_data(action)
            elif action.type == ActionType.SCROLL:
                result = await self._scroll(action)
            elif action.type == ActionType.HOVER:
                result = await self._hover(action)
            elif action.type == ActionType.SELECT:
                result = await self._select_option(action)
            elif action.type == ActionType.UPLOAD:
                result = await self._upload_file(action)
            elif action.type == ActionType.SCREENSHOT:
                result = await self._take_screenshot(action)
            else:
                raise ValueError(f"Unsupported action type: {action.type}")
            
            result["success"] = True
            result["action_type"] = action.type.value
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            result = {
                "success": False,
                "action_type": action.type.value,
                "error": str(e)
            }
        
        return result
    
    async def _navigate(self, action: AutomationAction) -> Dict[str, Any]:
        """Execute navigation action."""
        if not action.url:
            raise ValueError("URL is required for navigate action")
        
        # Simulate reading URL (if user would type it)
        if action.metadata.get("user_typed_url"):
            await self.behavioral_engine.simulate_reading(len(action.url))
        
        response = await self.page.goto(action.url, timeout=action.timeout * 1000)
        
        # Wait for page load with human-like delay
        await self.behavioral_engine.wait_with_human_delay("page_load")
        
        return {
            "url": action.url,
            "status": response.status if response else None,
            "final_url": self.page.url
        }
    
    async def _click(self, action: AutomationAction) -> Dict[str, Any]:
        """Execute click action with human-like mouse movement."""
        element = await self._find_element(action.selector, action.timeout)
        
        # Get element position for mouse movement
        box = await element.bounding_box()
        if box:
            # Calculate click position (center with slight randomization)
            import random
            click_x = box["x"] + box["width"] / 2 + random.uniform(-5, 5)
            click_y = box["y"] + box["height"] / 2 + random.uniform(-5, 5)
            
            # Generate human-like mouse movement
            movement_path = await self.behavioral_engine.move_mouse((click_x, click_y))
            
            # Simulate mouse movement (in real implementation, this would control actual mouse)
            for x, y, timestamp in movement_path[-5:]:  # Show last few points
                logger.debug(f"Mouse position: ({x:.1f}, {y:.1f}) at {timestamp:.3f}s")
        
        # Perform the click
        await element.click()
        
        return {
            "selector": action.selector,
            "element_found": True,
            "click_position": (click_x, click_y) if box else None
        }
    
    async def _type_text(self, action: AutomationAction) -> Dict[str, Any]:
        """Execute typing action with human-like patterns."""
        if not action.data:
            raise ValueError("Text data is required for type action")
        
        element = await self._find_element(action.selector, action.timeout)
        
        # Clear existing content first
        await element.clear()
        
        # Generate human-like typing sequence
        typing_sequence = await self.behavioral_engine.type_text(action.data)
        
        # Execute typing with realistic delays
        for char_or_action, delay in typing_sequence:
            if char_or_action == "backspace":
                await self.page.keyboard.press("Backspace")
            elif char_or_action == "pause":
                pass  # Delay is handled below
            else:
                await element.type(char_or_action)
            
            # Apply the delay
            await asyncio.sleep(delay)
        
        return {
            "selector": action.selector,
            "text_length": len(action.data),
            "keystrokes": len(typing_sequence)
        }
    
    async def _wait(self, action: AutomationAction) -> Dict[str, Any]:
        """Execute wait action."""
        wait_time = float(action.data) if action.data else 1.0
        
        # Use behavioral delay instead of simple sleep
        actual_delay = await self.behavioral_engine.wait_with_human_delay(
            "wait", 
            {"requested_duration": wait_time}
        )
        
        return {
            "requested_duration": wait_time,
            "actual_duration": actual_delay
        }
    
    async def _extract_data(self, action: AutomationAction) -> Dict[str, Any]:
        """Execute data extraction action."""
        element = await self._find_element(action.selector, action.timeout)
        
        # Determine what to extract
        attribute = action.metadata.get("attribute", "text")
        
        if attribute == "text":
            value = await element.text_content()
        elif attribute == "html":
            value = await element.inner_html()
        elif attribute == "value":
            value = await element.input_value()
        else:
            value = await element.get_attribute(attribute)
        
        # Simulate reading the extracted content
        if value:
            await self.behavioral_engine.simulate_reading(
                len(value), 
                action.metadata.get("complexity", "medium")
            )
        
        return {
            "selector": action.selector,
            "attribute": attribute,
            "value": value,
            "length": len(value) if value else 0
        }
    
    async def _scroll(self, action: AutomationAction) -> Dict[str, Any]:
        """Execute scroll action."""
        # Parse scroll parameters
        scroll_data = action.data or "0,300"  # Default: scroll down 300px
        
        try:
            if "," in scroll_data:
                delta_x, delta_y = map(int, scroll_data.split(","))
            else:
                delta_x, delta_y = 0, int(scroll_data)
        except ValueError:
            delta_x, delta_y = 0, 300
        
        # Perform scroll with human-like behavior
        if action.selector:
            # Scroll specific element
            element = await self._find_element(action.selector, action.timeout)
            await element.scroll_into_view_if_needed()
        else:
            # Scroll page
            await self.page.mouse.wheel(delta_x, delta_y)
        
        # Brief pause after scrolling
        await self.behavioral_engine.wait_with_human_delay("scroll")
        
        return {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "selector": action.selector
        }
    
    async def _hover(self, action: AutomationAction) -> Dict[str, Any]:
        """Execute hover action."""
        element = await self._find_element(action.selector, action.timeout)
        
        # Get element position for mouse movement
        box = await element.bounding_box()
        if box:
            hover_x = box["x"] + box["width"] / 2
            hover_y = box["y"] + box["height"] / 2
            
            # Generate mouse movement to hover position
            await self.behavioral_engine.move_mouse((hover_x, hover_y))
        
        await element.hover()
        
        # Brief pause while hovering
        await self.behavioral_engine.wait_with_human_delay("hover")
        
        return {
            "selector": action.selector,
            "hover_position": (hover_x, hover_y) if box else None
        }
    
    async def _select_option(self, action: AutomationAction) -> Dict[str, Any]:
        """Execute select option action."""
        if not action.data:
            raise ValueError("Option value is required for select action")
        
        element = await self._find_element(action.selector, action.timeout)
        
        # Simulate decision making for selection
        options = await element.locator("option").all()
        await self.behavioral_engine.simulate_decision_making(
            len(options), 
            action.metadata.get("complexity", "medium")
        )
        
        # Select the option
        await element.select_option(value=action.data)
        
        return {
            "selector": action.selector,
            "selected_value": action.data,
            "total_options": len(options)
        }
    
    async def _upload_file(self, action: AutomationAction) -> Dict[str, Any]:
        """Execute file upload action."""
        if not action.data:
            raise ValueError("File path is required for upload action")
        
        element = await self._find_element(action.selector, action.timeout)
        
        # Simulate file selection delay
        await self.behavioral_engine.simulate_decision_making(1, "simple")
        
        await element.set_input_files(action.data)
        
        return {
            "selector": action.selector,
            "file_path": action.data
        }
    
    async def _take_screenshot(self, action: AutomationAction) -> Dict[str, Any]:
        """Execute screenshot action."""
        screenshot_path = action.data or "screenshot.png"
        
        if action.selector:
            # Screenshot specific element
            element = await self._find_element(action.selector, action.timeout)
            await element.screenshot(path=screenshot_path)
        else:
            # Screenshot full page
            await self.page.screenshot(path=screenshot_path, full_page=True)
        
        return {
            "screenshot_path": screenshot_path,
            "selector": action.selector,
            "full_page": action.selector is None
        }
    
    async def _find_element(self, selector: str, timeout: int) -> Locator:
        """
        Find element with timeout and human-like behavior.
        
        Args:
            selector: CSS selector or XPath
            timeout: Timeout in seconds
            
        Returns:
            Element locator
            
        Raises:
            TimeoutError: If element not found within timeout
        """
        # Simulate visual search time
        await self.behavioral_engine.wait_with_human_delay(
            "element_search", 
            {"selector_complexity": len(selector)}
        )
        
        # Wait for element to be visible
        locator = self.page.locator(selector)
        await locator.wait_for(state="visible", timeout=timeout * 1000)
        
        return locator


class PageAnalyzer:
    """Analyzer for page content and structure."""
    
    def __init__(self, page: Page):
        """
        Initialize page analyzer.
        
        Args:
            page: Playwright page instance
        """
        self.page = page
    
    async def analyze_page_complexity(self) -> str:
        """
        Analyze page complexity for behavioral adjustments.
        
        Returns:
            Complexity level ('simple', 'medium', 'complex')
        """
        try:
            # Count various page elements
            element_counts = await self.page.evaluate("""
                () => {
                    return {
                        total_elements: document.querySelectorAll('*').length,
                        forms: document.querySelectorAll('form').length,
                        inputs: document.querySelectorAll('input, textarea, select').length,
                        links: document.querySelectorAll('a').length,
                        images: document.querySelectorAll('img').length,
                        scripts: document.querySelectorAll('script').length,
                        iframes: document.querySelectorAll('iframe').length
                    };
                }
            """)
            
            # Calculate complexity score
            complexity_score = 0
            
            # Base complexity from total elements
            if element_counts["total_elements"] > 1000:
                complexity_score += 3
            elif element_counts["total_elements"] > 500:
                complexity_score += 2
            elif element_counts["total_elements"] > 100:
                complexity_score += 1
            
            # Additional complexity factors
            if element_counts["forms"] > 3:
                complexity_score += 2
            elif element_counts["forms"] > 1:
                complexity_score += 1
            
            if element_counts["inputs"] > 10:
                complexity_score += 2
            elif element_counts["inputs"] > 5:
                complexity_score += 1
            
            if element_counts["iframes"] > 0:
                complexity_score += 2
            
            if element_counts["scripts"] > 20:
                complexity_score += 1
            
            # Determine complexity level
            if complexity_score >= 6:
                return "complex"
            elif complexity_score >= 3:
                return "medium"
            else:
                return "simple"
                
        except Exception as e:
            logger.warning(f"Failed to analyze page complexity: {e}")
            return "medium"  # Default to medium complexity
    
    async def detect_interactive_elements(self) -> List[Dict[str, Any]]:
        """
        Detect interactive elements on the page.
        
        Returns:
            List of interactive element information
        """
        try:
            elements = await self.page.evaluate("""
                () => {
                    const interactive = [];
                    
                    // Find clickable elements
                    const clickable = document.querySelectorAll(
                        'button, input[type="button"], input[type="submit"], a, [onclick], [role="button"]'
                    );
                    
                    clickable.forEach((el, index) => {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            interactive.push({
                                type: 'clickable',
                                tag: el.tagName.toLowerCase(),
                                text: el.textContent?.trim().substring(0, 50) || '',
                                id: el.id || '',
                                class: el.className || '',
                                position: {
                                    x: rect.left + rect.width / 2,
                                    y: rect.top + rect.height / 2
                                },
                                size: {
                                    width: rect.width,
                                    height: rect.height
                                }
                            });
                        }
                    });
                    
                    // Find input elements
                    const inputs = document.querySelectorAll('input, textarea, select');
                    inputs.forEach((el, index) => {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            interactive.push({
                                type: 'input',
                                tag: el.tagName.toLowerCase(),
                                input_type: el.type || 'text',
                                placeholder: el.placeholder || '',
                                id: el.id || '',
                                class: el.className || '',
                                position: {
                                    x: rect.left + rect.width / 2,
                                    y: rect.top + rect.height / 2
                                },
                                size: {
                                    width: rect.width,
                                    height: rect.height
                                }
                            });
                        }
                    });
                    
                    return interactive;
                }
            """)
            
            return elements
            
        except Exception as e:
            logger.warning(f"Failed to detect interactive elements: {e}")
            return []
    
    async def check_page_readiness(self) -> Dict[str, Any]:
        """
        Check if page is ready for automation.
        
        Returns:
            Page readiness information
        """
        try:
            readiness = await self.page.evaluate("""
                () => {
                    return {
                        document_ready: document.readyState === 'complete',
                        images_loaded: Array.from(document.images).every(img => img.complete),
                        no_pending_requests: performance.getEntriesByType('navigation')[0]?.loadEventEnd > 0,
                        jquery_ready: typeof jQuery !== 'undefined' ? jQuery.isReady : true,
                        visible_content: document.body?.offsetHeight > 0
                    };
                }
            """)
            
            # Calculate overall readiness score
            ready_count = sum(1 for value in readiness.values() if value)
            readiness["readiness_score"] = ready_count / len(readiness)
            readiness["is_ready"] = readiness["readiness_score"] >= 0.8
            
            return readiness
            
        except Exception as e:
            logger.warning(f"Failed to check page readiness: {e}")
            return {"is_ready": False, "readiness_score": 0.0}


async def execute_automation_sequence(
    page: Page, 
    actions: List[AutomationAction]
) -> List[Dict[str, Any]]:
    """
    Execute a sequence of automation actions with behavioral patterns.
    
    Args:
        page: Playwright page instance
        actions: List of actions to execute
        
    Returns:
        List of execution results
    """
    executor = AutomationExecutor(page)
    analyzer = PageAnalyzer(page)
    results = []
    
    # Analyze page complexity for behavioral adjustments
    complexity = await analyzer.analyze_page_complexity()
    logger.info(f"Page complexity: {complexity}")
    
    # Wait for page to be ready
    readiness = await analyzer.check_page_readiness()
    if not readiness["is_ready"]:
        logger.warning(f"Page may not be fully ready (score: {readiness['readiness_score']:.2f})")
    
    # Execute actions sequentially
    for i, action in enumerate(actions):
        logger.info(f"Executing action {i+1}/{len(actions)}: {action.type}")
        
        # Add complexity context to action
        action.metadata = action.metadata or {}
        action.metadata["page_complexity"] = complexity
        action.metadata["action_index"] = i
        action.metadata["total_actions"] = len(actions)
        
        # Execute action
        result = await executor.execute_action(action)
        results.append(result)
        
        # Stop on failure if not configured to continue
        if not result["success"] and not action.metadata.get("continue_on_error", False):
            logger.error(f"Action failed, stopping sequence: {result.get('error')}")
            break
        
        # Check if behavioral break is needed
        behavioral_engine = get_behavioral_engine()
        should_break, break_duration = await behavioral_engine.check_break_needed()
        
        if should_break:
            logger.info(f"Taking behavioral break ({break_duration:.1f}s)")
            await behavioral_engine.take_behavioral_break(break_duration)
    
    return results