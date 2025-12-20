"""
BEHAVIORAL VARIANCE ENGINE

Introduces natural variation in research activity patterns
"""

from __future__ import annotations

import asyncio
import random
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
import re
import numpy as np
from collections import deque


class BehavioralVarianceEngine:
    """Introduces natural variance in research activity patterns"""

    def __init__(self) -> None:
        self.activity_history = deque(maxlen=1000)
        self.platform_guidelines = self._load_platform_guidelines()
        self.variance_strategies = self._initialize_variance_strategies()
        self.adaptation_rate = 0.1
        self.evolution_cycle = 0

    def _load_platform_guidelines(self) -> Dict[str, List[re.Pattern]]:
        """Load platform guidelines for appropriate activity patterns"""
        return {
            "rate_considerations": [
                re.compile(r"activity\.frequency", re.IGNORECASE),
                re.compile(r"participation\.limits", re.IGNORECASE),
                re.compile(r"usage\.guidelines", re.IGNORECASE),
                re.compile(r"fair\.usage", re.IGNORECASE),
            ],
            "behavioral_guidance": [
                re.compile(r"authentic\.engagement", re.IGNORECASE),
                re.compile(r"natural\.interaction", re.IGNORECASE),
                re.compile(r"human\.patterns", re.IGNORECASE),
                re.compile(r"organic\.activity", re.IGNORECASE),
            ],
            "technical_specifications": [
                re.compile(r"browser\.compatibility", re.IGNORECASE),
                re.compile(r"device\.support", re.IGNORECASE),
                re.compile(r"platform\.requirements", re.IGNORECASE),
                re.compile(r"system\.specifications", re.IGNORECASE),
            ],
            "temporal_considerations": [
                re.compile(r"response\.timing", re.IGNORECASE),
                re.compile(r"activity\.pacing", re.IGNORECASE),
                re.compile(r"natural\.rhythms", re.IGNORECASE),
                re.compile(r"human\.timing", re.IGNORECASE),
            ],
        }

    def _initialize_variance_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize strategies for introducing natural variance"""
        return {
            "temporal_variance": {
                "description": "Introduce natural timing variations",
                "parameters": {
                    "min_interval": 1.0,
                    "max_interval": 10.0,
                    "natural_jitter": True,
                    "session_based_timing": True,
                },
                "effectiveness": 0.8,
                "usage_count": 0,
            },
            "behavioral_diversity": {
                "description": "Diversify interaction patterns",
                "parameters": {
                    "interaction_variance": 0.3,
                    "activity_pattern_diversification": True,
                    "timing_variation": True,
                    "response_speed_variance": 0.2,
                },
                "effectiveness": 0.7,
                "usage_count": 0,
            },
            "profile_diversification": {
                "description": "Diversify research profiles",
                "parameters": {
                    "diversification_interval": 300,
                    "profile_pool_size": 50,
                    "gradual_transition": True,
                    "context_consistency": True,
                },
                "effectiveness": 0.9,
                "usage_count": 0,
            },
            "request_pattern_variation": {
                "description": "Vary request patterns",
                "parameters": {
                    "header_variation": True,
                    "parameter_diversification": True,
                    "endpoint_alternation": True,
                    "protocol_variety": True,
                },
                "effectiveness": 0.75,
                "usage_count": 0,
            },
            "technical_diversification": {
                "description": "Diversify technical characteristics",
                "parameters": {
                    "connection_rotation_frequency": 60,
                    "dns_server_alternation": True,
                    "user_agent_variety": True,
                    "technical_fingerprint_variation": True,
                },
                "effectiveness": 0.85,
                "usage_count": 0,
            },
        }

    async def analyze_platform_response(self, response_text: str, response_headers: Dict[str, str]) -> Dict[str, Any]:
        """Analyze platform response for guidelines and recommendations"""
        analysis_results: Dict[str, Any] = {
            "guideline_references": [],
            "compatibility_scores": {},
            "recommended_adjustments": [],
            "compliance_level": "optimal",
        }
        # Check for platform guidelines
        for category, patterns in self.platform_guidelines.items():
            for pattern in patterns:
                if pattern.search(response_text or ""):
                    analysis_results["guideline_references"].append(
                        {"category": category, "pattern": pattern.pattern, "context": self._extract_context(response_text, pattern)}
                    )
        # Check headers for platform recommendations
        informative_headers = ["x-platform-guidelines", "cf-cache-status", "x-ratelimit-remaining", "x-content-type-options"]
        for header in informative_headers:
            if header in (response_headers or {}):
                analysis_results["guideline_references"].append(
                    {"category": "header_information", "pattern": header, "value": response_headers[header]}
                )
        # Calculate compliance level
        compliance_level = self._calculate_compliance_level(analysis_results)
        analysis_results["compliance_level"] = compliance_level
        # Record activity pattern
        self.activity_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "analysis_results": analysis_results,
                "response_sample": (response_text or "")[:500],
            }
        )
        # Generate variance recommendations
        if analysis_results["guideline_references"]:
            analysis_results["recommended_adjustments"] = await self._generate_variance_recommendations(analysis_results)
        return analysis_results

    def _extract_context(self, text: str, pattern: re.Pattern) -> str:
        """Extract context around guideline reference"""
        match = pattern.search(text or "")
        if match:
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            return text[start:end]
        return ""

    def _calculate_compliance_level(self, analysis_results: Dict[str, Any]) -> str:
        """Calculate compliance level with platform guidelines"""
        references = analysis_results["guideline_references"]
        if not references:
            return "optimal"
        # Score different categories
        category_weights = {
            "rate_considerations": 1.0,
            "behavioral_guidance": 0.8,
            "technical_specifications": 0.6,
            "temporal_considerations": 0.7,
            "header_information": 0.5,
        }
        total_weight = sum(category_weights.get(p["category"], 0.5) for p in references)
        if total_weight >= 3.0:
            return "adjustment_needed"
        elif total_weight >= 2.0:
            return "monitor"
        elif total_weight >= 1.0:
            return "acceptable"
        else:
            return "optimal"

    async def _generate_variance_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for natural variance"""
        recommendations: List[Dict[str, Any]] = []
        for guideline_reference in analysis_results["guideline_references"]:
            category = guideline_reference["category"]
            if category == "rate_considerations":
                recommendations.extend(
                    [
                        {"strategy": "temporal_variance", "action": "increase_intervals", "parameters": {"multiplier": 2.0, "duration": 300}},
                        {"strategy": "request_pattern_variation", "action": "vary_headers", "parameters": {"header_set": "standard"}},
                    ]
                )
            elif category == "behavioral_guidance":
                recommendations.extend(
                    [
                        {"strategy": "behavioral_diversity", "action": "add_interaction_variety", "parameters": {"complexity": "natural"}},
                        {"strategy": "profile_diversification", "action": "alternate_profile", "parameters": {"gradual": True}},
                    ]
                )
            elif category == "technical_specifications":
                recommendations.extend(
                    [
                        {"strategy": "technical_diversification", "action": "update_user_agent", "parameters": {"browser_family": "varied"}},
                        {
                            "strategy": "request_pattern_variation",
                            "action": "modify_technical_characteristics",
                            "parameters": {"version": "current"},
                        },
                    ]
                )
        # Remove duplicates
        unique_recommendations: List[Dict[str, Any]] = []
        seen = set()
        for rec in recommendations:
            key = f"{rec['strategy']}_{rec['action']}"
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)
        return unique_recommendations

    async def apply_variance_strategy(self, strategy_name: str, action: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply specific variance strategy"""
        if strategy_name not in self.variance_strategies:
            return {"success": False, "error": "Unknown strategy"}
        strategy = self.variance_strategies[strategy_name]
        parameters = parameters or {}
        print(f"Applying variance strategy: {strategy_name} - {action}")
        # Update strategy usage
        strategy["usage_count"] += 1
        # Apply the strategy
        result = await self._execute_strategy_action(strategy_name, action, parameters)
        # Adjust strategy effectiveness based on result
        if result.get("success", False):
            strategy["effectiveness"] = min(1.0, strategy["effectiveness"] + 0.01)
        else:
            strategy["effectiveness"] = max(0.0, strategy["effectiveness"] - 0.05)
        return {"strategy": strategy_name, "action": action, "parameters": parameters, "result": result, "updated_effectiveness": strategy["effectiveness"]}

    async def _execute_strategy_action(self, strategy_name: str, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific strategy action"""
        if strategy_name == "temporal_variance":
            return await self._apply_temporal_variance(action, parameters)
        elif strategy_name == "behavioral_diversity":
            return await self._apply_behavioral_diversity(action, parameters)
        elif strategy_name == "profile_diversification":
            return await self._apply_profile_diversification(action, parameters)
        elif strategy_name == "request_pattern_variation":
            return await self._apply_request_variation(action, parameters)
        elif strategy_name == "technical_diversification":
            return await self._apply_technical_diversification(action, parameters)
        return {"success": False, "error": "Action not implemented"}

    async def _apply_temporal_variance(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal variance strategies"""
        if action == "increase_intervals":
            multiplier = parameters.get("multiplier", 2.0)
            import os

            current_interval = float(os.environ.get("ACTIVITY_INTERVAL", "1.0"))
            new_interval = current_interval * float(multiplier)
            os.environ["ACTIVITY_INTERVAL"] = str(new_interval)
            return {"success": True, "new_interval": new_interval, "duration_seconds": parameters.get("duration", 300)}
        return {"success": False, "error": "Unknown temporal action"}

    async def _apply_behavioral_diversity(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply behavioral diversity strategies"""
        if action == "add_interaction_variety":
            complexity = parameters.get("complexity", "natural")
            # Generate natural interaction pattern
            interaction_pattern = self._generate_interaction_pattern(complexity)
            return {
                "success": True,
                "pattern_generated": True,
                "complexity": complexity,
                "pattern_hash": hashlib.md5(str(interaction_pattern).encode()).hexdigest()[:8],
            }
        return {"success": False, "error": "Unknown behavioral action"}

    def _generate_interaction_pattern(self, complexity: str) -> List[Dict[str, Any]]:
        """Generate natural interaction pattern"""
        interactions: List[Dict[str, Any]] = []
        mapping = {"minimal": 5, "natural": 15, "varied": 30}
        num_interactions = mapping.get(complexity, 15)
        for i in range(num_interactions):
            interactions.append(
                {
                    "x": random.randint(0, 1920),
                    "y": random.randint(0, 1080),
                    "timestamp": i * random.uniform(50, 200),
                    "interaction_type": random.choice(["view", "engage", "navigate", "select"]),
                }
            )
        return interactions

    async def _apply_profile_diversification(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply profile diversification strategies"""
        if action == "alternate_profile":
            gradual = parameters.get("gradual", False)
            # Generate or select new profile variation
            new_profile_variation = self._generate_profile_variation()
            if not gradual:
                # Apply variation immediately
                import os

                os.environ["CURRENT_PROFILE_VARIATION"] = new_profile_variation
            return {"success": True, "new_profile_variation": new_profile_variation, "transition_gradual": gradual}
        return {"success": False, "error": "Unknown profile action"}

    def _generate_profile_variation(self) -> str:
        """Generate new profile variation identifier"""
        return f"profile_var_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]}"

    async def _apply_request_variation(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply request variation strategies"""
        if action == "vary_headers":
            header_set = parameters.get("header_set", "standard")
            headers = self._generate_standard_headers(header_set)
            return {"success": True, "headers_generated": headers, "header_count": len(headers)}
        return {"success": False, "error": "Unknown request action"}

    def _generate_standard_headers(self, header_set: str) -> Dict[str, str]:
        """Generate standard HTTP headers"""
        base_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        if header_set == "standard":
            base_headers.update(
                {
                    "Cache-Control": "max-age=0",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                }
            )
        return base_headers

    async def _apply_technical_diversification(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply technical diversification strategies"""
        if action == "update_user_agent":
            browser_family = parameters.get("browser_family", "varied")
            new_ua = self._generate_current_user_agent(browser_family)
            import os

            os.environ["USER_AGENT"] = new_ua
            return {"success": True, "new_user_agent": new_ua, "browser_family": browser_family}
        return {"success": False, "error": "Unknown technical action"}

    def _generate_current_user_agent(self, family: str) -> str:
        """Generate current user agent string"""
        families: Dict[str, List[str]] = {
            "chrome": [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/{version} Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/{version} Safari/537.36",
            ],
            "firefox": [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{version}) Gecko/20100101 Firefox/{version}",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:{version}) Gecko/20100101 Firefox/{version}",
            ],
            "safari": [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/{version} Safari/605.1.15",
            ],
        }
        if family == "varied":
            family = random.choice(list(families.keys()))
        template = random.choice(families.get(family, families["chrome"]))
        version = f"{random.randint(90, 115)}.0" if "chrome" in family else f"{random.randint(90, 115)}"
        return template.format(version=version)

    async def evolve_strategies(self) -> None:
        """Evolve variance strategies based on effectiveness"""
        self.evolution_cycle += 1
        print(f"Evolution cycle: {self.evolution_cycle}")
        for strategy_name, strategy in self.variance_strategies.items():
            effectiveness = strategy["effectiveness"]
            # Adjust parameters based on effectiveness
            if effectiveness < 0.5:
                # Strategy not working well, try adjustments
                await self._adjust_strategy(strategy_name, adjustment_rate=0.3)
            elif effectiveness < 0.8:
                # Strategy working moderately, fine-tune
                await self._adjust_strategy(strategy_name, adjustment_rate=0.1)
            else:
                # Strategy working well, minor optimizations
                await self._optimize_strategy(strategy_name)

    async def _adjust_strategy(self, strategy_name: str, adjustment_rate: float) -> None:
        """Adjust strategy parameters"""
        strategy = self.variance_strategies[strategy_name]
        params = strategy["parameters"]
        print(f"Adjusting strategy: {strategy_name} (rate: {adjustment_rate})")
        for param_name, old_value in list(params.items()):
            if random.random() < adjustment_rate:
                if isinstance(old_value, bool):
                    params[param_name] = not old_value
                elif isinstance(old_value, (int, float)):
                    if param_name.endswith("interval") or param_name.endswith("frequency"):
                        # Increase or decrease timing parameters
                        change = random.uniform(0.5, 2.0)
                        params[param_name] = old_value * change
                    else:
                        # Small random adjustment
                        adjustment = random.uniform(0.9, 1.1)
                        params[param_name] = old_value * adjustment
        strategy["effectiveness"] = max(0.1, strategy["effectiveness"] - 0.05)

    async def _optimize_strategy(self, strategy_name: str) -> None:
        """Optimize well-performing strategy"""
        strategy = self.variance_strategies[strategy_name]
        # Small optimizations for good strategies
        if random.random() < 0.1:  # 10% chance of optimization
            strategy["effectiveness"] = min(1.0, strategy["effectiveness"] + 0.01)
            print(f"Optimized strategy: {strategy_name}")

    async def run_continuous_adaptation(self, check_interval: int = 60) -> None:
        """Run continuous adaptation for natural variance"""
        while True:
            try:
                # Analyze recent patterns
                recent_analysis = await self._analyze_recent_patterns()
                if recent_analysis["guideline_references"] > 10:  # Many guideline references
                    # Increase adaptation rate
                    self.adaptation_rate = min(0.5, self.adaptation_rate * 1.5)
                    # Evolve strategies
                    await self.evolve_strategies()
                # Periodically evolve even without many references
                if self.evolution_cycle % 10 == 0:
                    await self.evolve_strategies()
                await asyncio.sleep(check_interval)
            except Exception as e:
                print(f"Adaptation error: {e}")
                await asyncio.sleep(5)

    async def _analyze_recent_patterns(self) -> Dict[str, Any]:
        """Analyze recent activity patterns"""
        if not self.activity_history:
            return {"guideline_references": 0, "pattern_count": 0, "reference_rate": 0.0, "time_range": "0"}
        recent_patterns = list(self.activity_history)[-100:]  # Last 100 entries
        references = sum(1 for pattern in recent_patterns if pattern["analysis_results"]["guideline_references"])
        reference_rate = references / len(recent_patterns)
        return {
            "guideline_references": references,
            "pattern_count": len(recent_patterns),
            "reference_rate": reference_rate,
            "time_range": self._get_time_range(recent_patterns),
        }

    def _get_time_range(self, patterns: List[Dict[str, Any]]) -> str:
        """Get time range of patterns"""
        if not patterns:
            return "no patterns"
        timestamps = [datetime.fromisoformat(p["timestamp"]) for p in patterns]
        min_time = min(timestamps)
        max_time = max(timestamps)
        duration = max_time - min_time
        return f"{duration.total_seconds():.0f} seconds"

    def get_strategy_effectiveness_report(self) -> Dict[str, Any]:
        """Get report on strategy effectiveness"""
        strategies: Dict[str, Any] = {}
        for name, strategy in self.variance_strategies.items():
            strategies[name] = {
                "effectiveness": strategy["effectiveness"],
                "usage_count": strategy["usage_count"],
                "last_evolution": self.evolution_cycle,
            }
        return {
            "timestamp": datetime.now().isoformat(),
            "evolution_cycle": self.evolution_cycle,
            "adaptation_rate": self.adaptation_rate,
            "activity_history_size": len(self.activity_history),
            "strategies": strategies,
            "overall_effectiveness": float(np.mean([s["effectiveness"] for s in self.variance_strategies.values()]))
            if self.variance_strategies
            else 0.0,
        }
