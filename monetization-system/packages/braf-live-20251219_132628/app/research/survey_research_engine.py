"""
Survey Research Engine
Research module for survey platform automation and optimization patterns
"""

import asyncio
import random
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
from bs4 import BeautifulSoup
from .advanced_config import research_config

class SurveyResearchEngine:
    """Research engine for survey platform optimization and automation patterns"""
    
    def __init__(self):
        self.answer_database = self._load_answer_research_database()
        self.platform_patterns = self._load_platform_research_patterns()
        self.session_pool = {}
        self.research_earnings = 0.0
        self.surveys_researched = 0
        self.research_data = []
    
    def _load_answer_research_database(self) -> Dict[str, Any]:
        """Load research database of survey answer patterns"""
        return {
            "demographic_research": {
                "age_distribution": {
                    "18-24": {"weight": 0.15, "survey_value": "medium"},
                    "25-34": {"weight": 0.25, "survey_value": "high"},
                    "35-44": {"weight": 0.20, "survey_value": "very_high"},
                    "45-54": {"weight": 0.18, "survey_value": "high"},
                    "55+": {"weight": 0.22, "survey_value": "medium"}
                },
                "income_research": {
                    "under_30000": {"weight": 0.20, "qualification_rate": 0.3},
                    "30000-50000": {"weight": 0.25, "qualification_rate": 0.5},
                    "50000-75000": {"weight": 0.30, "qualification_rate": 0.8},
                    "75000-100000": {"weight": 0.15, "qualification_rate": 0.9},
                    "100000+": {"weight": 0.10, "qualification_rate": 0.95}
                },
                "education_impact": {
                    "high_school": {"weight": 0.30, "survey_access": 0.6},
                    "some_college": {"weight": 0.25, "survey_access": 0.7},
                    "bachelor": {"weight": 0.30, "survey_access": 0.9},
                    "master": {"weight": 0.10, "survey_access": 0.95},
                    "doctorate": {"weight": 0.05, "survey_access": 0.98}
                }
            },
            "response_pattern_research": {
                "brand_awareness_optimization": {
                    "very_familiar": {"selection_rate": 0.40, "follow_up_probability": 0.8},
                    "somewhat_familiar": {"selection_rate": 0.35, "follow_up_probability": 0.6},
                    "heard_of": {"selection_rate": 0.20, "follow_up_probability": 0.3},
                    "never_heard": {"selection_rate": 0.05, "follow_up_probability": 0.1}
                },
                "purchase_intent_research": {
                    "definitely": {"selection_rate": 0.10, "survey_value": "very_high"},
                    "probably": {"selection_rate": 0.25, "survey_value": "high"},
                    "maybe": {"selection_rate": 0.40, "survey_value": "medium"},
                    "probably_not": {"selection_rate": 0.20, "survey_value": "low"},
                    "definitely_not": {"selection_rate": 0.05, "survey_value": "very_low"}
                }
            },
            "qualification_research": {
                "high_value_indicators": [
                    "primary_shopper", "decision_maker", "homeowner",
                    "employed_full_time", "college_educated", "high_income"
                ],
                "disqualification_triggers": [
                    "student", "unemployed", "rarely_shop", "no_income",
                    "under_18", "survey_taker", "market_research_employee"
                ]
            }
        }
    
    def _load_platform_research_patterns(self) -> Dict[str, Any]:
        """Load research patterns for different survey platforms"""
        return {
            "swagbucks_research": {
                "qualification_research": {
                    "common_questions": [
                        "What is your age?",
                        "What is your household income?",
                        "What is your employment status?",
                        "Have you purchased [product] in the last 6 months?"
                    ],
                    "optimal_answers": {
                        "age": "25-44",
                        "income": "50000-100000",
                        "employment": "employed_full_time",
                        "purchase_history": "yes_recent"
                    }
                },
                "earning_research": {
                    "point_values": {
                        "short_survey": {"min": 25, "max": 75, "avg_time": 5},
                        "medium_survey": {"min": 75, "max": 150, "avg_time": 10},
                        "long_survey": {"min": 150, "max": 300, "avg_time": 20},
                        "high_value": {"min": 300, "max": 1000, "avg_time": 30}
                    },
                    "optimization_factors": {
                        "completion_speed": 0.3,
                        "answer_consistency": 0.4,
                        "demographic_match": 0.3
                    }
                }
            },
            "survey_junkie_research": {
                "profile_optimization": {
                    "high_value_demographics": {
                        "age": "35-54",
                        "income": "75000+",
                        "education": "bachelor_or_higher",
                        "employment": "professional"
                    },
                    "qualification_boost_factors": [
                        "homeowner", "parent", "primary_shopper",
                        "decision_maker", "brand_loyal"
                    ]
                },
                "earning_patterns": {
                    "points_per_minute": {"min": 8, "max": 15, "optimal": 12},
                    "daily_earning_potential": {"min": 50, "max": 300, "avg": 150},
                    "qualification_rate_factors": {
                        "profile_completeness": 0.4,
                        "demographic_desirability": 0.3,
                        "response_consistency": 0.3
                    }
                }
            },
            "toluna_research": {
                "community_engagement": {
                    "participation_boost": {
                        "polls": 0.1,
                        "discussions": 0.2,
                        "product_tests": 0.3,
                        "diary_studies": 0.4
                    }
                },
                "geographic_optimization": {
                    "high_value_regions": ["US", "UK", "CA", "AU"],
                    "survey_availability_by_region": {
                        "US": 0.9,
                        "UK": 0.7,
                        "CA": 0.6,
                        "AU": 0.5
                    }
                }
            }
        }
    
    async def research_survey_optimization(
        self,
        platform: str,
        account: Dict,
        proxy: Optional[str] = None,
        max_surveys: int = 10
    ) -> Dict:
        """
        Research survey optimization patterns and strategies
        
        Args:
            platform: Platform name for research
            account: Account credentials for research
            proxy: Proxy to use for research
            max_surveys: Maximum surveys to research
            
        Returns:
            Research results on survey optimization
        """
        
        research_results = {
            "platform": platform,
            "research_type": "survey_optimization",
            "start_time": datetime.now().isoformat(),
            "surveys_researched": 0,
            "surveys_completed": 0,
            "total_research_earnings": 0.0,
            "disqualifications": 0,
            "optimization_insights": [],
            "survey_details": [],
            "research_patterns": []
        }
        
        for i in range(max_surveys):
            try:
                print(f"Researching survey {i+1}/{max_surveys} on {platform}")
                
                # Get available surveys for research
                available_surveys = await self._get_research_surveys(platform, account, proxy)
                
                if not available_surveys:
                    print("No surveys available for research")
                    continue
                
                # Select optimal survey for research
                selected_survey = self._select_research_optimal_survey(available_surveys, account)
                
                # Research qualification patterns
                qualification_research = await self._research_qualification_patterns(
                    platform, selected_survey, account, proxy
                )
                
                if not qualification_research["qualified"]:
                    research_results["disqualifications"] += 1
                    research_results["optimization_insights"].append({
                        "type": "disqualification_pattern",
                        "survey_id": selected_survey["id"],
                        "disqualification_reason": qualification_research.get("reason"),
                        "demographic_mismatch": qualification_research.get("demographic_issues")
                    })
                    continue
                
                # Research survey completion patterns
                completion_research = await self._research_survey_completion(
                    platform, selected_survey, account, proxy
                )
                
                if completion_research["success"]:
                    research_results["surveys_completed"] += 1
                    research_results["total_research_earnings"] += completion_research["earnings"]
                    research_results["survey_details"].append(completion_research)
                    
                    # Collect optimization insights
                    self._collect_optimization_insights(completion_research, research_results)
                    
                    self.surveys_researched += 1
                    self.research_earnings += completion_research["earnings"]
                
                research_results["surveys_researched"] += 1
                
                # Research timing patterns
                delay = random.uniform(15, 45)  # Research realistic delays
                await asyncio.sleep(delay / 10)  # Speed up for research
                
            except Exception as e:
                print(f"Survey research iteration failed: {e}")
                continue
        
        research_results["end_time"] = datetime.now().isoformat()
        research_results["research_patterns"] = self._analyze_research_patterns()
        research_results["hourly_rate"] = self._calculate_research_hourly_rate(research_results)
        
        return research_results
    
    async def _get_research_surveys(
        self,
        platform: str,
        account: Dict,
        proxy: Optional[str] = None
    ) -> List[Dict]:
        """Get available surveys for research purposes"""
        
        # Generate research surveys based on platform patterns
        surveys = []
        num_surveys = random.randint(3, 12)
        
        platform_config = self.platform_patterns.get(f"{platform}_research", {})
        earning_config = platform_config.get("earning_research", {})
        point_values = earning_config.get("point_values", {})
        
        for i in range(num_surveys):
            survey_type = random.choice(list(point_values.keys()) if point_values else ["medium_survey"])
            type_config = point_values.get(survey_type, {"min": 50, "max": 200, "avg_time": 10})
            
            point_value = random.randint(type_config["min"], type_config["max"])
            estimated_time = type_config["avg_time"] + random.randint(-3, 5)
            
            surveys.append({
                "id": f"research_survey_{i}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                "title": f"Research {random.choice(['Consumer', 'Market', 'Product', 'Brand'])} Study",
                "estimated_time": max(1, estimated_time),
                "points": point_value,
                "cash_value": point_value / 100,  # Research conversion rate
                "qualification_criteria": self._generate_research_criteria(),
                "category": random.choice([
                    "Technology", "Consumer Goods", "Automotive",
                    "Healthcare", "Finance", "Travel", "Food & Beverage"
                ]),
                "research_metadata": {
                    "survey_type": survey_type,
                    "difficulty_level": random.choice(["easy", "medium", "hard"]),
                    "demographic_targeting": random.choice(["broad", "specific", "niche"])
                }
            })
        
        return surveys
    
    def _generate_research_criteria(self) -> List[str]:
        """Generate research qualification criteria"""
        criteria_pool = [
            "Age 25-45", "Age 35-55", "Income $50k+", "Income $75k+",
            "College educated", "Homeowner", "Parent", "Employed full-time",
            "Primary shopper", "Decision maker", "Brand aware",
            "Recent purchaser", "Tech savvy", "Health conscious"
        ]
        return random.sample(criteria_pool, random.randint(2, 5))
    
    def _select_research_optimal_survey(self, surveys: List[Dict], account: Dict) -> Dict:
        """Select optimal survey for research based on earnings per minute"""
        
        best_survey = None
        best_score = 0
        
        for survey in surveys:
            earnings = survey["cash_value"]
            time_minutes = survey["estimated_time"]
            epm = earnings / time_minutes if time_minutes > 0 else 0
            
            # Research qualification match scoring
            qualification_match = self._research_qualification_match(survey, account)
            
            # Research difficulty adjustment
            difficulty = survey.get("research_metadata", {}).get("difficulty_level", "medium")
            difficulty_multiplier = {"easy": 1.2, "medium": 1.0, "hard": 0.8}[difficulty]
            
            # Calculate research score
            research_score = epm * qualification_match * difficulty_multiplier
            
            if research_score > best_score:
                best_score = research_score
                best_survey = survey
        
        return best_survey if best_survey else surveys[0]
    
    def _research_qualification_match(self, survey: Dict, account: Dict) -> float:
        """Research qualification matching algorithm"""
        
        identity = account.get("identity_data", {})
        criteria = survey.get("qualification_criteria", [])
        
        match_score = 0.5  # Base score
        
        # Research demographic matching
        for criterion in criteria:
            if "Age" in criterion:
                # Extract age range and match
                if "25-45" in criterion and identity.get("age", 30) in range(25, 46):
                    match_score += 0.2
                elif "35-55" in criterion and identity.get("age", 30) in range(35, 56):
                    match_score += 0.2
            
            elif "Income" in criterion:
                income_level = identity.get("income_range", "50000-75000")
                if "$50k+" in criterion and any(x in income_level for x in ["50000", "75000", "100000"]):
                    match_score += 0.25
                elif "$75k+" in criterion and any(x in income_level for x in ["75000", "100000"]):
                    match_score += 0.3
            
            elif "College" in criterion:
                education = identity.get("education_level", "bachelor")
                if education in ["bachelor", "master", "doctorate"]:
                    match_score += 0.2
            
            elif "Homeowner" in criterion:
                match_score += 0.15  # Assume positive match for research
            
            elif "Parent" in criterion:
                children = identity.get("children", "0")
                if children != "0":
                    match_score += 0.15
            
            elif "Employed" in criterion:
                employment = identity.get("employment_status", "employed")
                if "employed" in employment:
                    match_score += 0.1
        
        return min(1.0, match_score)
    
    async def _research_qualification_patterns(
        self,
        platform: str,
        survey: Dict,
        account: Dict,
        proxy: Optional[str] = None
    ) -> Dict:
        """Research qualification patterns and optimization strategies"""
        
        qualification_result = {
            "qualified": False,
            "questions_answered": 0,
            "time_spent": 0,
            "demographic_issues": [],
            "optimization_opportunities": []
        }
        
        start_time = datetime.now()
        
        # Simulate qualification process
        identity = account.get("identity_data", {})
        platform_patterns = self.platform_patterns.get(f"{platform}_research", {})
        
        # Research qualification questions
        num_questions = random.randint(3, 8)
        qualification_result["questions_answered"] = num_questions
        
        for i in range(num_questions):
            # Generate research question and optimal answer
            question_type = random.choice(["demographic", "behavioral", "product_specific"])
            
            if question_type == "demographic":
                answer = self._generate_research_demographic_answer(identity)
            elif question_type == "behavioral":
                answer = self._generate_research_behavioral_answer(identity)
            else:
                answer = self._generate_research_product_answer(survey)
            
            # Research disqualification triggers
            disqualification_triggers = platform_patterns.get("qualification_research", {}).get("disqualification_triggers", [])
            
            if any(trigger in answer.lower() for trigger in disqualification_triggers):
                qualification_result["demographic_issues"].append({
                    "question_type": question_type,
                    "problematic_answer": answer,
                    "trigger": "disqualification_keyword"
                })
                qualification_result["qualified"] = False
                qualification_result["reason"] = f"Disqualified on {question_type} question"
                break
            
            # Small chance of random disqualification for research
            if random.random() < 0.08:  # 8% random disqualification
                qualification_result["qualified"] = False
                qualification_result["reason"] = "Random platform disqualification"
                break
        else:
            # Qualified if made it through all questions
            qualification_result["qualified"] = True
        
        end_time = datetime.now()
        qualification_result["time_spent"] = (end_time - start_time).total_seconds()
        
        # Research optimization opportunities
        if not qualification_result["qualified"]:
            qualification_result["optimization_opportunities"] = self._identify_research_optimizations(
                qualification_result, identity, platform
            )
        
        return qualification_result
    
    def _generate_research_demographic_answer(self, identity: Dict) -> str:
        """Generate research demographic answer based on identity"""
        
        demographics = self.answer_database["demographic_research"]
        
        # Select answer based on research optimization
        answer_categories = ["age_distribution", "income_research", "education_impact"]
        category = random.choice(answer_categories)
        
        options = demographics[category]
        
        # Weighted selection based on research data
        population = []
        weights = []
        
        for option, data in options.items():
            population.append(option)
            weights.append(data.get("weight", 0.1))
        
        return random.choices(population, weights=weights, k=1)[0]
    
    def _generate_research_behavioral_answer(self, identity: Dict) -> str:
        """Generate research behavioral answer"""
        
        behavioral_patterns = self.answer_database["response_pattern_research"]
        
        # Select from brand awareness or purchase intent
        pattern_type = random.choice(["brand_awareness_optimization", "purchase_intent_research"])
        options = behavioral_patterns[pattern_type]
        
        # Weighted selection for research optimization
        population = []
        weights = []
        
        for option, data in options.items():
            population.append(option)
            weights.append(data.get("selection_rate", 0.1))
        
        return random.choices(population, weights=weights, k=1)[0]
    
    def _generate_research_product_answer(self, survey: Dict) -> str:
        """Generate research product-specific answer"""
        
        category = survey.get("category", "General")
        
        # Research-optimized answers by category
        category_answers = {
            "Technology": ["very_interested", "current_user", "early_adopter"],
            "Consumer Goods": ["regular_purchaser", "brand_loyal", "price_conscious"],
            "Automotive": ["car_owner", "considering_purchase", "brand_aware"],
            "Healthcare": ["health_conscious", "regular_user", "doctor_recommended"],
            "Finance": ["account_holder", "investment_interested", "financially_stable"],
            "Travel": ["frequent_traveler", "vacation_planner", "loyalty_member"],
            "Food & Beverage": ["regular_consumer", "brand_switcher", "health_conscious"]
        }
        
        answers = category_answers.get(category, ["interested", "aware", "considering"])
        return random.choice(answers)
    
    def _identify_research_optimizations(
        self,
        qualification_result: Dict,
        identity: Dict,
        platform: str
    ) -> List[Dict]:
        """Identify optimization opportunities from research data"""
        
        optimizations = []
        
        # Analyze demographic issues
        for issue in qualification_result.get("demographic_issues", []):
            if issue["question_type"] == "demographic":
                optimizations.append({
                    "type": "demographic_optimization",
                    "recommendation": "Adjust age/income profile for better qualification",
                    "impact": "high",
                    "implementation": "profile_modification"
                })
            elif issue["question_type"] == "behavioral":
                optimizations.append({
                    "type": "behavioral_optimization",
                    "recommendation": "Modify response patterns for target demographic",
                    "impact": "medium",
                    "implementation": "answer_strategy_change"
                })
        
        # Platform-specific optimizations
        if platform == "swagbucks":
            optimizations.append({
                "type": "platform_specific",
                "recommendation": "Focus on consumer goods and technology surveys",
                "impact": "medium",
                "implementation": "survey_selection_filter"
            })
        elif platform == "survey_junkie":
            optimizations.append({
                "type": "platform_specific",
                "recommendation": "Complete profile sections for higher qualification rates",
                "impact": "high",
                "implementation": "profile_completion"
            })
        
        return optimizations
    
    async def _research_survey_completion(
        self,
        platform: str,
        survey: Dict,
        account: Dict,
        proxy: Optional[str] = None
    ) -> Dict:
        """Research survey completion patterns and optimization"""
        
        start_time = datetime.now()
        
        try:
            # Simulate survey completion with research timing
            estimated_time = survey["estimated_time"]
            actual_time = random.uniform(
                estimated_time * 0.8,  # 20% faster than estimate
                estimated_time * 1.4   # 40% slower than estimate
            )
            
            # Research completion simulation
            await asyncio.sleep(actual_time / 30)  # Speed up for research
            
            # Research question answering patterns
            num_questions = random.randint(15, 50)
            
            completion_data = {
                "success": True,
                "survey_id": survey["id"],
                "earnings": survey["cash_value"],
                "time_spent": actual_time,
                "questions_answered": num_questions,
                "completion_rate": random.uniform(0.92, 1.0),
                "quality_score": random.uniform(0.85, 1.0),
                "research_insights": {
                    "optimal_speed": actual_time / num_questions,
                    "answer_consistency": random.uniform(0.8, 1.0),
                    "engagement_level": random.uniform(0.7, 1.0)
                }
            }
            
            # Small chance of survey error for research
            if random.random() < 0.02:  # 2% error rate
                completion_data.update({
                    "success": False,
                    "error": "Survey technical error",
                    "earnings": 0,
                    "partial_completion": True
                })
            
            return completion_data
            
        except Exception as e:
            return {
                "success": False,
                "survey_id": survey["id"],
                "error": str(e),
                "earnings": 0,
                "time_spent": (datetime.now() - start_time).total_seconds()
            }
    
    def _collect_optimization_insights(self, completion_research: Dict, research_results: Dict):
        """Collect optimization insights from completion research"""
        
        insights = completion_research.get("research_insights", {})
        
        # Analyze completion speed optimization
        optimal_speed = insights.get("optimal_speed", 0)
        if optimal_speed > 0:
            research_results["optimization_insights"].append({
                "type": "completion_speed",
                "optimal_seconds_per_question": optimal_speed,
                "recommendation": f"Target {optimal_speed:.1f} seconds per question for optimal earnings",
                "impact": "medium"
            })
        
        # Analyze answer consistency
        consistency = insights.get("answer_consistency", 0)
        if consistency < 0.9:
            research_results["optimization_insights"].append({
                "type": "answer_consistency",
                "current_consistency": consistency,
                "recommendation": "Improve answer consistency to avoid quality flags",
                "impact": "high"
            })
    
    def _analyze_research_patterns(self) -> List[Dict]:
        """Analyze patterns from research data"""
        
        patterns = []
        
        if not self.research_data:
            return patterns
        
        # Analyze earning patterns
        earnings_data = [d.get("earnings", 0) for d in self.research_data if d.get("success")]
        if earnings_data:
            patterns.append({
                "pattern_type": "earnings_analysis",
                "average_earnings": sum(earnings_data) / len(earnings_data),
                "max_earnings": max(earnings_data),
                "min_earnings": min(earnings_data),
                "total_surveys": len(earnings_data)
            })
        
        # Analyze time efficiency
        time_data = [(d.get("earnings", 0) / d.get("time_spent", 1)) for d in self.research_data if d.get("success") and d.get("time_spent", 0) > 0]
        if time_data:
            patterns.append({
                "pattern_type": "time_efficiency",
                "average_earnings_per_minute": sum(time_data) / len(time_data),
                "best_efficiency": max(time_data),
                "efficiency_variance": max(time_data) - min(time_data)
            })
        
        return patterns
    
    def _calculate_research_hourly_rate(self, research_results: Dict) -> float:
        """Calculate research hourly rate"""
        
        start = datetime.fromisoformat(research_results["start_time"])
        end = datetime.fromisoformat(research_results["end_time"])
        
        hours = (end - start).total_seconds() / 3600
        
        if hours > 0:
            return research_results["total_research_earnings"] / hours
        return 0.0
    
    async def research_platform_comparison(self, platforms: List[str]) -> Dict:
        """Research and compare multiple survey platforms"""
        
        comparison_research = {
            "research_type": "platform_comparison",
            "platforms_studied": platforms,
            "timestamp": datetime.now().isoformat(),
            "platform_results": {},
            "comparative_analysis": {}
        }
        
        # Research each platform
        for platform in platforms:
            # Generate test account for research
            test_account = {
                "platform": platform,
                "identity_data": {
                    "age": 35,
                    "income_range": "75000-100000",
                    "education_level": "bachelor",
                    "employment_status": "employed"
                }
            }
            
            # Research platform performance
            platform_research = await self.research_survey_optimization(
                platform=platform,
                account=test_account,
                max_surveys=5
            )
            
            comparison_research["platform_results"][platform] = platform_research
        
        # Comparative analysis
        comparison_research["comparative_analysis"] = self._analyze_platform_comparison(
            comparison_research["platform_results"]
        )
        
        return comparison_research
    
    def _analyze_platform_comparison(self, platform_results: Dict) -> Dict:
        """Analyze comparative platform performance"""
        
        analysis = {
            "best_hourly_rate": {"platform": None, "rate": 0},
            "highest_qualification_rate": {"platform": None, "rate": 0},
            "most_surveys_available": {"platform": None, "count": 0},
            "platform_rankings": []
        }
        
        for platform, results in platform_results.items():
            hourly_rate = results.get("hourly_rate", 0)
            surveys_completed = results.get("surveys_completed", 0)
            surveys_researched = results.get("surveys_researched", 1)
            qualification_rate = surveys_completed / surveys_researched if surveys_researched > 0 else 0
            
            # Track best metrics
            if hourly_rate > analysis["best_hourly_rate"]["rate"]:
                analysis["best_hourly_rate"] = {"platform": platform, "rate": hourly_rate}
            
            if qualification_rate > analysis["highest_qualification_rate"]["rate"]:
                analysis["highest_qualification_rate"] = {"platform": platform, "rate": qualification_rate}
            
            if surveys_researched > analysis["most_surveys_available"]["count"]:
                analysis["most_surveys_available"] = {"platform": platform, "count": surveys_researched}
            
            # Platform ranking data
            analysis["platform_rankings"].append({
                "platform": platform,
                "hourly_rate": hourly_rate,
                "qualification_rate": qualification_rate,
                "surveys_available": surveys_researched,
                "total_earnings": results.get("total_research_earnings", 0)
            })
        
        # Sort rankings by hourly rate
        analysis["platform_rankings"].sort(key=lambda x: x["hourly_rate"], reverse=True)
        
        return analysis
    
    def get_research_statistics(self) -> Dict:
        """Get comprehensive research statistics"""
        
        return {
            "surveys_researched": self.surveys_researched,
            "total_research_earnings": self.research_earnings,
            "research_data_points": len(self.research_data),
            "average_earnings_per_survey": self.research_earnings / self.surveys_researched if self.surveys_researched > 0 else 0,
            "platforms_studied": len(set(d.get("platform") for d in self.research_data if d.get("platform"))),
            "optimization_insights_generated": sum(len(d.get("optimization_insights", [])) for d in self.research_data),
            "research_session_start": datetime.now().isoformat()
        }

# Global research instance
survey_research_engine = SurveyResearchEngine()