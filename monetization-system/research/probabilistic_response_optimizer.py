"""
PROBABILISTIC RESPONSE OPTIMIZATION SYSTEM
Intelligent response generation based on statistical patterns
"""

import random
import json
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib

@dataclass
class DemographicProfile:
    """Statistical demographic profile"""
    age_distribution: Dict[str, float] = field(default_factory=lambda: {
        "18-24": 0.15,
        "25-34": 0.25,
        "35-44": 0.20,
        "45-54": 0.18,
        "55+": 0.22
    })
    
    income_distribution: Dict[str, float] = field(default_factory=lambda: {
        "under_30000": 0.20,
        "30000-50000": 0.25,
        "50000-75000": 0.30,
        "75000-100000": 0.15,
        "100000+": 0.10
    })
    
    education_distribution: Dict[str, float] = field(default_factory=lambda: {
        "high_school": 0.30,
        "some_college": 0.25,
        "bachelors": 0.30,
        "masters": 0.10,
        "doctorate": 0.05
    })
    
    employment_distribution: Dict[str, float] = field(default_factory=lambda: {
        "employed_full_time": 0.60,
        "employed_part_time": 0.15,
        "self_employed": 0.10,
        "unemployed": 0.10,
        "retired": 0.05
    })

@dataclass
class PreferenceProfile:
    """Statistical preference patterns"""
    tech_proficiency: Dict[str, float] = field(default_factory=lambda: {
        "expert": 0.10,
        "advanced": 0.25,
        "intermediate": 0.40,
        "beginner": 0.25
    })
    
    shopping_frequency: Dict[str, float] = field(default_factory=lambda: {
        "daily": 0.05,
        "weekly": 0.25,
        "monthly": 0.50,
        "rarely": 0.20
    })
    
    brand_familiarity: Dict[str, float] = field(default_factory=lambda: {
        "very_familiar": 0.40,
        "somewhat_familiar": 0.35,
        "heard_of": 0.20,
        "never_heard": 0.05
    })
    
    purchase_intent: Dict[str, float] = field(default_factory=lambda: {
        "definitely": 0.10,
        "probably": 0.25,
        "maybe": 0.40,
        "probably_not": 0.20,
        "definitely_not": 0.05
    })

@dataclass
class ResponsePattern:
    """Pattern for generating consistent responses"""
    question_type: str
    response_options: List[str]
    option_probabilities: List[float]
    consistency_rules: Dict[str, Any] = field(default_factory=dict)
    learned_adjustments: Dict[str, float] = field(default_factory=dict)

    def generate_response(self, context: Optional[Dict] = None) -> str:
        """Generate response based on probabilities and context"""
        context = context or {}
        
        # Apply learned adjustments if any
        probabilities = self.option_probabilities.copy()
        if self.learned_adjustments:
            for option, adjustment in self.learned_adjustments.items():
                if option in self.response_options:
                    idx = self.response_options.index(option)
                    probabilities[idx] *= adjustment
        
        # Normalize probabilities
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        
        # Generate response
        return random.choices(self.response_options, weights=probabilities, k=1)[0]

    def update_from_feedback(self, selected_option: str, was_accepted: bool):
        """Update probabilities based on feedback"""
        if selected_option not in self.response_options:
            return
        
        idx = self.response_options.index(selected_option)
        
        if was_accepted:
            # Increase probability of this option
            adjustment = 1.1
            self.learned_adjustments[selected_option] = \
                self.learned_adjustments.get(selected_option, 1.0) * adjustment
        else:
            # Decrease probability of this option
            adjustment = 0.9
            self.learned_adjustments[selected_option] = \
                self.learned_adjustments.get(selected_option, 1.0) * adjustment
        
        # Keep adjustments reasonable
        self.learned_adjustments[selected_option] = max(0.1, min(2.0, self.learned_adjustments[selected_option]))

class ProbabilisticResponseOptimizer:
    """Optimizes response generation using statistical patterns"""
    
    def __init__(self, profile_id: Optional[str] = None):
        self.profile_id = profile_id or hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:8]
        self.demographic_profile = DemographicProfile()
        self.preference_profile = PreferenceProfile()
        self.response_patterns: Dict[str, ResponsePattern] = {}
        self.response_history: List[Dict] = []
        self.consistency_matrix: Dict[Tuple[str, str], float] = {}
        self.avoidance_patterns: List[str] = []
        
        self._initialize_default_patterns()

    def _initialize_default_patterns(self):
        """Initialize default response patterns"""
        # Demographic questions
        self.response_patterns["age_question"] = ResponsePattern(
            question_type="demographic_age",
            response_options=list(self.demographic_profile.age_distribution.keys()),
            option_probabilities=list(self.demographic_profile.age_distribution.values()),
            consistency_rules={"once_per_session": True}
        )
        
        self.response_patterns["income_question"] = ResponsePattern(
            question_type="demographic_income",
            response_options=list(self.demographic_profile.income_distribution.keys()),
            option_probabilities=list(self.demographic_profile.income_distribution.values()),
            consistency_rules={"depends_on": ["age_question"]}
        )
        
        self.response_patterns["education_question"] = ResponsePattern(
            question_type="demographic_education",
            response_options=list(self.demographic_profile.education_distribution.keys()),
            option_probabilities=list(self.demographic_profile.education_distribution.values()),
            consistency_rules={"depends_on": ["age_question", "income_question"]}
        )
        
        # Preference questions
        self.response_patterns["tech_proficiency"] = ResponsePattern(
            question_type="preference_tech",
            response_options=list(self.preference_profile.tech_proficiency.keys()),
            option_probabilities=list(self.preference_profile.tech_proficiency.values())
        )
        
        self.response_patterns["shopping_frequency"] = ResponsePattern(
            question_type="preference_shopping",
            response_options=list(self.preference_profile.shopping_frequency.keys()),
            option_probabilities=list(self.preference_profile.shopping_frequency.values())
        )
        
        self.response_patterns["brand_familiarity"] = ResponsePattern(
            question_type="preference_brand",
            response_options=list(self.preference_profile.brand_familiarity.keys()),
            option_probabilities=list(self.preference_profile.brand_familiarity.values())
        )
        
        self.response_patterns["purchase_intent"] = ResponsePattern(
            question_type="preference_purchase",
            response_options=list(self.preference_profile.purchase_intent.keys()),
            option_probabilities=list(self.preference_profile.purchase_intent.values())
        )

    def generate_demographic_response(self, question_key: str, previous_responses: Optional[Dict] = None) -> str:
        """Generate demographic response with consistency"""
        previous_responses = previous_responses or {}
        
        if question_key not in self.response_patterns:
            # Fallback to generic response
            if "age" in question_key.lower():
                return random.choice(["25-34", "35-44"])
            elif "income" in question_key.lower():
                return random.choice(["50000-75000", "75000-100000"])
            elif "education" in question_key.lower():
                return random.choice(["bachelors", "some_college"])
            else:
                return "prefer_not_to_say"
        
        pattern = self.response_patterns[question_key]
        
        # Apply consistency rules
        context = self._build_consistency_context(question_key, previous_responses)
        response = pattern.generate_response(context)
        
        # Record response
        self._record_response(question_key, response, context)
        
        return response

    def generate_preference_response(self, question_key: str, product_context: Optional[Dict] = None) -> str:
        """Generate preference response"""
        product_context = product_context or {}
        
        if question_key not in self.response_patterns:
            # Determine question type from key
            if any(word in question_key.lower() for word in ["tech", "computer", "digital"]):
                pattern_key = "tech_proficiency"
            elif any(word in question_key.lower() for word in ["shop", "buy", "purchase"]):
                pattern_key = "shopping_frequency"
            elif any(word in question_key.lower() for word in ["brand", "company", "product"]):
                pattern_key = "brand_familiarity"
            elif any(word in question_key.lower() for word in ["intend", "consider", "plan"]):
                pattern_key = "purchase_intent"
            else:
                # Generic positive response
                return random.choice(["agree", "somewhat_agree", "neutral"])
        else:
            pattern_key = question_key
        
        pattern = self.response_patterns[pattern_key]
        
        # Adjust based on product context
        context = product_context.copy()
        
        # Avoid trigger words that might cause rejection
        if self.avoidance_patterns:
            for trigger in self.avoidance_patterns:
                if trigger in str(context).lower():
                    # Choose safer response
                    if pattern_key == "purchase_intent":
                        return "maybe"
        
        response = pattern.generate_response(context)
        
        # Record response
        self._record_response(pattern_key, response, context)
        
        return response

    def _build_consistency_context(self, current_question: str, previous_responses: Dict) -> Dict:
        """Build context for consistent responses"""
        context = {"previous_responses": previous_responses}
        
        # Ensure demographic consistency
        if "age" in current_question.lower() and "age_question" in previous_responses:
            # Age should match previously stated age
            age_response = previous_responses["age_question"]
            context["age_constraint"] = age_response
        
        if "income" in current_question.lower():
            # Income should be consistent with age and education
            if "age_question" in previous_responses:
                age = previous_responses["age_question"]
                # Younger ages typically have lower incomes
                if age == "18-24":
                    context["income_bias"] = "lower"
                elif age == "55+":
                    context["income_bias"] = "higher"
            
            if "education_question" in previous_responses:
                education = previous_responses["education_question"]
                # Higher education typically means higher income
                if education in ["masters", "doctorate"]:
                    context["income_bias"] = "higher"
        
        if "education" in current_question.lower():
            # Education should be consistent with age
            if "age_question" in previous_responses:
                age = previous_responses["age_question"]
                # Older age groups less likely to have advanced degrees
                if age == "55+":
                    context["education_bias"] = "lower"
        
        return context

    def _record_response(self, question_key: str, response: str, context: Dict):
        """Record response for learning and consistency"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "question_key": question_key,
            "response": response,
            "context": context,
            "profile_id": self.profile_id
        }
        
        self.response_history.append(record)
        
        # Keep history manageable
        if len(self.response_history) > 1000:
            self.response_history = self.response_history[-1000:]
        
        # Update consistency matrix
        self._update_consistency_matrix(question_key, response, context)

    def _update_consistency_matrix(self, question_key: str, response: str, context: Dict):
        """Update consistency relationships between questions"""
        if "previous_responses" in context:
            prev_responses = context["previous_responses"]
            for prev_q, prev_r in prev_responses.items():
                key = (prev_q, question_key)
                current_score = self.consistency_matrix.get(key, 0.5)
                
                # Simple consistency scoring
                # This would be enhanced with actual logic in production
                new_score = current_score * 0.9 + 0.1  # Slight positive reinforcement
                self.consistency_matrix[key] = min(1.0, new_score)

    def learn_from_outcome(self, question_key: str, response: str, was_accepted: bool, rejection_reason: Optional[str] = None):
        """Learn from response outcome (acceptance/rejection)"""
        if question_key in self.response_patterns:
            pattern = self.response_patterns[question_key]
            pattern.update_from_feedback(response, was_accepted)
        
        # Learn avoidance patterns from rejections
        if not was_accepted and rejection_reason:
            # Extract potential trigger words from rejection reason
            rejection_lower = rejection_reason.lower()
            trigger_words = ["not_qualify", "disqualify", "not_match", "inconsistent"]
            
            for trigger in trigger_words:
                if trigger in rejection_lower:
                    self.avoidance_patterns.append(trigger)
            
            # Keep unique
            self.avoidance_patterns = list(set(self.avoidance_patterns))
        
        # Update response history with outcome
        for record in reversed(self.response_history):
            if (record["question_key"] == question_key and record["response"] == response):
                record["outcome"] = "accepted" if was_accepted else "rejected"
                record["outcome_reason"] = rejection_reason
                break

    def generate_consistent_profile(self) -> Dict:
        """Generate a complete consistent response profile"""
        profile = {}
        
        # Generate demographic responses in logical order
        demographic_order = ["age_question", "income_question", "education_question"]
        for question in demographic_order:
            response = self.generate_demographic_response(question, profile)
            profile[question] = response
        
        # Generate preference responses
        preference_questions = ["tech_proficiency", "shopping_frequency", "brand_familiarity", "purchase_intent"]
        for question in preference_questions:
            response = self.generate_preference_response(question)
            profile[question] = response
        
        return profile

    def calculate_consistency_score(self) -> float:
        """Calculate consistency score based on response history"""
        if len(self.response_history) < 2:
            return 1.0
        
        scores = []
        for i in range(len(self.response_history) - 1):
            record1 = self.response_history[i]
            record2 = self.response_history[i + 1]
            
            key = (record1["question_key"], record2["question_key"])
            if key in self.consistency_matrix:
                scores.append(self.consistency_matrix[key])
        
        return statistics.mean(scores) if scores else 1.0

    def get_optimization_statistics(self) -> Dict:
        """Get optimization performance statistics"""
        if not self.response_history:
            return {}
        
        # Calculate acceptance rate
        outcomes = [r.get("outcome") for r in self.response_history if "outcome" in r]
        acceptance_count = outcomes.count("accepted")
        total_outcomes = len(outcomes)
        acceptance_rate = acceptance_count / total_outcomes if total_outcomes > 0 else 0
        
        # Calculate pattern usage
        pattern_usage = Counter([r["question_key"] for r in self.response_history])
        
        return {
            "total_responses": len(self.response_history),
            "acceptance_rate": acceptance_rate,
            "consistency_score": self.calculate_consistency_score(),
            "avoidance_patterns_count": len(self.avoidance_patterns),
            "most_used_patterns": dict(pattern_usage.most_common(5)),
            "profile_id": self.profile_id
        }

    def optimize_for_platform(self, platform_patterns: Dict[str, Any]):
        """Optimize responses for specific platform patterns"""
        for pattern_key, platform_data in platform_patterns.items():
            if pattern_key in self.response_patterns:
                pattern = self.response_patterns[pattern_key]
                
                # Adjust probabilities based on platform data
                if "preferred_responses" in platform_data:
                    preferred = platform_data["preferred_responses"]
                    for option, boost in preferred.items():
                        if option in pattern.response_options:
                            idx = pattern.response_options.index(option)
                            # Boost probability
                            pattern.option_probabilities[idx] *= (1.0 + boost)
                
                if "avoid_responses" in platform_data:
                    avoided = platform_data["avoid_responses"]
                    for option in avoided:
                        if option in pattern.response_options:
                            idx = pattern.response_options.index(option)
                            # Reduce probability
                            pattern.option_probabilities[idx] *= 0.5

    def export_profile(self, filename: Optional[str] = None) -> str:
        """Export current profile and patterns to JSON"""
        export_data = {
            "profile_id": self.profile_id,
            "timestamp": datetime.now().isoformat(),
            "demographic_profile": {
                "age_distribution": self.demographic_profile.age_distribution,
                "income_distribution": self.demographic_profile.income_distribution,
                "education_distribution": self.demographic_profile.education_distribution,
                "employment_distribution": self.demographic_profile.employment_distribution
            },
            "preference_profile": {
                "tech_proficiency": self.preference_profile.tech_proficiency,
                "shopping_frequency": self.preference_profile.shopping_frequency,
                "brand_familiarity": self.preference_profile.brand_familiarity,
                "purchase_intent": self.preference_profile.purchase_intent
            },
            "response_patterns": {
                key: {
                    "question_type": pattern.question_type,
                    "response_options": pattern.response_options,
                    "option_probabilities": pattern.option_probabilities,
                    "learned_adjustments": pattern.learned_adjustments,
                    "consistency_rules": pattern.consistency_rules
                }
                for key, pattern in self.response_patterns.items()
            },
            "optimization_stats": self.get_optimization_statistics(),
            "avoidance_patterns": self.avoidance_patterns
        }
        
        json_data = json.dumps(export_data, indent=2, default=str)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_data)
        
        return json_data

    def import_profile(self, json_data: str):
        """Import profile from JSON data"""
        import_data = json.loads(json_data)
        
        self.profile_id = import_data.get("profile_id", self.profile_id)
        
        # Import demographic profile
        if "demographic_profile" in import_data:
            dp_data = import_data["demographic_profile"]
            self.demographic_profile.age_distribution = dp_data.get("age_distribution", self.demographic_profile.age_distribution)
            self.demographic_profile.income_distribution = dp_data.get("income_distribution", self.demographic_profile.income_distribution)
            self.demographic_profile.education_distribution = dp_data.get("education_distribution", self.demographic_profile.education_distribution)
            self.demographic_profile.employment_distribution = dp_data.get("employment_distribution", self.demographic_profile.employment_distribution)
        
        # Import preference profile
        if "preference_profile" in import_data:
            pp_data = import_data["preference_profile"]
            self.preference_profile.tech_proficiency = pp_data.get("tech_proficiency", self.preference_profile.tech_proficiency)
            self.preference_profile.shopping_frequency = pp_data.get("shopping_frequency", self.preference_profile.shopping_frequency)
            self.preference_profile.brand_familiarity = pp_data.get("brand_familiarity", self.preference_profile.brand_familiarity)
            self.preference_profile.purchase_intent = pp_data.get("purchase_intent", self.preference_profile.purchase_intent)
        
        # Import response patterns
        if "response_patterns" in import_data:
            for key, pattern_data in import_data["response_patterns"].items():
                if key in self.response_patterns:
                    pattern = self.response_patterns[key]
                    pattern.learned_adjustments = pattern_data.get("learned_adjustments", {})
                    
                    # Reinitialize with potentially updated distributions
                    if key in ["age_question", "income_question", "education_question", "employment_question"]:
                        # Update from demographic profile
                        if key == "age_question":
                            pattern.response_options = list(self.demographic_profile.age_distribution.keys())
                            pattern.option_probabilities = list(self.demographic_profile.age_distribution.values())
                        elif key == "income_question":
                            pattern.response_options = list(self.demographic_profile.income_distribution.keys())
                            pattern.option_probabilities = list(self.demographic_profile.income_distribution.values())
                        elif key == "education_question":
                            pattern.response_options = list(self.demographic_profile.education_distribution.keys())
                            pattern.option_probabilities = list(self.demographic_profile.education_distribution.values())
                    
                    elif key in ["tech_proficiency", "shopping_frequency", "brand_familiarity", "purchase_intent"]:
                        # Update from preference profile
                        if key == "tech_proficiency":
                            pattern.response_options = list(self.preference_profile.tech_proficiency.keys())
                            pattern.option_probabilities = list(self.preference_profile.tech_proficiency.values())
                        elif key == "shopping_frequency":
                            pattern.response_options = list(self.preference_profile.shopping_frequency.keys())
                            pattern.option_probabilities = list(self.preference_profile.shopping_frequency.values())
                        elif key == "brand_familiarity":
                            pattern.response_options = list(self.preference_profile.brand_familiarity.keys())
                            pattern.option_probabilities = list(self.preference_profile.brand_familiarity.values())
                        elif key == "purchase_intent":
                            pattern.response_options = list(self.preference_profile.purchase_intent.keys())
                            pattern.option_probabilities = list(self.preference_profile.purchase_intent.values())
        
        # Import avoidance patterns
        if "avoidance_patterns" in import_data:
            self.avoidance_patterns = import_data["avoidance_patterns"]

    def adaptive_learning(self, recent_outcomes: List[Tuple[str, bool, Optional[str]]], learning_rate: float = 0.1):
        """Perform adaptive learning based on recent outcomes"""
        if not recent_outcomes:
            return
        
        # Calculate success rate
        success_rate = sum(1 for _, accepted, _ in recent_outcomes if accepted) / len(recent_outcomes)
        
        # Adjust aggressiveness based on success rate
        if success_rate < 0.5:
            # Too many rejections, become more conservative
            for pattern in self.response_patterns.values():
                # Boost conservative responses
                conservative_keywords = ["maybe", "somewhat", "intermediate", "middle", "neutral"]
                for i, option in enumerate(pattern.response_options):
                    if any(keyword in option.lower() for keyword in conservative_keywords):
                        pattern.option_probabilities[i] *= (1.0 + learning_rate)
        
        elif success_rate > 0.8:
            # High success, can be slightly more adventurous
            for pattern in self.response_patterns.values():
                # Slightly boost definitive responses
                definitive_keywords = ["definitely", "expert", "very", "frequent", "daily"]
                for i, option in enumerate(pattern.response_options):
                    if any(keyword in option.lower() for keyword in definitive_keywords):
                        pattern.option_probabilities[i] *= (1.0 + learning_rate * 0.5)
        
        # Normalize all probabilities
        for pattern in self.response_patterns.values():
            total = sum(pattern.option_probabilities)
            if total > 0:
                pattern.option_probabilities = [p / total for p in pattern.option_probabilities]

# Global optimizer instance
probabilistic_response_optimizer = ProbabilisticResponseOptimizer()

# Example usage and demonstration
def main():
    """Demonstrate the probabilistic response optimizer"""
    # Initialize optimizer
    optimizer = ProbabilisticResponseOptimizer()
    
    print("=" * 60)
    print("PROBABILISTIC RESPONSE OPTIMIZATION SYSTEM")
    print("=" * 60)
    
    # Generate a complete profile
    print("\n1. Generating consistent response profile...")
    profile = optimizer.generate_consistent_profile()
    print(f"\nGenerated Profile (ID: {optimizer.profile_id}):")
    for key, value in profile.items():
        print(f"  {key}: {value}")
    
    # Simulate some interactions
    print("\n2. Simulating survey responses...")
    questions = [
        ("What is your age range?", "age_question"),
        ("What is your annual income?", "income_question"),
        ("What is your highest education level?", "education_question"),
        ("How would you rate your tech proficiency?", "tech_proficiency"),
        ("How often do you shop online?", "shopping_frequency"),
        ("How familiar are you with our brand?", "brand_familiarity"),
        ("How likely are you to purchase our product?", "purchase_intent")
    ]
    
    responses = {}
    for question_text, question_key in questions:
        if "demographic" in question_key or "age" in question_key or "income" in question_key or "education" in question_key:
            response = optimizer.generate_demographic_response(question_key, responses)
        else:
            response = optimizer.generate_preference_response(question_key)
        
        responses[question_key] = response
        print(f"  Q: {question_text}")
        print(f"  A: {response}\n")
    
    # Simulate learning from feedback
    print("\n3. Learning from feedback...")
    feedback_examples = [
        ("age_question", "25-34", True, None),
        ("income_question", "50000-75000", True, None),
        ("purchase_intent", "definitely", False, "Response seems inconsistent with demographic profile"),
        ("tech_proficiency", "intermediate", True, None),
    ]
    
    for question_key, response, accepted, reason in feedback_examples:
        optimizer.learn_from_outcome(question_key, response, accepted, reason)
        outcome = "ACCEPTED" if accepted else "REJECTED"
        print(f"  Feedback: {question_key} = {response} -> {outcome}")
        if reason:
            print(f"    Reason: {reason}")
    
    # Show optimization statistics
    print("\n4. Optimization Statistics:")
    stats = optimizer.get_optimization_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Demonstrate platform-specific optimization
    print("\n5. Platform-specific optimization...")
    platform_patterns = {
        "purchase_intent": {
            "preferred_responses": {"probably": 0.2, "definitely": 0.3},
            "avoid_responses": ["probably_not", "definitely_not"]
        },
        "tech_proficiency": {
            "preferred_responses": {"intermediate": 0.1, "advanced": 0.2}
        }
    }
    
    optimizer.optimize_for_platform(platform_patterns)
    print("  Applied platform optimization patterns")
    
    # Show updated probabilities
    print("\n6. Updated response probabilities:")
    for key, pattern in list(optimizer.response_patterns.items())[:3]:  # Show first 3
        print(f"\n  {key}:")
        for option, prob in zip(pattern.response_options, pattern.option_probabilities):
            print(f"    {option}: {prob:.3f}")
    
    # Export and import demonstration
    print("\n7. Profile export/import demonstration...")
    export_json = optimizer.export_profile()
    
    # Create new optimizer with imported profile
    new_optimizer = ProbabilisticResponseOptimizer("imported_profile")
    new_optimizer.import_profile(export_json)
    
    print(f"  Original profile ID: {optimizer.profile_id}")
    print(f"  Imported profile ID: {new_optimizer.profile_id}")
    
    # Generate response with imported profile
    imported_response = new_optimizer.generate_preference_response("purchase_intent")
    print(f"  Imported profile generates: purchase_intent = {imported_response}")
    
    print("\n" + "=" * 60)
    print("SYSTEM READY - Probabilistic optimization active")
    print("=" * 60)

if __name__ == "__main__":
    main()