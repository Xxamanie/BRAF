import re
from typing import Dict, List
from datetime import datetime

class ComplianceChecker:
    def __init__(self):
        self.platform_terms = {
            "swagbucks": {
                "allowed": ["human_completion", "single_account"],
                "prohibited": ["automation", "multiple_accounts", "false_information"],
                "rate_limits": {"surveys_per_day": 20, "minimum_time": 60}
            },
            "youtube": {
                "allowed": ["organic_watching", "engagement"],
                "prohibited": ["bot_traffic", "fake_views", "ad_click_fraud"],
                "rate_limits": {"videos_per_hour": 30, "watch_time_variation": True}
            }
        }
        
        self.risk_patterns = [
            {"pattern": r"bot|automation|script", "risk": "high"},
            {"pattern": r"fake|false|misleading", "risk": "high"},
            {"pattern": r"multiple.*account", "risk": "medium"},
            {"pattern": r"bypass.*limit", "risk": "high"}
        ]

    def check_automation_compliance(self, template_type: str,
                                   automation_config: Dict) -> Dict:
        """Check if automation complies with platform terms"""
        violations = []
        warnings = []
        
        # Check against platform terms
        if template_type in self.platform_terms:
            terms = self.platform_terms[template_type]
            
            # Check for prohibited activities
            for prohibited in terms["prohibited"]:
                if prohibited in str(automation_config).lower():
                    violations.append({
                        "type": "prohibited_activity",
                        "activity": prohibited,
                        "risk": "high",
                        "platform": template_type
                    })
            
            # Check rate limits
            if "rate_limits" in terms:
                for limit, value in terms["rate_limits"].items():
                    if limit in automation_config:
                        config_value = automation_config[limit]
                        if isinstance(value, int) and config_value > value:
                            violations.append({
                                "type": "rate_limit_exceeded",
                                "limit": limit,
                                "allowed": value,
                                "configured": config_value,
                                "risk": "medium"
                            })
        
        # Check for risk patterns
        config_str = str(automation_config).lower()
        for risk_pattern in self.risk_patterns:
            if re.search(risk_pattern["pattern"], config_str):
                warnings.append({
                    "type": "risk_pattern_detected",
                    "pattern": risk_pattern["pattern"],
                    "risk_level": risk_pattern["risk"],
                    "recommendation": "Review configuration"
                })
        
        compliance_score = 100 - (len(violations) * 20 + len(warnings) * 5)
        compliance_score = max(0, compliance_score)
        
        return {
            "compliant": len(violations) == 0,
            "score": compliance_score,
            "violations": violations,
            "warnings": warnings,
            "last_check": datetime.utcnow().isoformat(),
            "recommendations": self.generate_recommendations(violations, warnings)
        }

    def generate_recommendations(self, violations: List, warnings: List) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        for violation in violations:
            if violation["type"] == "prohibited_activity":
                recommendations.append(
                    f"Remove '{violation['activity']}' from configuration "
                    f"to comply with {violation['platform']} terms"
                )
            elif violation["type"] == "rate_limit_exceeded":
                recommendations.append(
                    f"Reduce {violation['limit']} from {violation['configured']} "
                    f"to {violation['allowed']} or lower"
                )
        
        for warning in warnings:
            recommendations.append(
                f"Review configuration for pattern: {warning['pattern']}"
            )
        
        return recommendations

    def monitor_activity(self, enterprise_id: str, activities: List[Dict]) -> Dict:
        """Monitor real-time activity for compliance"""
        suspicious_activities = []
        
        for activity in activities:
            # Check for abnormal patterns
            if self.is_abnormal_pattern(activity):
                suspicious_activities.append({
                    "activity": activity,
                    "reason": "abnormal_pattern",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Check for rate violations
            if self.exceeds_rate_limit(enterprise_id, activity):
                suspicious_activities.append({
                    "activity": activity,
                    "reason": "rate_limit_violation",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        risk_level = "low"
        if len(suspicious_activities) > 5:
            risk_level = "high"
        elif len(suspicious_activities) > 2:
            risk_level = "medium"
        
        return {
            "risk_level": risk_level,
            "suspicious_activities": suspicious_activities,
            "requires_intervention": risk_level == "high",
            "monitoring_period": "24h"
        }