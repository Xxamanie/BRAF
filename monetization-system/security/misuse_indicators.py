"""
Red Flags to Monitor
Technical Indicators of Misuse
"""

MISUSE_INDICATORS = {
    "technical": [
        "Rapid browser fingerprint rotation (>10/hour)",
        "Use of residential proxy networks",
        "Attempts to modify or disable security features",
        "Access to known fraud-related domains",
        "Patterns matching credential stuffing attacks"
    ],
    "behavioral": [
        "Unusual working hours (24/7 operation)",
        "Perfect task completion rates (100%)",
        "Lack of normal human errors/variance",
        "Rapid geographic switching",
        "Multiple accounts from same infrastructure"
    ],
    "financial": [
        "Structured transactions (just under reporting limits)",
        "Rapid withdrawal requests",
        "Funding from high-risk jurisdictions",
        "Integration with known mixing services",
        "Patterns matching money laundering"
    ]
}