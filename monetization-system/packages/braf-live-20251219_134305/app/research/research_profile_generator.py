"""
PROFILE GENERATION SYSTEM

Creates varied user profiles for research purposes
"""

from __future__ import annotations

import asyncio
import random
import string
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from faker import Faker


class ResearchProfileGenerator:
    """Generates varied user profiles for research."""

    def __init__(self) -> None:
        self.faker = Faker()
        self.session_cache: Dict[str, Any] = {}
        self.profiles_generated: int = 0
        self.generation_success_rate: float = 0.95

    async def generate_profile_batch(
        self,
        platform: str,
        count: int = 100,
        connection_list: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple research profiles.

        Returns:
            List of generated profile data
        """
        profiles: List[Dict[str, Any]] = []

        for i in range(count):
            try:
                # Generate research profile
                profile = self._generate_research_profile()

                # Rotate connection settings
                connection = self._get_connection(connection_list, i) if connection_list else None

                # Create platform-specific profile
                platform_profile = await self._create_platform_profile(platform, profile, connection)

                if platform_profile:
                    profiles.append(platform_profile)
                    self.profiles_generated += 1

                    # Save for research analysis
                    await self._save_profile_data(platform_profile)

                # Variable delay for realistic simulation
                delay = random.uniform(0.5, 2.0)
                await asyncio.sleep(delay)
            except Exception as e:  # pragma: no cover - defensive logging only
                print(f"Profile generation failed: {e}")
                continue

        return profiles

    def _generate_research_profile(self) -> Dict[str, Any]:
        """Generate comprehensive research profile."""
        first_name = self.faker.first_name()
        last_name = self.faker.last_name()
        username_base = f"{first_name.lower()}{last_name.lower()}{random.randint(1, 999)}"

        return {
            "first_name": first_name,
            "last_name": last_name,
            "email": f"{username_base}@{self.faker.free_email_domain()}",
            "password": self._generate_secure_password(),
            "username": username_base,
            "birth_date": self.faker.date_of_birth(minimum_age=18, maximum_age=65).strftime("%Y-%m-%d"),
            "street_address": self.faker.street_address(),
            "city": self.faker.city(),
            "state": self.faker.state_abbr(),
            "zip_code": self.faker.zipcode(),
            "country": "US",
            "phone_number": self.faker.phone_number(),
            "test_identifier": f"{random.randint(1000, 9999)}",
            "income_range": random.choice(["30000-50000", "50000-75000", "75000-100000"]),
            "education_level": random.choice(["high_school", "bachelor", "master", "doctorate"]),
            "employment_status": random.choice(["employed", "self_employed", "unemployed"]),
            "simulated_ip": self.faker.ipv4(),
            "user_agent": self.faker.user_agent(),
        }

    def _generate_secure_password(self, length: int = 12) -> str:
        """Generate secure password for testing."""
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        return "".join(random.choice(chars) for _ in range(length))

    def _get_connection(self, connection_list: List[str], index: int) -> str:
        """Rotate through connection list."""
        return connection_list[index % len(connection_list)]

    async def _create_platform_profile(
        self,
        platform: str,
        profile: Dict[str, Any],
        connection: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create platform-specific research profile."""
        if platform == "survey_platform_a":
            return await self._create_survey_profile(profile, connection)
        elif platform == "payment_platform_b":
            return await self._create_payment_profile(profile, connection)
        elif platform == "crypto_platform_c":
            return await self._create_crypto_profile(profile, connection)
        elif platform == "research_platform_d":
            return await self._create_research_profile(profile, connection)
        else:
            return await self._create_generic_profile(platform, profile, connection)

    async def _create_survey_profile(
        self,
        profile: Dict[str, Any],
        connection: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create survey platform research profile."""
        try:
            platform_profile: Dict[str, Any] = {
                "platform": "survey_platform_a",
                "username": profile["username"],
                "email": profile["email"],
                "password": profile["password"],
                "profile_completion": 100,
                "verification_status": "pending",
                "initial_research_points": 100,
                "created_at": datetime.now().isoformat(),
                "profile_data": profile,
                "security_test_answers": self._generate_test_answers(),
            }

            if random.random() < self.generation_success_rate:
                return platform_profile
        except Exception as e:  # pragma: no cover - defensive logging only
            print(f"Survey profile creation failed: {e}")
        return None

    async def _create_payment_profile(
        self,
        profile: Dict[str, Any],
        connection: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create payment platform test profile."""
        try:
            financial_details = {
                "simulated_bank": random.choice(["Chase", "Bank of America", "Wells Fargo", "Citibank"]),
                "test_account_number": f"{random.randint(100000000, 999999999)}",
                "test_routing_number": f"{random.randint(100000000, 999999999)}",
                "test_card_number": self._generate_test_card_number(),
                "test_expiry_date": f"{random.randint(1, 12):02d}/{random.randint(2025, 2030)}",
                "test_security_code": f"{random.randint(100, 999)}",
            }

            platform_profile: Dict[str, Any] = {
                "platform": "payment_platform_b",
                "email": profile["email"],
                "password": profile["password"],
                "simulated_financial_linked": True,
                "financial_test_details": financial_details,
                "verification_test_level": "verified" if random.random() > 0.3 else "unverified",
                "test_balance": random.uniform(0, 100),
                "created_at": datetime.now().isoformat(),
                "profile_data": profile,
            }

            if random.random() < self.generation_success_rate * 0.8:
                return platform_profile
        except Exception as e:  # pragma: no cover - defensive logging only
            print(f"Payment profile creation failed: {e}")
        return None

    async def _create_crypto_profile(
        self,
        profile: Dict[str, Any],
        connection: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create crypto platform test profile."""
        try:
            test_wallet_address = hashlib.sha256(
                f"{profile['email']}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:42]

            platform_profile: Dict[str, Any] = {
                "platform": "crypto_platform_c",
                "email": profile["email"],
                "password": profile["password"],
                "test_wallet_address": test_wallet_address,
                "test_verification_status": "verified" if random.random() > 0.4 else "pending",
                "test_trading_limits": {
                    "daily_test_buy": random.randint(1000, 10000),
                    "daily_test_sell": random.randint(1000, 10000),
                    "test_withdrawal_limit": random.randint(5000, 50000),
                },
                "created_at": datetime.now().isoformat(),
                "profile_data": profile,
            }

            if random.random() < self.generation_success_rate * 0.7:
                return platform_profile
        except Exception as e:  # pragma: no cover - defensive logging only
            print(f"Crypto profile creation failed: {e}")
        return None

    async def _create_research_profile(
        self,
        profile: Dict[str, Any],
        connection: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create research platform profile."""
        try:
            enhanced_profile = profile.copy()
            enhanced_profile.update(
                {
                    "household_income": random.choice(["75000-100000", "100000-150000"]),
                    "education": "bachelor" if random.random() > 0.3 else "master",
                    "employment": "employed_full_time",
                    "home_ownership": "own" if random.random() > 0.5 else "rent",
                    "marital_status": random.choice(["married", "single"]),
                    "children": random.choice(["0", "1", "2"]),
                    "interests": random.sample(
                        ["reading", "gardening", "traveling", "cooking", "sports"], 3
                    ),
                }
            )

            platform_profile: Dict[str, Any] = {
                "platform": "research_platform_d",
                "email": enhanced_profile["email"],
                "password": enhanced_profile["password"],
                "profile_quality_score": random.randint(80, 100),
                "research_qualification_rate": random.uniform(0.7, 0.95),
                "research_points_balance": random.randint(100, 1000),
                "created_at": datetime.now().isoformat(),
                "profile_data": enhanced_profile,
            }

            if random.random() < self.generation_success_rate:
                return platform_profile
        except Exception as e:  # pragma: no cover - defensive logging only
            print(f"Research profile creation failed: {e}")
        return None

    async def _create_generic_profile(
        self,
        platform: str,
        profile: Dict[str, Any],
        connection: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create generic research profile."""
        try:
            platform_profile: Dict[str, Any] = {
                "platform": platform,
                "email": profile["email"],
                "password": profile["password"],
                "username": profile["username"],
                "created_at": datetime.now().isoformat(),
                "profile_data": profile,
            }

            if random.random() < self.generation_success_rate * 0.9:
                return platform_profile
        except Exception as e:  # pragma: no cover - defensive logging only
            print(f"Generic profile creation failed: {e}")
        return None

    def _generate_test_answers(self) -> Dict[str, str]:
        """Generate test security answers."""
        questions = [
            "What is your test security question 1?",
            "What is your test security question 2?",
            "What is your test security question 3?",
            "What is your test security question 4?",
            "What is your test security question 5?",
        ]

        answers: Dict[str, str] = {}
        for question in random.sample(questions, 3):
            answers[question] = self.faker.word().capitalize()
        return answers

    def _generate_test_card_number(self) -> str:
        """Generate test card number for research using Luhn checksum."""

        def luhn_checksum(card_number: str) -> int:
            def digits_of(n: str) -> List[int]:
                return [int(d) for d in str(n)]

            digits = digits_of(card_number)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(str(d * 2)))
            return checksum % 10

        partial = "".join(str(random.randint(0, 9)) for _ in range(15))
        for check_digit in range(10):
            candidate = partial + str(check_digit)
            if luhn_checksum(candidate) == 0:
                return candidate
        return partial + "0"

    async def _save_profile_data(self, profile: Dict[str, Any]) -> None:
        """Save profile data for research using a simple XOR-based obfuscation."""
        try:
            def encode_data(data: str, key: int = 42) -> str:
                return "".join(chr(ord(c) ^ key) for c in data)

            filename = f"research_profiles_{datetime.now().strftime('%Y%m%d')}.data"
            encoded_data = encode_data(json.dumps(profile))
            with open(filename, "a", encoding="utf-8") as f:
                f.write(encoded_data + "\n")
        except Exception as e:  # pragma: no cover - defensive logging only
            print(f"Failed to save profile: {e}")

    async def get_research_stats(self) -> Dict[str, Any]:
        """Get profile generation statistics."""
        return {
            "total_profiles_generated": self.profiles_generated,
            "generation_success_rate": self.generation_success_rate,
            "estimated_research_value": self.profiles_generated * self._get_profile_value(),
            "platforms_researched": [
                "survey_platform_a",
                "payment_platform_b",
                "crypto_platform_c",
                "research_platform_d",
            ],
        }

    def _get_profile_value(self) -> float:
        """Estimate research value of generated profiles."""
        values = {
            "survey_platform_a": 10.0,
            "payment_platform_b": 50.0,
            "crypto_platform_c": 100.0,
            "research_platform_d": 5.0,
            "generic": 2.0,
        }
        return sum(values.values()) / len(values)
