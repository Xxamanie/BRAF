import asyncio
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List, Dict
import random
import json

class SurveyPlatform:
    def __init__(self, name: str, credentials: Dict):
        self.name = name
        self.credentials = credentials
        self.driver = None
        self.earnings = 0

class SurveyAutomation:
    def __init__(self):
        self.platforms = {
            "swagbucks": {
                "login_url": "https://www.swagbucks.com/login",
                "surveys_url": "https://www.swagbucks.com/surveys",
                "earning_rate": 0.5  # $ per survey
            },
            "surveyjunkie": {
                "login_url": "https://www.surveyjunkie.com/login",
                "surveys_url": "https://www.surveyjunkie.com/member-surveys",
                "earning_rate": 0.8
            },
            "toluna": {
                "login_url": "https://www.toluna.com/login",
                "surveys_url": "https://www.toluna.com/mysurveys",
                "earning_rate": 0.6
            }
        }
        
        self.ai_patterns = {
            "age": {"18-24": 0.3, "25-34": 0.4, "35-44": 0.2, "45+": 0.1},
            "income": {"<30k": 0.2, "30-60k": 0.4, "60-100k": 0.3, ">100k": 0.1},
            "employment": {"employed": 0.6, "student": 0.2, "unemployed": 0.1, "retired": 0.1}
        }

    async def initialize_driver(self):
        """Initialize browser with anti-detection features"""
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    def generate_ai_response(self, question: str, category: str) -> str:
        """AI-powered answer generation based on learned patterns"""
        if "age" in question.lower():
            return self.weighted_choice(self.ai_patterns["age"])
        elif "income" in question.lower():
            return self.weighted_choice(self.ai_patterns["income"])
        elif "employment" in question.lower():
            return self.weighted_choice(self.ai_patterns["employment"])
        elif "yes" in question.lower() or "no" in question.lower():
            return "Yes" if random.random() > 0.3 else "No"
        else:
            # Generic responses based on question type
            responses = ["Sometimes", "Often", "Rarely", "Always", "Never"]
            return random.choice(responses)

    def weighted_choice(self, choices: Dict):
        """Make weighted random choice"""
        total = sum(choices.values())
        r = random.uniform(0, total)
        current = 0
        for choice, weight in choices.items():
            if current + weight >= r:
                return choice
            current += weight

    async def complete_survey(self, platform: str, account: Dict):
        """Complete survey on specific platform"""
        try:
            platform_config = self.platforms[platform]
            
            # Login
            self.driver.get(platform_config["login_url"])
            await asyncio.sleep(2)
            
            # Fill login form
            username = self.driver.find_element(By.ID, "username")
            password = self.driver.find_element(By.ID, "password")
            username.send_keys(account["username"])
            password.send_keys(account["password"])
            
            login_btn = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            login_btn.click()
            await asyncio.sleep(3)
            
            # Navigate to surveys
            self.driver.get(platform_config["surveys_url"])
            await asyncio.sleep(3)
            
            # Find available surveys
            surveys = self.driver.find_elements(By.CLASS_NAME, "survey-item")
            
            for survey in surveys[:5]:  # Limit to 5 surveys per session
                try:
                    survey.click()
                    await asyncio.sleep(2)
                    
                    # Complete survey questions
                    questions = self.driver.find_elements(By.CLASS_NAME, "question")
                    
                    for question in questions:
                        question_text = question.text
                        category = self.detect_category(question_text)
                        answer = self.generate_ai_response(question_text, category)
                        
                        # Find answer options and select appropriate one
                        options = question.find_elements(By.TAG_NAME, "input")
                        for option in options:
                            if answer.lower() in option.get_attribute("value").lower():
                                option.click()
                                break
                        
                        await asyncio.sleep(random.uniform(1, 3))
                    
                    # Submit survey
                    submit_btn = self.driver.find_element(By.ID, "submit-survey")
                    submit_btn.click()
                    await asyncio.sleep(5)
                    
                    # Check if successful
                    if "thank you" in self.driver.page_source.lower():
                        self.earnings += platform_config["earning_rate"]
                        print(f"✓ Survey completed on {platform}: +${platform_config['earning_rate']}")
                
                except Exception as e:
                    print(f"Survey error: {e}")
                    continue
                
                # Rotate to next account if daily limit reached
                if self.earnings >= 10:  # $10 daily limit per account
                    print(f"Daily limit reached for {platform}, rotating account...")
                    break
        
        except Exception as e:
            print(f"Platform error {platform}: {e}")

    async def run_automation(self, accounts: List[Dict]):
        """Main automation loop"""
        for account in accounts:
            await self.initialize_driver()
            
            for platform in self.platforms.keys():
                if platform in account["platforms"]:
                    await self.complete_survey(platform, account)
            
            self.driver.quit()
            
            # Aggregation across platforms
            total_earnings = sum([self.platforms[p]["earning_rate"] for p in account["platforms"]])
            print(f"Total earnings for session: ${total_earnings}")
    
    def detect_category(self, question_text: str) -> str:
        """Detect question category for better response generation"""
        question_lower = question_text.lower()
        if any(word in question_lower for word in ["age", "old", "born"]):
            return "age"
        elif any(word in question_lower for word in ["income", "salary", "earn", "money"]):
            return "income"
        elif any(word in question_lower for word in ["job", "work", "employment", "career"]):
            return "employment"
        else:
            return "general"