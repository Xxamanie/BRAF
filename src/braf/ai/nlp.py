#!/usr/bin/env python3
"""
BRAF Natural Language Processing System
Advanced NLP for content analysis, instruction understanding, and intelligent automation
"""

import re
import spacy
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, pipeline
)
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class NLPEngine:
    """Advanced NLP engine for BRAF content analysis and understanding"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models
        self.intent_classifier = None
        self.entity_extractor = None
        self.sentiment_analyzer = None
        self.form_analyzer = None
        self.content_generator = None

        self._initialize_models()

        # NLP patterns and rules
        self.instruction_patterns = self._load_instruction_patterns()
        self.form_field_patterns = self._load_form_patterns()

    def _initialize_models(self):
        """Initialize NLP models"""
        try:
            # Load spaCy for basic NLP
            self.nlp = spacy.load("en_core_web_sm")

            # Intent classification (instruction understanding)
            self.intent_classifier = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )

            # Named Entity Recognition
            self.entity_extractor = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                device=0 if torch.cuda.is_available() else -1
            )

            # Sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )

            # Form analysis model (custom)
            self.form_analyzer = FormAnalyzer()

            # Content generation (for form filling)
            self.content_generator = ContentGenerator()

            logger.info("NLP models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {e}")
            # Fallback to basic spaCy
            self.nlp = spacy.load("en_core_web_sm")

    def analyze_page_content(self, text: str, html: str = None) -> Dict[str, Any]:
        """Comprehensive analysis of page content"""
        try:
            # Basic NLP processing
            doc = self.nlp(text)

            # Extract key information
            analysis = {
                'entities': self.extract_entities(text),
                'instructions': self.extract_instructions(text),
                'form_fields': self.identify_form_fields(html or text),
                'sentiment': self.analyze_sentiment(text),
                'topics': self.extract_topics(text),
                'actions_required': self.identify_required_actions(text),
                'content_type': self.classify_content_type(text),
                'complexity_score': self.calculate_complexity(text)
            }

            return analysis

        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {'error': str(e)}

    def extract_instructions(self, text: str) -> List[Dict[str, Any]]:
        """Extract actionable instructions from text"""
        instructions = []

        # Use pattern matching
        for pattern_name, pattern in self.instruction_patterns.items():
            matches = re.finditer(pattern['regex'], text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                instruction = {
                    'type': pattern_name,
                    'text': match.group(0).strip(),
                    'action': pattern['action'],
                    'priority': pattern['priority'],
                    'confidence': pattern['confidence']
                }
                instructions.append(instruction)

        # Use ML classification for complex instructions
        if self.intent_classifier:
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:
                    try:
                        result = self.intent_classifier(sentence, candidate_labels=[
                            "login_required", "form_filling", "navigation", "verification",
                            "payment_required", "account_creation", "data_entry"
                        ])
                        if result['scores'][0] > 0.7:  # High confidence
                            instructions.append({
                                'type': 'ml_classified',
                                'text': sentence,
                                'action': result['labels'][0],
                                'confidence': result['scores'][0]
                            })
                    except Exception as e:
                        logger.warning(f"ML instruction classification failed: {e}")

        return instructions

    def identify_form_fields(self, html: str) -> List[Dict[str, Any]]:
        """Identify and analyze form fields"""
        fields = []

        # Extract form elements from HTML
        if html:
            # Use regex to find form elements (simplified)
            input_pattern = r'<input[^>]*>'
            select_pattern = r'<select[^>]*>.*?</select>'
            textarea_pattern = r'<textarea[^>]*>.*?</textarea>'

            for pattern in [input_pattern, select_pattern, textarea_pattern]:
                matches = re.finditer(pattern, html, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    field_info = self._parse_form_element(match.group(0))
                    if field_info:
                        fields.append(field_info)

        # Analyze with ML if available
        if self.form_analyzer:
            fields = self.form_analyzer.enhance_field_analysis(fields, html)

        return fields

    def generate_form_responses(self, fields: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate appropriate responses for form fields"""
        responses = {}

        for field in fields:
            field_type = field.get('type', 'text')
            field_name = field.get('name', '')
            placeholder = field.get('placeholder', '')
            label = field.get('label', '')

            # Generate response based on field type and context
            response = self.content_generator.generate_field_response(
                field_type, field_name, placeholder, label
            )
            responses[field_name] = response

        return responses

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        if not self.entity_extractor:
            # Fallback to spaCy
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.8
                })
            return entities

        try:
            results = self.entity_extractor(text)
            entities = []
            for result in results:
                entities.append({
                    'text': result['word'],
                    'label': result['entity'],
                    'start': result['start'],
                    'end': result['end'],
                    'confidence': result['confidence']
                })
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of content"""
        if not self.sentiment_analyzer:
            return {'sentiment': 'neutral', 'confidence': 0.5}

        try:
            result = self.sentiment_analyzer(text[:512])  # Limit text length
            return {
                'sentiment': result[0]['label'],
                'confidence': result[0]['score']
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5}

    def extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        doc = self.nlp(text)

        # Simple keyword extraction (could be enhanced with topic modeling)
        topics = []
        keywords = ['login', 'register', 'sign up', 'payment', 'verification',
                   'security', 'account', 'password', 'email', 'phone']

        for keyword in keywords:
            if keyword.lower() in text.lower():
                topics.append(keyword)

        return list(set(topics))

    def identify_required_actions(self, text: str) -> List[Dict[str, Any]]:
        """Identify actions required by the user"""
        actions = []

        # Look for imperative sentences
        doc = self.nlp(text)
        for sent in doc.sents:
            # Check if sentence starts with imperative verbs
            first_token = sent[0]
            if first_token.pos_ == 'VERB' and first_token.tag_ in ['VB', 'VBP']:
                actions.append({
                    'action': first_token.lemma_,
                    'sentence': sent.text.strip(),
                    'urgency': self._assess_urgency(sent.text)
                })

        return actions

    def classify_content_type(self, text: str) -> str:
        """Classify the type of content"""
        # Simple classification based on keywords
        if any(word in text.lower() for word in ['login', 'sign in', 'username', 'password']):
            return 'login_page'
        elif any(word in text.lower() for word in ['register', 'sign up', 'create account']):
            return 'registration_page'
        elif any(word in text.lower() for word in ['payment', 'billing', 'credit card']):
            return 'payment_page'
        elif any(word in text.lower() for word in ['dashboard', 'profile', 'account']):
            return 'dashboard'
        else:
            return 'general_content'

    def calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        doc = self.nlp(text)

        # Factors: sentence length, vocabulary richness, structure
        avg_sentence_length = sum(len(sent) for sent in doc.sents) / len(list(doc.sents))
        unique_words = len(set(token.lemma_.lower() for token in doc if token.is_alpha))
        total_words = len([token for token in doc if token.is_alpha])

        if total_words == 0:
            return 0.0

        vocabulary_richness = unique_words / total_words

        # Complexity score (0-1)
        complexity = min(1.0, (avg_sentence_length / 20 + vocabulary_richness) / 2)
        return complexity

    def _load_instruction_patterns(self) -> Dict[str, Dict]:
        """Load regex patterns for instruction extraction"""
        return {
            'login_required': {
                'regex': r'(?:please|you must|you need to)?\s*(?:log in|login|sign in)(?:\s+to\s+.*?)?(?:\.|\n|$)',
                'action': 'login',
                'priority': 'high',
                'confidence': 0.9
            },
            'enter_code': {
                'regex': r'(?:enter|input|type)\s+(?:the|your)\s+(?:code|verification code|captcha)(?:\s+.*?)?(?:\.|\n|$)',
                'action': 'enter_verification_code',
                'priority': 'high',
                'confidence': 0.8
            },
            'fill_form': {
                'regex': r'(?:please|you must)?\s*(?:fill|complete|provide)\s+(?:the|your)\s+(?:form|information|details)(?:\s+.*?)?(?:\.|\n|$)',
                'action': 'fill_form',
                'priority': 'medium',
                'confidence': 0.7
            },
            'verify_email': {
                'regex': r'(?:check|verify)\s+(?:your email|email address)(?:\s+.*?)?(?:\.|\n|$)',
                'action': 'verify_email',
                'priority': 'medium',
                'confidence': 0.8
            },
            'upload_document': {
                'regex': r'(?:upload|submit|provide)\s+(?:a|your|the)\s+(?:document|file|photo|image)(?:\s+.*?)?(?:\.|\n|$)',
                'action': 'upload_document',
                'priority': 'medium',
                'confidence': 0.7
            }
        }

    def _load_form_patterns(self) -> Dict[str, str]:
        """Load patterns for form field identification"""
        return {
            'email': r'email|e-mail',
            'password': r'password|passcode',
            'name': r'(?:full )?name|firstname|lastname',
            'phone': r'phone|mobile|cell|telephone',
            'address': r'address|street|city|zip|postal',
            'date': r'date|birth|birthday|dob',
            'credit_card': r'credit.?card|card.?number|cvv|cvc'
        }

    def _parse_form_element(self, element_html: str) -> Optional[Dict[str, Any]]:
        """Parse HTML form element"""
        # Extract attributes
        type_match = re.search(r'type=["\']([^"\']+)["\']', element_html, re.IGNORECASE)
        name_match = re.search(r'name=["\']([^"\']+)["\']', element_html, re.IGNORECASE)
        id_match = re.search(r'id=["\']([^"\']+)["\']', element_html, re.IGNORECASE)
        placeholder_match = re.search(r'placeholder=["\']([^"\']+)["\']', element_html, re.IGNORECASE)
        required_match = re.search(r'required', element_html, re.IGNORECASE)

        field_info = {
            'type': type_match.group(1) if type_match else 'text',
            'name': name_match.group(1) if name_match else (id_match.group(1) if id_match else 'unknown'),
            'id': id_match.group(1) if id_match else None,
            'placeholder': placeholder_match.group(1) if placeholder_match else '',
            'required': bool(required_match)
        }

        # Determine field category
        field_info['category'] = self._categorize_field(field_info)

        return field_info

    def _categorize_field(self, field_info: Dict[str, str]) -> str:
        """Categorize form field type"""
        name = field_info.get('name', '').lower()
        placeholder = field_info.get('placeholder', '').lower()
        field_type = field_info.get('type', '')

        for category, pattern in self.form_field_patterns.items():
            if re.search(pattern, name) or re.search(pattern, placeholder):
                return category

        # Type-based categorization
        if field_type in ['email']:
            return 'email'
        elif field_type in ['password']:
            return 'password'
        elif field_type in ['tel', 'phone']:
            return 'phone'
        elif field_type in ['number']:
            return 'number'
        elif field_type in ['date']:
            return 'date'

        return 'text'

    def _assess_urgency(self, sentence: str) -> str:
        """Assess urgency level of an action"""
        urgent_words = ['immediately', 'now', 'urgent', 'critical', 'asap', 'quickly', 'fast']
        high_priority_words = ['must', 'required', 'necessary', 'important']

        if any(word in sentence.lower() for word in urgent_words):
            return 'urgent'
        elif any(word in sentence.lower() for word in high_priority_words):
            return 'high'
        else:
            return 'normal'

class FormAnalyzer:
    """Advanced form analysis using ML"""

    def __init__(self):
        self.field_classifier = None  # Would load ML model

    def enhance_field_analysis(self, fields: List[Dict], html: str) -> List[Dict]:
        """Enhance field analysis with ML insights"""
        # Analyze field relationships and dependencies
        for field in fields:
            field['dependencies'] = self._find_dependencies(field, fields)
            field['validation_rules'] = self._infer_validation(field, html)

        return fields

    def _find_dependencies(self, field: Dict, all_fields: List[Dict]) -> List[str]:
        """Find fields that this field depends on"""
        # Simple heuristic - could use ML
        dependencies = []
        field_name = field.get('name', '').lower()

        if 'confirm' in field_name:
            # This might be a confirmation field
            base_name = field_name.replace('confirm', '').replace('_', '').strip()
            for other_field in all_fields:
                if base_name in other_field.get('name', '').lower():
                    dependencies.append(other_field['name'])

        return dependencies

    def _infer_validation(self, field: Dict, html: str) -> List[str]:
        """Infer validation rules for field"""
        rules = []
        field_type = field.get('type', '')

        # Type-based rules
        if field_type == 'email':
            rules.extend(['email_format', 'required'])
        elif field_type == 'password':
            rules.extend(['min_length', 'complexity'])
        elif field.get('required'):
            rules.append('required')

        # Pattern-based rules from HTML
        if 'pattern=' in html:
            rules.append('pattern_match')

        return rules

class ContentGenerator:
    """Generate appropriate content for forms and interactions"""

    def __init__(self):
        self.templates = self._load_templates()

    def generate_field_response(self, field_type: str, field_name: str,
                              placeholder: str, label: str) -> str:
        """Generate appropriate response for form field"""
        # Use context to generate realistic responses
        context = f"{field_name} {placeholder} {label}".lower()

        if field_type == 'email':
            return self._generate_email(context)
        elif field_type == 'password':
            return self._generate_password()
        elif field_type == 'name':
            return self._generate_name(context)
        elif field_type == 'phone':
            return self._generate_phone()
        elif field_type == 'address':
            return self._generate_address()
        else:
            return self._generate_generic_text(field_type, context)

    def _generate_email(self, context: str) -> str:
        """Generate realistic email address"""
        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        if 'work' in context or 'business' in context:
            domains.insert(0, 'company.com')

        local_part = ''.join(['testuser', str(hash(context) % 1000)])
        domain = domains[hash(context) % len(domains)]

        return f"{local_part}@{domain}"

    def _generate_password(self) -> str:
        """Generate secure password"""
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits + string.punctuation
        password = ''.join(secrets.choice(alphabet) for i in range(12))
        return password

    def _generate_name(self, context: str) -> str:
        """Generate realistic name"""
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emma']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia']

        if 'first' in context:
            return first_names[hash(context) % len(first_names)]
        elif 'last' in context:
            return last_names[hash(context) % len(last_names)]
        else:
            first = first_names[hash(context) % len(first_names)]
            last = last_names[hash(context + 'last') % len(last_names)]
            return f"{first} {last}"

    def _generate_phone(self) -> str:
        """Generate realistic phone number"""
        import random
        area_codes = ['201', '202', '203', '204', '205']
        area = random.choice(area_codes)
        number = ''.join([str(random.randint(0, 9)) for _ in range(7)])
        return f"({area}) {number[:3]}-{number[3:]}"

    def _generate_address(self) -> str:
        """Generate realistic address"""
        streets = ['Main St', 'Oak Ave', 'Maple Dr', 'Cedar Ln']
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston']
        states = ['NY', 'CA', 'IL', 'TX']

        street_num = str(hash(self) % 9999 + 1)
        street = streets[hash(self) % len(streets)]
        city = cities[hash(self) % len(cities)]
        state = states[hash(self) % len(states)]
        zip_code = str(hash(self) % 90000 + 10000)

        return f"{street_num} {street}, {city}, {state} {zip_code}"

    def _generate_generic_text(self, field_type: str, context: str) -> str:
        """Generate generic text response"""
        if 'comment' in context or 'message' in context:
            return "This is a test message generated by BRAF AI."
        elif 'description' in context:
            return "Sample description for testing purposes."
        else:
            return f"Sample {field_type} input"

    def _load_templates(self) -> Dict[str, List[str]]:
        """Load response templates"""
        return {
            'email': ['user@example.com', 'test@test.com'],
            'name': ['John Doe', 'Jane Smith'],
            'phone': ['555-123-4567', '(555) 123-4567'],
            'address': ['123 Main St, Anytown, USA 12345']
        }

# Global instance
nlp_engine = NLPEngine()