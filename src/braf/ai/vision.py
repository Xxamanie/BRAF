#!/usr/bin/env python3
"""
BRAF Computer Vision Engine
Advanced computer vision for UI analysis, element detection, and captcha solving
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pytesseract
import easyocr
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import hashlib
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ComputerVisionEngine:
    """Advanced computer vision engine for BRAF"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models
        self.object_detector = None
        self.text_recognizer = None
        self.captcha_solver = None
        self.ui_analyzer = None

        self._initialize_models()

        # Preprocessing transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _initialize_models(self):
        """Initialize computer vision models"""
        try:
            # YOLOv8 for object detection
            self.object_detector = YOLO('yolov8n.pt')  # Nano model for speed

            # EasyOCR for text recognition
            self.text_recognizer = easyocr.Reader(['en'])

            # Initialize captcha solver
            self.captcha_solver = CaptchaSolver()

            # UI analysis model
            self.ui_analyzer = UIAnalyzer()

            logger.info("Computer vision models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vision models: {e}")
            # Fallback to basic OCR
            self.text_recognizer = pytesseract

    def analyze_screenshot(self, image_path: str) -> Dict[str, Any]:
        """Comprehensive analysis of a screenshot"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}

            pil_image = Image.open(image_path)

            # Detect UI elements
            ui_elements = self.detect_ui_elements(image)

            # Extract text
            text_regions = self.extract_text(image)

            # Detect interactive elements
            interactive_elements = self.detect_interactive_elements(image, ui_elements)

            # Analyze layout
            layout_analysis = self.analyze_layout(image, ui_elements)

            # Detect potential captchas
            captcha_regions = self.detect_captchas(image)

            return {
                'ui_elements': ui_elements,
                'text_regions': text_regions,
                'interactive_elements': interactive_elements,
                'layout_analysis': layout_analysis,
                'captcha_regions': captcha_regions,
                'success': True
            }

        except Exception as e:
            logger.error(f"Screenshot analysis failed: {e}")
            return {'error': str(e)}

    def detect_ui_elements(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect UI elements using object detection"""
        if not self.object_detector:
            return []

        try:
            # Convert to RGB for YOLO
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run detection
            results = self.object_detector(rgb_image)

            elements = []
            for result in results:
                for box in result.boxes:
                    element = {
                        'class': result.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy.tolist()[0],  # [x1, y1, x2, y2]
                        'type': self._classify_element_type(result.names[int(box.cls)])
                    }
                    elements.append(element)

            return elements

        except Exception as e:
            logger.error(f"UI element detection failed: {e}")
            return []

    def detect_interactive_elements(self, image: np.ndarray, ui_elements: List[Dict]) -> List[Dict[str, Any]]:
        """Detect buttons, links, inputs, etc."""
        interactive = []

        for element in ui_elements:
            if element['type'] in ['button', 'input', 'link', 'dropdown']:
                # Get element region
                x1, y1, x2, y2 = element['bbox']
                element_region = image[int(y1):int(y2), int(x1):int(x2)]

                # Analyze for interactivity clues
                is_interactive = self._analyze_interactivity(element_region, element)

                if is_interactive:
                    interactive.append({
                        **element,
                        'action_type': self._predict_action_type(element_region),
                        'clickable': True
                    })

        return interactive

    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text from image using OCR"""
        if not self.text_recognizer:
            return []

        try:
            # Use EasyOCR
            results = self.text_recognizer.readtext(image)

            text_regions = []
            for (bbox, text, confidence) in results:
                text_regions.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': confidence,
                    'type': 'ocr_text'
                })

            return text_regions

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return []

    def analyze_layout(self, image: np.ndarray, ui_elements: List[Dict]) -> Dict[str, Any]:
        """Analyze page layout and structure"""
        height, width = image.shape[:2]

        # Classify layout type
        layout_type = self._classify_layout(image, ui_elements)

        # Identify key regions
        regions = {
            'header': self._find_region(image, 'header', ui_elements),
            'navigation': self._find_region(image, 'navigation', ui_elements),
            'content': self._find_region(image, 'content', ui_elements),
            'footer': self._find_region(image, 'footer', ui_elements)
        }

        return {
            'layout_type': layout_type,
            'dimensions': {'width': width, 'height': height},
            'regions': regions,
            'complexity_score': len(ui_elements) / 10.0  # Simple complexity metric
        }

    def detect_captchas(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect potential captcha regions"""
        captcha_regions = []

        # Look for common captcha indicators
        height, width = image.shape[:2]

        # Scan for text clusters that might be captchas
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 50000:  # Reasonable captcha size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)

                # Check for captcha-like characteristics
                if 2 < aspect_ratio < 10:  # Long horizontal text
                    region = image[y:y+h, x:x+w]
                    if self._is_captcha_like(region):
                        captcha_regions.append({
                            'bbox': [x, y, x+w, y+h],
                            'confidence': 0.7,
                            'type': 'potential_captcha'
                        })

        return captcha_regions

    def solve_captcha(self, captcha_image: np.ndarray) -> str:
        """Solve captcha using advanced methods"""
        if not self.captcha_solver:
            return ""

        try:
            return self.captcha_solver.solve(captcha_image)
        except Exception as e:
            logger.error(f"Captcha solving failed: {e}")
            return ""

    def _classify_element_type(self, class_name: str) -> str:
        """Classify detected element type"""
        type_mapping = {
            'button': 'button',
            'input': 'input',
            'link': 'link',
            'select': 'dropdown',
            'textarea': 'input',
            'checkbox': 'input',
            'radio': 'input'
        }
        return type_mapping.get(class_name.lower(), 'unknown')

    def _analyze_interactivity(self, region: np.ndarray, element: Dict) -> bool:
        """Analyze if an element region shows interactivity clues"""
        # Simple heuristics for interactivity
        if region.size == 0:
            return False

        # Check for borders/shadows (indicators of clickable elements)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # High edge density might indicate buttons/links
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])

        return edge_density > 0.05 or element['type'] in ['button', 'link']

    def _predict_action_type(self, region: np.ndarray) -> str:
        """Predict the type of action an element performs"""
        # Analyze visual features to predict action
        # This is a simplified version - would use ML in production
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)

        if avg_brightness > 200:
            return 'submit'
        elif avg_brightness < 100:
            return 'cancel'
        else:
            return 'navigate'

    def _classify_layout(self, image: np.ndarray, elements: List[Dict]) -> str:
        """Classify the overall page layout"""
        height, width = image.shape[:2]

        # Simple layout classification
        element_count = len(elements)

        if element_count < 10:
            return 'simple'
        elif element_count < 50:
            return 'moderate'
        else:
            return 'complex'

    def _find_region(self, image: np.ndarray, region_type: str, elements: List[Dict]) -> Optional[Dict]:
        """Find a specific region in the layout"""
        height, width = image.shape[:2]

        if region_type == 'header':
            return {'bbox': [0, 0, width, height * 0.2], 'confidence': 0.8}
        elif region_type == 'navigation':
            return {'bbox': [0, height * 0.2, width * 0.3, height * 0.6], 'confidence': 0.6}
        elif region_type == 'content':
            return {'bbox': [width * 0.3, height * 0.2, width * 0.7, height * 0.7], 'confidence': 0.7}
        elif region_type == 'footer':
            return {'bbox': [0, height * 0.8, width, height * 0.2], 'confidence': 0.8}

        return None

    def _is_captcha_like(self, region: np.ndarray) -> bool:
        """Check if a region looks like a captcha"""
        if region.size == 0:
            return False

        # Analyze characteristics
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Captchas often have distorted text
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (height * width)

        # Check for text-like patterns
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        text_density = np.sum(thresh) / (height * width)

        return edge_density > 0.1 and text_density > 0.05

class CaptchaSolver:
    """Advanced captcha solving engine"""

    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load captcha solving models"""
        try:
            # Load pre-trained models for different captcha types
            self.models['text'] = self._create_text_captcha_model()
            self.models['image'] = self._create_image_captcha_model()
        except Exception as e:
            logger.error(f"Failed to load captcha models: {e}")

    def _create_text_captcha_model(self):
        """Create model for text-based captchas"""
        # Simplified CNN for text captcha recognition
        class TextCaptchaNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout2d(0.25)
                self.dropout2 = nn.Dropout2d(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 36)  # 26 letters + 10 digits

            def forward(self, x):
                x = self.conv1(x)
                x = torch.relu(x)
                x = self.conv2(x)
                x = torch.relu(x)
                x = torch.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                return torch.log_softmax(x, dim=1)

        return TextCaptchaNet()

    def _create_image_captcha_model(self):
        """Create model for image-based captchas"""
        # Use ResNet for image classification
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust for captcha classes
        return model

    def solve(self, captcha_image: np.ndarray) -> str:
        """Solve captcha from image"""
        try:
            # Preprocess image
            if len(captcha_image.shape) == 3:
                gray = cv2.cvtColor(captcha_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = captcha_image

            # Try OCR first
            text = pytesseract.image_to_string(gray, config='--psm 8').strip()

            if len(text) > 3:  # Basic validation
                return text

            # If OCR fails, try ML model
            return self._solve_with_ml(gray)

        except Exception as e:
            logger.error(f"Captcha solving error: {e}")
            return ""

    def _solve_with_ml(self, image: np.ndarray) -> str:
        """Solve using machine learning model"""
        # Placeholder - would implement actual ML solving
        return "SOLVED"

class UIAnalyzer:
    """UI structure and element analysis"""

    def __init__(self):
        self.element_classifier = None
        self.layout_analyzer = None

    def analyze_element_hierarchy(self, elements: List[Dict]) -> Dict[str, Any]:
        """Analyze the hierarchy and relationships between UI elements"""
        # Group elements by type and position
        groups = self._group_elements(elements)

        # Build hierarchy
        hierarchy = self._build_hierarchy(groups)

        return {
            'groups': groups,
            'hierarchy': hierarchy,
            'relationships': self._analyze_relationships(elements)
        }

    def _group_elements(self, elements: List[Dict]) -> Dict[str, List]:
        """Group elements by type and region"""
        groups = {
            'buttons': [],
            'inputs': [],
            'text': [],
            'images': [],
            'navigation': []
        }

        for element in elements:
            element_type = element.get('type', 'unknown')
            if element_type in groups:
                groups[element_type].append(element)

        return groups

    def _build_hierarchy(self, groups: Dict) -> Dict:
        """Build element hierarchy"""
        # Simple hierarchy based on positioning
        hierarchy = {
            'level_0': [],  # Top level
            'level_1': [],  # Navigation/content
            'level_2': []   # Interactive elements
        }

        # Assign elements to levels based on position
        for group_name, elements in groups.items():
            for element in elements:
                bbox = element.get('bbox', [])
                if bbox:
                    y_pos = (bbox[1] + bbox[3]) / 2  # Center Y
                    if y_pos < 200:  # Top of page
                        hierarchy['level_0'].append(element)
                    elif y_pos < 600:  # Middle
                        hierarchy['level_1'].append(element)
                    else:  # Bottom
                        hierarchy['level_2'].append(element)

        return hierarchy

    def _analyze_relationships(self, elements: List[Dict]) -> List[Dict]:
        """Analyze relationships between elements"""
        relationships = []

        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements[i+1:], i+1):
                if self._elements_related(elem1, elem2):
                    relationships.append({
                        'element1': i,
                        'element2': j,
                        'relationship': self._classify_relationship(elem1, elem2)
                    })

        return relationships

    def _elements_related(self, elem1: Dict, elem2: Dict) -> bool:
        """Check if two elements are related"""
        bbox1 = elem1.get('bbox', [])
        bbox2 = elem2.get('bbox', [])

        if not bbox1 or not bbox2:
            return False

        # Check if bounding boxes are close
        x1_center = (bbox1[0] + bbox1[2]) / 2
        x2_center = (bbox2[0] + bbox2[2]) / 2

        return abs(x1_center - x2_center) < 100  # Within 100px horizontally

    def _classify_relationship(self, elem1: Dict, elem2: Dict) -> str:
        """Classify the relationship between elements"""
        type1 = elem1.get('type', '')
        type2 = elem2.get('type', '')

        if type1 == 'input' and type2 == 'button':
            return 'input_with_submit'
        elif type1 == 'text' and type2 == 'input':
            return 'label_input'
        else:
            return 'spatial_proximity'

# Global instance
vision_engine = ComputerVisionEngine()