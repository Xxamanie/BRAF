"""
Human-like typing simulation with realistic errors and corrections.

This module simulates natural typing patterns including variable timing,
realistic errors, and correction behaviors that mimic human typing.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

from braf.core.models import BehavioralConfig


class HumanTyper:
    """Simulator for human-like typing behavior."""
    
    def __init__(self, wpm_range: Tuple[int, int] = (40, 80), error_rate: float = 0.02):
        """
        Initialize human typing simulator.
        
        Args:
            wpm_range: Range of words per minute (min, max)
            error_rate: Probability of making typing errors (0.0 to 1.0)
        """
        self.wpm_range = wpm_range
        self.error_rate = error_rate
        
        # Common typing errors (character substitutions)
        self.common_errors = {
            # QWERTY keyboard adjacent key errors
            'a': ['s', 'q', 'w'],
            'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'e', 'r', 'f', 'c', 'x'],
            'e': ['w', 'r', 'd', 's'],
            'f': ['d', 'r', 't', 'g', 'v', 'c'],
            'g': ['f', 't', 'y', 'h', 'b', 'v'],
            'h': ['g', 'y', 'u', 'j', 'n', 'b'],
            'i': ['u', 'o', 'k', 'j'],
            'j': ['h', 'u', 'i', 'k', 'm', 'n'],
            'k': ['j', 'i', 'o', 'l', 'm'],
            'l': ['k', 'o', 'p'],
            'm': ['n', 'j', 'k'],
            'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'p', 'l', 'k'],
            'p': ['o', 'l'],
            'q': ['w', 'a'],
            'r': ['e', 't', 'f', 'd'],
            's': ['a', 'w', 'e', 'd', 'x', 'z'],
            't': ['r', 'y', 'g', 'f'],
            'u': ['y', 'i', 'j', 'h'],
            'v': ['c', 'f', 'g', 'b'],
            'w': ['q', 'e', 's', 'a'],
            'x': ['z', 's', 'd', 'c'],
            'y': ['t', 'u', 'h', 'g'],
            'z': ['x', 's', 'a'],
            
            # Number row errors
            '1': ['2', 'q'],
            '2': ['1', '3', 'q', 'w'],
            '3': ['2', '4', 'w', 'e'],
            '4': ['3', '5', 'e', 'r'],
            '5': ['4', '6', 'r', 't'],
            '6': ['5', '7', 't', 'y'],
            '7': ['6', '8', 'y', 'u'],
            '8': ['7', '9', 'u', 'i'],
            '9': ['8', '0', 'i', 'o'],
            '0': ['9', 'o', 'p'],
        }
        
        # Common word-level errors
        self.word_errors = {
            'the': ['teh', 'hte'],
            'and': ['adn', 'nad'],
            'you': ['yuo', 'oyu'],
            'that': ['taht', 'htat'],
            'with': ['wiht', 'whit'],
            'have': ['ahve', 'hvae'],
            'this': ['tihs', 'htis'],
            'will': ['wlil', 'iwll'],
            'your': ['yuor', 'oyur'],
            'from': ['form', 'fomr'],
            'they': ['tehy', 'htey'],
            'know': ['konw', 'nkow'],
            'want': ['watn', 'wnat'],
            'been': ['bene', 'ebn'],
            'good': ['godo', 'ogod'],
            'much': ['muhc', 'mcuh'],
            'some': ['soem', 'smoe'],
            'time': ['tiem', 'itme'],
            'very': ['vrey', 'evrv'],
            'when': ['wehn', 'hwne'],
            'come': ['coem', 'ocme'],
            'here': ['heer', 'ehre'],
            'just': ['jsut', 'ujst'],
            'like': ['liek', 'ilke'],
            'long': ['logn', 'olgn'],
            'make': ['meka', 'amke'],
            'many': ['myna', 'anmy'],
            'over': ['oevr', 'voer'],
            'such': ['suhc', 'ushc'],
            'take': ['taek', 'atek'],
            'than': ['tahn', 'htan'],
            'them': ['tehm', 'htem'],
            'well': ['wlel', 'ewll'],
            'were': ['wree', 'ewre']
        }
    
    def calculate_keystroke_delay(self, char: str, prev_char: Optional[str] = None) -> float:
        """
        Calculate realistic delay between keystrokes.
        
        Args:
            char: Current character being typed
            prev_char: Previous character (for digraph timing)
            
        Returns:
            Delay in seconds
        """
        # Base delay from WPM
        base_wpm = random.uniform(*self.wpm_range)
        base_delay = 60.0 / (base_wpm * 5)  # 5 characters per word average
        
        # Adjust for character type
        if char.isalpha():
            char_multiplier = 1.0
        elif char.isdigit():
            char_multiplier = 1.2  # Numbers are slightly slower
        elif char in ' \t':
            char_multiplier = 0.8  # Spaces are faster
        elif char in '.,!?;:':
            char_multiplier = 1.1  # Punctuation slightly slower
        else:
            char_multiplier = 1.3  # Special characters slower
        
        # Digraph adjustments (common letter combinations)
        digraph_multiplier = 1.0
        if prev_char:
            digraph = prev_char.lower() + char.lower()
            
            # Fast digraphs (common combinations)
            fast_digraphs = ['th', 'he', 'in', 'er', 'an', 're', 'ed', 'nd', 'ou', 'ea']
            if digraph in fast_digraphs:
                digraph_multiplier = 0.85
            
            # Slow digraphs (awkward combinations)
            elif digraph in ['qw', 'xz', 'zx', 'qx', 'jq', 'vx']:
                digraph_multiplier = 1.4
            
            # Same finger combinations (slower)
            same_finger_pairs = [
                ['q', 'a', 'z'], ['w', 's', 'x'], ['e', 'd', 'c'], ['r', 'f', 'v'],
                ['t', 'g', 'b'], ['y', 'h', 'n'], ['u', 'j', 'm'], ['i', 'k'],
                ['o', 'l'], ['p']
            ]
            
            for finger_group in same_finger_pairs:
                if prev_char.lower() in finger_group and char.lower() in finger_group:
                    digraph_multiplier = 1.3
                    break
        
        # Add random variation
        variation = random.uniform(0.7, 1.3)
        
        delay = base_delay * char_multiplier * digraph_multiplier * variation
        
        # Ensure minimum and maximum delays
        return max(0.05, min(0.5, delay))
    
    def should_make_error(self, char: str, word_position: int, word_length: int) -> bool:
        """
        Determine if an error should be made at this position.
        
        Args:
            char: Character being typed
            word_position: Position within the current word
            word_length: Total length of the word
            
        Returns:
            True if an error should be made
        """
        # Base error probability
        error_prob = self.error_rate
        
        # Increase error probability for:
        # - Longer words
        if word_length > 8:
            error_prob *= 1.5
        
        # - Middle of words (less attention)
        if 0.2 < word_position / word_length < 0.8:
            error_prob *= 1.2
        
        # - Difficult characters
        if char.lower() in 'qxzj':
            error_prob *= 1.3
        
        # - Numbers and special characters
        if not char.isalpha():
            error_prob *= 1.4
        
        return random.random() < error_prob
    
    def generate_character_error(self, char: str) -> str:
        """
        Generate a realistic typing error for a character.
        
        Args:
            char: Original character
            
        Returns:
            Error character
        """
        char_lower = char.lower()
        
        # Use adjacent key errors if available
        if char_lower in self.common_errors:
            error_char = random.choice(self.common_errors[char_lower])
            
            # Preserve case
            if char.isupper():
                return error_char.upper()
            else:
                return error_char
        
        # Fallback to random similar character
        if char.isalpha():
            # Random letter
            return random.choice('abcdefghijklmnopqrstuvwxyz')
        elif char.isdigit():
            # Random digit
            return random.choice('0123456789')
        else:
            # Return original for special characters
            return char
    
    def generate_word_error(self, word: str) -> Optional[str]:
        """
        Generate a word-level error (transposition, etc.).
        
        Args:
            word: Original word
            
        Returns:
            Error word or None if no error should be made
        """
        word_lower = word.lower()
        
        # Check for common word errors
        if word_lower in self.word_errors:
            if random.random() < 0.3:  # 30% chance for word-level error
                error_word = random.choice(self.word_errors[word_lower])
                
                # Preserve case pattern
                if word.isupper():
                    return error_word.upper()
                elif word.istitle():
                    return error_word.capitalize()
                else:
                    return error_word
        
        # Generate transposition error for longer words
        if len(word) > 3 and random.random() < 0.1:
            chars = list(word)
            # Swap two adjacent characters
            pos = random.randint(0, len(chars) - 2)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return ''.join(chars)
        
        return None
    
    def simulate_correction(self, error_length: int) -> List[Tuple[str, float]]:
        """
        Simulate error correction behavior (backspaces and retyping).
        
        Args:
            error_length: Number of characters to correct
            
        Returns:
            List of (action, delay) tuples for correction
        """
        corrections = []
        
        # Pause before correction (realization delay)
        realization_delay = random.uniform(0.2, 0.8)
        corrections.append(('pause', realization_delay))
        
        # Backspace the error
        for _ in range(error_length):
            backspace_delay = random.uniform(0.1, 0.25)
            corrections.append(('backspace', backspace_delay))
        
        # Small pause before retyping
        retype_delay = random.uniform(0.1, 0.3)
        corrections.append(('pause', retype_delay))
        
        return corrections


class TypingSession:
    """Manages a typing session with realistic behavior patterns."""
    
    def __init__(self, config: Optional[BehavioralConfig] = None):
        """
        Initialize typing session.
        
        Args:
            config: Behavioral configuration for typing parameters
        """
        if config:
            wpm_range = (config.typing_speed_wpm - 20, config.typing_speed_wpm + 20)
            error_rate = config.error_rate
        else:
            wpm_range = (40, 80)
            error_rate = 0.02
        
        self.typer = HumanTyper(wpm_range, error_rate)
        self.session_fatigue = 0.0  # Increases over time
        self.words_typed = 0
        self.errors_made = 0
        self.corrections_made = 0
    
    def type_text(self, text: str) -> List[Tuple[str, float]]:
        """
        Generate typing sequence for given text.
        
        Args:
            text: Text to type
            
        Returns:
            List of (character/action, delay) tuples
        """
        typing_sequence = []
        words = text.split()
        
        for word_idx, word in enumerate(words):
            # Add space before word (except first)
            if word_idx > 0:
                space_delay = self.typer.calculate_keystroke_delay(' ')
                typing_sequence.append((' ', space_delay))
            
            # Check for word-level error
            word_error = self.typer.generate_word_error(word)
            if word_error:
                # Type the error word
                word_sequence = self._type_word(word_error)
                typing_sequence.extend(word_sequence)
                
                # Realize error and correct it
                correction_sequence = self.typer.simulate_correction(len(word_error))
                typing_sequence.extend(correction_sequence)
                
                # Type correct word
                correct_sequence = self._type_word(word)
                typing_sequence.extend(correct_sequence)
                
                self.errors_made += 1
                self.corrections_made += 1
            else:
                # Type word normally (may still have character errors)
                word_sequence = self._type_word(word)
                typing_sequence.extend(word_sequence)
            
            self.words_typed += 1
            self._update_fatigue()
        
        return typing_sequence
    
    def _type_word(self, word: str) -> List[Tuple[str, float]]:
        """
        Generate typing sequence for a single word.
        
        Args:
            word: Word to type
            
        Returns:
            List of (character, delay) tuples
        """
        sequence = []
        prev_char = None
        
        for char_idx, char in enumerate(word):
            # Calculate delay with fatigue
            base_delay = self.typer.calculate_keystroke_delay(char, prev_char)
            fatigue_multiplier = 1.0 + self.session_fatigue * 0.5
            delay = base_delay * fatigue_multiplier
            
            # Check for character error
            if self.typer.should_make_error(char, char_idx, len(word)):
                error_char = self.typer.generate_character_error(char)
                sequence.append((error_char, delay))
                
                # Correction behavior
                if random.random() < 0.8:  # 80% chance to correct immediately
                    correction_sequence = self.typer.simulate_correction(1)
                    sequence.extend(correction_sequence)
                    
                    # Type correct character
                    correct_delay = self.typer.calculate_keystroke_delay(char, prev_char)
                    sequence.append((char, correct_delay))
                    
                    self.corrections_made += 1
                
                self.errors_made += 1
            else:
                sequence.append((char, delay))
            
            prev_char = char
        
        return sequence
    
    def _update_fatigue(self):
        """Update session fatigue based on words typed."""
        # Fatigue increases slowly over time
        self.session_fatigue = min(0.3, self.words_typed * 0.001)
    
    def get_session_stats(self) -> Dict:
        """
        Get statistics for the typing session.
        
        Returns:
            Dictionary of session statistics
        """
        accuracy = 1.0 - (self.errors_made / max(1, self.words_typed * 5))  # Assume 5 chars per word
        
        return {
            "words_typed": self.words_typed,
            "errors_made": self.errors_made,
            "corrections_made": self.corrections_made,
            "accuracy": accuracy,
            "session_fatigue": self.session_fatigue,
            "correction_rate": self.corrections_made / max(1, self.errors_made)
        }


def simulate_human_typing(text: str, config: Optional[BehavioralConfig] = None) -> List[Tuple[str, float]]:
    """
    Convenience function to simulate human typing for text.
    
    Args:
        text: Text to type
        config: Optional behavioral configuration
        
    Returns:
        List of (character/action, delay) tuples
    """
    session = TypingSession(config)
    return session.type_text(text)


def calculate_typing_metrics(typing_sequence: List[Tuple[str, float]]) -> Dict:
    """
    Calculate metrics for typing sequence analysis.
    
    Args:
        typing_sequence: Sequence of (character/action, delay) tuples
        
    Returns:
        Dictionary of typing metrics
    """
    total_time = sum(delay for _, delay in typing_sequence)
    character_count = len([char for char, _ in typing_sequence if len(char) == 1 and char.isalnum()])
    
    if total_time > 0 and character_count > 0:
        chars_per_second = character_count / total_time
        wpm = (chars_per_second * 60) / 5  # 5 characters per word average
    else:
        chars_per_second = 0
        wpm = 0
    
    delays = [delay for _, delay in typing_sequence]
    
    return {
        "total_time": total_time,
        "character_count": character_count,
        "chars_per_second": chars_per_second,
        "words_per_minute": wpm,
        "average_delay": sum(delays) / len(delays) if delays else 0,
        "min_delay": min(delays) if delays else 0,
        "max_delay": max(delays) if delays else 0,
        "total_keystrokes": len(typing_sequence)
    }