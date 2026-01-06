"""
Human-like mouse movement generation using Bezier curves with realistic noise.

This module implements smooth, natural mouse movements that mimic human behavior
with proper acceleration, deceleration, and random variations.
"""

import math
import random
from typing import List, Tuple

import numpy as np
from scipy.special import comb

# Type alias for coordinate points
Point = Tuple[float, float]


class BezierMouseMovement:
    """Generator for human-like mouse movements using Bezier curves."""
    
    def __init__(self, noise_factor: float = 0.5, min_velocity: float = 0.5, max_velocity: float = 1.5):
        """
        Initialize Bezier mouse movement generator.
        
        Args:
            noise_factor: Amount of random noise to add (0.0 to 1.0)
            min_velocity: Minimum velocity multiplier
            max_velocity: Maximum velocity multiplier
        """
        self.noise_factor = noise_factor
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
    
    def generate_movement_path(
        self, 
        start: Point, 
        end: Point, 
        num_points: int = 100,
        control_points: int = 4
    ) -> List[Tuple[float, float, float]]:
        """
        Generate smooth mouse movement path from start to end point.
        
        Args:
            start: Starting coordinate (x, y)
            end: Ending coordinate (x, y)
            num_points: Number of points in the path
            control_points: Number of Bezier control points
            
        Returns:
            List of (x, y, timestamp) tuples representing the movement path
        """
        # Generate control points for natural curve
        bezier_points = self._generate_control_points(start, end, control_points)
        
        # Generate base Bezier curve
        curve_points = self._generate_bezier_curve(bezier_points, num_points)
        
        # Add realistic noise
        noisy_points = self._add_perlin_noise(curve_points)
        
        # Apply velocity profile (acceleration/deceleration)
        timed_points = self._apply_velocity_profile(noisy_points)
        
        return timed_points
    
    def _generate_control_points(self, start: Point, end: Point, num_controls: int) -> List[Point]:
        """
        Generate control points for Bezier curve with natural variation.
        
        Args:
            start: Starting point
            end: Ending point
            num_controls: Number of control points to generate
            
        Returns:
            List of control points including start and end
        """
        points = [start]
        
        # Calculate distance and direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Generate intermediate control points
        for i in range(1, num_controls - 1):
            # Linear interpolation with random offset
            t = i / (num_controls - 1)
            
            # Base position along straight line
            base_x = start[0] + t * dx
            base_y = start[1] + t * dy
            
            # Add perpendicular offset for natural curve
            perpendicular_offset = distance * 0.1 * random.uniform(-1, 1)
            
            # Calculate perpendicular direction
            if distance > 0:
                perp_x = -dy / distance
                perp_y = dx / distance
            else:
                perp_x = perp_y = 0
            
            # Apply offset
            control_x = base_x + perp_x * perpendicular_offset
            control_y = base_y + perp_y * perpendicular_offset
            
            # Add some random variation
            control_x += random.uniform(-distance * 0.05, distance * 0.05)
            control_y += random.uniform(-distance * 0.05, distance * 0.05)
            
            points.append((control_x, control_y))
        
        points.append(end)
        return points
    
    def _generate_bezier_curve(self, control_points: List[Point], num_points: int) -> List[Point]:
        """
        Generate Bezier curve from control points.
        
        Args:
            control_points: List of control points
            num_points: Number of points to generate
            
        Returns:
            List of points along the Bezier curve
        """
        n = len(control_points) - 1
        t_values = np.linspace(0.0, 1.0, num_points)
        
        curve_points = []
        
        for t in t_values:
            x = y = 0.0
            
            # Calculate Bezier curve point using Bernstein polynomials
            for i, (px, py) in enumerate(control_points):
                bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
                x += px * bernstein
                y += py * bernstein
            
            curve_points.append((x, y))
        
        return curve_points
    
    def _add_perlin_noise(self, points: List[Point]) -> List[Point]:
        """
        Add Perlin-like noise to create natural variations.
        
        Args:
            points: List of curve points
            
        Returns:
            List of points with added noise
        """
        if self.noise_factor <= 0:
            return points
        
        noisy_points = []
        
        for i, (x, y) in enumerate(points):
            # Generate noise that decreases towards the end (more precision near target)
            progress = i / len(points)
            noise_scale = self.noise_factor * (1 - np.exp(-5 * progress))
            
            # Add Gaussian noise
            noise_x = random.gauss(0, noise_scale)
            noise_y = random.gauss(0, noise_scale)
            
            noisy_points.append((x + noise_x, y + noise_y))
        
        return noisy_points
    
    def _apply_velocity_profile(self, points: List[Point]) -> List[Tuple[float, float, float]]:
        """
        Apply realistic velocity profile with acceleration and deceleration.
        
        Args:
            points: List of movement points
            
        Returns:
            List of (x, y, timestamp) tuples
        """
        if len(points) < 2:
            return [(points[0][0], points[0][1], 0.0)] if points else []
        
        # Calculate distances between consecutive points
        distances = []
        total_distance = 0
        
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            distance = math.sqrt(dx * dx + dy * dy)
            distances.append(distance)
            total_distance += distance
        
        # Generate velocity profile (ease-in-out)
        timed_points = [(points[0][0], points[0][1], 0.0)]
        current_time = 0.0
        
        # Random base velocity
        base_velocity = random.uniform(self.min_velocity, self.max_velocity)
        
        for i, distance in enumerate(distances):
            progress = i / len(distances)
            
            # Ease-in-out velocity curve
            if progress < 0.5:
                # Acceleration phase
                velocity_multiplier = 2 * progress * progress
            else:
                # Deceleration phase
                velocity_multiplier = 1 - 2 * (progress - 0.5) * (progress - 0.5)
            
            # Apply velocity with some randomness
            velocity = base_velocity * (0.5 + 0.5 * velocity_multiplier)
            velocity *= random.uniform(0.8, 1.2)  # Add variation
            
            # Calculate time for this segment
            if velocity > 0:
                segment_time = distance / velocity
            else:
                segment_time = 0.01  # Minimum time
            
            current_time += segment_time
            
            timed_points.append((
                points[i + 1][0],
                points[i + 1][1],
                current_time
            ))
        
        return timed_points


class MouseMovementOptimizer:
    """Optimizer for mouse movement paths to reduce detection risk."""
    
    @staticmethod
    def optimize_path_for_stealth(path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Optimize movement path to reduce bot detection risk.
        
        Args:
            path: Original movement path
            
        Returns:
            Optimized movement path
        """
        if len(path) < 3:
            return path
        
        optimized = [path[0]]  # Keep start point
        
        for i in range(1, len(path) - 1):
            prev_point = optimized[-1]
            curr_point = path[i]
            next_point = path[i + 1]
            
            # Check for unnatural straight lines
            if MouseMovementOptimizer._is_too_straight(prev_point, curr_point, next_point):
                # Add slight deviation
                deviation_x = random.uniform(-2, 2)
                deviation_y = random.uniform(-2, 2)
                
                optimized.append((
                    curr_point[0] + deviation_x,
                    curr_point[1] + deviation_y,
                    curr_point[2]
                ))
            else:
                optimized.append(curr_point)
        
        optimized.append(path[-1])  # Keep end point
        return optimized
    
    @staticmethod
    def _is_too_straight(p1: Tuple[float, float, float], 
                        p2: Tuple[float, float, float], 
                        p3: Tuple[float, float, float]) -> bool:
        """
        Check if three consecutive points form too straight a line.
        
        Args:
            p1, p2, p3: Three consecutive points
            
        Returns:
            True if the line is suspiciously straight
        """
        # Calculate angle between vectors
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Calculate magnitudes
        mag1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
        mag2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
        
        if mag1 == 0 or mag2 == 0:
            return False
        
        # Calculate dot product and angle
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        cos_angle = dot_product / (mag1 * mag2)
        
        # Clamp to valid range for acos
        cos_angle = max(-1, min(1, cos_angle))
        angle = math.acos(cos_angle)
        
        # Consider it too straight if angle is very small (< 5 degrees)
        return angle < math.radians(5)


def generate_human_mouse_movement(
    start: Point, 
    end: Point, 
    noise_factor: float = 0.5,
    num_points: int = 100
) -> List[Tuple[float, float, float]]:
    """
    Convenience function to generate human-like mouse movement.
    
    Args:
        start: Starting coordinate
        end: Ending coordinate
        noise_factor: Amount of noise to add (0.0 to 1.0)
        num_points: Number of points in the path
        
    Returns:
        List of (x, y, timestamp) tuples
    """
    generator = BezierMouseMovement(noise_factor=noise_factor)
    path = generator.generate_movement_path(start, end, num_points)
    
    # Optimize for stealth
    optimized_path = MouseMovementOptimizer.optimize_path_for_stealth(path)
    
    return optimized_path


def calculate_movement_metrics(path: List[Tuple[float, float, float]]) -> dict:
    """
    Calculate metrics for mouse movement path analysis.
    
    Args:
        path: Movement path with timestamps
        
    Returns:
        Dictionary of movement metrics
    """
    if len(path) < 2:
        return {"total_distance": 0, "total_time": 0, "average_velocity": 0}
    
    total_distance = 0
    velocities = []
    
    for i in range(1, len(path)):
        # Calculate distance
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        distance = math.sqrt(dx * dx + dy * dy)
        total_distance += distance
        
        # Calculate velocity
        time_diff = path[i][2] - path[i-1][2]
        if time_diff > 0:
            velocity = distance / time_diff
            velocities.append(velocity)
    
    total_time = path[-1][2] - path[0][2]
    average_velocity = sum(velocities) / len(velocities) if velocities else 0
    max_velocity = max(velocities) if velocities else 0
    min_velocity = min(velocities) if velocities else 0
    
    return {
        "total_distance": total_distance,
        "total_time": total_time,
        "average_velocity": average_velocity,
        "max_velocity": max_velocity,
        "min_velocity": min_velocity,
        "velocity_variance": np.var(velocities) if velocities else 0,
        "num_points": len(path)
    }
