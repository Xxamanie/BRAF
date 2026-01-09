#!/usr/bin/env python3
"""
BRAF Optimization Problem Solver
Specialized solver for optimization problems with validated benchmarks
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import random
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class OptimizationProblem:
    """Represents an optimization problem with validation"""
    name: str
    objective_function: Callable
    constraints: List[Callable] = None
    bounds: List[Tuple[float, float]] = None
    dimension: int = None
    problem_type: str = "continuous"  # continuous, discrete, combinatorial
    global_minimum: Optional[float] = None  # Known optimal value for validation
    optimal_solution: Optional[np.ndarray] = None  # Known optimal solution
    benchmark_name: str = None  # CEC, BBOB, etc.
    difficulty_level: str = "medium"  # easy, medium, hard, very_hard

@dataclass
class SolverResult:
    """Result from optimization solver with validation"""
    solution: np.ndarray
    objective_value: float
    convergence_history: List[float]
    computation_time: float
    method_used: str
    iterations: int
    function_evaluations: int
    constraint_violations: int = 0
    feasibility_score: float = 1.0  # 1.0 = fully feasible
    optimality_gap: Optional[float] = None  # Gap to known optimum
    benchmark_score: Optional[float] = None  # Standardized benchmark score

class OptimizationSolver:
    """Validated optimization problem solver with benchmarks"""

    def __init__(self):
        self.benchmarks = self._load_benchmarks()
        self.solvers = {
            'differential_evolution': self._differential_evolution,
            'particle_swarm': self._particle_swarm_optimization,
            'simulated_annealing': self._simulated_annealing,
            'genetic_algorithm': self._genetic_algorithm,
            'bayesian_optimization': self._bayesian_optimization,
            'gradient_descent': self._gradient_descent,
            'newton_method': self._newton_method
        }

        self.benchmark_results = defaultdict(list)

    def solve(self, problem: OptimizationProblem, method: str = "auto",
             max_iterations: int = 1000, tolerance: float = 1e-6) -> SolverResult:
        """Solve optimization problem with specified method"""

        start_time = datetime.now()

        # Auto-select method if requested
        if method == "auto":
            method = self._select_optimal_method(problem)

        # Validate problem
        self._validate_problem(problem)

        # Solve using selected method
        if method not in self.solvers:
            raise ValueError(f"Unknown solver method: {method}")

        solver_func = self.solvers[method]
        result = solver_func(problem, max_iterations, tolerance)

        # Validate result
        result = self._validate_result(result, problem)

        # Compute benchmark score if applicable
        if problem.benchmark_name:
            result.benchmark_score = self._compute_benchmark_score(result, problem)

        # Log performance
        computation_time = (datetime.now() - start_time).total_seconds()
        result.computation_time = computation_time

        logger.info(".6f"
                   f"Benchmark: {result.benchmark_score}")

        # Store benchmark result
        self.benchmark_results[problem.name].append({
            'method': method,
            'result': result,
            'problem': problem,
            'timestamp': datetime.now()
        })

        return result

    def benchmark_solver(self, benchmark_name: str, methods: List[str] = None) -> Dict[str, Any]:
        """Run solver on standard benchmark suite"""

        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        if methods is None:
            methods = list(self.solvers.keys())

        benchmark_problems = self.benchmarks[benchmark_name]
        results = defaultdict(list)

        for problem in benchmark_problems:
            for method in methods:
                try:
                    result = self.solve(problem, method)
                    results[method].append(result)
                    logger.info(f"Completed {problem.name} with {method}")
                except Exception as e:
                    logger.warning(f"Failed {problem.name} with {method}: {e}")

        # Compute aggregate statistics
        summary = {}
        for method, method_results in results.items():
            if method_results:
                objective_values = [r.objective_value for r in method_results]
                computation_times = [r.computation_time for r in method_results]
                optimality_gaps = [r.optimality_gap for r in method_results if r.optimality_gap is not None]

                summary[method] = {
                    'mean_objective': np.mean(objective_values),
                    'std_objective': np.std(objective_values),
                    'mean_time': np.mean(computation_times),
                    'mean_optimality_gap': np.mean(optimality_gaps) if optimality_gaps else None,
                    'success_rate': len([r for r in method_results if r.optimality_gap and r.optimality_gap < 0.01]) / len(method_results),
                    'problems_solved': len(method_results)
                }

        return {
            'benchmark_name': benchmark_name,
            'methods_tested': methods,
            'problems_tested': len(benchmark_problems),
            'results': summary,
            'best_method': min(summary.keys(), key=lambda k: summary[k]['mean_objective']) if summary else None
        }

    def _differential_evolution(self, problem: OptimizationProblem,
                              max_iterations: int, tolerance: float) -> SolverResult:
        """Differential Evolution optimization"""

        population_size = min(50, 10 * problem.dimension)
        cr = 0.9  # Crossover rate
        f = 0.8   # Differential weight

        # Initialize population
        if problem.bounds:
            population = np.array([np.random.uniform(low, high, problem.dimension)
                                 for low, high in problem.bounds])
        else:
            population = np.random.randn(population_size, problem.dimension)

        fitness = np.array([problem.objective_function(ind) for ind in population])
        convergence_history = [np.min(fitness)]

        for iteration in range(max_iterations):
            new_population = population.copy()

            for i in range(population_size):
                # Select three random individuals different from i
                indices = [j for j in range(population_size) if j != i]
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Mutation
                mutant = population[a] + f * (population[b] - population[c])

                # Crossover
                trial = population[i].copy()
                crossover_points = np.random.rand(problem.dimension) < cr

                if not np.any(crossover_points):
                    crossover_points[np.random.randint(problem.dimension)] = True

                trial[crossover_points] = mutant[crossover_points]

                # Selection
                trial_fitness = problem.objective_function(trial)

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

            population = new_population
            convergence_history.append(np.min(fitness))

            # Check convergence
            if len(convergence_history) > 10:
                recent_change = abs(convergence_history[-1] - convergence_history[-10])
                if recent_change < tolerance:
                    break

        best_idx = np.argmin(fitness)
        return SolverResult(
            solution=population[best_idx],
            objective_value=fitness[best_idx],
            convergence_history=convergence_history,
            computation_time=0.0,  # Will be set by caller
            method_used="differential_evolution",
            iterations=iteration + 1,
            function_evaluations=population_size * (iteration + 1)
        )

    def _particle_swarm_optimization(self, problem: OptimizationProblem,
                                   max_iterations: int, tolerance: float) -> SolverResult:
        """Particle Swarm Optimization"""

        num_particles = min(30, 5 * problem.dimension)
        w = 0.7  # Inertia weight
        c1 = 1.4  # Personal acceleration
        c2 = 1.4  # Social acceleration

        # Initialize particles
        if problem.bounds:
            positions = np.array([np.random.uniform(low, high, problem.dimension)
                                for low, high in problem.bounds for _ in range(num_particles)])
            positions = positions.reshape(num_particles, problem.dimension)
        else:
            positions = np.random.randn(num_particles, problem.dimension)

        velocities = np.random.randn(num_particles, problem.dimension) * 0.1

        # Personal bests
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([problem.objective_function(pos) for pos in positions])

        # Global best
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        convergence_history = [global_best_score]

        for iteration in range(max_iterations):
            for i in range(num_particles):
                r1, r2 = np.random.rand(2)

                # Update velocity
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social

                # Update position
                positions[i] += velocities[i]

                # Clip to bounds if specified
                if problem.bounds:
                    for d in range(problem.dimension):
                        low, high = problem.bounds[d]
                        positions[i, d] = np.clip(positions[i, d], low, high)

                # Update personal best
                current_score = problem.objective_function(positions[i])
                if current_score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i].copy()
                    personal_best_scores[i] = current_score

                    # Update global best
                    if current_score < global_best_score:
                        global_best_position = positions[i].copy()
                        global_best_score = current_score

            convergence_history.append(global_best_score)

            # Check convergence
            if len(convergence_history) > 5 and abs(convergence_history[-1] - convergence_history[-5]) < tolerance:
                break

        return SolverResult(
            solution=global_best_position,
            objective_value=global_best_score,
            convergence_history=convergence_history,
            computation_time=0.0,
            method_used="particle_swarm",
            iterations=iteration + 1,
            function_evaluations=num_particles * (iteration + 1)
        )

    def _simulated_annealing(self, problem: OptimizationProblem,
                           max_iterations: int, tolerance: float) -> SolverResult:
        """Simulated Annealing optimization"""

        initial_temp = 100.0
        final_temp = 0.01
        alpha = 0.95  # Cooling rate

        # Initial solution
        if problem.bounds:
            current_solution = np.array([np.random.uniform(low, high)
                                       for low, high in problem.bounds])
        else:
            current_solution = np.random.randn(problem.dimension)

        current_energy = problem.objective_function(current_solution)
        best_solution = current_solution.copy()
        best_energy = current_energy

        convergence_history = [best_energy]
        temperature = initial_temp

        for iteration in range(max_iterations):
            # Generate neighbor solution
            neighbor = current_solution + np.random.normal(0, 0.1, problem.dimension)

            # Clip to bounds
            if problem.bounds:
                for d in range(problem.dimension):
                    low, high = problem.bounds[d]
                    neighbor[d] = np.clip(neighbor[d], low, high)

            neighbor_energy = problem.objective_function(neighbor)

            # Acceptance probability
            delta_energy = neighbor_energy - current_energy
            if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
                current_solution = neighbor
                current_energy = neighbor_energy

                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy

            # Cool down
            temperature *= alpha
            if temperature < final_temp:
                break

            if iteration % 10 == 0:
                convergence_history.append(best_energy)

        return SolverResult(
            solution=best_solution,
            objective_value=best_energy,
            convergence_history=convergence_history,
            computation_time=0.0,
            method_used="simulated_annealing",
            iterations=iteration + 1,
            function_evaluations=iteration + 1
        )

    def _genetic_algorithm(self, problem: OptimizationProblem,
                         max_iterations: int, tolerance: float) -> SolverResult:
        """Genetic Algorithm optimization"""

        population_size = min(50, 10 * problem.dimension)
        mutation_rate = 0.1
        crossover_rate = 0.8

        # Initialize population
        if problem.bounds:
            population = np.array([np.random.uniform(low, high, problem.dimension)
                                 for low, high in problem.bounds for _ in range(population_size)])
            population = population.reshape(population_size, problem.dimension)
        else:
            population = np.random.randn(population_size, problem.dimension)

        fitness = np.array([problem.objective_function(ind) for ind in population])
        convergence_history = [np.min(fitness)]

        for iteration in range(max_iterations):
            # Selection (tournament)
            selected = []
            for _ in range(population_size):
                tournament = np.random.choice(population_size, 3, replace=False)
                winner = tournament[np.argmin(fitness[tournament])]
                selected.append(population[winner])

            # Crossover
            new_population = []
            for i in range(0, population_size, 2):
                parent1, parent2 = selected[i], selected[min(i+1, population_size-1)]

                if np.random.rand() < crossover_rate:
                    crossover_point = np.random.randint(1, problem.dimension)
                    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                new_population.extend([child1, child2])

            new_population = np.array(new_population[:population_size])

            # Mutation
            for i in range(population_size):
                for d in range(problem.dimension):
                    if np.random.rand() < mutation_rate:
                        if problem.bounds:
                            low, high = problem.bounds[d]
                            new_population[i, d] = np.random.uniform(low, high)
                        else:
                            new_population[i, d] += np.random.normal(0, 0.1)

            population = new_population
            fitness = np.array([problem.objective_function(ind) for ind in population])
            convergence_history.append(np.min(fitness))

            # Check convergence
            if len(convergence_history) > 10 and abs(convergence_history[-1] - convergence_history[-5]) < tolerance:
                break

        best_idx = np.argmin(fitness)
        return SolverResult(
            solution=population[best_idx],
            objective_value=fitness[best_idx],
            convergence_history=convergence_history,
            computation_time=0.0,
            method_used="genetic_algorithm",
            iterations=iteration + 1,
            function_evaluations=population_size * (iteration + 1)
        )

    def _bayesian_optimization(self, problem: OptimizationProblem,
                             max_iterations: int, tolerance: float) -> SolverResult:
        """Bayesian Optimization using Gaussian Processes"""

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            from scipy.stats import norm

            # Initial random samples
            n_initial = min(5, max_iterations // 5)
            X_samples = []
            y_samples = []

            for _ in range(n_initial):
                if problem.bounds:
                    x = np.array([np.random.uniform(low, high) for low, high in problem.bounds])
                else:
                    x = np.random.randn(problem.dimension)
                X_samples.append(x)
                y_samples.append(problem.objective_function(x))

            X_samples = np.array(X_samples)
            y_samples = np.array(y_samples)

            convergence_history = [np.min(y_samples)]

            for iteration in range(max_iterations - n_initial):
                # Fit GP model
                kernel = Matern(nu=2.5)
                gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
                gp.fit(X_samples, y_samples)

                # Acquisition function (Expected Improvement)
                def expected_improvement(x):
                    x = x.reshape(1, -1)
                    mu, sigma = gp.predict(x, return_std=True)
                    mu = mu[0]
                    sigma = sigma[0]

                    current_best = np.min(y_samples)
                    z = (current_best - mu) / sigma if sigma > 0 else 0
                    ei = (current_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                    return -ei  # Minimize negative EI

                # Optimize acquisition function
                from scipy.optimize import minimize

                bounds = problem.bounds if problem.bounds else [(-2, 2)] * problem.dimension
                result = minimize(expected_improvement, np.random.randn(problem.dimension),
                                bounds=bounds, method='L-BFGS-B')

                next_x = result.x
                next_y = problem.objective_function(next_x)

                # Add to samples
                X_samples = np.vstack([X_samples, next_x])
                y_samples = np.append(y_samples, next_y)

                convergence_history.append(np.min(y_samples))

                # Check convergence
                if len(convergence_history) > 5 and abs(convergence_history[-1] - convergence_history[-5]) < tolerance:
                    break

            best_idx = np.argmin(y_samples)
            return SolverResult(
                solution=X_samples[best_idx],
                objective_value=y_samples[best_idx],
                convergence_history=convergence_history,
                computation_time=0.0,
                method_used="bayesian_optimization",
                iterations=iteration + n_initial,
                function_evaluations=len(y_samples)
            )

        except ImportError:
            logger.warning("Bayesian optimization requires scikit-learn")
            return self._differential_evolution(problem, max_iterations, tolerance)

    def _gradient_descent(self, problem: OptimizationProblem,
                        max_iterations: int, tolerance: float) -> SolverResult:
        """Gradient Descent optimization"""

        # Initialize
        if problem.bounds:
            x = np.array([np.random.uniform(low, high) for low, high in problem.bounds])
        else:
            x = np.random.randn(problem.dimension)

        learning_rate = 0.01
        convergence_history = []

        for iteration in range(max_iterations):
            # Compute gradient (numerical approximation)
            grad = self._numerical_gradient(problem.objective_function, x)

            # Update
            x = x - learning_rate * grad

            # Clip to bounds
            if problem.bounds:
                for d in range(problem.dimension):
                    low, high = problem.bounds[d]
                    x[d] = np.clip(x[d], low, high)

            obj_value = problem.objective_function(x)
            convergence_history.append(obj_value)

            # Adaptive learning rate
            if len(convergence_history) > 1 and convergence_history[-1] > convergence_history[-2]:
                learning_rate *= 0.5

            # Check convergence
            if len(convergence_history) > 5 and abs(convergence_history[-1] - convergence_history[-5]) < tolerance:
                break

        return SolverResult(
            solution=x,
            objective_value=convergence_history[-1],
            convergence_history=convergence_history,
            computation_time=0.0,
            method_used="gradient_descent",
            iterations=iteration + 1,
            function_evaluations=len(convergence_history)
        )

    def _newton_method(self, problem: OptimizationProblem,
                      max_iterations: int, tolerance: float) -> SolverResult:
        """Newton's Method optimization"""

        # Initialize
        if problem.bounds:
            x = np.array([np.random.uniform(low, high) for low, high in problem.bounds])
        else:
            x = np.random.randn(problem.dimension)

        convergence_history = []

        for iteration in range(max_iterations):
            # Compute gradient and Hessian (numerical approximation)
            grad = self._numerical_gradient(problem.objective_function, x)
            hessian = self._numerical_hessian(problem.objective_function, x)

            # Newton's update
            try:
                delta_x = np.linalg.solve(hessian, grad)
                x = x - delta_x

                # Clip to bounds
                if problem.bounds:
                    for d in range(problem.dimension):
                        low, high = problem.bounds[d]
                        x[d] = np.clip(x[d], low, high)

            except np.linalg.LinAlgError:
                # Hessian not invertible, fall back to gradient descent
                x = x - 0.01 * grad

            obj_value = problem.objective_function(x)
            convergence_history.append(obj_value)

            # Check convergence
            if len(convergence_history) > 1 and abs(convergence_history[-1] - convergence_history[-2]) < tolerance:
                break

        return SolverResult(
            solution=x,
            objective_value=convergence_history[-1],
            convergence_history=convergence_history,
            computation_time=0.0,
            method_used="newton_method",
            iterations=iteration + 1,
            function_evaluations=len(convergence_history)
        )

    def _numerical_gradient(self, func: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """Compute numerical gradient"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return grad

    def _numerical_hessian(self, func: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """Compute numerical Hessian"""
        n = len(x)
        hessian = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
                x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
                x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
                x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h

                hessian[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * h * h)

        return hessian

    def _select_optimal_method(self, problem: OptimizationProblem) -> str:
        """Select optimal solver method based on problem characteristics"""

        if problem.problem_type == "continuous" and problem.dimension:
            if problem.dimension <= 5:
                return "newton_method"  # Good for low-dimensional smooth problems
            elif problem.dimension <= 20:
                return "bayesian_optimization"  # Good for medium-dimensional problems
            else:
                return "particle_swarm"  # Scales better for high dimensions

        elif problem.problem_type == "discrete":
            return "genetic_algorithm"

        elif problem.problem_type == "combinatorial":
            return "simulated_annealing"

        else:
            return "differential_evolution"  # General purpose

    def _validate_problem(self, problem: OptimizationProblem):
        """Validate problem definition"""
        if not callable(problem.objective_function):
            raise ValueError("Objective function must be callable")

        if problem.dimension is None:
            # Try to infer dimension
            test_input = np.random.randn(5)
            try:
                result = problem.objective_function(test_input)
                problem.dimension = 5
            except:
                raise ValueError("Cannot infer problem dimension")

    def _validate_result(self, result: SolverResult, problem: OptimizationProblem) -> SolverResult:
        """Validate and enhance solver result"""

        # Check constraints
        if problem.constraints:
            violations = 0
            for constraint in problem.constraints:
                if not constraint(result.solution):
                    violations += 1

            result.constraint_violations = violations
            result.feasibility_score = max(0, 1.0 - violations / len(problem.constraints))

        # Compute optimality gap if known optimum exists
        if problem.global_minimum is not None:
            result.optimality_gap = abs(result.objective_value - problem.global_minimum)

        return result

    def _compute_benchmark_score(self, result: SolverResult, problem: OptimizationProblem) -> Optional[float]:
        """Compute standardized benchmark score"""
        if not hasattr(problem, 'benchmark_name') or problem.global_minimum is None:
            return None

        # Normalized score (0 = optimal, higher = worse)
        if problem.global_minimum == 0:
            score = abs(result.objective_value)
        else:
            score = abs(result.objective_value - problem.global_minimum) / abs(problem.global_minimum)

        # Penalize constraint violations
        score *= (1 + result.constraint_violations)

        # Penalize computation time (normalized)
        time_penalty = min(1.0, result.computation_time / 60.0)  # Cap at 1 minute
        score *= (1 + time_penalty)

        return score

    def _load_benchmarks(self) -> Dict[str, List[OptimizationProblem]]:
        """Load standard optimization benchmarks"""

        benchmarks = {
            'cec2017': self._create_cec2017_benchmarks(),
            'bbob': self._create_bbob_benchmarks(),
            'simple': self._create_simple_test_problems()
        }

        return benchmarks

    def _create_cec2017_benchmarks(self) -> List[OptimizationProblem]:
        """Create CEC 2017 benchmark problems"""
        problems = []

        # F1: Shifted and Rotated Bent Cigar Function
        def bent_cigar(x):
            z = x - np.array([100] * len(x))  # Shift
            return z[0]**2 + 1e6 * np.sum(z[1:]**2)

        problems.append(OptimizationProblem(
            name="cec2017_f1",
            objective_function=bent_cigar,
            bounds=[(-100, 100) for _ in range(10)],
            dimension=10,
            global_minimum=0.0,
            benchmark_name="cec2017",
            difficulty_level="hard"
        ))

        # F2: Shifted and Rotated Schwefel's Function
        def schwefel(x):
            z = x - np.array([420.9687] * len(x))
            return 418.9829 * len(x) - np.sum(z * np.sin(np.sqrt(np.abs(z))))

        problems.append(OptimizationProblem(
            name="cec2017_f2",
            objective_function=schwefel,
            bounds=[(-500, 500) for _ in range(10)],
            dimension=10,
            global_minimum=0.0,
            benchmark_name="cec2017",
            difficulty_level="hard"
        ))

        return problems

    def _create_bbob_benchmarks(self) -> List[OptimizationProblem]:
        """Create BBOB benchmark problems"""
        problems = []

        # Sphere function (separable)
        def sphere(x):
            return np.sum(x**2)

        problems.append(OptimizationProblem(
            name="bbob_sphere",
            objective_function=sphere,
            bounds=[(-5, 5) for _ in range(10)],
            dimension=10,
            global_minimum=0.0,
            benchmark_name="bbob",
            difficulty_level="easy"
        ))

        # Rosenbrock function (non-separable)
        def rosenbrock(x):
            return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

        problems.append(OptimizationProblem(
            name="bbob_rosenbrock",
            objective_function=rosenbrock,
            bounds=[(-2, 2) for _ in range(10)],
            dimension=10,
            global_minimum=0.0,
            benchmark_name="bbob",
            difficulty_level="medium"
        ))

        return problems

    def _create_simple_test_problems(self) -> List[OptimizationProblem]:
        """Create simple test problems for validation"""
        problems = []

        # Quadratic bowl
        def quadratic(x):
            return np.sum(x**2) + np.sum(x) + 10

        problems.append(OptimizationProblem(
            name="test_quadratic",
            objective_function=quadratic,
            bounds=[(-10, 10) for _ in range(5)],
            dimension=5,
            global_minimum=10.0,  # At x = [-0.5] * 5
            benchmark_name="simple",
            difficulty_level="easy"
        ))

        # Rastrigin function (multimodal)
        def rastrigin(x):
            return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

        problems.append(OptimizationProblem(
            name="test_rastrigin",
            objective_function=rastrigin,
            bounds=[(-5.12, 5.12) for _ in range(5)],
            dimension=5,
            global_minimum=0.0,
            benchmark_name="simple",
            difficulty_level="hard"
        ))

        return problems

    def get_solver_stats(self) -> Dict[str, Any]:
        """Get solver performance statistics"""
        total_runs = sum(len(results) for results in self.benchmark_results.values())

        if total_runs == 0:
            return {'total_runs': 0}

        method_performance = defaultdict(list)

        for problem_results in self.benchmark_results.values():
            for result in problem_results:
                method = result['method']
                solver_result = result['result']
                method_performance[method].append({
                    'success': solver_result.objective_value < 1e-3 if hasattr(solver_result, 'objective_value') else False,
                    'time': solver_result.computation_time,
                    'iterations': solver_result.iterations
                })

        stats = {}
        for method, performances in method_performance.items():
            success_rate = np.mean([p['success'] for p in performances])
            avg_time = np.mean([p['time'] for p in performances])
            avg_iterations = np.mean([p['iterations'] for p in performances])

            stats[method] = {
                'success_rate': success_rate,
                'avg_time': avg_time,
                'avg_iterations': avg_iterations,
                'total_runs': len(performances)
            }

        stats['total_runs'] = total_runs
        stats['best_method'] = max(stats.keys(), key=lambda k: stats[k]['success_rate'])

        return stats

# Global optimization solver
optimization_solver = OptimizationSolver()</content>
</xai:function_call">Create Advanced Learning from Limited Data (Few-Shot Learning)