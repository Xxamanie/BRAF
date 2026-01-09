#!/usr/bin/env python3
"""
BRAF Self-Evolving Algorithms
Genetic programming and evolutionary algorithms for code and strategy evolution
"""

import random
import ast
import inspect
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime
import hashlib
import json
from pathlib import Path
import copy

logger = logging.getLogger(__name__)

@dataclass
class Genome:
    """Genetic representation of a strategy or algorithm"""
    genes: List[Any] = field(default_factory=list)
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    genome_id: str = field(default_factory=lambda: hashlib.md5(str(random.random()).encode()).hexdigest()[:8])

@dataclass
class StrategyChromosome:
    """Chromosome representing a complete strategy"""
    decision_logic: Genome
    action_selection: Genome
    risk_management: Genome
    adaptation_rules: Genome
    fitness_score: float = 0.0

class GeneticOperator:
    """Genetic operators for evolution"""

    def __init__(self):
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8

    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform crossover between two genomes"""
        if random.random() > self.crossover_rate or len(parent1.genes) != len(parent2.genes):
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        # Single point crossover
        crossover_point = random.randint(1, len(parent1.genes) - 1)

        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]

        child1 = Genome(
            genes=child1_genes,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id]
        )

        child2 = Genome(
            genes=child2_genes,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id]
        )

        return child1, child2

    def mutate(self, genome: Genome) -> Genome:
        """Apply mutation to genome"""
        mutated = copy.deepcopy(genome)

        for i in range(len(mutated.genes)):
            if random.random() < self.mutation_rate:
                # Apply random mutation based on gene type
                gene = mutated.genes[i]

                if isinstance(gene, (int, float)):
                    # Numeric mutation
                    mutation_strength = random.gauss(0, 0.1)
                    mutated.genes[i] = gene + mutation_strength
                    mutated.mutation_history.append(f"Numeric mutation at {i}: {gene} -> {mutated.genes[i]}")

                elif isinstance(gene, str):
                    # String mutation - character substitution
                    if len(gene) > 0:
                        pos = random.randint(0, len(gene) - 1)
                        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                        new_char = random.choice(chars)
                        mutated.genes[i] = gene[:pos] + new_char + gene[pos+1:]
                        mutated.mutation_history.append(f"String mutation at {i}: {gene} -> {mutated.genes[i]}")

                elif isinstance(gene, list):
                    # List mutation - add/remove element
                    if random.random() < 0.5 and len(gene) > 0:
                        # Remove random element
                        remove_idx = random.randint(0, len(gene) - 1)
                        removed = gene.pop(remove_idx)
                        mutated.mutation_history.append(f"List removal at {i}: removed {removed}")
                    else:
                        # Add random element
                        new_element = self._generate_random_gene()
                        gene.append(new_element)
                        mutated.mutation_history.append(f"List addition at {i}: added {new_element}")

        mutated.generation += 1
        return mutated

    def _generate_random_gene(self) -> Any:
        """Generate a random gene of appropriate type"""
        gene_types = [int, float, str, list]

        gene_type = random.choice(gene_types)

        if gene_type == int:
            return random.randint(-100, 100)
        elif gene_type == float:
            return random.uniform(-10.0, 10.0)
        elif gene_type == str:
            length = random.randint(1, 10)
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
        elif gene_type == list:
            return [random.randint(0, 10) for _ in range(random.randint(1, 5))]

class CodeEvolutionEngine:
    """Evolutionary engine for code generation and optimization"""

    def __init__(self):
        self.genetic_ops = GeneticOperator()
        self.code_templates = self._load_code_templates()
        self.fitness_functions = {}

    def evolve_code_snippet(self, base_code: str, fitness_function: Callable,
                          generations: int = 50, population_size: int = 20) -> str:
        """Evolve a code snippet using genetic programming"""
        # Parse base code into AST
        try:
            tree = ast.parse(base_code)
        except SyntaxError:
            logger.error("Invalid base code syntax")
            return base_code

        # Create initial population
        population = self._create_initial_population(tree, population_size)

        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                try:
                    code = self._genome_to_code(individual)
                    fitness = fitness_function(code)
                    individual.fitness_score = fitness
                    fitness_scores.append(fitness)
                except Exception as e:
                    individual.fitness_score = 0.0
                    fitness_scores.append(0.0)

            # Select parents
            parents = self._tournament_selection(population, k=5)

            # Create next generation
            next_population = []

            while len(next_population) < population_size:
                if len(parents) >= 2:
                    parent1, parent2 = random.sample(parents, 2)
                    child1, child2 = self.genetic_ops.crossover(parent1, parent2)

                    child1 = self.genetic_ops.mutate(child1)
                    child2 = self.genetic_ops.mutate(child2)

                    next_population.extend([child1, child2])

            population = next_population[:population_size]

            # Log progress
            best_fitness = max(fitness_scores) if fitness_scores else 0
            logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}")

        # Return best individual
        best_individual = max(population, key=lambda x: x.fitness_score)
        return self._genome_to_code(best_individual)

    def evolve_strategy(self, strategy_template: Dict[str, Any],
                       fitness_evaluator: Callable) -> StrategyChromosome:
        """Evolve a complete strategy using genetic algorithms"""
        # Create initial population of strategies
        population_size = 20
        population = []

        for _ in range(population_size):
            strategy = StrategyChromosome(
                decision_logic=self._create_random_genome('decision_logic'),
                action_selection=self._create_random_genome('action_selection'),
                risk_management=self._create_random_genome('risk_management'),
                adaptation_rules=self._create_random_genome('adaptation_rules')
            )
            population.append(strategy)

        generations = 30

        for generation in range(generations):
            # Evaluate fitness for each strategy
            for strategy in population:
                strategy.fitness_score = fitness_evaluator(strategy)

            # Sort by fitness
            population.sort(key=lambda x: x.fitness_score, reverse=True)

            # Keep top performers
            elite_count = 4
            elites = population[:elite_count]

            # Create next generation
            next_population = elites.copy()

            while len(next_population) < population_size:
                # Select parents from elites
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)

                # Crossover
                child = self._crossover_strategies(parent1, parent2)

                # Mutate
                child = self._mutate_strategy(child)

                next_population.append(child)

            population = next_population

            best_fitness = population[0].fitness_score
            logger.info(f"Strategy Evolution Generation {generation}: Best fitness = {best_fitness:.4f}")

        return population[0]

    def _create_initial_population(self, base_tree: ast.AST, size: int) -> List[Genome]:
        """Create initial population from base AST"""
        population = []

        for _ in range(size):
            # Create genome representing AST modifications
            genes = self._ast_to_genes(base_tree)
            genome = Genome(genes=genes)
            population.append(genome)

        return population

    def _ast_to_genes(self, tree: ast.AST) -> List[Any]:
        """Convert AST to genetic representation"""
        genes = []

        # Simple representation - extract constants and names
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                genes.append(node.value)
            elif isinstance(node, ast.Name):
                genes.append(node.id)
            elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
                genes.append(node.n)

        return genes[:50]  # Limit gene count

    def _genome_to_code(self, genome: Genome) -> str:
        """Convert genome back to code"""
        # This is a simplified reconstruction
        # In practice, would need more sophisticated AST reconstruction

        # Start with template
        template = self.code_templates.get('decision_function', '''
def decision_function(context):
    score = 0.0
    if 'login' in str(context):
        score += 1.0
    if len(str(context)) > 100:
        score += 0.5
    return score > 1.0
''')

        # Apply genetic modifications (simplified)
        code_lines = template.split('\n')
        for gene in genome.genes[:5]:  # Use first few genes
            if isinstance(gene, (int, float)) and len(code_lines) > 2:
                # Modify a numeric constant
                line_idx = random.randint(1, len(code_lines) - 2)
                if 'score' in code_lines[line_idx]:
                    code_lines[line_idx] = code_lines[line_idx].replace('1.0', str(gene))

        return '\n'.join(code_lines)

    def _tournament_selection(self, population: List[Genome], k: int) -> List[Genome]:
        """Tournament selection"""
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(population, min(k, len(population)))
            winner = max(tournament, key=lambda x: x.fitness_score)
            selected.append(winner)
        return selected

    def _load_code_templates(self) -> Dict[str, str]:
        """Load code templates for evolution"""
        return {
            'decision_function': '''
def decision_function(context):
    score = 0.0
    if 'login' in str(context):
        score += 1.0
    if len(str(context)) > 100:
        score += 0.5
    return score > 1.0
''',
            'action_selector': '''
def select_action(state, actions):
    if state.get('urgent', False):
        return actions[0]  # First action for urgent
    return random.choice(actions)
''',
            'risk_calculator': '''
def calculate_risk(factors):
    risk = 0.0
    for factor in factors:
        if factor > 0.7:
            risk += 0.3
    return min(risk, 1.0)
'''
        }

    def _create_random_genome(self, genome_type: str) -> Genome:
        """Create a random genome for given type"""
        if genome_type == 'decision_logic':
            genes = [random.uniform(0, 2) for _ in range(10)]  # Thresholds and weights
        elif genome_type == 'action_selection':
            genes = [random.choice(['random', 'greedy', 'epsilon_greedy']) for _ in range(5)]
        elif genome_type == 'risk_management':
            genes = [random.uniform(0, 1) for _ in range(8)]  # Risk thresholds
        elif genome_type == 'adaptation_rules':
            genes = [random.randint(0, 10) for _ in range(6)]  # Adaptation parameters
        else:
            genes = [random.random() for _ in range(10)]

        return Genome(genes=genes)

    def _crossover_strategies(self, parent1: StrategyChromosome,
                            parent2: StrategyChromosome) -> StrategyChromosome:
        """Crossover between strategy chromosomes"""
        child = StrategyChromosome(
            decision_logic=self.genetic_ops.crossover(parent1.decision_logic, parent2.decision_logic)[0],
            action_selection=self.genetic_ops.crossover(parent1.action_selection, parent2.action_selection)[0],
            risk_management=self.genetic_ops.crossover(parent1.risk_management, parent2.risk_management)[0],
            adaptation_rules=self.genetic_ops.crossover(parent1.adaptation_rules, parent2.adaptation_rules)[0]
        )
        return child

    def _mutate_strategy(self, strategy: StrategyChromosome) -> StrategyChromosome:
        """Apply mutation to strategy"""
        mutated = copy.deepcopy(strategy)
        mutated.decision_logic = self.genetic_ops.mutate(mutated.decision_logic)
        mutated.action_selection = self.genetic_ops.mutate(mutated.action_selection)
        mutated.risk_management = self.genetic_ops.mutate(mutated.risk_management)
        mutated.adaptation_rules = self.genetic_ops.mutate(mutated.adaptation_rules)
        return mutated

class SelfEvolvingSystem:
    """Main self-evolving system that continuously improves BRAF"""

    def __init__(self):
        self.code_evolver = CodeEvolutionEngine()
        self.active_strategies = {}
        self.evolution_history = []
        self.performance_threshold = 0.8  # Trigger evolution when performance drops

    async def continuous_evolution(self):
        """Run continuous evolution loop"""
        while True:
            try:
                # Monitor performance
                current_performance = await self._evaluate_system_performance()

                # Check if evolution is needed
                if current_performance['overall_score'] < self.performance_threshold:
                    logger.info("Performance below threshold, triggering evolution...")

                    # Evolve decision making
                    new_decision_code = self.code_evolver.evolve_code_snippet(
                        self._get_current_decision_code(),
                        self._evaluate_decision_fitness
                    )

                    # Evolve strategy
                    new_strategy = self.code_evolver.evolve_strategy(
                        self._get_current_strategy_template(),
                        self._evaluate_strategy_fitness
                    )

                    # Deploy improvements
                    await self._deploy_evolved_code(new_decision_code, new_strategy)

                    self.evolution_history.append({
                        'timestamp': datetime.now(),
                        'performance_before': current_performance,
                        'improvements': ['decision_logic', 'strategy']
                    })

                # Wait before next evolution cycle
                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Evolution cycle failed: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes

    async def _evaluate_system_performance(self) -> Dict[str, Any]:
        """Evaluate overall system performance"""
        # This would integrate with monitoring systems
        # Simplified evaluation
        return {
            'overall_score': random.uniform(0.5, 0.9),  # Placeholder
            'success_rate': random.uniform(0.7, 0.95),
            'adaptation_speed': random.uniform(0.6, 0.9),
            'risk_level': random.uniform(0.1, 0.4)
        }

    def _get_current_decision_code(self) -> str:
        """Get current decision making code"""
        return '''
def decision_function(context):
    score = 0.0
    if 'login' in str(context):
        score += 1.0
    if len(str(context)) > 100:
        score += 0.5
    return score > 1.0
'''

    def _get_current_strategy_template(self) -> Dict[str, Any]:
        """Get current strategy template"""
        return {
            'decision_logic': {'type': 'threshold_based'},
            'action_selection': {'type': 'epsilon_greedy'},
            'risk_management': {'type': 'dynamic_threshold'},
            'adaptation_rules': {'type': 'performance_based'}
        }

    def _evaluate_decision_fitness(self, code: str) -> float:
        """Evaluate fitness of evolved decision code"""
        try:
            # Execute code and test on sample data
            local_vars = {}
            exec(code, {}, local_vars)

            decision_func = local_vars.get('decision_function')
            if not decision_func:
                return 0.0

            # Test on sample contexts
            test_cases = [
                ({'url': 'login.example.com'}, True),
                ({'url': 'dashboard.example.com'}, True),
                ({'url': 'api.example.com/data'}, False),
                ({'url': 'home.example.com'}, False)
            ]

            correct = 0
            for context, expected in test_cases:
                try:
                    result = decision_func(context)
                    if result == expected:
                        correct += 1
                except:
                    pass

            return correct / len(test_cases)

        except Exception as e:
            logger.warning(f"Decision fitness evaluation failed: {e}")
            return 0.0

    def _evaluate_strategy_fitness(self, strategy: StrategyChromosome) -> float:
        """Evaluate fitness of evolved strategy"""
        # Simplified strategy evaluation
        fitness_components = []

        # Decision logic fitness
        dl_fitness = len(strategy.decision_logic.genes) / 20.0  # Prefer more complex logic
        fitness_components.append(dl_fitness)

        # Action selection fitness
        as_fitness = strategy.action_selection.fitness_score if hasattr(strategy.action_selection, 'fitness_score') else 0.5
        fitness_components.append(as_fitness)

        # Risk management fitness
        rm_fitness = 1.0 - sum(strategy.risk_management.genes) / len(strategy.risk_management.genes)  # Lower risk thresholds preferred
        fitness_components.append(rm_fitness)

        return sum(fitness_components) / len(fitness_components)

    async def _deploy_evolved_code(self, decision_code: str, strategy: StrategyChromosome):
        """Deploy evolved code and strategies"""
        try:
            # Save evolved decision code
            with open('src/braf/ai/evolved_decisions.py', 'w') as f:
                f.write(decision_code)

            # Update active strategies
            self.active_strategies['decision_making'] = decision_code
            self.active_strategies['strategy'] = strategy

            logger.info("Evolved code and strategies deployed successfully")

        except Exception as e:
            logger.error(f"Failed to deploy evolved code: {e}")

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get evolution statistics"""
        return {
            'total_evolution_cycles': len(self.evolution_history),
            'last_evolution': self.evolution_history[-1] if self.evolution_history else None,
            'active_improvements': list(self.active_strategies.keys()),
            'evolution_trends': self._analyze_evolution_trends()
        }

    def _analyze_evolution_trends(self) -> Dict[str, Any]:
        """Analyze evolution improvement trends"""
        if len(self.evolution_history) < 2:
            return {'trend': 'insufficient_data'}

        recent_performances = [h['performance_before']['overall_score'] for h in self.evolution_history[-10:]]
        trend = 'improving' if recent_performances[-1] > recent_performances[0] else 'declining'

        return {
            'direction': trend,
            'magnitude': abs(recent_performances[-1] - recent_performances[0]),
            'consistency': len([p for p in recent_performances if p > 0.8]) / len(recent_performances)
        }

# Global instances
evolution_engine = CodeEvolutionEngine()
self_evolving_system = SelfEvolvingSystem()