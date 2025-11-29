"""
Main Orchestrator for the Hybrid Code Generation System.

This module coordinates all five phases of the pipeline:
1. Multi-Strategy Generation (4 LLM calls)
2. Detection and Scoring (0 LLM calls)
3. Early Exit Check (score ≥ 0.95)
4. Strategy Evolution (2 LLM calls if no early exit)
5. Adaptive Mitigation (0-2 LLM calls if needed)
"""

import logging
import time
from typing import List, Optional

from .models.data_models import (
    TestCase, ProblemResult, EvaluationResult, ExperimentConfig,
    HallucinationType
)
from .strategy_generator.strategy_generator import StrategyGenerator
from .code_generator.code_generator import CodeGenerator
from .detection_engine.detection_engine import DetectionEngine
from .evolution_engine.strategy_evolution_engine import StrategyEvolutionEngine
from .mitigation_engine.mitigation_engine import MitigationEngine
from .llm.base_interface import BaseLLMInterface

logger = logging.getLogger(__name__)


class HybridCodeGenerator:
    """
    Main orchestrator for the hybrid code generation system.
    
    Coordinates all five phases of the pipeline with strict LLM call tracking
    and adaptive execution based on solution quality.
    """
    
    def __init__(
        self,
        llm: BaseLLMInterface,
        config: ExperimentConfig
    ):
        """
        Initialize the hybrid code generator with all components.
        
        Args:
            llm: LLM interface for all generation tasks
            config: Experiment configuration
        """
        self.config = config
        self.llm = llm
        
        # Initialize all components
        self.strategy_generator = StrategyGenerator(llm)
        self.code_generator = CodeGenerator(llm)
        self.detection_engine = DetectionEngine(config)
        self.evolution_engine = StrategyEvolutionEngine(llm)
        self.mitigation_engine = MitigationEngine(llm, self.detection_engine, config)
        
        logger.info(
            f"HybridCodeGenerator initialized with config: "
            f"ablation_mode={config.ablation_mode}, "
            f"score_threshold={config.score_threshold}, "
            f"max_mitigation_iterations={config.max_mitigation_iterations}"
        )
    
    def solve_problem(
        self,
        problem_id: str,
        problem: str,
        test_cases: List[TestCase]
    ) -> ProblemResult:
        """
        Solve a programming problem using the five-phase pipeline.
        
        This is the main entry point that orchestrates:
        - Phase 1: Multi-strategy generation (4 LLM calls)
        - Phase 2: Detection and scoring (0 LLM calls)
        - Phase 3: Early exit check (score ≥ 0.95)
        - Phase 4: Strategy evolution (2 LLM calls if no early exit)
        - Phase 5: Adaptive mitigation (0-2 LLM calls if needed)
        
        Args:
            problem_id: Unique identifier for the problem
            problem: Problem description
            test_cases: List of test cases for evaluation
            
        Returns:
            ProblemResult with final solution and execution metrics
        """
        logger.info("🚀" + "="*79)
        logger.info(f"🚀 STARTING PROBLEM: {problem_id}")
        logger.info("🚀" + "="*79)
        logger.info(f"PROBLEM DESCRIPTION:")
        logger.info(problem)
        logger.info(f"TEST CASES: {len(test_cases)} total")
        for i, tc in enumerate(test_cases[:3], 1):
            logger.info(f"  Test {i}: Input={tc.input} → Expected={tc.expected_output}")
        if len(test_cases) > 3:
            logger.info(f"  ... and {len(test_cases) - 3} more test cases")
        logger.info("🚀" + "="*79)
        
        start_time = time.time()
        
        # Initialize LLM call counter
        llm_calls = 0
        
        # Track solutions at each phase
        initial_evaluations: List[EvaluationResult] = []
        evolved_evaluation: Optional[EvaluationResult] = None
        early_exit = False
        mitigation_iterations = 0
        
        try:
            # ===== PHASE 1: Multi-Strategy Generation (4 LLM calls) =====
            logger.info("📋" + "="*79)
            logger.info("📋 PHASE 1: Multi-Strategy Generation (4 LLM calls)")
            logger.info("📋" + "="*79)
            
            # Generate 3 diverse strategies (1 LLM call)
            strategies = self.strategy_generator.generate_strategies(problem, test_cases)
            llm_calls += 1
            logger.info(f"Generated {len(strategies)} strategies (LLM calls: {llm_calls})")
            
            # Generate code for each strategy (3 LLM calls)
            solutions = self.code_generator.generate_all_codes(strategies, problem, test_cases)
            llm_calls += 3
            logger.info(
                f"Generated {len(solutions)} code implementations (LLM calls: {llm_calls})"
            )
            
            # ===== PHASE 2: Detection and Scoring (0 LLM calls) =====
            logger.info("🔍" + "="*79)
            logger.info("🔍 PHASE 2: Detection and Scoring (0 LLM calls)")
            logger.info("🔍" + "="*79)
            
            initial_evaluations = []
            for i, solution in enumerate(solutions):
                evaluation = self.detection_engine.evaluate_solution(solution, test_cases)
                initial_evaluations.append(evaluation)
                logger.debug(
                    f"Solution {i+1} evaluated: "
                    f"score={evaluation.composite_score:.3f}, "
                    f"test_pass_rate={evaluation.detection_results.test_pass_rate:.1%}, "
                    f"hallucination={evaluation.detection_results.hallucination_type.value}"
                )
            
            # Sort by test pass rate first, then composite score (prioritize working solutions)
            initial_evaluations.sort(
                key=lambda e: (e.detection_results.test_pass_rate, e.composite_score), 
                reverse=True
            )
            best_initial = initial_evaluations[0]
            
            logger.info(
                f"Best initial solution: score={best_initial.composite_score:.3f}, "
                f"strategy='{best_initial.solution.strategy.name}'"
            )
            
            # ===== PHASE 3: Early Exit Check =====
            logger.info("Phase 3: Early Exit Check")
            
            # Early exit requires BOTH good score AND all tests passing
            score_meets_threshold = best_initial.composite_score >= self.config.score_threshold
            all_tests_pass = best_initial.detection_results.test_pass_rate == 1.0
            
            if score_meets_threshold and all_tests_pass:
                early_exit = True
                final_solution = best_initial
                
                logger.info(
                    f"Early exit triggered! Score {best_initial.composite_score:.3f} "
                    f">= threshold {self.config.score_threshold:.3f} AND all tests pass "
                    f"({best_initial.detection_results.test_pass_rate:.1%})"
                )
                logger.info(f"Total LLM calls: {llm_calls}")
                
                # Log early exit event
                self._log_early_exit(problem_id, best_initial, llm_calls)
                
            else:
                # Log specific reason for no early exit
                if not score_meets_threshold:
                    logger.info(
                        f"No early exit: score {best_initial.composite_score:.3f} "
                        f"< threshold {self.config.score_threshold:.3f}"
                    )
                elif not all_tests_pass:
                    logger.info(
                        f"No early exit: test pass rate {best_initial.detection_results.test_pass_rate:.1%} "
                        f"< 100% (score {best_initial.composite_score:.3f} meets threshold)"
                    )
                
                # ===== PHASE 4: Strategy Evolution (2 LLM calls) =====
                logger.info("Phase 4: Strategy Evolution")
                
                # Analyze top 2 solutions
                analysis = self.evolution_engine.analyze_solutions(initial_evaluations)
                
                # Generate evolved strategy (1 LLM call)
                evolved_strategy = self.evolution_engine.generate_evolved_strategy(
                    analysis, problem, test_cases
                )
                llm_calls += 1
                logger.info(
                    f"Generated evolved strategy: '{evolved_strategy.name}' "
                    f"(LLM calls: {llm_calls})"
                )
                
                # Generate evolved code (1 LLM call)
                evolved_solution = self.evolution_engine.generate_evolved_code(
                    evolved_strategy, problem, test_cases
                )
                llm_calls += 1
                logger.info(f"Generated evolved code (LLM calls: {llm_calls})")
                
                # Evaluate evolved solution
                evolved_evaluation = self.detection_engine.evaluate_solution(
                    evolved_solution, test_cases
                )
                logger.info(
                    f"Evolved solution evaluated: score={evolved_evaluation.composite_score:.3f}"
                )
                
                # Combine all candidates (initial + evolved)
                all_candidates = initial_evaluations + [evolved_evaluation]
                
                # Select best solution from all candidates (prioritize test-passing solutions)
                best_solution_eval = max(
                    all_candidates, 
                    key=lambda e: (e.detection_results.test_pass_rate, e.composite_score)
                )
                
                logger.info(
                    f"Best solution after evolution: "
                    f"score={best_solution_eval.composite_score:.3f}, "
                    f"strategy='{best_solution_eval.solution.strategy.name}'"
                )
                
                # ===== PHASE 5: Adaptive Mitigation (0-2 LLM calls) =====
                logger.info("Phase 5: Adaptive Mitigation")
                
                # Check if mitigation is needed
                needs_mitigation = (
                    best_solution_eval.detection_results.hallucination_type != HallucinationType.NO_HALLUCINATION
                    or best_solution_eval.detection_results.test_pass_rate < 1.0
                )
                
                if needs_mitigation:
                    logger.info(
                        f"Mitigation needed: "
                        f"hallucination={best_solution_eval.detection_results.hallucination_type.value}, "
                        f"test_pass_rate={best_solution_eval.detection_results.test_pass_rate:.1%}"
                    )
                    
                    # Apply iterative mitigation (0-2 LLM calls)
                    mitigated_solution, mitigation_iterations = self.mitigation_engine.iterative_mitigation(
                        best_solution_eval.solution,
                        problem,
                        test_cases,
                        max_iterations=self.config.max_mitigation_iterations
                    )
                    llm_calls += mitigation_iterations
                    
                    logger.info(
                        f"Mitigation complete: {mitigation_iterations} iterations used "
                        f"(LLM calls: {llm_calls})"
                    )
                    
                    # Evaluate final solution
                    final_solution = self.detection_engine.evaluate_solution(
                        mitigated_solution, test_cases
                    )
                    
                    logger.info(
                        f"Final solution after mitigation: score={final_solution.composite_score:.3f}"
                    )
                    
                else:
                    logger.info("No mitigation needed - solution is already high quality")
                    final_solution = best_solution_eval
                    mitigation_iterations = 0
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Enforce maximum LLM call limit
            if llm_calls > 8:
                logger.warning(
                    f"LLM call limit exceeded: {llm_calls} > 8. "
                    f"This should not happen in normal operation."
                )
            
            # Create problem result
            result = ProblemResult(
                problem_id=problem_id,
                problem_description=problem,
                initial_solutions=initial_evaluations,
                early_exit=early_exit,
                evolved_solution=evolved_evaluation,
                final_solution=final_solution,
                mitigation_iterations=mitigation_iterations,
                total_llm_calls=llm_calls,
                execution_time_seconds=execution_time
            )
            
            logger.info(
                f"Problem {problem_id} complete: "
                f"score={final_solution.composite_score:.3f}, "
                f"llm_calls={llm_calls}, "
                f"time={execution_time:.1f}s, "
                f"early_exit={early_exit}"
            )
            
            # Log LLM call distribution
            self._log_llm_call_distribution(llm_calls)
            
            return result
            
        except Exception as e:
            logger.error(f"Error solving problem {problem_id}: {str(e)}", exc_info=True)
            raise
    
    def _log_early_exit(
        self,
        problem_id: str,
        solution: EvaluationResult,
        llm_calls: int
    ) -> None:
        """
        Log early exit event for performance analysis.
        
        Args:
            problem_id: Problem identifier
            solution: Solution that triggered early exit
            llm_calls: Number of LLM calls used (should be 4)
        """
        logger.info(
            f"EARLY_EXIT: problem={problem_id}, "
            f"score={solution.composite_score:.3f}, "
            f"strategy='{solution.solution.strategy.name}', "
            f"llm_calls={llm_calls}, "
            f"test_pass_rate={solution.detection_results.test_pass_rate:.1%}, "
            f"hallucination={solution.detection_results.hallucination_type.value}"
        )
    
    def _log_llm_call_distribution(self, llm_calls: int) -> None:
        """
        Log LLM call count for distribution analysis.
        
        Expected distribution:
        - 4 calls: Early exit (~15% of problems)
        - 6 calls: Evolution without mitigation (~60% of remaining)
        - 7 calls: Evolution + 1 mitigation (~30% of remaining)
        - 8 calls: Evolution + 2 mitigations (~10% of remaining)
        
        Args:
            llm_calls: Number of LLM calls used for this problem
        """
        logger.info(f"LLM_CALL_COUNT: {llm_calls}")
        
        # Validate expected call counts
        if llm_calls not in [4, 6, 7, 8]:
            logger.warning(
                f"Unexpected LLM call count: {llm_calls}. "
                f"Expected one of [4, 6, 7, 8]"
            )
