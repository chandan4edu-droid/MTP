"""Experiment runner for evaluating the hybrid code generation system."""

import logging
import time
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from .datasets.dataset_loaders import load_dataset, Problem
from .orchestrator import HybridCodeGenerator
from .models.data_models import ExperimentConfig, ProblemResult
from .evaluation.evaluation_reporter import EvaluationReporter
from .llm.mistral_llm import MistralLLM


class ExperimentRunner:
    """
    Runs experiments on code generation benchmarks with support for
    ablation studies and comprehensive logging.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize LLM
        self.llm = MistralLLM(
            api_token=config.llm_api_key if hasattr(config, 'llm_api_key') else None,
            model_name=config.llm_model if hasattr(config, 'llm_model') else None
        )
        
        self.orchestrator = HybridCodeGenerator(self.llm, config)
        self.reporter = EvaluationReporter(config)
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("ExperimentRunner")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            output_dir / f"experiment_{self.config.dataset}_{self.config.ablation_mode}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def run_experiment(
        self,
        dataset_path: Optional[str] = None,
        max_problems: Optional[int] = None
    ) -> List[ProblemResult]:
        """
        Run the experiment on the specified dataset.
        
        Args:
            dataset_path: Optional custom path to dataset file
            max_problems: Optional limit on number of problems to process
            
        Returns:
            List of ProblemResult objects
        """
        self.logger.info(f"Starting experiment: {self.config.dataset} - {self.config.ablation_mode}")
        self.logger.info(f"Configuration: {self.config}")
        
        # Load dataset
        try:
            problems = load_dataset(self.config.dataset, dataset_path, interactive=getattr(self.config, 'interactive', False))
            self.logger.info(f"Loaded {len(problems)} problems from {self.config.dataset}")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
        
        # Limit number of problems if specified
        if max_problems is not None:
            problems = problems[:max_problems]
            self.logger.info(f"Limited to {max_problems} problems")
        
        # Process each problem
        results = []
 <<-----------------------------------------i added by me----------------------------------------------------------------------------------------------->
        for i, problem in enumerate(tqdm(problems, desc="Processing problems")):
            try:
                self.logger.info(f"Processing problem {i+1}/{len(problems)}: {problem.task_id}")
 <-------------------------------------------------------------------------------------------------------------------------------------------------------------------->       
                # Solve the problem
                result = self._solve_problem(problem)
                results.append(result)
                
                # Log the result
                self.reporter.log_problem_result(result)
                self.logger.info(
                    f"Completed {problem.task_id}: "
                    f"score={result.final_solution.composite_score:.3f}, "
                    f"llm_calls={result.total_llm_calls}, "
                    f"time={result.execution_time_seconds:.1f}s"
                )
                
            except Exception as e:
                self.logger.error(f"Error processing {problem.task_id}: {e}", exc_info=True)
                # Continue processing remaining problems
                continue
        
        self.logger.info(f"Experiment completed: {len(results)}/{len(problems)} problems processed")
        
        # Generate report
        self._generate_report(results)
        
        return results
    
    def _solve_problem(self, problem: Problem) -> ProblemResult:
        """
        Solve a single problem using the orchestrator.
        
        Args:
            problem: Problem to solve
            
        Returns:
            ProblemResult with solution and metrics
        """
        # Convert problem to test cases
        test_cases = problem.to_test_cases()
        
        # Solve using orchestrator
        result = self.orchestrator.solve_problem(
            problem_id=problem.task_id,
            problem=problem.prompt,
            test_cases=test_cases
        )
        
        # Add problem metadata
        result.problem_id = problem.task_id
        result.problem_description = problem.prompt
        
        return result
    
    def _generate_report(self, results: List[ProblemResult]) -> None:
        """
        Generate and save experiment report.
        
        Args:
            results: List of problem results
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate main report
        report_path = output_dir / f"report_{self.config.dataset}_{self.config.ablation_mode}.json"
        self.reporter.generate_report(str(report_path))
        self.logger.info(f"Report saved to {report_path}")
        
        # Compute and log aggregate metrics
        metrics = self.reporter.compute_aggregate_metrics()
        self.logger.info("=" * 80)
        self.logger.info("EXPERIMENT RESULTS")
        self.logger.info("=" * 80)
        self.logger.info(f"Dataset: {self.config.dataset}")
        self.logger.info(f"Ablation Mode: {self.config.ablation_mode}")
        self.logger.info(f"Total Problems: {metrics.total_problems}")
        self.logger.info(f"Accuracy: {metrics.accuracy:.2%}")
        self.logger.info(f"Average LLM Calls: {metrics.avg_llm_calls:.2f}")
        self.logger.info(f"Early Exit Rate: {metrics.early_exit_rate:.2%}")
        self.logger.info(f"Evolution Success Rate: {metrics.evolution_success_rate:.2%}")
        self.logger.info(f"Mitigation Success Rate: {metrics.mitigation_success_rate:.2%}")
        self.logger.info("Call Distribution:")
        for calls, rate in sorted(metrics.call_distribution.items()):
            self.logger.info(f"  {calls} calls: {rate:.2%}")
        self.logger.info("=" * 80)


def run_ablation_study(
    base_config: ExperimentConfig,
    dataset_path: Optional[str] = None,
    max_problems: Optional[int] = None
) -> None:
    """
    Run ablation study with all four configurations.
    
    Args:
        base_config: Base configuration to use
        dataset_path: Optional custom path to dataset file
        max_problems: Optional limit on number of problems to process
    """
    ablation_modes = ["baseline", "multi_strategy", "with_evolution", "full_system"]
    all_results = []
    
    for mode in ablation_modes:
        print(f"\n{'='*80}")
        print(f"Running ablation mode: {mode}")
        print(f"{'='*80}\n")
        
        # Create config for this mode
        config = ExperimentConfig(
            dataset=base_config.dataset,
            ablation_mode=mode,
            score_threshold=base_config.score_threshold,
            max_mitigation_iterations=base_config.max_mitigation_iterations,
            llm_model=base_config.llm_model,
            llm_api_key=base_config.llm_api_key,
            output_dir=base_config.output_dir,
            log_level=base_config.log_level
        )
        
        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run_experiment(dataset_path, max_problems)
        all_results.append((mode, results))
    
    # Generate comparison report
    print(f"\n{'='*80}")
    print("Generating ablation study comparison report")
    print(f"{'='*80}\n")
    
    reporter = EvaluationReporter(base_config)
    for mode, results in all_results:
        for result in results:
            reporter.log_problem_result(result)
    
    output_dir = Path(base_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / f"ablation_study_{base_config.dataset}.json"
    reporter.export_ablation_study(all_results, str(comparison_path))
    print(f"Ablation study report saved to {comparison_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run code generation experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HumanEval",
        choices=["HumanEval", "MBPP"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--ablation-mode",
        type=str,
        default="full_system",
        choices=["baseline", "multi_strategy", "with_evolution", "full_system"],
        help="Ablation mode"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Custom path to dataset file"
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Maximum number of problems to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--run-ablation-study",
        action="store_true",
        help="Run full ablation study with all configurations"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode to create custom problems"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = ExperimentConfig(
        dataset=args.dataset,
        ablation_mode=args.ablation_mode,
        output_dir=args.output_dir,
        log_level=args.log_level
    )
    
    # Add interactive flag to config
    config.interactive = args.interactive
    
    if args.run_ablation_study:
        # Run ablation study
        run_ablation_study(config, args.dataset_path, args.max_problems)
    else:
        # Run single experiment
        runner = ExperimentRunner(config)
        runner.run_experiment(args.dataset_path, args.max_problems)
