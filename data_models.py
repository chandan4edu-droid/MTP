"""Core data models for the Hybrid Code Generation System."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple


class HallucinationType(Enum):
    """Types of code hallucinations that can be detected."""
    NO_HALLUCINATION = "no_hallucination"
    DEAD_CODE = "dead_code"
    SYNTACTIC_INCORRECTNESS = "syntactic_incorrectness"
    LOGICAL_ERROR = "logical_error"
    ROBUSTNESS_ISSUE = "robustness_issue"
    SECURITY_VULNERABILITY = "security_vulnerability"


@dataclass
class TestCase:
    """Represents a single test case with input and expected output."""
    input: str
    expected_output: str


@dataclass
class Strategy:
    """Represents a high-level algorithmic strategy."""
    name: str
    description: str
    approach_type: str


@dataclass
class Solution:
    """Represents a complete solution with strategy and code."""
    strategy: Strategy
    code: str
    generation_prompt: Optional[str] = None  # Prompt that generated this code


@dataclass
class Vulnerability:
    """Represents a security vulnerability detected in code."""
    type: str
    line_number: int
    description: str
    severity: str = "MEDIUM"
    confidence: str = "MEDIUM"


@dataclass
class FailedTest:
    """Represents a failed test case with details."""
    input: str
    expected: str
    actual: str


@dataclass
class DetectionResults:
    """Results from all detection mechanisms."""
    hallucination_type: HallucinationType
    dead_code_score: float
    security_score: float
    vulnerabilities: List[Vulnerability]
    test_pass_rate: float
    failed_tests: List[FailedTest]


@dataclass
class EvaluationResult:
    """Complete evaluation of a solution including detection and scoring."""
    solution: Solution
    detection_results: DetectionResults
    composite_score: float


@dataclass
class AnalysisReport:
    """Analysis of top solutions for strategy evolution."""
    top_solutions: List[Tuple[Solution, EvaluationResult]]
    best_approach_types: List[str]
    common_failures: List[FailedTest]
    strengths: List[str]
    weaknesses: List[str]


@dataclass
class ProblemResult:
    """Complete results for a single problem."""
    problem_id: str
    problem_description: str
    initial_solutions: List[EvaluationResult]
    early_exit: bool
    evolved_solution: Optional[EvaluationResult]
    final_solution: EvaluationResult
    mitigation_iterations: int
    total_llm_calls: int
    execution_time_seconds: float


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all problems."""
    total_problems: int
    accuracy: float
    avg_llm_calls: float
    call_distribution: Dict[int, float]  # {4: 0.15, 6: 0.60, 7: 0.30, 8: 0.10}
    early_exit_rate: float
    evolution_success_rate: float
    mitigation_success_rate: float
    hallucination_rates: Dict[str, float]  # {hallucination_type: rate}


@dataclass
class ExperimentConfig:
    """Configuration for experiments and pipeline execution."""
    # Dataset configuration
    dataset: str = "HumanEval"  # "HumanEval" or "MBPP"
    
    # Ablation mode
    ablation_mode: str = "full_system"  # "baseline", "multi_strategy", "with_evolution", "full_system"
    
    # Pipeline thresholds
    score_threshold: float = 0.45  # Early exit threshold
    max_mitigation_iterations: int = 2
    
    # LLM configuration
    llm_model: str = "mistral-medium"
    llm_api_key: Optional[str] = None
    llm_max_retries: int = 3
    llm_retry_delays: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    
    # Detection configuration
    codebert_model_path: Optional[str] = None
    test_timeout_seconds: int = 5
    memory_limit_mb: int = 512
    cpu_time_limit_seconds: int = 5
    
    # Scoring weights
    test_weight: float = 0.5
    quality_weight: float = 0.3
    security_weight: float = 0.2
    
    # Output configuration
    output_dir: str = "results"
    log_level: str = "INFO"
