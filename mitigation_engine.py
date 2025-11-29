"""
Mitigation Engine for applying targeted fixes to solutions with detected issues.

This module implements adaptive fallback mitigation that:
1. Generates specific detection feedback based on hallucination type
2. Applies iterative refinement (max 2 iterations)
3. Terminates early when issues are resolved
4. Only invokes when the best solution has issues
"""

import logging
from typing import List, Tuple
from hybrid_system.models.data_models import (
    Solution, TestCase, EvaluationResult, DetectionResults,
    HallucinationType, Strategy, ExperimentConfig
)
from hybrid_system.llm.base_interface import BaseLLMInterface
from hybrid_system.detection_engine.detection_engine import DetectionEngine

logger = logging.getLogger(__name__)


class MitigationEngine:
    """
    Mitigation engine that applies targeted fixes based on detection feedback.
    
    The engine:
    - Generates specific feedback for each hallucination type
    - Applies iterative refinement using LLM calls
    - Terminates early when hallucinations are resolved and test pass rate = 1.0
    - Limits to maximum 2 iterations
    """
    
    def __init__(
        self,
        llm_interface: BaseLLMInterface,
        detection_engine: DetectionEngine,
        config: ExperimentConfig
    ):
        """
        Initialize the mitigation engine.
        
        Args:
            llm_interface: LLM interface for generating fixes
            detection_engine: Detection engine for re-evaluating solutions
            config: Experiment configuration
        """
        self.llm = llm_interface
        self.detection_engine = detection_engine
        self.config = config
        
        logger.info("MitigationEngine initialized")
    
    def generate_detection_feedback(self, evaluation: EvaluationResult) -> str:
        """
        Generate specific, actionable feedback based on hallucination type.
        
        This method creates targeted feedback for each hallucination type:
        - dead_code: Line numbers and variable names from static analysis
        - syntactic_incorrectness: Syntax error messages with line numbers
        - logical_error: Failed test inputs, expected outputs, actual outputs
        - robustness_issue: Edge case descriptions and failure modes
        - security_vulnerability: Vulnerability types and affected lines
        
        Args:
            evaluation: Evaluation result with detection results
            
        Returns:
            Formatted feedback string for the LLM
        """
        detection = evaluation.detection_results
        hallucination_type = detection.hallucination_type
        
        logger.debug(f"Generating feedback for hallucination type: {hallucination_type.value}")
        
        if hallucination_type == HallucinationType.NO_HALLUCINATION:
            # No hallucination, but might have test failures
            if detection.test_pass_rate < 1.0:
                return self._generate_test_failure_feedback(detection)
            else:
                return "No specific issues detected."
        
        elif hallucination_type == HallucinationType.DEAD_CODE:
            return self._generate_dead_code_feedback(detection)
        
        elif hallucination_type == HallucinationType.SYNTACTIC_INCORRECTNESS:
            return self._generate_syntax_error_feedback(detection)
        
        elif hallucination_type == HallucinationType.LOGICAL_ERROR:
            return self._generate_logical_error_feedback(detection)
        
        elif hallucination_type == HallucinationType.ROBUSTNESS_ISSUE:
            return self._generate_robustness_feedback(detection)
        
        elif hallucination_type == HallucinationType.SECURITY_VULNERABILITY:
            return self._generate_security_feedback(detection)
        
        else:
            return f"Unknown hallucination type: {hallucination_type.value}"
    
    def _generate_dead_code_feedback(self, detection: DetectionResults) -> str:
        """Generate feedback for dead code issues."""
        feedback_parts = ["DEAD CODE DETECTED:\n"]
        
        # Add dead code score
        feedback_parts.append(f"Dead code score: {detection.dead_code_score:.2f} (higher = more dead code)\n")
        
        # Add general guidance
        feedback_parts.append("\nThe code contains unused or unreachable segments:")
        feedback_parts.append("- Remove unused variables and functions")
        feedback_parts.append("- Eliminate unreachable code blocks")
        feedback_parts.append("- Clean up redundant imports")
        feedback_parts.append("- Remove empty code blocks\n")
        
        # Add test failure info if present
        if detection.failed_tests:
            feedback_parts.append(f"\nAdditionally, {len(detection.failed_tests)} test(s) are failing.")
            feedback_parts.append("Ensure the cleaned code still passes all tests.")
        
        return "\n".join(feedback_parts)
    
    def _generate_syntax_error_feedback(self, detection: DetectionResults) -> str:
        """Generate feedback for syntax errors."""
        feedback_parts = ["SYNTAX ERRORS DETECTED:\n"]
        
        # Check if we have failed tests that might contain syntax error info
        if detection.failed_tests:
            feedback_parts.append("The code contains syntax errors that prevent execution:")
            for i, failed_test in enumerate(detection.failed_tests[:3], 1):  # Show first 3
                if "SyntaxError" in failed_test.actual or "IndentationError" in failed_test.actual:
                    feedback_parts.append(f"\nError {i}: {failed_test.actual}")
        else:
            feedback_parts.append("The code contains syntax errors. Common issues:")
            feedback_parts.append("- Missing colons after function/class definitions")
            feedback_parts.append("- Improper indentation")
            feedback_parts.append("- Unclosed brackets, parentheses, or quotes")
            feedback_parts.append("- Invalid Python statements")
        
        feedback_parts.append("\nFix all syntax errors to make the code executable.")
        
        return "\n".join(feedback_parts)
    
    def _generate_logical_error_feedback(self, detection: DetectionResults) -> str:
        """Generate feedback for logical errors."""
        feedback_parts = ["LOGICAL ERRORS DETECTED:\n"]
        
        feedback_parts.append(f"Test pass rate: {detection.test_pass_rate:.1%}")
        feedback_parts.append(f"Failed tests: {len(detection.failed_tests)}\n")
        
        if detection.failed_tests:
            feedback_parts.append("The code executes but produces incorrect outputs:\n")
            
            # Show details for up to 5 failed tests
            for i, failed_test in enumerate(detection.failed_tests[:5], 1):
                feedback_parts.append(f"Failed Test {i}:")
                feedback_parts.append(f"  Input: {failed_test.input}")
                feedback_parts.append(f"  Expected: {failed_test.expected}")
                feedback_parts.append(f"  Actual: {failed_test.actual}")
                feedback_parts.append("")
            
            if len(detection.failed_tests) > 5:
                feedback_parts.append(f"... and {len(detection.failed_tests) - 5} more failed tests")
        
        feedback_parts.append("Review the algorithm logic and fix the incorrect computations.")
        
        return "\n".join(feedback_parts)
    
    def _generate_robustness_feedback(self, detection: DetectionResults) -> str:
        """Generate feedback for robustness issues."""
        feedback_parts = ["ROBUSTNESS ISSUES DETECTED:\n"]
        
        feedback_parts.append("The code fails under edge cases or unexpected inputs.\n")
        
        if detection.failed_tests:
            feedback_parts.append(f"Failed {len(detection.failed_tests)} test(s) due to:")
            
            # Analyze failure patterns
            has_exceptions = any("Error" in ft.actual or "Exception" in ft.actual 
                               for ft in detection.failed_tests)
            has_none = any("None" in ft.actual for ft in detection.failed_tests)
            
            if has_exceptions:
                feedback_parts.append("- Unhandled exceptions (add try-except blocks)")
            if has_none:
                feedback_parts.append("- Unexpected None values (add null checks)")
            
            feedback_parts.append("\nFailed test examples:")
            for i, failed_test in enumerate(detection.failed_tests[:3], 1):
                feedback_parts.append(f"\nTest {i}:")
                feedback_parts.append(f"  Input: {failed_test.input}")
                feedback_parts.append(f"  Error: {failed_test.actual}")
        
        feedback_parts.append("\nAdd proper error handling and input validation:")
        feedback_parts.append("- Handle edge cases (empty inputs, None values, boundary conditions)")
        feedback_parts.append("- Add try-except blocks for potential exceptions")
        feedback_parts.append("- Validate inputs before processing")
        
        return "\n".join(feedback_parts)
    
    def _generate_security_feedback(self, detection: DetectionResults) -> str:
        """Generate feedback for security vulnerabilities."""
        feedback_parts = ["SECURITY VULNERABILITIES DETECTED:\n"]
        
        feedback_parts.append(f"Security score: {detection.security_score:.2f}")
        feedback_parts.append(f"Vulnerabilities found: {len(detection.vulnerabilities)}\n")
        
        if detection.vulnerabilities:
            feedback_parts.append("Specific vulnerabilities:\n")
            
            for i, vuln in enumerate(detection.vulnerabilities, 1):
                feedback_parts.append(f"{i}. {vuln.type} (Severity: {vuln.severity})")
                feedback_parts.append(f"   Line {vuln.line_number}: {vuln.description}")
                feedback_parts.append("")
        
        feedback_parts.append("Fix these security issues:")
        feedback_parts.append("- Sanitize all user inputs")
        feedback_parts.append("- Avoid using eval() or exec()")
        feedback_parts.append("- Use parameterized queries for database operations")
        feedback_parts.append("- Validate and escape shell commands")
        
        return "\n".join(feedback_parts)
    
    def _generate_test_failure_feedback(self, detection: DetectionResults) -> str:
        """Generate feedback when there are test failures but no hallucination."""
        feedback_parts = ["TEST FAILURES DETECTED:\n"]
        
        feedback_parts.append(f"Test pass rate: {detection.test_pass_rate:.1%}")
        feedback_parts.append(f"Failed tests: {len(detection.failed_tests)}\n")
        
        if detection.failed_tests:
            feedback_parts.append("Failed test details:\n")
            
            for i, failed_test in enumerate(detection.failed_tests[:5], 1):
                feedback_parts.append(f"Test {i}:")
                feedback_parts.append(f"  Input: {failed_test.input}")
                feedback_parts.append(f"  Expected: {failed_test.expected}")
                feedback_parts.append(f"  Actual: {failed_test.actual}")
                feedback_parts.append("")
        
        feedback_parts.append("Fix the code to pass all test cases.")
        
        return "\n".join(feedback_parts)
    
    def apply_mitigation(
        self,
        solution: Solution,
        evaluation: EvaluationResult,
        problem: str,
        test_cases: List[TestCase]
    ) -> Solution:
        """
        Apply one mitigation iteration using 1 LLM call.
        
        This method:
        1. Generates detection feedback
        2. Creates a mitigation prompt with current code and feedback
        3. Calls LLM to generate fixed code
        4. Parses and returns the new solution
        
        Args:
            solution: Current solution to fix
            evaluation: Evaluation results with detection feedback
            problem: Original problem description
            test_cases: Test cases for context
            
        Returns:
            New solution with attempted fixes
        """
        logger.info("Applying mitigation iteration")
        
        # Generate specific feedback
        feedback = self.generate_detection_feedback(evaluation)
        
        # Create mitigation prompt
        prompt = self._create_mitigation_prompt(
            solution.code,
            evaluation.detection_results.hallucination_type,
            feedback,
            problem,
            test_cases
        )
        
        # LOG PROMPT  
        logger.info("="*80)
        logger.info("MITIGATION ENGINE - LLM CALL")
        logger.info("="*80)
        logger.info("CURRENT CODE BEING FIXED:")
        logger.info("-" * 40)
        logger.info(solution.code)
        logger.info("-" * 40)
        logger.info(f"DETECTED ISSUE: {evaluation.detection_results.hallucination_type.value}")
        logger.info(f"FEEDBACK: {feedback}")
        logger.info("PROMPT SENT TO LLM:")
        logger.info("-" * 40)
        logger.info(prompt)
        logger.info("-" * 40)
        
        # Call LLM to generate fixed code
        logger.debug("Calling LLM for mitigation")
        response = self.llm.generate(prompt, max_tokens=2048, temperature=0.7)
        
        # LOG RESPONSE
        logger.info("RESPONSE RECEIVED FROM LLM:")
        logger.info("-" * 40)
        logger.info(response)
        logger.info("-" * 40)
        
        # Parse the fixed code
        fixed_code = self._parse_code_from_response(response)
        
        # LOG FIXED CODE
        logger.info("FIXED CODE:")
        logger.info("-" * 40)
        logger.info(fixed_code)
        logger.info("-" * 40)
        
        # Create new solution with same strategy but fixed code
        fixed_solution = Solution(
            strategy=solution.strategy,
            code=fixed_code,
            generation_prompt=prompt  # Store the mitigation prompt
        )
        
        logger.info("MITIGATION SOLUTION CREATED:")
        logger.info(f"  Strategy: {fixed_solution.strategy.name}")
        logger.info(f"  Fixed code length: {len(fixed_solution.code)} chars")
        logger.info("="*80)
        
        logger.info("Mitigation iteration complete")
        
        return fixed_solution
    
    def _create_mitigation_prompt(
        self,
        current_code: str,
        hallucination_type: HallucinationType,
        feedback: str,
        problem: str,
        test_cases: List[TestCase]
    ) -> str:
        """
        Create a prompt for the LLM to fix the code.
        
        Args:
            current_code: Current code with issues
            hallucination_type: Type of hallucination detected
            feedback: Specific feedback about the issues
            problem: Original problem description
            test_cases: Test cases for context
            
        Returns:
            Formatted prompt for the LLM
        """
        # Format test cases
        test_cases_str = "\n".join([
            f"Input: {tc.input}\nExpected Output: {tc.expected_output}"
            for tc in test_cases[:3]  # Show first 3 test cases
        ])
        
        prompt = f"""The following code has been detected to have issues:

CURRENT CODE:
```python
{current_code}
```

DETECTED ISSUE: {hallucination_type.value}

SPECIFIC FEEDBACK:
{feedback}

ORIGINAL PROBLEM:
{problem}

TEST CASES (examples):
{test_cases_str}

Please fix the code to address these specific issues. Requirements:
1. Fix all identified problems
2. Ensure the code passes all test cases
3. Maintain the same function signature and interface
4. Keep the code clean and efficient

Provide ONLY the corrected Python code without additional explanation.
"""
        
        return prompt
    
    def _parse_code_from_response(self, response: str) -> str:
        """
        Parse Python code from LLM response.
        
        Handles:
        - Code wrapped in markdown code blocks
        - Code with or without language specifier
        - Plain code without markdown
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted Python code
        """
        # Try to extract code from markdown blocks
        if "```python" in response:
            # Extract code between ```python and ```
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        elif "```" in response:
            # Extract code between ``` and ```
            start = response.find("```") + len("```")
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # If no markdown blocks, return the whole response
        return response.strip()
    
    def iterative_mitigation(
        self,
        solution: Solution,
        problem: str,
        test_cases: List[TestCase],
        max_iterations: int = 2
    ) -> Tuple[Solution, int]:
        """
        Apply iterative mitigation with early termination.
        
        This method:
        1. Applies up to max_iterations mitigation iterations
        2. Re-evaluates after each iteration
        3. Terminates early if hallucinations resolved and test pass rate = 1.0
        4. Returns the best solution found
        
        Args:
            solution: Initial solution to improve
            problem: Original problem description
            test_cases: Test cases for evaluation
            max_iterations: Maximum number of iterations (default: 2)
            
        Returns:
            Tuple of (best_solution, iterations_used)
        """
        logger.info(f"Starting iterative mitigation (max {max_iterations} iterations)")
        
        current_solution = solution
        best_solution = solution
        best_evaluation = self.detection_engine.evaluate_solution(solution, test_cases)
        best_score = best_evaluation.composite_score
        
        iterations_used = 0
        
        for iteration in range(max_iterations):
            logger.info(f"Mitigation iteration {iteration + 1}/{max_iterations}")
            
            # Evaluate current solution
            current_evaluation = self.detection_engine.evaluate_solution(
                current_solution, test_cases
            )
            
            # Check for early termination
            if (current_evaluation.detection_results.hallucination_type == HallucinationType.NO_HALLUCINATION
                and current_evaluation.detection_results.test_pass_rate == 1.0):
                logger.info(
                    f"Early termination: hallucinations resolved and all tests pass "
                    f"(score: {current_evaluation.composite_score:.3f})"
                )
                return current_solution, iterations_used
            
            # Apply mitigation
            try:
                mitigated_solution = self.apply_mitigation(
                    current_solution,
                    current_evaluation,
                    problem,
                    test_cases
                )
                iterations_used += 1
                
                # Evaluate mitigated solution
                mitigated_evaluation = self.detection_engine.evaluate_solution(
                    mitigated_solution, test_cases
                )
                
                logger.info(
                    f"Iteration {iteration + 1} complete: "
                    f"score {current_evaluation.composite_score:.3f} -> "
                    f"{mitigated_evaluation.composite_score:.3f}"
                )
                
                # Update best solution if improved
                if mitigated_evaluation.composite_score > best_score:
                    best_solution = mitigated_solution
                    best_evaluation = mitigated_evaluation
                    best_score = mitigated_evaluation.composite_score
                    logger.debug(f"New best solution found: score={best_score:.3f}")
                
                # Update current solution for next iteration
                current_solution = mitigated_solution
                
                # Check for early termination after mitigation
                if (mitigated_evaluation.detection_results.hallucination_type == HallucinationType.NO_HALLUCINATION
                    and mitigated_evaluation.detection_results.test_pass_rate == 1.0):
                    logger.info(
                        f"Early termination after iteration {iteration + 1}: "
                        f"hallucinations resolved and all tests pass "
                        f"(score: {mitigated_evaluation.composite_score:.3f})"
                    )
                    return mitigated_solution, iterations_used
                
            except Exception as e:
                logger.error(f"Mitigation iteration {iteration + 1} failed: {str(e)}")
                # Continue with current solution
                continue
        
        logger.info(
            f"Mitigation complete: {iterations_used} iterations used, "
            f"best score: {best_score:.3f}"
        )
        
        return best_solution, iterations_used
