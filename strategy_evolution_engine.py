"""
Strategy Evolution Engine for combining successful strategies.

This module analyzes top-performing solutions and creates evolved hybrid strategies
that combine strengths while avoiding identified weaknesses.
"""

import logging
from typing import List, Tuple

from ..llm.base_interface import BaseLLMInterface
from ..models.data_models import (
    Solution, Strategy, EvaluationResult, AnalysisReport,
    TestCase, HallucinationType
)

logger = logging.getLogger(__name__)


class StrategyEvolutionEngine:
    """
    Analyzes top solutions and generates evolved hybrid strategies.
    
    This engine implements detection-guided strategy evolution by:
    1. Analyzing top 2 solutions to identify strengths and weaknesses
    2. Generating evolved strategies that combine successful approaches
    3. Implementing evolved strategies as executable code
    """
    
    def __init__(self, llm: BaseLLMInterface):
        """
        Initialize the Strategy Evolution Engine.
        
        Args:
            llm: LLM interface for generating evolved strategies
        """
        self.llm = llm
        logger.info("Initialized StrategyEvolutionEngine")
    
    def analyze_solutions(
        self,
        evaluations: List[EvaluationResult]
    ) -> AnalysisReport:
        """
        Analyze solutions to identify top 2 and extract insights.
        
        This method:
        1. Identifies top 2 solutions by composite score
        2. Extracts approach types that pass most tests
        3. Identifies specific hallucination types from each solution
        4. Finds common failed test cases across solutions
        5. Determines unique strengths of each approach
        
        Args:
            evaluations: List of evaluation results to analyze
            
        Returns:
            AnalysisReport with insights for strategy evolution
        """
        logger.info(f"Analyzing {len(evaluations)} solutions for evolution")
        
        # Sort by composite score and select top 2
        sorted_evals = sorted(
            evaluations,
            key=lambda e: e.composite_score,
            reverse=True
        )
        top_2 = sorted_evals[:2]
        
        logger.debug(
            f"Top 2 solutions: "
            f"1) {top_2[0].solution.strategy.name} (score={top_2[0].composite_score:.3f}), "
            f"2) {top_2[1].solution.strategy.name} (score={top_2[1].composite_score:.3f})"
        )
        
        # Extract approach types that pass most tests
        best_approach_types = self._identify_best_approaches(top_2)
        
        # Find common failed test cases
        common_failures = self._find_common_failures(top_2)
        
        # Identify strengths and weaknesses
        strengths = self._identify_strengths(top_2)
        weaknesses = self._identify_weaknesses(top_2)
        
        report = AnalysisReport(
            top_solutions=[(e.solution, e) for e in top_2],
            best_approach_types=best_approach_types,
            common_failures=common_failures,
            strengths=strengths,
            weaknesses=weaknesses
        )
        
        logger.info(
            f"Analysis complete: "
            f"best_approaches={best_approach_types}, "
            f"common_failures={len(common_failures)}, "
            f"strengths={len(strengths)}, "
            f"weaknesses={len(weaknesses)}"
        )
        
        return report
    
    def _identify_best_approaches(
        self,
        top_solutions: List[EvaluationResult]
    ) -> List[str]:
        """
        Identify approach types that pass the most tests.
        
        Args:
            top_solutions: Top 2 evaluation results
            
        Returns:
            List of approach types ordered by test pass rate
        """
        # Create list of (approach_type, test_pass_rate) tuples
        approaches = [
            (
                eval_result.solution.strategy.approach_type,
                eval_result.detection_results.test_pass_rate
            )
            for eval_result in top_solutions
        ]
        
        # Sort by test pass rate (descending)
        approaches.sort(key=lambda x: x[1], reverse=True)
        
        # Return approach types
        approach_types = [approach for approach, _ in approaches]
        
        logger.debug(f"Best approach types: {approach_types}")
        return approach_types
    
    def _find_common_failures(
        self,
        top_solutions: List[EvaluationResult]
    ) -> List:
        """
        Find test cases that failed in multiple solutions.
        
        Args:
            top_solutions: Top 2 evaluation results
            
        Returns:
            List of FailedTest objects that appear in multiple solutions
        """
        if len(top_solutions) < 2:
            return []
        
        # Get failed tests from each solution
        failed_1 = top_solutions[0].detection_results.failed_tests
        failed_2 = top_solutions[1].detection_results.failed_tests
        
        # Find common failures by comparing inputs
        common = []
        for test1 in failed_1:
            for test2 in failed_2:
                if test1.input == test2.input:
                    # This test failed in both solutions
                    common.append(test1)
                    break
        
        logger.debug(f"Found {len(common)} common failed tests")
        return common
    
    def _identify_strengths(
        self,
        top_solutions: List[EvaluationResult]
    ) -> List[str]:
        """
        Identify unique strengths of each approach.
        
        Strengths include:
        - High test pass rate
        - No hallucinations
        - Good security score
        - Low dead code
        - Specific algorithmic advantages
        
        Args:
            top_solutions: Top 2 evaluation results
            
        Returns:
            List of strength descriptions
        """
        strengths = []
        
        for i, eval_result in enumerate(top_solutions):
            solution_num = i + 1
            strategy = eval_result.solution.strategy
            detection = eval_result.detection_results
            
            # High test pass rate
            if detection.test_pass_rate >= 0.8:
                strengths.append(
                    f"Solution {solution_num} ({strategy.name}): "
                    f"High test pass rate ({detection.test_pass_rate:.1%})"
                )
            
            # No hallucinations
            if detection.hallucination_type == HallucinationType.NO_HALLUCINATION:
                strengths.append(
                    f"Solution {solution_num} ({strategy.name}): "
                    f"No hallucinations detected"
                )
            
            # Good security
            if detection.security_score >= 0.9:
                strengths.append(
                    f"Solution {solution_num} ({strategy.name}): "
                    f"Strong security ({detection.security_score:.1%})"
                )
            
            # Low dead code
            if detection.dead_code_score <= 0.2:
                strengths.append(
                    f"Solution {solution_num} ({strategy.name}): "
                    f"Minimal dead code"
                )
            
            # Approach-specific strengths
            approach_type = strategy.approach_type.lower()
            if 'recursive' in approach_type and detection.test_pass_rate > 0.5:
                strengths.append(
                    f"Solution {solution_num} ({strategy.name}): "
                    f"Effective recursive approach"
                )
            elif 'iterative' in approach_type and detection.test_pass_rate > 0.5:
                strengths.append(
                    f"Solution {solution_num} ({strategy.name}): "
                    f"Efficient iterative approach"
                )
            elif 'data' in approach_type and detection.test_pass_rate > 0.5:
                strengths.append(
                    f"Solution {solution_num} ({strategy.name}): "
                    f"Optimized data structure usage"
                )
        
        logger.debug(f"Identified {len(strengths)} strengths")
        return strengths
    
    def _identify_weaknesses(
        self,
        top_solutions: List[EvaluationResult]
    ) -> List[str]:
        """
        Identify weaknesses to avoid in evolved strategy.
        
        Weaknesses include:
        - Specific hallucination types
        - Low test pass rate
        - Security vulnerabilities
        - High dead code
        
        Args:
            top_solutions: Top 2 evaluation results
            
        Returns:
            List of weakness descriptions
        """
        weaknesses = []
        
        for i, eval_result in enumerate(top_solutions):
            solution_num = i + 1
            strategy = eval_result.solution.strategy
            detection = eval_result.detection_results
            
            # Hallucination types
            if detection.hallucination_type != HallucinationType.NO_HALLUCINATION:
                weaknesses.append(
                    f"Solution {solution_num} ({strategy.name}): "
                    f"{detection.hallucination_type.value}"
                )
            
            # Low test pass rate
            if detection.test_pass_rate < 0.8:
                weaknesses.append(
                    f"Solution {solution_num} ({strategy.name}): "
                    f"Low test pass rate ({detection.test_pass_rate:.1%})"
                )
            
            # Security issues
            if detection.security_score < 0.8:
                weaknesses.append(
                    f"Solution {solution_num} ({strategy.name}): "
                    f"Security concerns ({len(detection.vulnerabilities)} vulnerabilities)"
                )
            
            # Dead code
            if detection.dead_code_score > 0.3:
                weaknesses.append(
                    f"Solution {solution_num} ({strategy.name}): "
                    f"Significant dead code (score={detection.dead_code_score:.2f})"
                )
        
        logger.debug(f"Identified {len(weaknesses)} weaknesses")
        return weaknesses
    
    def generate_evolved_strategy(
        self,
        analysis: AnalysisReport,
        problem: str,
        test_cases: List[TestCase]
    ) -> Strategy:
        """
        Generate evolved hybrid strategy based on analysis.
        
        Uses 1 LLM call to create a strategy that:
        - Combines strengths of top approaches
        - Avoids identified weaknesses
        - Addresses common failure patterns
        
        Args:
            analysis: Analysis report from analyze_solutions()
            problem: Original problem description
            test_cases: Test cases for the problem
            
        Returns:
            Evolved Strategy object
            
        Raises:
            Exception: If strategy generation fails
        """
        logger.info("Generating evolved hybrid strategy")
        
        try:
            # Create prompt for evolved strategy
            prompt = self._create_evolution_prompt(analysis, problem, test_cases)
            
            # LOG PROMPT
            logger.info("="*80)
            logger.info("STRATEGY EVOLUTION - LLM CALL 5/8")
            logger.info("="*80)
            logger.info("PROMPT SENT TO LLM:")
            logger.info("-" * 40)
            logger.info(prompt)
            logger.info("-" * 40)
            
            # LOG EVOLUTION PROMPT
            logger.info("EVOLUTION PROMPT SENT TO LLM:")
            logger.info("=" * 80)
            logger.info(prompt)
            logger.info("=" * 80)
            
            # Generate evolved strategy (1 LLM call)
            response = self.llm.generate(prompt, max_tokens=1024, temperature=0.7)
            
            # LOG RESPONSE
            logger.info("RESPONSE RECEIVED FROM LLM:")
            logger.info("-" * 40)
            logger.info(response)
            logger.info("-" * 40)
            
            # Parse strategy from response
            strategy = self._parse_evolved_strategy(response)
            
            # LOG PARSED STRATEGY
            logger.info("PARSED EVOLVED STRATEGY:")
            logger.info(f"  Name: {strategy.name}")
            logger.info(f"  Approach: {strategy.approach_type}")
            logger.info(f"  Description: {strategy.description}")
            logger.info("="*80)
            
            logger.info(f"Successfully generated evolved strategy: {strategy.name}")
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to generate evolved strategy: {str(e)}")
            raise
    
    def _create_evolution_prompt(
        self,
        analysis: AnalysisReport,
        problem: str,
        test_cases: List[TestCase]
    ) -> str:
        """
        Create prompt for generating evolved strategy.
        
        Args:
            analysis: Analysis report with insights
            problem: Problem description
            test_cases: Test cases
            
        Returns:
            Formatted prompt string
        """
        # Extract information from top 2 solutions
        sol1, eval1 = analysis.top_solutions[0]
        sol2, eval2 = analysis.top_solutions[1]
        
        # Format test cases (first 3)
        test_cases_str = "\n".join([
            f"Input: {tc.input}\nExpected: {tc.expected_output}"
            for tc in test_cases[:3]
        ])
        
        # Format common failures
        if analysis.common_failures:
            failures_str = "\n".join([
                f"Input: {f.input}, Expected: {f.expected}, Got: {f.actual}"
                for f in analysis.common_failures[:3]
            ])
        else:
            failures_str = "None"
        
        # Format strengths
        strengths_str = "\n".join([f"- {s}" for s in analysis.strengths])
        
        # Format weaknesses
        weaknesses_str = "\n".join([f"- {w}" for w in analysis.weaknesses])
        
        prompt = f"""Given the following programming problem:

{problem}

Test cases:
{test_cases_str}

Two solutions have been generated with the following results:

Solution 1:
Strategy: {sol1.strategy.name}
Approach Type: {sol1.strategy.approach_type}
Description: {sol1.strategy.description}
Test Pass Rate: {eval1.detection_results.test_pass_rate:.1%}
Hallucination Type: {eval1.detection_results.hallucination_type.value}
Composite Score: {eval1.composite_score:.3f}

Solution 2:
Strategy: {sol2.strategy.name}
Approach Type: {sol2.strategy.approach_type}
Description: {sol2.strategy.description}
Test Pass Rate: {eval2.detection_results.test_pass_rate:.1%}
Hallucination Type: {eval2.detection_results.hallucination_type.value}
Composite Score: {eval2.composite_score:.3f}

Common Failed Tests:
{failures_str}

Identified Strengths:
{strengths_str}

Identified Weaknesses:
{weaknesses_str}

Create an evolved hybrid strategy that:
1. Combines the strengths of both approaches
2. Avoids the identified weaknesses
3. Addresses the common failure patterns
4. Uses insights from the best-performing approach types: {', '.join(analysis.best_approach_types)}

Provide your response in the following format:
STRATEGY NAME: [concise name for the hybrid strategy]
APPROACH TYPE: [hybrid approach type]
DESCRIPTION: [detailed description of how the hybrid strategy works, explaining how it combines strengths and avoids weaknesses]
"""
        return prompt
    
    def _parse_evolved_strategy(self, response: str) -> Strategy:
        """
        Parse evolved strategy from LLM response.
        
        Expected format:
        STRATEGY NAME: [name]
        APPROACH TYPE: [type]
        DESCRIPTION: [description]
        
        Args:
            response: LLM response text
            
        Returns:
            Strategy object
            
        Raises:
            ValueError: If response cannot be parsed
        """
        lines = response.strip().split('\n')
        
        # LOG PARSING DETAILS
        logger.info("PARSING EVOLVED STRATEGY RESPONSE:")
        logger.info(f"Total lines to parse: {len(lines)}")
        for i, line in enumerate(lines[:20]):  # Log first 20 lines
            logger.info(f"Line {i}: '{line}'")
        if len(lines) > 20:
            logger.info(f"... and {len(lines) - 20} more lines")
        
        name = None
        approach_type = None
        description_lines = []
        in_description = False
        
        for line in lines:
            line = line.strip()
            
            # Handle different formats: "STRATEGY NAME:", "**STRATEGY NAME:**", etc.
            if 'STRATEGY NAME' in line.upper():
                # Find the actual content after the marker
                line_upper = line.upper()
                start_pos = line_upper.find('STRATEGY NAME') + len('STRATEGY NAME')
                content = line[start_pos:].lstrip(':*').strip()
                # Remove markdown formatting
                content = content.replace('**', '').strip()
                logger.info(f"Found STRATEGY NAME line: '{line}' -> extracted: '{content}'")
                if content:
                    name = content
                    logger.info(f"Set name to: '{name}'")
            elif 'APPROACH TYPE' in line.upper():
                line_upper = line.upper()
                start_pos = line_upper.find('APPROACH TYPE') + len('APPROACH TYPE')
                content = line[start_pos:].lstrip(':*').strip()
                content = content.replace('**', '').strip()
                logger.info(f"Found APPROACH TYPE line: '{line}' -> extracted: '{content}'")
                if content:
                    approach_type = content
                    logger.info(f"Set approach_type to: '{approach_type}'")
            elif 'DESCRIPTION' in line.upper():
                line_upper = line.upper()
                start_pos = line_upper.find('DESCRIPTION') + len('DESCRIPTION')
                content = line[start_pos:].lstrip(':*').strip()
                content = content.replace('**', '').strip()
                logger.info(f"Found DESCRIPTION line: '{line}' -> extracted: '{content}'")
                if content:
                    description_lines.append(content)
                    logger.info(f"Added to description: '{content}'")
                in_description = True
                logger.info("Set in_description = True")
            elif in_description and line and not line.startswith('```'):
                # Skip lines that look like headers or section breaks
                if not any(marker in line.upper() for marker in ['STRATEGY NAME', 'APPROACH TYPE', 'DESCRIPTION']):
                    description_lines.append(line)
        
        # Log what we extracted
        logger.info("PRIMARY PARSING RESULTS:")
        logger.info(f"  Name: '{name}'")
        logger.info(f"  Approach Type: '{approach_type}'")
        logger.info(f"  Description Lines: {len(description_lines)} lines")
        for i, line in enumerate(description_lines[:5]):  # Log first 5 description lines
            logger.info(f"    Desc[{i}]: '{line}'")
        
        # If we couldn't extract fields with the main parser, try fallback parsing
        if not name or not approach_type or not description_lines:
            logger.warning("Primary parsing failed, trying fallback methods...")
            name, approach_type, description_lines = self._fallback_strategy_parsing(response)
        
        # Validate required fields
        if not name:
            logger.error(f"Could not extract strategy name from response: {response[:500]}...")
            raise ValueError("Could not extract strategy name from response")
        if not approach_type:
            logger.error(f"Could not extract approach type from response: {response[:500]}...")
            raise ValueError("Could not extract approach type from response")
        if not description_lines:
            logger.error(f"Could not extract description from response: {response[:500]}...")
            raise ValueError("Could not extract description from response")
        
        description = ' '.join(description_lines)
        
        # Debug logging
        logger.debug(f"Parsed strategy - Name: '{name}', Type: '{approach_type}', Desc: '{description[:100]}...'")
        
        strategy = Strategy(
            name=name,
            description=description,
            approach_type=approach_type
        )
        
        logger.debug(
            f"Parsed evolved strategy: name='{name}', "
            f"approach='{approach_type}', "
            f"description_len={len(description)}"
        )
        
        return strategy
    
    def _fallback_strategy_parsing(self, response: str) -> tuple:
        """
        Fallback parsing method for when standard parsing fails.
        Uses more aggressive pattern matching and heuristics.
        
        Args:
            response: LLM response text
            
        Returns:
            Tuple of (name, approach_type, description_lines)
        """
        import re
        
        name = None
        approach_type = None
        description_lines = []
        
        # Try to find patterns like "Name: ...", "**Name:**", etc.
        name_patterns = [
            r'(?i)(?:strategy\s+)?name\s*[:\*]*\s*([^\n]+)',
            r'(?i)\*\*([^*]+)\*\*.*(?:strategy|approach)',
            r'(?i)^([^:\n]+)(?:\s*strategy|\s*approach)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, response)
            if match:
                candidate = match.group(1).strip()
                # Clean up the candidate
                candidate = re.sub(r'[*:]', '', candidate).strip()
                if len(candidate) > 3 and len(candidate) < 100:  # Reasonable length
                    name = candidate
                    break
        
        # Try to find approach type patterns
        approach_patterns = [
            r'(?i)approach\s+type\s*[:\*]*\s*([^\n]+)',
            r'(?i)type\s*[:\*]*\s*([^\n]+)',
            r'(?i)\b(iterative|recursive|divide_and_conquer|data_structure_optimized|hybrid|greedy|mathematical)\b',
        ]
        
        for pattern in approach_patterns:
            match = re.search(pattern, response)
            if match:
                candidate = match.group(1).strip()
                candidate = re.sub(r'[*:]', '', candidate).strip()
                if candidate:
                    approach_type = candidate
                    break
        
        # If still no approach type, try to infer from common terms
        if not approach_type:
            if 'hybrid' in response.lower():
                approach_type = 'hybrid'
            elif 'iterative' in response.lower():
                approach_type = 'iterative'
            elif 'recursive' in response.lower():
                approach_type = 'recursive'
            else:
                approach_type = 'hybrid'  # Default fallback
        
        # Extract description - take everything after description marker or use full response
        desc_match = re.search(r'(?i)description\s*[:\*]*\s*(.+)', response, re.DOTALL)
        if desc_match:
            desc_text = desc_match.group(1).strip()
            # Clean up and split into lines
            desc_text = re.sub(r'\*+', '', desc_text)  # Remove markdown
            description_lines = [line.strip() for line in desc_text.split('\n') if line.strip()]
        else:
            # Use the whole response as description if no clear structure
            clean_response = re.sub(r'\*+', '', response)
            description_lines = [line.strip() for line in clean_response.split('\n') if line.strip()]
        
        # Provide defaults if still missing
        if not name:
            name = "Evolved Hybrid Strategy"
        if not approach_type:
            approach_type = "hybrid"
        if not description_lines:
            description_lines = ["Hybrid strategy combining multiple approaches"]
        
        logger.info(f"Fallback parsing extracted - Name: '{name}', Type: '{approach_type}'")
        return name, approach_type, description_lines
    
    def generate_evolved_code(
        self,
        evolved_strategy: Strategy,
        problem: str,
        test_cases: List[TestCase]
    ) -> Solution:
        """
        Generate code implementation for evolved strategy.
        
        Uses 1 LLM call to implement the evolved strategy as executable code.
        This is similar to regular code generation but emphasizes the hybrid nature.
        
        Args:
            evolved_strategy: Evolved strategy to implement
            problem: Problem description
            test_cases: Test cases
            
        Returns:
            Solution object with evolved code
            
        Raises:
            Exception: If code generation fails
        """
        logger.info(f"Generating code for evolved strategy: {evolved_strategy.name}")
        
        try:
            # Create prompt for code generation
            prompt = self._create_code_generation_prompt(
                evolved_strategy, problem, test_cases
            )
            
            # LOG PROMPT
            logger.info("="*80)
            logger.info("EVOLVED CODE GENERATION - LLM CALL 6/8")
            logger.info("="*80)
            logger.info("PROMPT SENT TO LLM:")
            logger.info("-" * 40)
            logger.info(prompt)
            logger.info("-" * 40)
            
            # Generate code (1 LLM call)
            response = self.llm.generate(prompt, max_tokens=2048, temperature=0.7)
            
            # LOG RESPONSE
            logger.info("RESPONSE RECEIVED FROM LLM:")
            logger.info("-" * 40)
            logger.info(response)
            logger.info("-" * 40)
            
            # Extract code from response
            code = self._extract_code(response)
            
            # LOG EXTRACTED CODE
            logger.info("EXTRACTED EVOLVED CODE:")
            logger.info("-" * 40)
            logger.info(code)
            logger.info("-" * 40)
            
            # Create solution
            solution = Solution(strategy=evolved_strategy, code=code, generation_prompt=prompt)
            
            # LOG FINAL SOLUTION
            logger.info("EVOLVED SOLUTION CREATED:")
            logger.info(f"  Strategy: {solution.strategy.name}")
            logger.info(f"  Approach: {solution.strategy.approach_type}")
            logger.info(f"  Code length: {len(solution.code)} chars")
            logger.info("="*80)
            
            logger.info(
                f"Successfully generated evolved code ({len(code)} chars)"
            )
            
            return solution
            
        except Exception as e:
            logger.error(f"Failed to generate evolved code: {str(e)}")
            raise
    
    def _create_code_generation_prompt(
        self,
        strategy: Strategy,
        problem: str,
        test_cases: List[TestCase]
    ) -> str:
        """
        Create prompt for generating code from evolved strategy.
        
        Args:
            strategy: Evolved strategy to implement
            problem: Problem description
            test_cases: Test cases
            
        Returns:
            Formatted prompt string
        """
        # Format test cases
        test_cases_str = "\n".join([
            f"Input: {tc.input}\nExpected Output: {tc.expected_output}"
            for tc in test_cases[:3]
        ])
        
        prompt = f"""Given the following programming problem:

{problem}

Evolved Hybrid Strategy to implement:
Name: {strategy.name}
Approach Type: {strategy.approach_type}
Description: {strategy.description}

Test cases:
{test_cases_str}

Generate a complete Python implementation following this evolved hybrid strategy. Requirements:
1. Implement the hybrid approach as described in the strategy
2. Include proper error handling for edge cases
3. Add brief comments explaining key logic
4. Ensure the code is syntactically correct and executable
5. Focus on combining the strengths mentioned in the strategy description

Provide ONLY the Python code without additional explanation. You may use markdown code blocks if desired.
"""
        return prompt
    
    def _extract_code(self, response: str) -> str:
        """
        Extract Python code from LLM response.
        
        Handles markdown code blocks and plain code.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted Python code
            
        Raises:
            ValueError: If no valid code can be extracted
        """
        import re
        
        # Try to extract from markdown code blocks
        code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            code = max(matches, key=len) if len(matches) > 1 else matches[0]
            logger.debug(f"Extracted code from markdown block ({len(code)} chars)")
            return code.strip()
        
        # Try to find function definitions
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if re.match(r'^\s*(def|class)\s+', line):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            code = '\n'.join(code_lines).strip()
            logger.debug(f"Extracted code from function definitions ({len(code)} chars)")
            return code
        
        # Last resort
        if 'def ' in response:
            logger.warning("Could not identify code blocks, returning entire response")
            return response.strip()
        
        raise ValueError("Could not extract valid Python code from LLM response")
