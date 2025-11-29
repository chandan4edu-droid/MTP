"""Code Generator for implementing strategies as executable Python code."""

import logging
import re
from typing import List

from ..llm.base_interface import BaseLLMInterface
from ..models.data_models import Strategy, Solution, TestCase

logger = logging.getLogger(__name__)


class CodeGenerator:
    """
    Generates executable Python code implementations from strategy descriptions.
    
    Takes strategy descriptions and converts them into complete Python implementations
    that can be executed and tested. Handles code extraction from LLM responses,
    including markdown code blocks and other formatting.
    """
    
    def __init__(self, llm: BaseLLMInterface):
        """
        Initialize the Code Generator.
        
        Args:
            llm: LLM interface for generating code
        """
        self.llm = llm
        logger.info("Initialized CodeGenerator")
    
    def _create_prompt(
        self,
        strategy: Strategy,
        problem: str,
        test_cases: List[TestCase]
    ) -> str:
        """
        Create the prompt for generating code from a strategy.
        
        Args:
            strategy: Strategy to implement
            problem: Problem description
            test_cases: List of test cases
            
        Returns:
            Formatted prompt string
        """
        # Format test cases
        test_cases_str = "\n".join([
            f"Input: {tc.input}\nExpected Output: {tc.expected_output}"
            for tc in test_cases[:3]  # Include first 3 test cases for context
        ])
        
        prompt = f"""Given the following programming problem:

{problem}

Strategy to implement:
Name: {strategy.name}
Approach: {strategy.approach_type}
Description: {strategy.description}

Test cases:
{test_cases_str}

Generate a complete Python implementation following this strategy. Requirements:
1. Include the complete function implementation with proper signature
2. Add appropriate error handling for edge cases

# added by me 
<---------------------------------------------------------------------------------------->
3. Include brief comments explaining key logic
4. Ensure the code is syntactically correct and executable
5. Make sure the implementation follows the strategy description
<------------------------------------------------------------------------------------------>

Provide ONLY the Python code without additional explanation. You may use markdown code blocks if desired.
"""
        return prompt
    
    def _extract_code(self, response: str) -> str:
        """
        Extract Python code from LLM response.
        
        Handles various formats:
        - Markdown code blocks (```python ... ```)
        - Plain code
        - Mixed text and code
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Extracted Python code
            
        Raises:
            ValueError: If no valid code can be extracted
        """
        # Try to extract from markdown code blocks first
        code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            # Use the first (or largest) code block
            code = max(matches, key=len) if len(matches) > 1 else matches[0]
            logger.debug(f"Extracted code from markdown block ({len(code)} chars)")
            return code.strip()
        
        # If no markdown blocks, try to find code by looking for function definitions
        # Look for lines starting with 'def ' or 'class '
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Start capturing when we see a function or class definition
            if re.match(r'^\s*(def|class)\s+', line):
                in_code = True
            
            # Capture lines while in code mode
            if in_code:
                code_lines.append(line)
                
                # Stop if we hit a line that looks like explanatory text
                # (not indented and doesn't start with def/class/import/from/#)
                if line.strip() and not re.match(r'^\s*(def|class|import|from|#|@|\w+\s*=)', line):
                    if not line.startswith(' ') and not line.startswith('\t'):
                        # This might be explanatory text, but keep the line if it's part of code
                        if not any(keyword in line.lower() for keyword in ['def ', 'class ', 'return', '=']):
                            break
        
        if code_lines:
            code = '\n'.join(code_lines).strip()
            logger.debug(f"Extracted code from function definitions ({len(code)} chars)")
            return code
        
        # Last resort: return the entire response if it looks like code
        # (contains 'def ' or has significant indentation)
        if 'def ' in response or any(line.startswith('    ') for line in lines):
            logger.warning("Could not identify code blocks, returning entire response")
            return response.strip()
        
        # Complete failure
        raise ValueError("Could not extract valid Python code from LLM response")
    
    def _validate_code(self, code: str) -> bool:
        """
        Perform basic validation on extracted code.
        
        Args:
            code: Python code string
            
        Returns:
            True if code appears valid, False otherwise
        """
        # Check for minimum requirements
        if not code or len(code.strip()) < 10:
            logger.warning("Code is too short or empty")
            return False
        
        # Check for function definition
        if 'def ' not in code:
            logger.warning("Code does not contain a function definition")
            return False
        
        # Try to compile (basic syntax check)
        try:
            compile(code, '<string>', 'exec')
            logger.debug("Code passed basic syntax validation")
            return True
        except SyntaxError as e:
            logger.warning(f"Code has syntax error: {str(e)}")
            return False
    
    def generate_code(
        self,
        strategy: Strategy,
        problem: str,
        test_cases: List[TestCase],
        llm_call_num: int = 0
    ) -> Solution:
        """
        Generate code implementation for a single strategy.
        
        Uses exactly 1 LLM call. If code extraction or validation fails,
        raises an exception to be handled by the caller.
        
        Args:
            strategy: Strategy to implement
            problem: Problem description
            test_cases: List of test cases
            
        Returns:
            Solution object containing strategy and code
            
        Raises:
            Exception: If code generation, extraction, or validation fails
        """
        logger.info(f"Generating code for strategy: {strategy.name}")
        
        try:
            # Create prompt
            prompt = self._create_prompt(strategy, problem, test_cases)
            
            # LOG PROMPT
            logger.info("="*80)
            logger.info(f"CODE GENERATION - LLM CALL {llm_call_num}/4 - Strategy: {strategy.name}")
            logger.info("="*80)
<--------------------------------------------------------Added by me----------------------------------------------------------------------------->
            logger.info("PROMPT SENT TO LLM:")
            logger.info("-" * 40)
            logger.info(prompt)
            logger.info("-" * 40)
<--------------------------------------------------------Added byme----------------------------------------------------------------------------->
 
            
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
            logger.info("EXTRACTED CODE:")
            logger.info("-" * 40)
            logger.info(code)
            logger.info("-" * 40)
            
            # Validate code
            if not self._validate_code(code):
                raise ValueError("Generated code failed validation")
            
            # Create solution
            solution = Solution(strategy=strategy, code=code, generation_prompt=prompt)
            
            # LOG FINAL SOLUTION
            logger.info("FINAL SOLUTION CREATED:")
            
         #   ** added by me **
            logger.info(f"  Strategy: {solution.strategy.name}")
            logger.info(f"  Code length: {len(solution.code)} chars")
         #   ** added by me **
            logger.info("="*80)
            
            logger.info(
                f"Successfully generated code for strategy '{strategy.name}' "
                f"({len(code)} chars)"
            )
            
            return solution
            
        except Exception as e:
            logger.error(
                f"Failed to generate code for strategy '{strategy.name}': {str(e)}"
            )
            raise
    
    def generate_all_codes(
        self,
        strategies: List[Strategy],
        problem: str,
        test_cases: List[TestCase]
    ) -> List[Solution]:
        """
        Generate code implementations for all strategies.
        
        Uses exactly 3 LLM calls (one per strategy). If any strategy fails,
        it is skipped and logged, but the process continues for remaining strategies.
        
        Args:
            strategies: List of strategies to implement (should be 3)
            problem: Problem description
            test_cases: List of test cases
            
        Returns:
            List of Solution objects (may be fewer than 3 if some fail)
        """
        logger.info(f"Generating code for {len(strategies)} strategies")
        
        solutions = []
<-----------------------------------------i added by me for proper logging------------------------------------------------------>      
        for i, strategy in enumerate(strategies):
            try:
                llm_call_num = i + 2
                solution = self.generate_code(strategy, problem, test_cases, llm_call_num)
                solutions.append(solution)
                logger.info(f"Successfully generated solution {i+1}/{len(strategies)}")
                
            except Exception as e:
                logger.error(
                    f"Failed to generate code for strategy {i+1}/{len(strategies)} "
                    f"('{strategy.name}'): {str(e)}"
                )
                logger.info("Continuing with remaining strategies")
                continue
        
        logger.info(
            f"Code generation complete: {len(solutions)}/{len(strategies)} successful"
        )
        
        if len(solutions) == 0:
            logger.error("All code generation attempts failed")
            raise RuntimeError("Failed to generate code for any strategy")
        
        return solutions
