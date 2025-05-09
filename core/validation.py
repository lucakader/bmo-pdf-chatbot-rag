from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
import logging
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)

class HallucinationCheck(BaseModel):
    """Check if the generated response contains hallucinations."""
    is_hallucination: bool = Field(description="Whether the response contains hallucinations")
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Reasoning for the hallucination check")
    verified_claims: Optional[List[str]] = Field(description="List of claims that were verified in the context", default=None)
    unverified_claims: Optional[List[str]] = Field(description="List of claims that could not be verified in the context", default=None)

class ResponseValidator:
    """Validate generated responses for hallucinations."""
    
    def __init__(self, llm, confidence_threshold: float = 0.6, max_timeout: int = 30):
        """Initialize the response validator."""
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.max_timeout = max_timeout
        
    def check_hallucination(
        self, 
        response: str, 
        context: str, 
        question: str
    ) -> Optional[HallucinationCheck]:
        """
        Check if the response contains hallucinations with adaptive timeout.
        
        Args:
            response: The generated response.
            context: The context used to generate the response.
            question: The user's question.
            
        Returns:
            Optional[HallucinationCheck]: The hallucination check result if successful, None otherwise.
        """
        try:
            # Input validation
            if not response or not context or not question:
                logger.warning("Missing input for hallucination check")
                return None
                
            logger.info("Running hallucination check")
            
            # Calculate adaptive timeout based on content length
            base_timeout = 5  # Base timeout in seconds
            context_length = len(context)
            response_length = len(response)
            
            # Adjust timeout based on content size (1 additional second per 1000 chars)
            content_size_factor = (context_length + response_length) // 1000
            adaptive_timeout = min(
                base_timeout + content_size_factor,
                self.max_timeout  # Cap at max_timeout seconds
            )
            
            logger.info(f"Using adaptive timeout of {adaptive_timeout}s for hallucination check")
                
            # Create a prompt for hallucination checking with confidence score
            hallucination_prompt = self._create_hallucination_prompt()
            
            # Define the chain for hallucination checking with structured output
            hallucination_chain = (
                hallucination_prompt 
                | self.llm.with_structured_output(HallucinationCheck)
            )
            
            # Run the chain with timeout protection
            try:
                result = hallucination_chain.invoke({
                    "context": context,
                    "question": question,
                    "response": response
                }, timeout=adaptive_timeout)
                
                # Validate result fields
                if not hasattr(result, 'confidence_score') or not isinstance(result.confidence_score, float):
                    logger.warning("Invalid hallucination check result: missing or invalid confidence score")
                    return None
                    
                # Ensure confidence score is within range
                result.confidence_score = max(0.0, min(1.0, result.confidence_score))
                
                # Ensure lists are initialized
                if result.verified_claims is None:
                    result.verified_claims = []
                if result.unverified_claims is None:
                    result.unverified_claims = []
                
                logger.info(f"Hallucination check complete. Is hallucination: {result.is_hallucination}, Score: {result.confidence_score}")
                logger.info(f"Verified claims: {len(result.verified_claims)}, Unverified claims: {len(result.unverified_claims)}")
                
                return result
            except FuturesTimeoutError:
                logger.warning(f"Hallucination check timed out after {adaptive_timeout}s")
                return None
            except Exception as inner_e:
                logger.error(f"Error during hallucination check invocation: {str(inner_e)}")
                if "rate limit" in str(inner_e).lower():
                    logger.warning("Rate limit hit during hallucination check, skipping")
                return None
                
        except Exception as e:
            logger.error(f"Error checking hallucination: {str(e)}")
            return None
            
    def _create_hallucination_prompt(self):
        """Create a prompt for hallucination checking."""
        from langchain.prompts import ChatPromptTemplate
        
        return ChatPromptTemplate.from_template("""
        You are a critical evaluator that checks for hallucinations in AI-generated responses.
        
        Context from knowledge base:
        {context}
        
        Question: {question}
        Response: {response}
        
        Task 1: Extract key factual claims from the response.
        
        Task 2: For each claim, determine if it is supported by the context.
        Create two lists:
        1. Verified claims - claims that are directly supported by the context
        2. Unverified claims - claims that cannot be verified from the context
        
        Task 3: Evaluate if the response contains any information not supported by the context.
        Assign a confidence score on a scale of 0 to 1, where:
        - 0.0-0.2: Most of the response is unsupported by the context
        - 0.3-0.5: Significant parts are unsupported by the context
        - 0.6-0.8: Minor inaccuracies or small unsupported details
        - 0.9-1.0: Response is fully supported by the context
        
        Be conservative - only mark as hallucination if it clearly contains facts not in the context.
        """)
    
    def validate_response(
        self, 
        response: str, 
        context: str, 
        question: str,
        hallucination_result: Optional[HallucinationCheck] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Validate the response against the context to detect and mitigate hallucinations.
        
        Args:
            response: Generated response
            context: Retrieved context
            question: Original question
            hallucination_result: Optional result from hallucination check
            
        Returns:
            A tuple of (validated_response, validation_info)
        """
        if not response:
            logger.warning("Empty response provided for validation")
            return "I couldn't generate a valid response. Please try again.", {
                "has_citations": False,
                "warning": "Empty response provided"
            }
            
        # Extract and check citations
        lines = response.split('\n')
        validated_response = []
        citations_found = False
        citations_section = []
        
        for i, line in enumerate(lines):
            # Check if we've reached the sources section - allow different formats
            if (line.lower().startswith('sources:') or 
                line.lower().startswith('source:') or 
                line.lower() == 'sources' or 
                line.lower() == 'references:' or
                line.lower() == 'references'):
                citations_found = True
                citations_section = lines[i:]
                break
            validated_response.append(line)
        
        # Build validation info
        validation_info = {
            'has_citations': citations_found,
        }
        
        # Add hallucination check information if available
        if hallucination_result:
            validation_info['hallucination_check'] = {
                'is_hallucination': hallucination_result.is_hallucination,
                'confidence_score': hallucination_result.confidence_score,
                'reasoning': hallucination_result.reasoning,
            }
            
            # Add verified/unverified claims if available
            if hallucination_result.verified_claims:
                validation_info['verified_claims'] = hallucination_result.verified_claims
            if hallucination_result.unverified_claims:
                validation_info['unverified_claims'] = hallucination_result.unverified_claims
                
                # For unverified claims, add a warning to the response
                if len(hallucination_result.unverified_claims) > 0:
                    unverified_warning = "\n\n⚠️ **Caution**: The following claims could not be verified from the source material:\n"
                    for claim in hallucination_result.unverified_claims:
                        unverified_warning += f"- {claim}\n"
                    validated_response.append(unverified_warning)
        
        # If no sources section, add a warning
        if not citations_found:
            validation_info['warning'] = 'Response does not cite specific sources'
            # Look for implicit citations in brackets like [Source 1]
            if any('[Source' in line for line in validated_response):
                validation_info['has_implicit_citations'] = True
                logger.info("Response has implicit citations but no formal sources section")
            else:
                validated_response.append('\n\n⚠️ Note: This response does not cite specific sources and may be less reliable.')
        else:
            validation_info['citations'] = citations_section
            validated_response.extend(citations_section)
        
        return '\n'.join(validated_response), validation_info
        
    def generate_fallback_response(
        self, 
        query: str, 
        confidence_score: float, 
        reasoning: str
    ) -> str:
        """
        Generate a fallback response when confidence is too low.
        
        Args:
            query: The original query
            confidence_score: The confidence score from hallucination check
            reasoning: The reasoning from hallucination check
            
        Returns:
            A fallback response explaining why we can't provide a reliable answer
        """
        confidence_percent = int(confidence_score * 100)
        
        return f"""I don't have enough reliable information to answer this question confidently.

Your question: "{query}"

Based on the documents I have access to, I cannot provide a satisfactory answer with sufficient confidence (current confidence: {confidence_percent}%).

Reason: {reasoning}

Please try rephrasing your question to focus on topics covered in the documents, or consult additional sources for this information.
""" 