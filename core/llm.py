from typing import Dict, Any, Optional, Type, Callable
from abc import ABC, abstractmethod
import os
import logging
import time
from collections import OrderedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Define metrics - import directly from monitoring.metrics
from monitoring.metrics import LLM_CALLS, METRICS_REGISTRY, TOKEN_USAGE

# Define LLM-specific metrics
LLM_LATENCY = Histogram('chatbot_llm_response_time_seconds', 'LLM response time in seconds', 
                       ['model'], buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0], registry=METRICS_REGISTRY)

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def get_llm(self, **kwargs):
        """Get the LLM instance."""
        pass
        
    @abstractmethod
    def with_structured_output(self, output_class):
        """Get an LLM that returns structured output."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI implementation of LLM provider."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0, api_key: Optional[str] = None):
        """Initialize OpenAI provider."""
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.llm = self._initialize_llm()
        logger.info(f"Initialized OpenAI LLM with model: {model_name}, temperature: {temperature}")
        
    def _initialize_llm(self):
        """Initialize the OpenAI LLM."""
        try:
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                api_key=self.api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {str(e)}")
            raise RuntimeError(f"Failed to initialize OpenAI LLM: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        start_time = time.time()
        try:
            # Handle additional parameters
            temperature = kwargs.get('temperature', self.temperature)
            if temperature != self.temperature:
                llm = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=temperature,
                    api_key=self.api_key
                )
            else:
                llm = self.llm
                
            # Create a simple chain
            chain = (
                ChatPromptTemplate.from_template(prompt)
                | llm
                | StrOutputParser()
            )
            
            # Generate response
            response = chain.invoke({})
            
            # Record metrics
            LLM_CALLS.labels(model=self.model_name, status='success').inc()
            
            # Estimate token usage (rough approximation)
            self._record_token_usage(prompt, response, self.model_name)
            
            return response
        except Exception as e:
            logger.error(f"Error generating response from LLM: {str(e)}")
            LLM_CALLS.labels(model=self.model_name, status='error').inc()
            raise
        finally:
            # Record latency
            LLM_LATENCY.labels(model=self.model_name).observe(time.time() - start_time)
    
    def _record_token_usage(self, prompt: str, response: str, model: str):
        """Estimate and record token usage."""
        try:
            # Simple approximation: ~4 chars per token
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(response) // 4
            
            # Record token usage
            TOKEN_USAGE.labels(operation='prompt', model=model).inc(prompt_tokens)
            TOKEN_USAGE.labels(operation='completion', model=model).inc(completion_tokens)
            TOKEN_USAGE.labels(operation='total', model=model).inc(prompt_tokens + completion_tokens)
        except Exception as e:
            logger.warning(f"Error recording token usage: {str(e)}")
    
    def get_llm(self, **kwargs):
        """Get the LLM instance."""
        # If parameters are different, create a new instance
        temperature = kwargs.get('temperature', self.temperature)
        if temperature != self.temperature:
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=temperature,
                api_key=self.api_key
            )
        return self.llm
        
    def with_structured_output(self, output_class: Type):
        """Get an LLM that returns structured output."""
        return self.llm.with_structured_output(output_class)
        
    def create_rag_chain(self, prompt):
        """Create a RAG chain with the LLM."""
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

class CachedLLMProvider(LLMProvider):
    """LLM provider with caching support."""
    
    def __init__(self, base_provider: LLMProvider, cache_size: int = 100):
        """Initialize cached LLM provider."""
        self.base_provider = base_provider
        self.cache = OrderedDict()  # Use OrderedDict for LRU cache implementation
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM with caching."""
        # Create cache key from prompt and parameters
        cache_key = self._create_cache_key(prompt, kwargs)
        
        # Check cache
        if cache_key in self.cache:
            # Move the item to the end (most recently used position)
            value = self.cache.pop(cache_key)
            self.cache[cache_key] = value
            
            self.cache_hits += 1
            logger.info(f"Cache hit! Hits: {self.cache_hits}, Misses: {self.cache_misses}")
            return value
            
        # Cache miss
        self.cache_misses += 1
        response = self.base_provider.generate(prompt, **kwargs)
        
        # Update cache
        self._update_cache(cache_key, response)
        
        return response
    
    def get_llm(self, **kwargs):
        """Get the LLM instance."""
        return self.base_provider.get_llm(**kwargs)
        
    def with_structured_output(self, output_class: Type):
        """Get an LLM that returns structured output."""
        return self.base_provider.with_structured_output(output_class)
        
    def create_rag_chain(self, prompt):
        """Create a RAG chain with the LLM by delegating to the base provider."""
        return self.base_provider.create_rag_chain(prompt)
        
    def _create_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """Create a unique cache key based on prompt and parameters."""
        try:
            # Create a string representation of parameters
            params_str = ",".join([f"{k}={v}" for k, v in sorted(params.items())])
            
            # Combine prompt and parameters
            key = f"{prompt}|{params_str}"
            
            # Use a hash for shorter keys
            import hashlib
            return hashlib.md5(key.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Error creating cache key: {str(e)}")
            # Fallback to a simple key based on the length of the prompt
            return f"prompt:{len(prompt)}:{hash(prompt)}"
        
    def _update_cache(self, key: str, value: str):
        """Update the cache with a new value using proper LRU policy."""
        # If key exists, remove it first to update its position
        if key in self.cache:
            del self.cache[key]
            
        # If cache is full, remove least recently used entry (first item)
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
            
        # Add new entry (most recently used)
        self.cache[key] = value
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "size": len(self.cache),
            "max_size": self.cache_size,
            "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }
        
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        logger.info("Cache cleared") 