import logging
import time
import socket
import functools
from typing import Dict, Any, Optional, Callable
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server, REGISTRY, CollectorRegistry

logger = logging.getLogger(__name__)

METRICS_REGISTRY = CollectorRegistry()

# Define application metrics with custom registry
REQUEST_COUNT = Counter('chatbot_requests_total', 'Total number of requests', ['status'], registry=METRICS_REGISTRY)
RESPONSE_TIME = Histogram('chatbot_response_time_seconds', 'Response time in seconds', 
                       ['operation'], buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0], registry=METRICS_REGISTRY)
RETRIEVAL_COUNT = Counter('chatbot_retrieval_count_total', 'Number of retrievals from vector store', 
                       ['source'], registry=METRICS_REGISTRY)
HALLUCINATION_GAUGE = Gauge('chatbot_hallucination_score', 'Hallucination score of responses', registry=METRICS_REGISTRY)
USER_SATISFACTION = Gauge('chatbot_user_satisfaction', 'User satisfaction score', registry=METRICS_REGISTRY)
LLM_CALLS = Counter('chatbot_llm_requests_total', 'Total LLM API calls', ['model', 'status'], registry=METRICS_REGISTRY)
TOKEN_USAGE = Counter('chatbot_token_usage_total', 'Total token usage', ['operation', 'model'], registry=METRICS_REGISTRY)

# Cache metrics
CACHE_HITS = Counter('chatbot_cache_hits_total', 'Total cache hits', ['cache_type'], registry=METRICS_REGISTRY)
CACHE_MISSES = Counter('chatbot_cache_misses_total', 'Total cache misses', ['cache_type'], registry=METRICS_REGISTRY)
CACHE_SIZE = Gauge('chatbot_cache_size', 'Current cache size', ['cache_type'], registry=METRICS_REGISTRY)

class MetricsManager:
    """Manages application metrics."""
    
    def __init__(self, metrics_port: int = 8099, enable_metrics: bool = True):
        """Initialize metrics manager."""
        self.metrics_port = metrics_port
        self.enable_metrics = enable_metrics
        self.server_started = False
        
    def start_metrics_server(self, addr: str = '0.0.0.0'):
        """Start the metrics server if not already running."""
        if not self.enable_metrics:
            logger.info("Metrics are disabled, not starting server")
            return False
            
        if self.server_started:
            logger.info("Metrics server already started")
            return True
            
        try:
            # Start the metrics server with our custom registry
            logger.info(f"Starting metrics server on {addr}:{self.metrics_port}")
            start_http_server(self.metrics_port, addr=addr, registry=METRICS_REGISTRY)
            
            # Verify server is running
            try:
                with socket.create_connection((addr if addr != '0.0.0.0' else 'localhost', 
                                               self.metrics_port), timeout=2) as s:
                    logger.info(f"Successfully connected to metrics server on {addr}:{self.metrics_port}")
            except Exception as conn_e:
                logger.warning(f"Could not connect to metrics server for verification: {conn_e}")
                
            self.server_started = True
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False
            
    def set_user_satisfaction(self, value: Optional[float] = None):
        """
        Set user satisfaction metric.
        
        Args:
            value: Satisfaction value on a 0-100 scale
        """
        if not self.enable_metrics:
            return
            
        if value is not None:
            USER_SATISFACTION.set(value)
            logger.info(f"Set user satisfaction to {value}/100")
            
    def record_hallucination_score(self, score: float):
        """Record hallucination score."""
        if not self.enable_metrics:
            return
            
        HALLUCINATION_GAUGE.set(score)
        logger.info(f"Recorded hallucination score: {score}")
        
    def record_retrieval(self, source: str = 'vector_store'):
        """Record a retrieval operation."""
        if not self.enable_metrics:
            return
            
        RETRIEVAL_COUNT.labels(source=source).inc()
        
    def record_llm_call(self, model: str, status: str = 'success'):
        """Record an LLM API call."""
        if not self.enable_metrics:
            return
            
        LLM_CALLS.labels(model=model, status=status).inc()
        
    def record_token_usage(self, operation: str, model: str, tokens: int):
        """Record token usage."""
        if not self.enable_metrics:
            return
            
        TOKEN_USAGE.labels(operation=operation, model=model).inc(tokens)
        
    def record_cache_hit(self, cache_type: str = 'response'):
        """Record a cache hit."""
        if not self.enable_metrics:
            return
            
        CACHE_HITS.labels(cache_type=cache_type).inc()
        
    def record_cache_miss(self, cache_type: str = 'response'):
        """Record a cache miss."""
        if not self.enable_metrics:
            return
            
        CACHE_MISSES.labels(cache_type=cache_type).inc()
        
    def update_cache_size(self, cache_type: str, size: int):
        """Update cache size metric."""
        if not self.enable_metrics:
            return
            
        CACHE_SIZE.labels(cache_type=cache_type).set(size)
        
def timing_decorator(operation_name: str):
    """Decorator to measure and record operation time."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                REQUEST_COUNT.labels(status='success').inc()
                return result
            except Exception as e:
                REQUEST_COUNT.labels(status='error').inc()
                raise e
            finally:
                RESPONSE_TIME.labels(operation=operation_name).observe(
                    time.time() - start_time
                )
        return wrapper
    return decorator 