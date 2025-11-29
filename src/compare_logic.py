"""
Compare Logic Module
Orchestrates parallel model calls and result aggregation.
"""

import concurrent.futures
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from src.openrouter_client import OpenRouterClient, ModelResponse
from src.cache_manager import get_cache_manager, CachedResponse
from src.model_metadata import get_metadata_manager, ModelInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Container for a single model's comparison result."""
    model_id: str
    model_name: str
    output_text: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    error: Optional[str] = None
    success: bool = True
    from_cache: bool = False
    color_index: int = 0  # For UI coloring


@dataclass
class ComparisonSession:
    """Container for a complete comparison session."""
    prompt: str
    temperature: float
    timestamp: str
    results: List[ComparisonResult] = field(default_factory=list)
    total_latency_ms: float = 0
    successful_count: int = 0
    failed_count: int = 0


def run_single_model(
    client: OpenRouterClient,
    model_id: str,
    model_name: str,
    prompt: str,
    temperature: float,
    use_cache: bool = True,
    color_index: int = 0
) -> ComparisonResult:
    """
    Run a single model and return the result.
    
    Args:
        client: OpenRouter client instance
        model_id: The model ID to run
        model_name: Human-readable model name
        prompt: User prompt
        temperature: Sampling temperature
        use_cache: Whether to check cache first
        color_index: Index for UI coloring
        
    Returns:
        ComparisonResult with the output or error
    """
    cache_manager = get_cache_manager()
    
    # Check cache first
    if use_cache:
        cached = cache_manager.get(model_id, prompt, temperature)
        if cached:
            logger.info(f"Cache hit for {model_id}")
            return ComparisonResult(
                model_id=model_id,
                model_name=model_name,
                output_text=cached.output_text,
                latency_ms=0,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                success=True,
                from_cache=True,
                color_index=color_index
            )
    
    # Make API call
    response = client.chat_completion(
        model_id=model_id,
        prompt=prompt,
        temperature=temperature
    )
    
    # Cache successful responses
    if response.success and use_cache:
        cache_manager.set(
            model_id=model_id,
            prompt=prompt,
            output_text=response.output_text,
            temperature=temperature,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens
        )
    
    return ComparisonResult(
        model_id=model_id,
        model_name=model_name,
        output_text=response.output_text,
        latency_ms=response.latency_ms,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        total_tokens=response.total_tokens,
        error=response.error,
        success=response.success,
        from_cache=False,
        color_index=color_index
    )


def run_comparison(
    api_key: str,
    prompt: str,
    model_ids: List[str],
    temperature: float = 0.7,
    use_cache: bool = True,
    max_workers: int = 5,
    progress_callback: Optional[callable] = None
) -> ComparisonSession:
    """
    Run a comparison across multiple models.
    
    Args:
        api_key: OpenRouter API key
        prompt: User prompt to send to all models
        model_ids: List of model IDs to compare
        temperature: Sampling temperature
        use_cache: Whether to use response caching
        max_workers: Maximum concurrent API calls
        progress_callback: Optional callback for progress updates (receives completed count)
        
    Returns:
        ComparisonSession with all results
    """
    client = OpenRouterClient(api_key=api_key)
    metadata = get_metadata_manager()
    
    session = ComparisonSession(
        prompt=prompt,
        temperature=temperature,
        timestamp=datetime.now().isoformat()
    )
    
    # Get model names
    model_names = {}
    for model_id in model_ids:
        info = metadata.get_model(model_id)
        model_names[model_id] = info.name if info else model_id
    
    completed = 0
    
    # Run models in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(
                run_single_model,
                client,
                model_id,
                model_names[model_id],
                prompt,
                temperature,
                use_cache,
                i  # color_index
            ): model_id
            for i, model_id in enumerate(model_ids)
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            model_id = future_to_model[future]
            try:
                result = future.result()
                session.results.append(result)
                
                if result.success:
                    session.successful_count += 1
                    session.total_latency_ms += result.latency_ms
                else:
                    session.failed_count += 1
                    
            except Exception as e:
                logger.error(f"Exception for {model_id}: {e}")
                session.results.append(ComparisonResult(
                    model_id=model_id,
                    model_name=model_names[model_id],
                    output_text="",
                    latency_ms=0,
                    error=str(e),
                    success=False,
                    color_index=model_ids.index(model_id)
                ))
                session.failed_count += 1
            
            completed += 1
            if progress_callback:
                progress_callback(completed)
    
    # Sort results by original order (color_index)
    session.results.sort(key=lambda r: r.color_index)
    
    logger.info(f"Comparison complete: {session.successful_count} succeeded, {session.failed_count} failed")
    
    return session


def estimate_cost(
    prompt: str,
    model_ids: List[str],
    estimated_output_tokens: int = 500
) -> Dict[str, Any]:
    """
    Estimate the cost of running a comparison.
    
    Args:
        prompt: User prompt
        model_ids: List of model IDs
        estimated_output_tokens: Estimated output tokens per model
        
    Returns:
        Cost estimation breakdown
    """
    metadata = get_metadata_manager()
    
    # Rough token count (4 chars per token)
    estimated_input_tokens = len(prompt) // 4
    
    costs = []
    total_cost = 0
    
    for model_id in model_ids:
        info = metadata.get_model(model_id)
        if info:
            input_cost = estimated_input_tokens * info.input_cost_per_token
            output_cost = estimated_output_tokens * info.output_cost_per_token
            model_cost = input_cost + output_cost
            
            costs.append({
                "model_id": model_id,
                "model_name": info.name,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": model_cost
            })
            
            total_cost += model_cost
        else:
            costs.append({
                "model_id": model_id,
                "model_name": model_id,
                "input_cost": 0,
                "output_cost": 0,
                "total_cost": 0,
                "unknown": True
            })
    
    return {
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "models": costs,
        "total_estimated_cost": total_cost
    }
