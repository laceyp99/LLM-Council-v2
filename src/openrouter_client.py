"""
OpenRouter API Client Module
Handles API calls with headers, retries, latency timing, and error handling.
"""

import time
import requests
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Container for model response data."""
    model_id: str
    output_text: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    error: Optional[str] = None
    success: bool = True


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    MODELS_ENDPOINT = "/models"
    CHAT_ENDPOINT = "/chat/completions"
    
    def __init__(
        self,
        api_key: str,
        site_url: str = "http://localhost:8501",
        site_name: str = "LLM Council",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            site_url: Your site URL for OpenRouter headers
            site_name: Your site name for OpenRouter headers
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def _get_headers(self) -> Dict[str, str]:
        """Build request headers for OpenRouter API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json"
        }
    
    def fetch_models(self) -> Tuple[bool, Any]:
        """
        Fetch available models from OpenRouter.
        
        Returns:
            Tuple of (success, data/error_message)
        """
        try:
            response = requests.get(
                f"{self.BASE_URL}{self.MODELS_ENDPOINT}",
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()
            return True, response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch models: {e}")
            return False, str(e)
    
    def chat_completion(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> ModelResponse:
        """
        Send a chat completion request to a specific model.
        
        Args:
            model_id: The OpenRouter model ID
            prompt: User prompt text
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt
            
        Returns:
            ModelResponse with results or error
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        last_error = None
        
        for attempt in range(self.max_retries):
            start_time = time.perf_counter()
            
            try:
                response = requests.post(
                    f"{self.BASE_URL}{self.CHAT_ENDPOINT}",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=120  # Long timeout for slow models
                )
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited for {model_id}, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                # Extract response data
                choices = data.get("choices", [])
                if not choices:
                    return ModelResponse(
                        model_id=model_id,
                        output_text="",
                        latency_ms=latency_ms,
                        error="No response choices returned",
                        success=False
                    )
                
                output_text = choices[0].get("message", {}).get("content", "")
                
                # Extract token usage if available
                usage = data.get("usage", {})
                
                return ModelResponse(
                    model_id=model_id,
                    output_text=output_text,
                    latency_ms=latency_ms,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    success=True
                )
                
            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                logger.warning(f"Timeout for {model_id}, attempt {attempt + 1}/{self.max_retries}")
                
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"Request failed for {model_id}: {e}, attempt {attempt + 1}/{self.max_retries}")
                
                # Don't retry on client errors (4xx except 429)
                if hasattr(e, 'response') and e.response is not None:
                    if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                        break
                        
            # Exponential backoff
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (2 ** attempt))
        
        # All retries failed
        return ModelResponse(
            model_id=model_id,
            output_text="",
            latency_ms=0,
            error=last_error or "Unknown error",
            success=False
        )


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validate an OpenRouter API key by making a test request.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not api_key or not api_key.strip():
        return False, "API key is empty"
    
    client = OpenRouterClient(api_key=api_key)
    success, result = client.fetch_models()
    
    if success:
        return True, "API key is valid"
    else:
        return False, f"API key validation failed: {result}"
