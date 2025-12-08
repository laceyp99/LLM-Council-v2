"""
Model Metadata Module
Fetches and caches OpenRouter model specifications.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
import logging

from src.openrouter_client import OpenRouterClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default cache file location
PROJECT_DIR = Path(__file__).parent.parent
CACHE_FILE = PROJECT_DIR / "data" / "models_cache.json"
CACHE_EXPIRY_HOURS = 24  # Refresh cache after this many hours


@dataclass
class ModelInfo:
    """Container for model metadata."""
    id: str
    name: str
    context_length: int
    input_cost_per_token: float  # Cost per token in USD
    output_cost_per_token: float  # Cost per token in USD
    description: str = ""
    architecture: str = ""
    top_provider: str = ""
    
    @property
    def input_cost_per_million(self) -> float:
        """Cost per million input tokens."""
        return self.input_cost_per_token * 1_000_000
    
    @property
    def output_cost_per_million(self) -> float:
        """Cost per million output tokens."""
        return self.output_cost_per_token * 1_000_000
    
    def format_display_name(self) -> str:
        """Format model name with cost info for dropdown display."""
        ctx = f"{self.context_length // 1000}K" if self.context_length >= 1000 else str(self.context_length)
        in_cost = f"${self.input_cost_per_million:.2f}"
        out_cost = f"${self.output_cost_per_million:.2f}"
        return f"{self.name} | {ctx} ctx | In: {in_cost}/M | Out: {out_cost}/M"


class ModelMetadataManager:
    """Manages fetching and caching of OpenRouter model metadata."""
    
    def __init__(self, cache_file: Optional[Path] = None):
        """
        Initialize the metadata manager.
        
        Args:
            cache_file: Path to the cache JSON file
        """
        self.cache_file = cache_file or CACHE_FILE
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, ModelInfo] = {}
        self._last_fetch: float = 0
        
    def _load_cache(self) -> bool:
        """Load models from cache file."""
        if not self.cache_file.exists():
            return False
            
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self._last_fetch = data.get("timestamp", 0)
            models_data = data.get("models", [])
            
            self._models = {}
            for m in models_data:
                self._models[m["id"]] = ModelInfo(**m)
                
            logger.info(f"Loaded {len(self._models)} models from cache")
            return True
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    
    def _save_cache(self) -> None:
        """Save models to cache file."""
        try:
            data = {
                "timestamp": self._last_fetch,
                "models": [asdict(m) for m in self._models.values()]
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(self._models)} models to cache")
            
        except IOError as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._models:
            return False
            
        age_hours = (time.time() - self._last_fetch) / 3600
        return age_hours < CACHE_EXPIRY_HOURS
    
    def _parse_models(self, api_response: Dict[str, Any]) -> None:
        """Parse API response into ModelInfo objects."""
        models_data = api_response.get("data", [])
        
        self._models = {}
        for model in models_data:
            try:
                model_id = model.get("id", "")
                if not model_id:
                    continue
                    
                # Extract pricing info
                pricing = model.get("pricing", {})
                input_cost = float(pricing.get("prompt", 0) or 0)
                output_cost = float(pricing.get("completion", 0) or 0)
                
                # Extract context length
                context_length = model.get("context_length", 4096)
                
                # Get architecture info
                architecture = model.get("architecture", {})
                arch_str = architecture.get("modality", "")
                
                info = ModelInfo(
                    id=model_id,
                    name=model.get("name", model_id),
                    context_length=context_length,
                    input_cost_per_token=input_cost,
                    output_cost_per_token=output_cost,
                    description=model.get("description", ""),
                    architecture=arch_str,
                    top_provider=model.get("top_provider", {}).get("name", "")
                )
                
                self._models[model_id] = info
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse model {model.get('id', 'unknown')}: {e}")
                
        self._last_fetch = time.time()
        logger.info(f"Parsed {len(self._models)} models from API")
    
    def fetch_models(self, api_key: str, force_refresh: bool = False) -> bool:
        """
        Fetch models from OpenRouter API.
        
        Args:
            api_key: OpenRouter API key
            force_refresh: If True, bypass cache
            
        Returns:
            True if models were successfully loaded
        """
        # Try cache first
        if not force_refresh:
            self._load_cache()
            if self._is_cache_valid():
                return True
        
        # Fetch from API
        client = OpenRouterClient(api_key=api_key)
        success, result = client.fetch_models()
        
        if success:
            self._parse_models(result)
            self._save_cache()
            return True
        else:
            # If API fails, try to use stale cache
            if self._models:
                logger.warning("API fetch failed, using stale cache")
                return True
            logger.error(f"Failed to fetch models: {result}")
            return False
    
    def get_all_models(self) -> List[ModelInfo]:
        """Get all available models."""
        return list(self._models.values())
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a specific model by ID."""
        return self._models.get(model_id)
    
    def search_models(self, query: str) -> List[ModelInfo]:
        """
        Search models by name or ID.
        
        Args:
            query: Search string
            
        Returns:
            List of matching models
        """
        query = query.lower()
        return [
            m for m in self._models.values()
            if query in m.id.lower() or query in m.name.lower()
        ]
    
    def get_models_sorted_by_cost(self, ascending: bool = True) -> List[ModelInfo]:
        """Get models sorted by input cost."""
        return sorted(
            self._models.values(),
            key=lambda m: m.input_cost_per_token,
            reverse=not ascending
        )
    
    def get_models_by_context_length(self, min_context: int = 0) -> List[ModelInfo]:
        """Get models with at least the specified context length."""
        return [m for m in self._models.values() if m.context_length >= min_context]


# Singleton instance for app-wide use
_metadata_manager: Optional[ModelMetadataManager] = None


def get_metadata_manager() -> ModelMetadataManager:
    """Get the singleton metadata manager instance."""
    global _metadata_manager
    if _metadata_manager is None:
        _metadata_manager = ModelMetadataManager()
    return _metadata_manager
