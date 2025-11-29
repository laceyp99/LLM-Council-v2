"""
Cache Manager Module
Handles optional caching of model outputs for identical prompt-model pairs.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default cache file location
CACHE_FILE = Path(__file__).parent / "data" / "response_cache.json"
MAX_CACHE_ENTRIES = 1000  # Maximum number of cached responses


@dataclass
class CachedResponse:
    """Container for a cached model response."""
    model_id: str
    prompt_hash: str
    output_text: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    timestamp: str
    temperature: float


class CacheManager:
    """Manages caching of model responses."""
    
    def __init__(self, cache_file: Optional[Path] = None, enabled: bool = True):
        """
        Initialize the cache manager.
        
        Args:
            cache_file: Path to the cache JSON file
            enabled: Whether caching is enabled
        """
        self.cache_file = cache_file or CACHE_FILE
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self._cache: Dict[str, CachedResponse] = {}
        self._lock = threading.Lock()
        self._load_cache()
    
    @staticmethod
    def _generate_key(model_id: str, prompt: str, temperature: float) -> str:
        """Generate a unique cache key for a prompt-model pair."""
        content = f"{model_id}|{prompt}|{temperature:.2f}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        """Generate a hash of the prompt for storage."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    def _load_cache(self) -> None:
        """Load cache from file."""
        if not self.cache_file.exists():
            logger.info("No cache file found, starting fresh")
            return
            
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cache_data = data.get("cache", {})
            for key, entry in cache_data.items():
                self._cache[key] = CachedResponse(
                    model_id=entry["model_id"],
                    prompt_hash=entry["prompt_hash"],
                    output_text=entry["output_text"],
                    input_tokens=entry.get("input_tokens"),
                    output_tokens=entry.get("output_tokens"),
                    timestamp=entry["timestamp"],
                    temperature=entry.get("temperature", 0.7)
                )
                
            logger.info(f"Loaded {len(self._cache)} cached responses")
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load cache: {e}")
            self._cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to file."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "entry_count": len(self._cache),
                "cache": {
                    key: asdict(entry)
                    for key, entry in self._cache.items()
                }
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved {len(self._cache)} cached responses")
            
        except IOError as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _prune_cache(self) -> None:
        """Remove oldest entries if cache is too large."""
        if len(self._cache) <= MAX_CACHE_ENTRIES:
            return
            
        # Sort by timestamp and remove oldest
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].timestamp
        )
        
        entries_to_remove = len(self._cache) - MAX_CACHE_ENTRIES
        for key, _ in sorted_entries[:entries_to_remove]:
            del self._cache[key]
            
        logger.info(f"Pruned {entries_to_remove} old cache entries")
    
    def get(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.7
    ) -> Optional[CachedResponse]:
        """
        Get a cached response if available.
        
        Args:
            model_id: The model ID
            prompt: The user prompt
            temperature: The temperature setting
            
        Returns:
            CachedResponse if found, None otherwise
        """
        if not self.enabled:
            return None
            
        key = self._generate_key(model_id, prompt, temperature)
        return self._cache.get(key)
    
    def set(
        self,
        model_id: str,
        prompt: str,
        output_text: str,
        temperature: float = 0.7,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None
    ) -> None:
        """
        Cache a model response.
        
        Args:
            model_id: The model ID
            prompt: The user prompt
            output_text: The model's output
            temperature: The temperature setting
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        if not self.enabled:
            return
            
        with self._lock:
            key = self._generate_key(model_id, prompt, temperature)
            
            self._cache[key] = CachedResponse(
                model_id=model_id,
                prompt_hash=self._hash_prompt(prompt),
                output_text=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                timestamp=datetime.now().isoformat(),
                temperature=temperature
            )
            
            self._prune_cache()
            self._save_cache()
            
            logger.debug(f"Cached response for {model_id}")
    
    def clear(self) -> None:
        """Clear all cached responses."""
        with self._lock:
            self._cache = {}
            self._save_cache()
            logger.info("Cache cleared")
    
    def clear_model(self, model_id: str) -> int:
        """
        Clear all cached responses for a specific model.
        
        Args:
            model_id: The model ID to clear
            
        Returns:
            Number of entries removed
        """
        with self._lock:
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if entry.model_id == model_id
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
                
            if keys_to_remove:
                self._save_cache()
                
            logger.info(f"Cleared {len(keys_to_remove)} cache entries for {model_id}")
            return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        models = {}
        for entry in self._cache.values():
            if entry.model_id not in models:
                models[entry.model_id] = 0
            models[entry.model_id] += 1
            
        return {
            "total_entries": len(self._cache),
            "max_entries": MAX_CACHE_ENTRIES,
            "enabled": self.enabled,
            "entries_by_model": models
        }
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable caching."""
        self.enabled = enabled
        logger.info(f"Caching {'enabled' if enabled else 'disabled'}")


# Singleton instance for app-wide use
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the singleton cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
