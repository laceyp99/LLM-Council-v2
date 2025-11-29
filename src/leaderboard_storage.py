"""
Leaderboard Storage Module
Handles vote persistence and leaderboard calculations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default votes file location
VOTES_FILE = Path(__file__).parent / "data" / "votes.json"


@dataclass
class ModelScore:
    """Container for a model's voting scores."""
    model_id: str
    model_name: str
    best_votes: int = 0
    worst_votes: int = 0
    
    @property
    def net_score(self) -> int:
        """Calculate net score (best minus worst)."""
        return self.best_votes - self.worst_votes
    
    @property
    def total_votes(self) -> int:
        """Total number of votes received."""
        return self.best_votes + self.worst_votes


class LeaderboardStorage:
    """Manages vote storage and leaderboard calculations."""
    
    def __init__(self, votes_file: Optional[Path] = None):
        """
        Initialize the leaderboard storage.
        
        Args:
            votes_file: Path to the votes JSON file
        """
        self.votes_file = votes_file or VOTES_FILE
        self.votes_file.parent.mkdir(parents=True, exist_ok=True)
        self._scores: Dict[str, ModelScore] = {}
        self._lock = threading.Lock()
        self._load_votes()
        
    def _load_votes(self) -> None:
        """Load votes from file."""
        if not self.votes_file.exists():
            logger.info("No votes file found, starting fresh")
            return
            
        try:
            with open(self.votes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            scores_data = data.get("scores", {})
            for model_id, score_dict in scores_data.items():
                self._scores[model_id] = ModelScore(
                    model_id=model_id,
                    model_name=score_dict.get("model_name", model_id),
                    best_votes=score_dict.get("best_votes", 0),
                    worst_votes=score_dict.get("worst_votes", 0)
                )
                
            logger.info(f"Loaded votes for {len(self._scores)} models")
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load votes: {e}")
            self._scores = {}
    
    def _save_votes(self) -> None:
        """Save votes to file."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "scores": {
                    model_id: {
                        "model_name": score.model_name,
                        "best_votes": score.best_votes,
                        "worst_votes": score.worst_votes
                    }
                    for model_id, score in self._scores.items()
                }
            }
            
            with open(self.votes_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved votes for {len(self._scores)} models")
            
        except IOError as e:
            logger.error(f"Failed to save votes: {e}")
    
    def vote_best(self, model_id: str, model_name: Optional[str] = None) -> None:
        """
        Record a 'best' vote for a model.
        
        Args:
            model_id: The model ID
            model_name: Human-readable model name
        """
        with self._lock:
            if model_id not in self._scores:
                self._scores[model_id] = ModelScore(
                    model_id=model_id,
                    model_name=model_name or model_id
                )
            
            self._scores[model_id].best_votes += 1
            
            # Update name if provided
            if model_name:
                self._scores[model_id].model_name = model_name
                
            self._save_votes()
            logger.info(f"Recorded 'best' vote for {model_id}")
    
    def vote_worst(self, model_id: str, model_name: Optional[str] = None) -> None:
        """
        Record a 'worst' vote for a model.
        
        Args:
            model_id: The model ID
            model_name: Human-readable model name
        """
        with self._lock:
            if model_id not in self._scores:
                self._scores[model_id] = ModelScore(
                    model_id=model_id,
                    model_name=model_name or model_id
                )
            
            self._scores[model_id].worst_votes += 1
            
            # Update name if provided
            if model_name:
                self._scores[model_id].model_name = model_name
                
            self._save_votes()
            logger.info(f"Recorded 'worst' vote for {model_id}")
    
    def get_score(self, model_id: str) -> Optional[ModelScore]:
        """Get the score for a specific model."""
        return self._scores.get(model_id)
    
    def get_leaderboard(self, sort_by: str = "net_score") -> List[ModelScore]:
        """
        Get all models sorted by score.
        
        Args:
            sort_by: Field to sort by ('net_score', 'best_votes', 'worst_votes', 'total_votes')
            
        Returns:
            List of ModelScore objects sorted by the specified field
        """
        scores = list(self._scores.values())
        
        if sort_by == "net_score":
            scores.sort(key=lambda s: s.net_score, reverse=True)
        elif sort_by == "best_votes":
            scores.sort(key=lambda s: s.best_votes, reverse=True)
        elif sort_by == "worst_votes":
            scores.sort(key=lambda s: s.worst_votes, reverse=True)
        elif sort_by == "total_votes":
            scores.sort(key=lambda s: s.total_votes, reverse=True)
            
        return scores
    
    def get_top_models(self, n: int = 10) -> List[ModelScore]:
        """Get the top N models by net score."""
        return self.get_leaderboard("net_score")[:n]
    
    def get_bottom_models(self, n: int = 10) -> List[ModelScore]:
        """Get the bottom N models by net score."""
        return self.get_leaderboard("net_score")[-n:][::-1]
    
    def reset_votes(self) -> None:
        """Reset all votes (use with caution)."""
        with self._lock:
            self._scores = {}
            self._save_votes()
            logger.warning("All votes have been reset")
    
    def export_leaderboard(self) -> Dict:
        """Export leaderboard data for display."""
        leaderboard = self.get_leaderboard()
        return {
            "last_updated": datetime.now().isoformat(),
            "total_models": len(leaderboard),
            "leaderboard": [
                {
                    "rank": i + 1,
                    "model_id": s.model_id,
                    "model_name": s.model_name,
                    "net_score": s.net_score,
                    "best_votes": s.best_votes,
                    "worst_votes": s.worst_votes,
                    "total_votes": s.total_votes
                }
                for i, s in enumerate(leaderboard)
            ]
        }


# Singleton instance for app-wide use
_leaderboard_storage: Optional[LeaderboardStorage] = None


def get_leaderboard_storage() -> LeaderboardStorage:
    """Get the singleton leaderboard storage instance."""
    global _leaderboard_storage
    if _leaderboard_storage is None:
        _leaderboard_storage = LeaderboardStorage()
    return _leaderboard_storage
