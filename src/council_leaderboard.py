"""
Council Leaderboard Storage Module
Handles vote persistence for council voting (model-to-model votes).
This is separate from user votes - it tracks when models vote for each other's responses.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default council votes file location
COUNCIL_VOTES_FILE = Path(__file__).parent / "data" / "council_votes.json"


@dataclass
class CouncilModelScore:
    """Container for a model's council voting scores."""
    model_id: str
    model_name: str
    votes_received: int = 0  # Times this model's response was voted as best
    times_participated: int = 0  # Times this model participated in voting sessions
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate as percentage."""
        if self.times_participated == 0:
            return 0.0
        return (self.votes_received / self.times_participated) * 100


class CouncilLeaderboardStorage:
    """Manages council vote storage and leaderboard calculations."""
    
    def __init__(self, votes_file: Optional[Path] = None):
        """
        Initialize the council leaderboard storage.
        
        Args:
            votes_file: Path to the council votes JSON file
        """
        self.votes_file = votes_file or COUNCIL_VOTES_FILE
        self.votes_file.parent.mkdir(parents=True, exist_ok=True)
        self._scores: Dict[str, CouncilModelScore] = {}
        self._lock = threading.Lock()
        self._load_votes()
        
    def _load_votes(self) -> None:
        """Load votes from file."""
        if not self.votes_file.exists():
            logger.info("No council votes file found, starting fresh")
            return
            
        try:
            with open(self.votes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            scores_data = data.get("scores", {})
            for model_id, score_dict in scores_data.items():
                self._scores[model_id] = CouncilModelScore(
                    model_id=model_id,
                    model_name=score_dict.get("model_name", model_id),
                    votes_received=score_dict.get("votes_received", 0),
                    times_participated=score_dict.get("times_participated", 0)
                )
                
            logger.info(f"Loaded council votes for {len(self._scores)} models")
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load council votes: {e}")
            self._scores = {}
    
    def _save_votes(self) -> None:
        """Save votes to file."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "scores": {
                    model_id: {
                        "model_name": score.model_name,
                        "votes_received": score.votes_received,
                        "times_participated": score.times_participated
                    }
                    for model_id, score in self._scores.items()
                }
            }
            
            with open(self.votes_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved council votes for {len(self._scores)} models")
            
        except IOError as e:
            logger.error(f"Failed to save council votes: {e}")
    
    def record_council_vote(self, voted_for_model_id: str, voted_for_model_name: Optional[str] = None) -> None:
        """
        Record a vote that a model received from another model in council voting.
        
        Args:
            voted_for_model_id: The model ID that received the vote
            voted_for_model_name: Human-readable model name
        """
        with self._lock:
            if voted_for_model_id not in self._scores:
                self._scores[voted_for_model_id] = CouncilModelScore(
                    model_id=voted_for_model_id,
                    model_name=voted_for_model_name or voted_for_model_id
                )
            
            self._scores[voted_for_model_id].votes_received += 1
            
            # Update name if provided
            if voted_for_model_name:
                self._scores[voted_for_model_id].model_name = voted_for_model_name
                
            self._save_votes()
            logger.info(f"Recorded council vote for {voted_for_model_id}")
    
    def record_participation(self, model_ids: List[str], model_names: Dict[str, str]) -> None:
        """
        Record that models participated in a council voting session.
        
        Args:
            model_ids: List of model IDs that participated
            model_names: Dict mapping model_id to model_name
        """
        with self._lock:
            for model_id in model_ids:
                model_name = model_names.get(model_id, model_id)
                
                if model_id not in self._scores:
                    self._scores[model_id] = CouncilModelScore(
                        model_id=model_id,
                        model_name=model_name
                    )
                
                self._scores[model_id].times_participated += 1
                
                # Update name if we have it
                if model_name:
                    self._scores[model_id].model_name = model_name
                    
            self._save_votes()
            logger.info(f"Recorded participation for {len(model_ids)} models")
    
    def get_score(self, model_id: str) -> Optional[CouncilModelScore]:
        """Get the score for a specific model."""
        return self._scores.get(model_id)
    
    def get_leaderboard(self, sort_by: str = "votes_received") -> List[CouncilModelScore]:
        """
        Get all models sorted by score.
        
        Args:
            sort_by: Field to sort by ('votes_received', 'times_participated', 'win_rate')
            
        Returns:
            List of CouncilModelScore objects sorted by the specified field
        """
        scores = list(self._scores.values())
        
        if sort_by == "votes_received":
            scores.sort(key=lambda s: s.votes_received, reverse=True)
        elif sort_by == "times_participated":
            scores.sort(key=lambda s: s.times_participated, reverse=True)
        elif sort_by == "win_rate":
            scores.sort(key=lambda s: s.win_rate, reverse=True)
            
        return scores
    
    def get_top_models(self, n: int = 10) -> List[CouncilModelScore]:
        """Get the top N models by votes received."""
        return self.get_leaderboard("votes_received")[:n]
    
    def reset_votes(self) -> None:
        """Reset all council votes (use with caution)."""
        with self._lock:
            self._scores = {}
            self._save_votes()
            logger.warning("All council votes have been reset")
    
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
                    "votes_received": s.votes_received,
                    "times_participated": s.times_participated,
                    "win_rate": round(s.win_rate, 1)
                }
                for i, s in enumerate(leaderboard)
            ]
        }


# Singleton instance for app-wide use
_council_leaderboard_storage: Optional[CouncilLeaderboardStorage] = None


def get_council_leaderboard_storage() -> CouncilLeaderboardStorage:
    """Get the singleton council leaderboard storage instance."""
    global _council_leaderboard_storage
    if _council_leaderboard_storage is None:
        _council_leaderboard_storage = CouncilLeaderboardStorage()
    return _council_leaderboard_storage
