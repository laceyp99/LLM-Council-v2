"""
Voting Logic Module
Handles the second step where models vote on the best response from the initial comparison.
"""

import concurrent.futures
import random
import string
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re

from src.openrouter_client import OpenRouterClient, ModelResponse
from src.model_metadata import get_metadata_manager
from src.council_leaderboard import get_council_leaderboard_storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VotingResult:
    """Result from a single model's vote."""
    voter_model_id: str
    voter_model_name: str
    voted_for_label: str  # e.g., "A", "B", "C"
    voted_for_model_id: Optional[str] = None  # Resolved after voting
    voted_for_model_name: Optional[str] = None  # Resolved after voting
    explanation: Optional[str] = None
    latency_ms: float = 0
    error: Optional[str] = None
    success: bool = True


@dataclass
class VotingSession:
    """Container for the complete voting session."""
    original_prompt: str
    timestamp: str
    label_to_model: Dict[str, str] = field(default_factory=dict)  # e.g., {"A": "openai/gpt-4", ...}
    label_to_response: Dict[str, str] = field(default_factory=dict)  # e.g., {"A": "response text", ...}
    voting_results: List[VotingResult] = field(default_factory=list)
    vote_counts: Dict[str, int] = field(default_factory=dict)  # e.g., {"A": 3, "B": 1}
    total_votes: int = 0
    successful_votes: int = 0
    failed_votes: int = 0


def generate_option_labels(count: int) -> List[str]:
    """Generate option labels A, B, C, ... for the given count."""
    return list(string.ascii_uppercase[:count])


def build_voting_prompt(
    original_prompt: str,
    responses: Dict[str, str],  # label -> response text
    labels: List[str]
) -> str:
    """
    Build the voting prompt that asks models to choose the best response.
    
    Args:
        original_prompt: The original user prompt
        responses: Dictionary mapping labels to response texts
        labels: List of labels in order (e.g., ["A", "B", "C"])
        
    Returns:
        The formatted voting prompt
    """
    options_text = "\n\n".join([
        f"=== Response {label} ===\n{responses[label]}"
        for label in labels
    ])
    
    label_list = ", ".join(labels[:-1]) + f", or {labels[-1]}" if len(labels) > 1 else labels[0]
    
    voting_prompt = f"""You are evaluating responses from different AI assistants to the following prompt:

--- ORIGINAL PROMPT ---
{original_prompt}
--- END ORIGINAL PROMPT ---

Below are the responses from different assistants, labeled {label_list}:

{options_text}

=== YOUR TASK ===
Analyze each response and choose the BEST one. Consider:
1. Accuracy and correctness
2. Completeness of the answer
3. Clarity and helpfulness
4. Following the original prompt's instructions

You MUST respond with ONLY a single letter ({label_list}) representing your choice for the best response.
Do not include any explanation, just the letter.

Your choice:"""
    
    return voting_prompt


def parse_vote(response_text: str, valid_labels: List[str]) -> Optional[str]:
    """
    Parse the model's response to extract the voted label.
    
    Args:
        response_text: The model's response
        valid_labels: List of valid labels (e.g., ["A", "B", "C"])
        
    Returns:
        The voted label or None if parsing failed
    """
    if not response_text:
        return None
    
    # Clean up the response
    cleaned = response_text.strip().upper()
    
    # First, try to match just the letter
    if cleaned in valid_labels:
        return cleaned
    
    # Try to find a letter at the start
    if cleaned and cleaned[0] in valid_labels:
        return cleaned[0]
    
    # Try to find "Response X" or "Option X" pattern
    for label in valid_labels:
        patterns = [
            rf'\b{label}\b',  # Just the letter as a word
            rf'response\s*{label}',  # "Response A"
            rf'option\s*{label}',  # "Option A"
            rf'choice[:\s]*{label}',  # "Choice: A"
        ]
        for pattern in patterns:
            if re.search(pattern, cleaned, re.IGNORECASE):
                return label
    
    # Look for explicit statements like "I choose A" or "The best is A"
    for label in valid_labels:
        patterns = [
            rf'choose\s*{label}',
            rf'select\s*{label}',
            rf'pick\s*{label}',
            rf'best\s+(?:is\s+)?{label}',
            rf'vote\s+(?:for\s+)?{label}',
        ]
        for pattern in patterns:
            if re.search(pattern, cleaned, re.IGNORECASE):
                return label
    
    return None


def run_single_vote(
    client: OpenRouterClient,
    voter_model_id: str,
    voter_model_name: str,
    voting_prompt: str,
    valid_labels: List[str],
    temperature: float = 0.3  # Lower temperature for more deterministic voting
) -> VotingResult:
    """
    Have a single model vote on the best response.
    
    Args:
        client: OpenRouter client instance
        voter_model_id: The model ID that is voting
        voter_model_name: Human-readable model name
        voting_prompt: The formatted voting prompt
        valid_labels: List of valid vote labels
        temperature: Sampling temperature (lower for consistency)
        
    Returns:
        VotingResult with the vote or error
    """
    # Make API call
    response = client.chat_completion(
        model_id=voter_model_id,
        prompt=voting_prompt,
        temperature=temperature,
        max_tokens=50  # We only need a single letter response
    )
    
    if not response.success:
        return VotingResult(
            voter_model_id=voter_model_id,
            voter_model_name=voter_model_name,
            voted_for_label="",
            latency_ms=response.latency_ms,
            error=response.error,
            success=False
        )
    
    # Parse the vote
    voted_label = parse_vote(response.output_text, valid_labels)
    
    if not voted_label:
        return VotingResult(
            voter_model_id=voter_model_id,
            voter_model_name=voter_model_name,
            voted_for_label="",
            explanation=response.output_text,
            latency_ms=response.latency_ms,
            error=f"Could not parse vote from response: {response.output_text[:100]}",
            success=False
        )
    
    return VotingResult(
        voter_model_id=voter_model_id,
        voter_model_name=voter_model_name,
        voted_for_label=voted_label,
        explanation=response.output_text,
        latency_ms=response.latency_ms,
        success=True
    )


def run_voting_session(
    api_key: str,
    original_prompt: str,
    model_responses: Dict[str, str],  # model_id -> response_text
    model_names: Dict[str, str],  # model_id -> model_name
    temperature: float = 0.3,
    max_workers: int = 5,
    progress_callback: Optional[callable] = None
) -> VotingSession:
    """
    Run the voting session where all models vote on the best response.
    
    Args:
        api_key: OpenRouter API key
        original_prompt: The original user prompt
        model_responses: Dictionary mapping model IDs to their response texts
        model_names: Dictionary mapping model IDs to their display names
        temperature: Sampling temperature for voting
        max_workers: Maximum concurrent API calls
        progress_callback: Optional callback for progress updates
        
    Returns:
        VotingSession with all voting results
    """
    client = OpenRouterClient(api_key=api_key)
    
    # Generate labels and create mappings
    model_ids = list(model_responses.keys())
    labels = generate_option_labels(len(model_ids))
    
    # Shuffle for unbiased presentation (models won't know which response is which)
    shuffled_indices = list(range(len(model_ids)))
    random.shuffle(shuffled_indices)
    
    label_to_model = {}
    label_to_response = {}
    
    for i, idx in enumerate(shuffled_indices):
        label = labels[i]
        model_id = model_ids[idx]
        label_to_model[label] = model_id
        label_to_response[label] = model_responses[model_id]
    
    # Build the voting prompt
    voting_prompt = build_voting_prompt(
        original_prompt=original_prompt,
        responses=label_to_response,
        labels=labels
    )
    
    session = VotingSession(
        original_prompt=original_prompt,
        timestamp=datetime.now().isoformat(),
        label_to_model=label_to_model,
        label_to_response=label_to_response,
        vote_counts={label: 0 for label in labels}
    )
    
    completed = 0
    
    # Run voting in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(
                run_single_vote,
                client,
                model_id,
                model_names.get(model_id, model_id),
                voting_prompt,
                labels,
                temperature
            ): model_id
            for model_id in model_ids
        }
        
        for future in concurrent.futures.as_completed(future_to_model):
            model_id = future_to_model[future]
            try:
                result = future.result()
                
                # Resolve voted-for model info
                if result.success and result.voted_for_label:
                    voted_model_id = label_to_model.get(result.voted_for_label)
                    result.voted_for_model_id = voted_model_id
                    result.voted_for_model_name = model_names.get(voted_model_id, voted_model_id)
                    session.vote_counts[result.voted_for_label] += 1
                    session.successful_votes += 1
                else:
                    session.failed_votes += 1
                
                session.voting_results.append(result)
                session.total_votes += 1
                
            except Exception as e:
                logger.error(f"Exception for voting by {model_id}: {e}")
                session.voting_results.append(VotingResult(
                    voter_model_id=model_id,
                    voter_model_name=model_names.get(model_id, model_id),
                    voted_for_label="",
                    error=str(e),
                    success=False
                ))
                session.failed_votes += 1
                session.total_votes += 1
            
            completed += 1
            if progress_callback:
                progress_callback(completed)
    
    # Record votes to the council leaderboard
    _record_council_votes(session, model_names)
    
    logger.info(f"Voting complete: {session.successful_votes} succeeded, {session.failed_votes} failed")
    
    return session


def _record_council_votes(session: VotingSession, model_names: Dict[str, str]) -> None:
    """
    Record the voting results to the council leaderboard.
    
    Args:
        session: The completed voting session
        model_names: Dictionary mapping model IDs to display names
    """
    council_storage = get_council_leaderboard_storage()
    
    # Record participation for all models that were in the voting session
    participating_model_ids = list(session.label_to_model.values())
    council_storage.record_participation(participating_model_ids, model_names)
    
    # Record each successful vote
    for result in session.voting_results:
        if result.success and result.voted_for_model_id:
            council_storage.record_council_vote(
                voted_for_model_id=result.voted_for_model_id,
                voted_for_model_name=result.voted_for_model_name
            )
    
    logger.info(f"Recorded {session.successful_votes} council votes to leaderboard")


def get_voting_winner(session: VotingSession) -> Optional[Dict[str, Any]]:
    """
    Get the winning response from a voting session.
    
    Args:
        session: The completed voting session
        
    Returns:
        Dictionary with winner info or None if no clear winner
    """
    if not session.vote_counts or session.successful_votes == 0:
        return None
    
    # Find the label(s) with most votes
    max_votes = max(session.vote_counts.values())
    if max_votes == 0:
        return None
    
    winners = [
        label for label, count in session.vote_counts.items()
        if count == max_votes
    ]
    
    # If there's a tie, report it
    if len(winners) > 1:
        return {
            "is_tie": True,
            "tied_labels": winners,
            "tied_model_ids": [session.label_to_model[l] for l in winners],
            "vote_count": max_votes,
            "total_votes": session.successful_votes
        }
    
    winner_label = winners[0]
    winner_model_id = session.label_to_model[winner_label]
    
    return {
        "is_tie": False,
        "winner_label": winner_label,
        "winner_model_id": winner_model_id,
        "vote_count": max_votes,
        "total_votes": session.successful_votes
    }
