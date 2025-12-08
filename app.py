"""
LLM Council V2 - AI Model Comparison App
Main Streamlit application entry point.
"""

import os
from dotenv import load_dotenv
import streamlit as st
from typing import List, Optional

# Import local modules
from src.openrouter_client import validate_api_key
from src.model_metadata import get_metadata_manager, ModelInfo
from src.leaderboard_storage import get_leaderboard_storage
from src.council_leaderboard import get_council_leaderboard_storage
from src.cache_manager import get_cache_manager
from src.compare_logic import run_comparison, run_comparison_with_voting, ComparisonSession
from src.voting_logic import VotingSession, get_voting_winner
from src.ui_components import (
    inject_custom_css,
    render_model_selector,
    render_results_grid,
    render_leaderboard_table,
    render_settings_sidebar,
    get_color
)

# Page configuration
st.set_page_config(
    page_title="LLM Council - AI Comparison",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    # Check for environment variable
    load_dotenv(".env")
    env_api_key = os.environ.get("OPENROUTER_API_KEY", "")
    
    defaults = {
        "api_key": "",
        "use_env_api_key": bool(env_api_key),  # Default to env var if available
        "env_api_key_available": bool(env_api_key),
        "temperature": 0.7,
        "cache_enabled": True,
        "council_voting_enabled": True,  # Enable council voting by default
        "model_slots": ["", ""],  # Start with 2 empty slots
        "comparison_results": None,
        "voted_this_session": set(),
        "models_loaded": False,
        "force_refresh_models": False,
        "model_filter": "",
        "current_page": "compare",
        "api_key_error": None,
        "anonymous_mode": True  # Hide model names by default for unbiased evaluation
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_effective_api_key() -> str:
    """Get the API key to use based on current settings."""
    if st.session_state.use_env_api_key:
        return os.environ.get("OPENROUTER_API_KEY", "")
    return st.session_state.api_key


def load_models(api_key: str = None) -> bool:
    """Load models from OpenRouter API."""
    effective_key = api_key or get_effective_api_key()
    if not effective_key:
        return False
    
    metadata = get_metadata_manager()
    force_refresh = st.session_state.get("force_refresh_models", False)
    
    success = metadata.fetch_models(
        api_key=effective_key,
        force_refresh=force_refresh
    )
    
    if success:
        st.session_state.models_loaded = True
        st.session_state.force_refresh_models = False
    
    return success


def add_model_slot():
    """Add a new model slot."""
    st.session_state.model_slots.append("")


def remove_model_slot(index: int):
    """Remove a model slot by index."""
    if len(st.session_state.model_slots) > 1:
        st.session_state.model_slots.pop(index)
        st.rerun()


def update_model_slot(index: int, model_id: str):
    """Update a model slot with the selected model."""
    if index < len(st.session_state.model_slots):
        st.session_state.model_slots[index] = model_id


def handle_vote_best(model_id: str, model_name: str):
    """Handle a 'best' vote."""
    storage = get_leaderboard_storage()
    storage.vote_best(model_id, model_name)
    st.session_state.voted_this_session.add(model_id)
    st.success(f"✅ Voted '{model_name}' as best!")


def handle_vote_worst(model_id: str, model_name: str):
    """Handle a 'worst' vote."""
    storage = get_leaderboard_storage()
    storage.vote_worst(model_id, model_name)
    st.session_state.voted_this_session.add(model_id)
    st.warning(f"Voted '{model_name}' as worst")


def render_compare_page():
    """Render the main comparison page."""
    st.title("⚖️ LLM Council")
    st.markdown("Compare AI model responses side by side")
    
    # Show API key error if there was one from a previous attempt
    if st.session_state.api_key_error:
        st.error(st.session_state.api_key_error)
        st.session_state.api_key_error = None
    
    # Get effective API key
    effective_key = get_effective_api_key()
    
    # Try to load models if we have an API key
    metadata = get_metadata_manager()
    all_models = metadata.get_all_models()
    
    if not all_models and effective_key:
        # Try to load models
        if not st.session_state.models_loaded:
            with st.spinner("Loading models from OpenRouter..."):
                if not load_models(effective_key):
                    st.warning("⚠️ Failed to load models. Please check your API key in the sidebar.")
        all_models = metadata.get_all_models()
    
    # Show info if no models available (but don't block UI)
    models_available = bool(all_models)
    
    if not models_available:
        st.info("💡 **To load the model list:** Enter your OpenRouter API key in the sidebar, or click the button below to use your environment variable if available.")
        
        # Offer to load models with env var if available
        env_key = os.environ.get("OPENROUTER_API_KEY", "")
        if env_key and not effective_key:
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🔑 Use Environment API Key", use_container_width=True):
                    st.session_state.use_env_api_key = True
                    with st.spinner("Loading models..."):
                        if load_models(env_key):
                            st.rerun()
                        else:
                            st.error("Failed to load models. Check your OPENROUTER_API_KEY environment variable.")
            with col2:
                st.caption("✓ Detected `OPENROUTER_API_KEY` environment variable")
    
    # Model filter
    if models_available:
        st.text_input(
            "🔍 Filter models",
            key="model_filter",
            placeholder="Type to filter models by name...",
            help="Filter the model dropdown by name"
        )
        
        filter_text = st.session_state.model_filter
        
        # Create a dict for quick model lookup by ID
        all_models_map = {m.id: m for m in all_models}
        
        # Filter models for display
        if filter_text:
            filtered_models = [
                m for m in all_models
                if filter_text.lower() in m.id.lower() or filter_text.lower() in m.name.lower()
            ]
        else:
            filtered_models = all_models
        
        # Sort by name
        filtered_models.sort(key=lambda m: m.name.lower())
        
        st.markdown(f"**{len(filtered_models)}** models available")
    else:
        filter_text = ""
        all_models_map = {}
        filtered_models = []
    
    # Model slots section
    st.subheader("Select Models to Compare")
    
    for i, slot_value in enumerate(st.session_state.model_slots):
        color = get_color(i)
        
        col1, col2 = st.columns([11, 1])
        
        with col1:
            st.markdown(
                f'<div style="width: 100%; height: 4px; background-color: {color["border"]}; '
                f'border-radius: 2px; margin-bottom: 5px;"></div>',
                unsafe_allow_html=True
            )
            
            if models_available:
                # Dropdown mode - we have models loaded
                # Build options list - include currently selected model even if not in filter
                slot_models = list(filtered_models)
                currently_selected_model = None
                
                if slot_value and slot_value in all_models_map:
                    currently_selected_model = all_models_map[slot_value]
                    # Add to list if not already there (filtered out)
                    if currently_selected_model not in slot_models:
                        slot_models.insert(0, currently_selected_model)
                
                options = ["-- Select a model --"] + [m.format_display_name() for m in slot_models]
                model_map = {m.format_display_name(): m.id for m in slot_models}
                
                # Find current index
                current_index = 0
                if currently_selected_model:
                    display_name = currently_selected_model.format_display_name()
                    if display_name in options:
                        current_index = options.index(display_name)
                
                selected = st.selectbox(
                    f"Model {i + 1}",
                    options=options,
                    index=current_index,
                    key=f"model_select_{i}",
                    label_visibility="collapsed"
                )
                
                if selected != "-- Select a model --":
                    new_model_id = model_map.get(selected, "")
                    if new_model_id != slot_value:
                        st.session_state.model_slots[i] = new_model_id
                elif selected == "-- Select a model --" and slot_value:
                    # User explicitly deselected - clear the slot
                    st.session_state.model_slots[i] = ""
            else:
                # No models loaded - show disabled placeholder
                st.selectbox(
                    f"Model {i + 1}",
                    options=["-- Load models first --"],
                    disabled=True,
                    key=f"model_select_disabled_{i}",
                    label_visibility="collapsed"
                )
        
        with col2:
            if len(st.session_state.model_slots) > 1:
                if st.button("✕", key=f"remove_{i}", help="Remove this model"):
                    remove_model_slot(i)
    
    # Add model button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("➕ Add Model", use_container_width=True, disabled=not models_available):
            add_model_slot()
            st.rerun()
    
    st.divider()
    
    # Prompt input
    st.subheader("Enter Your Prompt")
    
    prompt = st.text_area(
        "Prompt",
        height=150,
        placeholder="Enter your prompt here... All selected models will receive this same prompt.",
        label_visibility="collapsed"
    )
    
    # Get selected model IDs
    selected_models = [m for m in st.session_state.model_slots if m]
    
    # Submit button
    col1, col2 = st.columns([1, 3])
    with col1:
        submit = st.button(
            "🚀 Run Comparison",
            type="primary",
            use_container_width=True,
            disabled=not prompt or len(selected_models) < 1
        )
    
    if submit and prompt and selected_models:
        # Check for API key before running
        effective_key = get_effective_api_key()
        if not effective_key:
            st.session_state.api_key_error = "⚠️ Please enter your OpenRouter API key in the sidebar before running a comparison."
            st.rerun()
        else:
            # Reset votes for new comparison
            st.session_state.voted_this_session = set()
            
            # Determine if we should run voting (2+ models and voting enabled)
            run_voting = len(selected_models) >= 2 and st.session_state.council_voting_enabled
            
            # Run comparison
            with st.spinner(f"Running comparison across {len(selected_models)} models..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(completed):
                    progress_bar.progress(completed / len(selected_models))
                    status_text.text(f"Getting responses: {completed}/{len(selected_models)}")
                
                cache_mgr = get_cache_manager()
                cache_mgr.set_enabled(st.session_state.cache_enabled)
                
                if run_voting:
                    # Two-step process: comparison + voting
                    def update_voting_progress(completed):
                        progress_bar.progress(completed / len(selected_models))
                        status_text.text(f"Council voting: {completed}/{len(selected_models)}")
                    
                    session = run_comparison_with_voting(
                        api_key=effective_key,
                        prompt=prompt,
                        model_ids=selected_models,
                        temperature=st.session_state.temperature,
                        use_cache=st.session_state.cache_enabled,
                        progress_callback=update_progress,
                        voting_progress_callback=update_voting_progress
                    )
                else:
                    # Single step: just comparison
                    session = run_comparison(
                        api_key=effective_key,
                        prompt=prompt,
                        model_ids=selected_models,
                        temperature=st.session_state.temperature,
                        use_cache=st.session_state.cache_enabled,
                        progress_callback=update_progress
                    )
                
                st.session_state.comparison_results = session
                progress_bar.empty()
                status_text.empty()
    
    # Display results
    if st.session_state.comparison_results:
        session: ComparisonSession = st.session_state.comparison_results
        
        st.divider()
        st.subheader("Results")
        
        # Anonymous mode toggle
        anon_col1, anon_col2 = st.columns([1, 3])
        with anon_col1:
            show_names = st.toggle(
                "🔓 Reveal model names",
                value=not st.session_state.anonymous_mode,
                help="Toggle to show/hide model names for unbiased evaluation"
            )
            st.session_state.anonymous_mode = not show_names
        with anon_col2:
            if st.session_state.anonymous_mode:
                st.caption("🎭 Anonymous mode: Model names hidden for unbiased voting")
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Successful", session.successful_count)
        with col2:
            st.metric("Failed", session.failed_count)
        with col3:
            avg_latency = session.total_latency_ms / max(session.successful_count, 1)
            st.metric("Avg Latency", f"{avg_latency:.0f}ms")
        
        # Determine columns based on result count
        num_results = len(session.results)
        if num_results <= 2:
            columns = num_results
        elif num_results <= 4:
            columns = 2
        else:
            columns = 3
        
        # Check which models already voted
        voted = st.session_state.voted_this_session
        
        # Render results
        render_results_grid(
            results=session.results,
            columns=columns,
            on_vote_best=handle_vote_best,
            on_vote_worst=handle_vote_worst,
            votes_disabled=False,
            anonymous_mode=st.session_state.anonymous_mode
        )
        
        # Show voted models
        if voted:
            st.info(f"You've voted on: {', '.join(voted)}")
        
        # Show Council Voting Results if available
        if session.voting_enabled and session.voting_session:
            render_voting_results(session.voting_session, st.session_state.anonymous_mode)


def render_voting_results(voting_session: VotingSession, anonymous_mode: bool = True):
    """Render the council voting results section."""
    st.divider()
    st.subheader("🗳️ Council Voting Results")
    st.caption("Each model voted for the response they consider best (anonymized)")
    
    # Get winner info
    winner = get_voting_winner(voting_session)
    
    if not winner:
        st.warning("No valid votes were cast.")
        return
    
    # Display winner announcement
    if winner["is_tie"]:
        tied_count = len(winner["tied_labels"])
        st.info(f"🤝 **It's a tie!** {tied_count} responses received {winner['vote_count']} votes each.")
    else:
        winner_label = winner["winner_label"]
        winner_model_id = winner["winner_model_id"]
        
        # Find the model name from the voting session
        winner_name = None
        for result in voting_session.voting_results:
            if result.voted_for_model_id == winner_model_id:
                winner_name = result.voted_for_model_name
                break
        
        if anonymous_mode:
            st.success(f"🏆 **Council Winner: Response {winner_label}** with {winner['vote_count']}/{winner['total_votes']} votes")
        else:
            display_name = winner_name or winner_model_id
            st.success(f"🏆 **Council Winner: {display_name}** (Response {winner_label}) with {winner['vote_count']}/{winner['total_votes']} votes")
    
    # Vote breakdown
    st.markdown("#### Vote Breakdown")
    
    # Create columns for vote counts
    labels = list(voting_session.vote_counts.keys())
    cols = st.columns(len(labels))
    
    for i, label in enumerate(labels):
        vote_count = voting_session.vote_counts[label]
        model_id = voting_session.label_to_model.get(label, "Unknown")
        
        with cols[i]:
            # Find model name
            model_name = None
            for result in voting_session.voting_results:
                if result.voted_for_model_id == model_id:
                    model_name = result.voted_for_model_name
                    break
            
            if anonymous_mode:
                st.metric(f"Response {label}", f"{vote_count} votes")
            else:
                display_name = model_name or model_id
                # Truncate long names
                if len(display_name) > 20:
                    display_name = display_name[:17] + "..."
                st.metric(display_name, f"{vote_count} votes", help=f"Response {label}")
    
    # Show individual votes in an expander
    with st.expander("📋 See how each model voted"):
        for result in voting_session.voting_results:
            if result.success:
                voter = result.voter_model_name if not anonymous_mode else f"Voter (hidden)"
                voted_for = result.voted_for_label
                voted_for_name = result.voted_for_model_name if not anonymous_mode else f"Response {voted_for}"
                
                if anonymous_mode:
                    st.markdown(f"• A model voted for **Response {voted_for}**")
                else:
                    st.markdown(f"• **{voter}** voted for **{voted_for_name}** (Response {voted_for})")
            else:
                voter = result.voter_model_name if not anonymous_mode else "A model"
                st.markdown(f"• ❌ {voter} - Vote failed: {result.error}")


def render_leaderboard_page():
    """Render the user voting leaderboard page."""
    st.title("🏆 User Voting Leaderboard")
    st.markdown("Rankings based on **your manual votes** across comparison sessions")
    
    storage = get_leaderboard_storage()
    leaderboard_data = storage.export_leaderboard()
    
    # Sort options
    col1, col2 = st.columns([2, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["Net Score", "Best Votes", "Worst Votes", "Total Votes"],
            label_visibility="collapsed"
        )
    
    sort_map = {
        "Net Score": "net_score",
        "Best Votes": "best_votes",
        "Worst Votes": "worst_votes",
        "Total Votes": "total_votes"
    }
    
    # Re-fetch with new sort
    sorted_leaderboard = storage.get_leaderboard(sort_by=sort_map[sort_by])
    leaderboard_data["leaderboard"] = [
        {
            "rank": i + 1,
            "model_id": s.model_id,
            "model_name": s.model_name,
            "net_score": s.net_score,
            "best_votes": s.best_votes,
            "worst_votes": s.worst_votes,
            "total_votes": s.total_votes
        }
        for i, s in enumerate(sorted_leaderboard)
    ]
    
    render_leaderboard_table(leaderboard_data)
    
    # Reset button
    st.divider()
    
    with st.expander("⚠️ Danger Zone"):
        st.warning("This will permanently delete all user vote data.")
        if st.button("Reset All Votes", type="secondary"):
            storage.reset_votes()
            st.success("All votes have been reset!")
            st.rerun()


def render_council_leaderboard_page():
    """Render the council voting leaderboard page."""
    st.title("🗳️ Council Voting Leaderboard")
    st.markdown("Rankings based on **model-to-model votes** - when models vote for each other's responses")
    
    storage = get_council_leaderboard_storage()
    leaderboard_data = storage.export_leaderboard()
    
    entries = leaderboard_data.get("leaderboard", [])
    
    if not entries:
        st.info("No council votes recorded yet. Run a comparison with 2+ models and Council Voting enabled to start building the leaderboard!")
        return
    
    # Sort options
    col1, col2 = st.columns([2, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["Votes Received", "Times Participated", "Win Rate"],
            label_visibility="collapsed"
        )
    
    sort_map = {
        "Votes Received": "votes_received",
        "Times Participated": "times_participated",
        "Win Rate": "win_rate"
    }
    
    # Re-fetch with new sort
    sorted_leaderboard = storage.get_leaderboard(sort_by=sort_map[sort_by])
    
    st.markdown(f"**{len(sorted_leaderboard)} models ranked** | Last updated: {leaderboard_data.get('last_updated', 'Unknown')[:10]}")
    
    # Table header
    cols = st.columns([1, 4, 2, 2, 2])
    with cols[0]:
        st.markdown("**Rank**")
    with cols[1]:
        st.markdown("**Model**")
    with cols[2]:
        st.markdown("**Votes Received**")
    with cols[3]:
        st.markdown("**Sessions**")
    with cols[4]:
        st.markdown("**Win Rate**")
    
    st.divider()
    
    # Table rows
    for i, score in enumerate(sorted_leaderboard):
        rank = i + 1
        
        cols = st.columns([1, 4, 2, 2, 2])
        
        with cols[0]:
            # Medal for top 3
            if rank == 1:
                st.markdown("🥇")
            elif rank == 2:
                st.markdown("🥈")
            elif rank == 3:
                st.markdown("🥉")
            else:
                st.markdown(f"#{rank}")
        
        with cols[1]:
            st.markdown(score.model_name)
        
        with cols[2]:
            st.markdown(f"**{score.votes_received}**")
        
        with cols[3]:
            st.markdown(str(score.times_participated))
        
        with cols[4]:
            win_rate = score.win_rate
            if win_rate >= 50:
                st.markdown(f"🟢 **{win_rate:.1f}%**")
            elif win_rate >= 25:
                st.markdown(f"🟡 **{win_rate:.1f}%**")
            else:
                st.markdown(f"⚪ **{win_rate:.1f}%**")
    
    # Reset button
    st.divider()
    
    with st.expander("⚠️ Danger Zone"):
        st.warning("This will permanently delete all council vote data.")
        if st.button("Reset Council Votes", type="secondary"):
            storage.reset_votes()
            st.success("All council votes have been reset!")
            st.rerun()


def main():
    """Main application entry point."""
    # Initialize
    init_session_state()
    inject_custom_css()
    
    # Navigation
    st.sidebar.header("📍 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Compare Models", "User Leaderboard", "Council Leaderboard"],
        label_visibility="collapsed"
    )

    st.sidebar.divider()

    # Sidebar settings
    new_api_key, new_temperature, new_cache_enabled, new_use_env, new_council_voting = render_settings_sidebar(
        api_key=st.session_state.api_key,
        temperature=st.session_state.temperature,
        cache_enabled=st.session_state.cache_enabled,
        use_env_api_key=st.session_state.use_env_api_key,
        env_api_key_available=st.session_state.env_api_key_available,
        council_voting_enabled=st.session_state.council_voting_enabled
    )
    
    # Update settings
    if new_api_key != st.session_state.api_key:
        st.session_state.api_key = new_api_key
        st.session_state.models_loaded = False
    
    if new_use_env != st.session_state.use_env_api_key:
        st.session_state.use_env_api_key = new_use_env
        st.session_state.models_loaded = False
    
    st.session_state.temperature = new_temperature
    st.session_state.cache_enabled = new_cache_enabled
    st.session_state.council_voting_enabled = new_council_voting
    
    # Sidebar info
    st.sidebar.divider()
    
    # Show cache stats (always visible now)
    cache_mgr = get_cache_manager()
    stats = cache_mgr.get_stats()
    st.sidebar.caption(f"📦 Cache: {stats['total_entries']} entries")
    
    # Show vote stats
    storage = get_leaderboard_storage()
    leaderboard = storage.get_leaderboard()
    total_votes = sum(s.total_votes for s in leaderboard)
    st.sidebar.caption(f"👤 User votes: {total_votes}")
    
    # Show council vote stats
    council_storage = get_council_leaderboard_storage()
    council_leaderboard = council_storage.get_leaderboard()
    total_council_votes = sum(s.votes_received for s in council_leaderboard)
    st.sidebar.caption(f"🗳️ Council votes: {total_council_votes}")
    
    # Render selected page
    if page == "Compare Models":
        render_compare_page()
    elif page == "User Leaderboard":
        render_leaderboard_page()
    elif page == "Council Leaderboard":
        render_council_leaderboard_page()


if __name__ == "__main__":
    main()
