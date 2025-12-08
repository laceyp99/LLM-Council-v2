"""
UI Components Module
Reusable Streamlit UI components with styling for the comparison app.
"""

import streamlit as st
from typing import List, Optional, Callable
from src.compare_logic import ComparisonResult
from src.model_metadata import ModelInfo

# Color palette for model cards (tints and borders)
MODEL_COLORS = [
    {"bg": "#e3f2fd", "border": "#1976d2", "text": "#0d47a1"},  # Blue
    {"bg": "#f3e5f5", "border": "#7b1fa2", "text": "#4a148c"},  # Purple
    {"bg": "#e8f5e9", "border": "#388e3c", "text": "#1b5e20"},  # Green
    {"bg": "#fff3e0", "border": "#f57c00", "text": "#e65100"},  # Orange
    {"bg": "#fce4ec", "border": "#c2185b", "text": "#880e4f"},  # Pink
    {"bg": "#e0f7fa", "border": "#0097a7", "text": "#006064"},  # Cyan
    {"bg": "#fff8e1", "border": "#ffa000", "text": "#ff6f00"},  # Amber
    {"bg": "#f1f8e9", "border": "#689f38", "text": "#33691e"},  # Light Green
    {"bg": "#ede7f6", "border": "#512da8", "text": "#311b92"},  # Deep Purple
    {"bg": "#e1f5fe", "border": "#0288d1", "text": "#01579b"},  # Light Blue
]

TRUNCATE_LENGTH = 200  # Characters to show in collapsed preview


def get_color(index: int) -> dict:
    """Get color scheme for a model index."""
    return MODEL_COLORS[index % len(MODEL_COLORS)]


def inject_custom_css():
    """Inject custom CSS for styling."""
    st.markdown("""
    <style>
    .model-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-width: 2px;
        border-style: solid;
    }
    
    .model-header {
        font-size: 1.1em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .model-stats {
        font-size: 0.85em;
        color: #666;
        margin-bottom: 10px;
    }
    
    .model-output {
        background-color: rgba(255,255,255,0.7);
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    
    .copy-text {
        font-family: monospace;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .error-card {
        background-color: #ffebee;
        border-color: #c62828;
    }
    
    .cache-badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
    }
    
    .vote-button {
        margin-right: 5px;
    }
    
    .leaderboard-row {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    
    .leaderboard-positive {
        background-color: #e8f5e9;
    }
    
    .leaderboard-negative {
        background-color: #ffebee;
    }
    
    .leaderboard-neutral {
        background-color: #f5f5f5;
    }
    
    .truncated-text {
        color: #666;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)


def render_model_selector(
    models: List[ModelInfo],
    key: str,
    selected_id: Optional[str] = None,
    filter_text: str = ""
) -> Optional[str]:
    """
    Render a model selector dropdown with metadata.
    
    Args:
        models: List of available models
        key: Unique key for the selectbox
        selected_id: Pre-selected model ID
        filter_text: Text to filter models by
        
    Returns:
        Selected model ID or None
    """
    # Filter models if filter text provided
    if filter_text:
        filter_lower = filter_text.lower()
        filtered_models = [
            m for m in models
            if filter_lower in m.id.lower() or filter_lower in m.name.lower()
        ]
    else:
        filtered_models = models
    
    if not filtered_models:
        st.warning("No models match your filter")
        return None
    
    # Build options
    options = ["-- Select a model --"] + [m.format_display_name() for m in filtered_models]
    model_map = {m.format_display_name(): m.id for m in filtered_models}
    
    # Find default index
    default_index = 0
    if selected_id:
        for i, m in enumerate(filtered_models):
            if m.id == selected_id:
                default_index = i + 1  # +1 for the placeholder
                break
    
    selected = st.selectbox(
        "Model",
        options=options,
        index=default_index,
        key=key,
        label_visibility="collapsed"
    )
    
    if selected == "-- Select a model --":
        return None
    
    return model_map.get(selected)


def render_model_slot(
    slot_index: int,
    models: List[ModelInfo],
    selected_model_id: Optional[str] = None,
    on_remove: Optional[Callable] = None,
    filter_text: str = ""
) -> Optional[str]:
    """
    Render a single model slot with selector and remove button.
    
    Args:
        slot_index: Index of this slot
        models: List of available models
        selected_model_id: Currently selected model
        on_remove: Callback when remove is clicked
        filter_text: Text to filter models by
        
    Returns:
        Selected model ID or None
    """
    color = get_color(slot_index)
    
    col1, col2 = st.columns([9, 1])
    
    with col1:
        st.markdown(
            f'<div style="width: 100%; height: 4px; background-color: {color["border"]}; '
            f'border-radius: 2px; margin-bottom: 5px;"></div>',
            unsafe_allow_html=True
        )
        selected = render_model_selector(
            models=models,
            key=f"model_select_{slot_index}",
            selected_id=selected_model_id,
            filter_text=filter_text
        )
    
    with col2:
        if st.button("✕", key=f"remove_{slot_index}", help="Remove this model"):
            if on_remove:
                on_remove(slot_index)
    
    return selected

# Anonymous labels for models
ANONYMOUS_LABELS = [
    "Model A", "Model B", "Model C", "Model D", "Model E",
    "Model F", "Model G", "Model H", "Model I", "Model J"
]


def get_anonymous_label(index: int) -> str:
    """Get an anonymous label for a model by index."""
    if index < len(ANONYMOUS_LABELS):
        return ANONYMOUS_LABELS[index]
    return f"Model {index + 1}"


def render_result_card(
    result: ComparisonResult,
    index: int,
    on_vote_best: Optional[Callable] = None,
    on_vote_worst: Optional[Callable] = None,
    votes_disabled: bool = False,
    already_voted: bool = False,
    anonymous_mode: bool = False
):
    """
    Render a result card for a model response.
    
    Args:
        result: The comparison result to display
        index: Index of this result (for unique keys)
        on_vote_best: Callback for best vote
        on_vote_worst: Callback for worst vote
        votes_disabled: Whether voting is disabled
        already_voted: Whether this model was already voted on
        anonymous_mode: Whether to hide the model name
    """
    color = get_color(result.color_index)
    
    # Determine display name based on anonymous mode
    display_name = get_anonymous_label(index) if anonymous_mode else result.model_name
    
    # Card container
    if result.success:
        card_style = f"background-color: {color['bg']}; border-color: {color['border']};"
    else:
        card_style = "background-color: #ffebee; border-color: #c62828;"
    
    st.markdown(
        f'<div class="model-card" style="{card_style}">',
        unsafe_allow_html=True
    )
    
    # Header with model name (or anonymous label)
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        st.markdown(
            f'<span class="model-header" style="color: {color["text"]}">{display_name}</span>',
            unsafe_allow_html=True
        )
    
    with header_col2:
        if result.from_cache:
            st.markdown('<span class="cache-badge">📦 Cached</span>', unsafe_allow_html=True)
    
    # Stats line
    stats_parts = []
    if result.latency_ms is not None:
        stats_parts.append(f"⏱️ {result.latency_ms:.0f}ms")
    if result.total_tokens:
        stats_parts.append(f"📊 {result.total_tokens} tokens")
    elif result.output_tokens:
        stats_parts.append(f"📊 {result.output_tokens} tokens")
    
    if stats_parts:
        st.markdown(
            f'<div class="model-stats">{" • ".join(stats_parts)}</div>',
            unsafe_allow_html=True
        )
    
    # Output section
    if result.success and result.output_text:
        output_text = result.output_text
        
        # Truncated preview for collapsed state
        truncated = output_text[:TRUNCATE_LENGTH]
        if len(output_text) > TRUNCATE_LENGTH:
            truncated += "..."
        
        with st.expander(f"📝 Response ({len(output_text)} chars)", expanded=True):
            # View mode toggle
            view_mode = st.radio(
                "View",
                ["Markdown", "Raw"],
                horizontal=True,
                key=f"view_mode_{result.model_id}_{index}",
                label_visibility="collapsed"
            )
            
            # Display based on view mode
            if view_mode == "Markdown":
                st.markdown(output_text)
            else:
                st.code(output_text, language=None)
    else:
        st.error(f"❌ Error: {result.error}")
    
    # Vote buttons
    if result.success:
        vote_col1, vote_col2, vote_col3 = st.columns([1, 1, 2])
        
        # Disable if already voted on this model
        is_disabled = votes_disabled or already_voted
        
        with vote_col1:
            if st.button(
                "👍 Best",
                key=f"best_{result.model_id}_{index}",
                disabled=is_disabled,
                use_container_width=True
            ):
                if on_vote_best:
                    on_vote_best(result.model_id, result.model_name)
        
        with vote_col2:
            if st.button(
                "👎 Worst",
                key=f"worst_{result.model_id}_{index}",
                disabled=is_disabled,
                use_container_width=True
            ):
                if on_vote_worst:
                    on_vote_worst(result.model_id, result.model_name)
        
        with vote_col3:
            if already_voted:
                st.caption("✓ Voted")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_results_grid(
    results: List[ComparisonResult],
    columns: int = 2,
    on_vote_best: Optional[Callable] = None,
    on_vote_worst: Optional[Callable] = None,
    votes_disabled: bool = False,
    anonymous_mode: bool = False
):
    """
    Render results in a responsive grid.
    
    Args:
        results: List of comparison results
        columns: Number of columns
        on_vote_best: Callback for best vote
        on_vote_worst: Callback for worst vote
        votes_disabled: Whether voting is disabled
        anonymous_mode: Whether to hide model names
    """
    # Get voted models from session state
    voted_models = st.session_state.get("voted_this_session", set())
    
    # Create columns
    cols = st.columns(columns)
    
    for i, result in enumerate(results):
        col_index = i % columns
        with cols[col_index]:
            render_result_card(
                result=result,
                index=i,
                on_vote_best=on_vote_best,
                on_vote_worst=on_vote_worst,
                votes_disabled=votes_disabled,
                already_voted=result.model_id in voted_models,
                anonymous_mode=anonymous_mode
            )


def render_leaderboard_table(leaderboard_data: dict):
    """
    Render the leaderboard as a styled table.
    
    Args:
        leaderboard_data: Leaderboard export data
    """
    entries = leaderboard_data.get("leaderboard", [])
    
    if not entries:
        st.info("No votes recorded yet. Compare some models to start building the leaderboard!")
        return
    
    st.markdown(f"**{len(entries)} models ranked** | Last updated: {leaderboard_data.get('last_updated', 'Unknown')}")
    
    # Table header
    cols = st.columns([1, 4, 2, 2, 2, 2])
    with cols[0]:
        st.markdown("**Rank**")
    with cols[1]:
        st.markdown("**Model**")
    with cols[2]:
        st.markdown("**Net Score**")
    with cols[3]:
        st.markdown("**👍 Best**")
    with cols[4]:
        st.markdown("**👎 Worst**")
    with cols[5]:
        st.markdown("**Total**")
    
    st.divider()
    
    # Table rows
    for entry in entries:
        rank = entry["rank"]
        net_score = entry["net_score"]
        
        # Determine row style
        if net_score > 0:
            row_class = "leaderboard-positive"
        elif net_score < 0:
            row_class = "leaderboard-negative"
        else:
            row_class = "leaderboard-neutral"
        
        cols = st.columns([1, 4, 2, 2, 2, 2])
        
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
            st.markdown(entry["model_name"])
        
        with cols[2]:
            score_display = f"+{net_score}" if net_score > 0 else str(net_score)
            if net_score > 0:
                st.markdown(f"🟢 **{score_display}**")
            elif net_score < 0:
                st.markdown(f"🔴 **{score_display}**")
            else:
                st.markdown(f"⚪ **{score_display}**")
        
        with cols[3]:
            st.markdown(str(entry["best_votes"]))
        
        with cols[4]:
            st.markdown(str(entry["worst_votes"]))
        
        with cols[5]:
            st.markdown(str(entry["total_votes"]))


def render_settings_sidebar(
    api_key: str,
    temperature: float,
    cache_enabled: bool,
    use_env_api_key: bool = False,
    env_api_key_available: bool = False,
    council_voting_enabled: bool = True,
    on_api_key_change: Optional[Callable] = None,
    on_temperature_change: Optional[Callable] = None,
    on_cache_toggle: Optional[Callable] = None
) -> tuple:
    """
    Render the settings sidebar.
    
    Returns:
        Tuple of (api_key, temperature, cache_enabled, use_env_api_key, council_voting_enabled)
    """
    st.sidebar.header("⚙️ Settings")
    
    # Environment variable toggle
    st.sidebar.markdown("**🔑 API Key**")
    
    if env_api_key_available:
        new_use_env = st.sidebar.checkbox(
            "Use environment variable",
            value=use_env_api_key,
            help="Use the API key from your system's OPENROUTER_API_KEY environment variable"
        )
        
        if new_use_env:
            st.sidebar.success("✓ Using environment variable")
            new_api_key = api_key  # Keep existing value but don't use it
        else:
            new_api_key = st.sidebar.text_input(
                "OpenRouter API Key",
                value=api_key,
                type="password",
                help="Your OpenRouter API key"
            )
    else:
        new_use_env = False
        st.sidebar.caption("💡 Tip: Set OPENROUTER_API_KEY environment variable for automatic API key detection")
        new_api_key = st.sidebar.text_input(
            "OpenRouter API Key",
            value=api_key,
            type="password",
            help="Your OpenRouter API key",
            placeholder="sk-or-..."
        )
    
    st.sidebar.divider()
    
    new_temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=temperature,
        step=0.1,
        help="Higher values make output more random"
    )
    
    new_cache_enabled = st.sidebar.checkbox(
        "Enable Response Cache",
        value=cache_enabled,
        help="Cache responses for identical prompts"
    )
    
    new_council_voting = st.sidebar.checkbox(
        "🗳️ Enable Council Voting",
        value=council_voting_enabled,
        help="After responses, each model votes for the best answer (requires 2+ models)"
    )
    
    st.sidebar.divider()
    
    if st.sidebar.button("🔄 Refresh Models"):
        st.session_state["force_refresh_models"] = True
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear Cache"):
        from src.cache_manager import get_cache_manager
        get_cache_manager().clear()
        st.sidebar.success("Cache cleared!")
    
    return new_api_key, new_temperature, new_cache_enabled, new_use_env, new_council_voting
