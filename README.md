# LLM Council v2

![header](header.png "LLM Council v2")


A modular AI comparison app built with Python and Streamlit. Compare responses from multiple OpenRouter models side by side and vote on the best outputs to build your own model leaderboard.

## Features

- ⚖️ **Side-by-side comparison** - Run the same prompt through multiple AI models simultaneously
- 🎭 **Anonymous mode** - Model names hidden by default for unbiased evaluation (toggle to reveal)
- ⚡ **Parallel execution** - Fast comparisons with concurrent API calls
- 🗳️ **Voting system** - Vote on best and worst responses to track model performance
- 🏆 **Leaderboard** - See cumulative rankings based on your votes (net score = best - worst)
- 💾 **Response caching** - Optional caching to save costs on repeated prompts

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/laceyp99/LLM-Council-v2.git
cd LLM-Council-v2
```

Create a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

**Option A: Environment variable (recommended for developers)**

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key
```

Get your API key from [OpenRouter](https://openrouter.ai/keys).

**Option B: Enter in the app**

You can also enter your API key directly in the Streamlit sidebar when running the app.

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

### Comparing Models

1. Select models to compare using the dropdown slots (use the filter to search by name)
2. Add more model slots with the "➕ Add Model" button
3. Enter your prompt in the text area
4. Click "🚀 Run Comparison"
5. View results side-by-side with markdown rendering or raw text

### Voting

After a comparison:
- Click 👍 **Best** on the response you prefer
- Click 👎 **Worst** on the response you like least
- Votes persist across sessions and build your personal leaderboard

### Leaderboard

- View all models ranked by net score (Best votes - Worst votes)
- Sort by different metrics (net score, best votes, worst votes, total votes)
- Reset all votes if needed

## Tech Stack

- **Language:** Python 3.10+
- **API:** OpenRouter (access to 100+ LLM models)
- **Frontend:** Streamlit
- **Storage:** JSON files (local, in `data/` directory)
- **Package Management:** pip

## Project Structure

```
llm-council-v2/
├── src/
│   ├── __init__.py
│   ├── openrouter_client.py
│   ├── model_metadata.py
│   ├── compare_logic.py
│   ├── leaderboard_storage.py
│   ├── cache_manager.py
│   └── ui_components.py
├── data/
│   └── .gitkeep
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Settings

Available in the sidebar:

- **Temperature**: Control response randomness (0 = deterministic, 2 = very random)
- **Enable Cache**: Toggle response caching for identical prompt+model combinations
- **Refresh Models**: Reload the model list from OpenRouter
- **Clear Cache**: Remove all cached responses

## Data Storage

All data is stored locally in JSON files in the `data/` directory:

- **models_cache.json**: Cached model metadata (auto-refreshes every 24 hours)
- **votes.json**: Your voting history
- **response_cache.json**: Cached model responses

No data is sent to external servers beyond the OpenRouter API calls.

## License

MIT License - feel free to use and modify as needed.