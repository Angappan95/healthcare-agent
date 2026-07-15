# Healthcare Agent - Multi-Agent Orchestration System

A sophisticated healthcare information retrieval system using **LangGraph supervisor pattern** to intelligently coordinate between PubMed literature search and web search, providing comprehensive, evidence-based responses with source attribution.

## 🌟 Features

### Multi-Agent Coordination
- **Supervisor Agent:** LLM-based intelligent routing that analyzes query intent
- **PubMed Agent:** Searches peer-reviewed medical literature from NCBI PubMed
- **WebSearch Agent:** Searches credible health websites via Tavily API (CDC, NIH, Mayo Clinic, etc.)
- **Synthesis Agent:** Merges multi-source information with credibility-weighted conflict resolution

### Advanced Capabilities
- **Intelligent Routing:** Automatically determines which agent(s) to invoke based on query type
- **Graceful Degradation:** Returns partial results if one agent fails
- **Circuit Breaker Pattern:** Prevents cascading failures with automatic recovery
- **Conversation History:** SQLite checkpointing for multi-turn conversations
- **Source Attribution:** Every response includes citations and credibility scores
- **Backward Compatible:** Legacy linear workflow still supported

### Resilience & Quality
- Retry logic with exponential backoff (2 retries)
- Error handling without workflow failure
- Credibility scoring (PubMed: 1.0, Credible Web: 0.8, General Web: 0.5)
- Medical disclaimers on all responses
- No PHI (Personal Health Information) storage

## 🏗️ Architecture

```
User Query
    ↓
Supervisor Agent (LLM analyzes intent)
    ↓
┌─────────────────────┬─────────────────────┐
↓                     ↓                     ↓
PubMed Agent    WebSearch Agent    Both (sequential)
(Peer-reviewed)  (Credible sources)
    ↓                     ↓                     ↓
    └─────────────────────┴─────────────────────┘
                    ↓
            Synthesis Agent
        (Merge + Cite + Resolve conflicts)
                    ↓
            Final Response
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for comprehensive technical documentation.

## 🚀 Quick Start

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repo-url>
   cd healthcare-agent
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

### Configuration

Set required API keys:
```bash
# Required for LLM (choose one)
export GOOGLE_API_KEY="AIza-xxxxx"      # Gemini
# OR
export OPENAI_API_KEY="sk-xxxxx"        # OpenAI

# Required for WebSearch Agent
export TAVILY_API_KEY="tvly-xxxxx"      # Get from https://tavily.com
```

### Usage

```python
from src.workflow import run_workflow

# Simple query
result = run_workflow("What are treatments for type 2 diabetes?")
print(result['response'])
```

### Multi-Turn Conversations

The system uses **LangGraph's built-in message handling** with `add_messages` reducer and `SummarizationMiddleware` to maintain conversation context across multiple queries:

```python
from src.workflow import run_workflow, get_conversation_turns, print_conversation_history

# Use the same thread_id to maintain context across turns
thread_id = "user-session-123"

# Turn 1: Initial question
result1 = run_workflow(
    user_query="What are the main treatments for type 2 diabetes?",
    thread_id=thread_id
)

# Turn 2: Follow-up question (system remembers previous context)
result2 = run_workflow(
    user_query="What are the side effects of those medications?",  # References previous turn
    thread_id=thread_id
)

# Turn 3: Another follow-up
result3 = run_workflow(
    user_query="Are there lifestyle changes that can help?",
    thread_id=thread_id
)

# Access conversation history from messages
from langchain_core.messages import HumanMessage, AIMessage

messages = result3['messages']
for msg in messages:
    if isinstance(msg, HumanMessage):
        print(f"User: {msg.content}")
    elif isinstance(msg, AIMessage):
        print(f"Assistant: {msg.content}")
        print(f"Sources: {msg.additional_kwargs.get('sources_used', {})}")

# Or use helper functions
turns = get_conversation_turns(result3)
print_conversation_history(result3)
```

**Key Features:**
- **`add_messages` reducer**: Automatically appends new messages to conversation
- **BaseMessage types**: Uses LangChain's `HumanMessage` and `AIMessage` for proper typing
- **SummarizationMiddleware**: Automatically summarizes old messages when tokens exceed 2000
  - Token-based trigger prevents context window overflow
  - Keeps last 2 messages (1 turn) intact while summarizing older context
  - Efficient memory management for long consultations
- **Metadata support**: AI responses include `sources_used` in `additional_kwargs`
- **Persistence**: History persisted via SQLite checkpointing (survives restarts)

See [`examples/multi_turn_conversation.py`](examples/multi_turn_conversation.py) for a complete demo.

## 📊 How Routing Works

The Supervisor Agent uses LLM to analyze queries and route intelligently:

| Query Type | Example | Routing Decision |
|------------|---------|------------------|
| **Research/Clinical** | "What do clinical trials show about drug X?" | → PubMed Agent |
| **General Health** | "What are symptoms of diabetes?" | → WebSearch Agent |
| **Treatment Info** | "How is condition X treated?" | → Both Agents |
| **Complex Medical** | "Relationship between diabetes and heart disease" | → Both Agents |

## 🧪 Testing

Run unit tests:
```bash
pytest test/test_supervisor_workflow.py -v
```

Run all tests:
```bash
pytest test/ -v
```

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Comprehensive system design, state flow, agent responsibilities
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - What was built, features, usage examples
- **[SETUP.md](SETUP.md)** - Detailed installation and configuration
- **[WORKFLOW.md](WORKFLOW.md)** - Original workflow documentation
- **[CHECKPOINTING_GUIDE.md](CHECKPOINTING_GUIDE.md)** - Conversation history management

## 🎯 Example Queries

```python
# Research-focused (routes to PubMed)
run_workflow("What are the mechanisms of metformin in type 2 diabetes?")

# General health (routes to WebSearch)
run_workflow("What are common flu symptoms and remedies?")

# Complex topic (routes to Both)
run_workflow("What are the latest treatments for COVID-19 long-haul symptoms?")
```

## 🔧 Configuration Options

Edit `config.py` to customize:

```python
# Agent Behavior
MAX_AGENT_ITERATIONS = 5          # Prevent infinite loops
ENABLE_PARALLEL_AGENTS = True     # Allow parallel execution
AGENT_TIMEOUT = 30                # Seconds per agent

# Search Configuration
TAVILY_MAX_RESULTS = 5            # Web search results
ARTICLE_CHAR_LIMIT = 500          # Chars per article in prompts

# Retry & Resilience
MAX_RETRIES = 2                   # Retry attempts
RETRY_BACKOFF_FACTOR = 2          # Exponential backoff
```

## 📈 Performance

| Metric | Sequential | Parallel |
|--------|-----------|----------|
| **PubMed only** | 4-6s | - |
| **WebSearch only** | 3-5s | - |
| **Both agents** | 8-12s | 5-9s (planned) |

## 🚧 Known Limitations

1. **Parallel execution** currently sequential (LangGraph limitation)
2. **No query caching** (repeated queries hit APIs)
3. **SQLite checkpointing** may need PostgreSQL for production scale

## 🛠️ Troubleshooting

### "Tavily API key not configured"
```bash
export TAVILY_API_KEY="your-key-here"
```

### "ModuleNotFoundError: No module named 'langgraph_checkpoint_sqlite'"
```bash
pip install langgraph-checkpoint-sqlite tavily-python
```

### "Circuit breaker open"
Wait 60 seconds for automatic reset, or check agent logs for underlying error.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is for educational purposes.

## ⚠️ Disclaimer

**This system provides health information for educational purposes only and should not be considered medical advice. Always consult with a qualified healthcare professional for medical concerns.**

## 🙏 Acknowledgments

- **LangGraph** - Multi-agent orchestration framework
- **NCBI PubMed** - Medical literature database
- **Tavily** - Web search API
- **Google Gemini / OpenAI** - Language models

---

**Version:** 1.0.0  
**Last Updated:** 2026-07-15
