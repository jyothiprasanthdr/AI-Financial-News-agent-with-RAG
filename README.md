# AI-Financial-News-agent-with-RAG

AI Financial News Agent with Retrieval-Augmented Generation (RAG)

Overview
--------

This repository implements an end-to-end Retrieval-Augmented Generation (RAG)
application for real-time financial news analysis, ticker extraction, and
workflow automation. The system integrates LangGraph, Qdrant (vector database),
OpenAI/Gemini embeddings, and a Yahoo Finance client, and exposes a Streamlit
user interface for interactive use.

Features
--------
- Retrieval-Augmented Generation (RAG) using vector embeddings to surface
	relevant financial articles and historical context.
- Multi-step LangGraph workflow that:
	- extracts a ticker from the user query
	- performs semantic search over Qdrant
	- fetches the latest market price (with fallback logic)
	- summarizes returned articles
	- generates a final response
- Streamlit frontend for interactive queries, summarized articles, and
	real-time stock prices.
- Modular architecture with separate clients, pipeline, and UI components.

Project structure
-----------------

- AI_Chatbot/
	- clients/ — Qdrant client, Yahoo data fetcher
	- pipeline/ — LangGraph nodes (search, summarize, fetch, format)
	- misc/ — shared helpers and utilities
	- app_streamlit.py — Streamlit UI entrypoint
- rag_stock_chatbot.py — script-style runner for programmatic usage
- Dockerfile — container build instructions
- requirements.txt — Python dependencies
- README.md — this file

Installation
------------

1. Clone the repository

	 git clone https://github.com/jyothiprasanthdr/AI-Financial-News-agent-with-RAG
	 cd AI-Financial-News-agent-with-RAG

2. Create and activate a Python virtual environment

	 python3 -m venv env
	 source env/bin/activate   # macOS / Linux
	 env\Scripts\activate     # Windows

3. Install dependencies

	 pip install -r requirements.txt

Configuration
-------------

Create a `.env` file in the project root containing the required keys:

```
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

Running the Streamlit app
-------------------------

Start the Streamlit UI:

```
streamlit run AI_Chatbot/app_streamlit.py
```

LangGraph workflow (simplified)
-------------------------------

Core workflow nodes:

- `extract_ticker`
- `semantic_search`
- `yahoo_fetch_with_fallback`
- `summarize_articles`
- `summarize`

The compiled graph produces a final state that the app presents to the user.

Vector store (Qdrant)
---------------------

To run Qdrant locally:

```
docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

Or use Qdrant Cloud for a managed deployment.

Optional: running the API
-------------------------

Run the script for programmatic usage:

```
python rag_stock_chatbot.py
```

Docker support
--------------

Build the image:

```
docker build -t financial-agent .
```

Run the container:

```
docker run -p 8000:8000 financial-agent
```

Notes and troubleshooting
------------------------

- Ensure environment variables are set before running the app.
- If Qdrant is not reachable, check that the container is running and that
	`QDRANT_HOST`/`QDRANT_PORT` match your setup.
- If you encounter LangGraph/runtime compatibility issues, consult the
	project's docs and consider checking the graph/node return types.

