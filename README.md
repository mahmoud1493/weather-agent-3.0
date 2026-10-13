# 🌦️ Weather Agent with Conversation Memory

A Streamlit weather assistant built with LangChain, Groq, Gemini, OpenWeatherMap, and LangGraph thread-based conversation memory.

**🔗 Live demo:** [weather-agent-3.0](https://weather-agent-3-mi7773.streamlit.app/)

## Features

- Uses Groq as the primary language model.
- Uses Gemini as a fallback when Groq is unavailable or rate-limited.
- Detects the user's approximate location from their IP address.
- Gets current weather data from OpenWeatherMap.
- Supports weather questions with or without a city.
- Displays temperature in Celsius and Fahrenheit.
- Keeps conversation context using a persistent `thread_id`.
- Shows agent tool calls and response structure.
- Allows the user to clear the chat and start a new conversation.

## Project structure

```text
.
├── app.py
├── tools.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
