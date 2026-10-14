import logging
import os
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from tools import get_location, get_weather


logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

load_dotenv()


st.set_page_config(
    page_title="Weather Agent",
    page_icon="🌦️",
)

st.title("🌦️ Weather Agent")

st.caption(
    "A Groq-first weather agent with Gemini fallback and "
    "persistent conversation memory."
)


# ------------------------------------
# Load Streamlit Cloud secrets
# ------------------------------------

def load_streamlit_secrets_into_environment():
    """
    Load secrets from Streamlit Community Cloud into environment variables.

    Locally, the application uses the .env file.
    On Streamlit Community Cloud, it uses st.secrets.
    """

    required_keys = [
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "OPENWEATHER_API_KEY",
    ]

    for key in required_keys:
        try:
            value = st.secrets.get(key)

            if value:
                os.environ[key] = str(value)

        except Exception:
            # Locally, .env will be used instead.
            pass


load_streamlit_secrets_into_environment()


# ------------------------------------
# Agent configuration
# ------------------------------------

SYSTEM_PROMPT = """
You are a helpful weather assistant.

YOUR WORKFLOW:

1. If the user asks about weather WITHOUT specifying a location, you MUST:
   - First call get_location() to find their location.
   - Then call get_weather(city) with that location.

2. If the user provides a city, call get_weather(city) directly.

3. Use the weather tool result.
   Do not invent weather data.

4. Present the following information whenever available:
   - City
   - Country
   - Weather condition
   - Temperature in Celsius
   - Temperature in Fahrenheit
   - Humidity
   - Wind speed
"""


@st.cache_resource
def build_agent():
    """
    Build the agent once and reuse it across Streamlit reruns.
    """

    # Groq is the primary model.
    groq = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
    )

    # Gemini is the fallback model.
    gemini = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=1.0,
        max_retries=0,
    )

    # Groq is tried first.
    # Gemini is used automatically if Groq fails.
    return create_agent(
        model=groq,
        tools=[
            get_weather,
            get_location,
        ],
        middleware=[
            ModelFallbackMiddleware(gemini),
        ],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


agent = build_agent()


# ------------------------------------
# Helper functions
# ------------------------------------

def content_to_text(content):
    """
    Convert LangChain message content into text
    that Streamlit can display.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def message_summary(message):
    """
    Create a readable summary of a LangChain message.
    """

    return {
        "message_class": type(message).__name__,
        "message_type": getattr(message, "type", None),
        "content": content_to_text(
            getattr(message, "content", None)
        ),
        "tool_calls": getattr(message, "tool_calls", None),
        "additional_kwargs": getattr(
            message,
            "additional_kwargs",
            {},
        ),
        "response_metadata": getattr(
            message,
            "response_metadata",
            {},
        ),
    }


# ------------------------------------
# Streamlit session state
# ------------------------------------

# A thread ID is created once for this browser session.
# Every question in this session uses the same ID.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = (
        f"weather-thread-{uuid4()}"
    )


# These messages are used to display the chat in Streamlit.
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# This counts the messages already returned by the LangGraph agent.
if "agent_message_count" not in st.session_state:
    st.session_state.agent_message_count = 0


# ------------------------------------
# Sidebar
# ------------------------------------

with st.sidebar:
    st.subheader("Conversation")

    st.caption(
        f"Thread ID: `{st.session_state.thread_id}`"
    )

    st.info(
        "All questions in this browser session use the same "
        "thread ID, so the agent can remember the conversation."
    )

    if st.button("🗑️ Clear chat and start a new thread"):
        st.session_state.chat_messages = []
        st.session_state.agent_message_count = 0
        st.session_state.thread_id = (
            f"weather-thread-{uuid4()}"
        )

        st.rerun()


# ------------------------------------
# Display previous chat messages
# ------------------------------------

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ------------------------------------
# New user query
# ------------------------------------

if prompt := st.chat_input(
    "Ask about the current weather..."
):
    # Display and store the user's message.
    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking the weather..."):
            try:
                previous_message_count = (
                    st.session_state.agent_message_count
                )

                # The checkpointer remembers previous messages
                # associated with this thread_id.
                #
                # Only the new user message is sent here.
                result = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id": (
                                st.session_state.thread_id
                            ),
                        }
                    },
                )

                all_agent_messages = result["messages"]

                new_steps = all_agent_messages[
                    previous_message_count:
                ]

                st.session_state.agent_message_count = (
                    len(all_agent_messages)
                )

                answer = content_to_text(
                    all_agent_messages[-1].content
                )

            except Exception as error:
                answer = (
                    f"Sorry, something went wrong: {error}"
                )
                new_steps = []

        st.markdown(answer)

        # Store the assistant's answer for the next rerun.
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        # ------------------------------------
        # Show agent steps
        # ------------------------------------

        if new_steps:
            with st.expander("🔧 Agent steps"):
                model_names = set()

                for message in new_steps:
                    metadata = (
                        getattr(
                            message,
                            "response_metadata",
                            {},
                        )
                        or {}
                    )

                    model_name = metadata.get(
                        "model_name"
                    )

                    if model_name:
                        model_names.add(str(model_name))

                    if getattr(message, "type", None) == "ai":
                        tool_calls = (
                            getattr(
                                message,
                                "tool_calls",
                                [],
                            )
                            or []
                        )

                        for call in tool_calls:
                            st.markdown(
                                f"**Called** "
                                f"`{call.get('name')}` "
                                f"with "
                                f"`{call.get('args')}`"
                            )

                    elif getattr(message, "type", None) == "tool":
                        tool_name = getattr(
                            message,
                            "name",
                            "tool",
                        )

                        st.markdown(
                            f"**Result from** "
                            f"`{tool_name}`:"
                        )

                        st.code(
                            content_to_text(
                                message.content
                            )
                        )

                if model_names:
                    st.caption(
                        "Answered by: "
                        + ", ".join(
                            sorted(model_names)
                        )
                    )

        # ------------------------------------
        # Show response structure
        # ------------------------------------

        if new_steps:
            with st.expander(
                "🔎 Response structure"
            ):
                st.write(
                    "Response type:",
                    type(result).__name__,
                )

                st.write(
                    "Response keys:",
                    list(result.keys()),
                )

                st.write(
                    "Total messages in thread:",
                    len(all_agent_messages),
                )

                for index, message in enumerate(
                    new_steps
                ):
                    st.markdown(
                        f"**Message {index} in this response**"
                    )

                    st.json(
                        message_summary(message)
                    )
