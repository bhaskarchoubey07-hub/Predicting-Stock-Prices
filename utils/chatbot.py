from __future__ import annotations

import openai
import streamlit as st

def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    if len(st.session_state.messages) > 10:
        st.session_state.messages.pop(0)

def get_ai_response(user_input: str, context: str = "") -> str:
    """
    Generates a response from the AI financial advisor.
    Uses OpenAI API if available, otherwise falls back to rule-based responses.
    """
    api_key = st.secrets.get("OPENAI_API_KEY")
    if api_key:
        openai.api_key = api_key
        try:
            # Augment messages with context and history
            messages = st.session_state.get("messages", [])[-5:]  # Keep last 5 messages for context
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a sophisticated financial advisor AI. Context: {context}"},
                    *st.session_state.messages,
                    {"role": "system", "content": f"You are a helpful Fintech AI assistant. Current context: {context}"},
                    *messages,
                    {"role": "user", "content": user_input}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Sorry, I encountered an error with the AI service: {e}"

    # Fallback Smart Responses
    input_lower = user_input.lower()
    if "bullish" in input_lower: return "Bullish sentiment often coincides with the 20-day MA staying above the 50-day MA."
    if "risk" in input_lower: return "To mitigate risk, focus on stocks with a high Sharpe ratio and strong fundamentals."
    return "I am your FinAI assistant. Ask me about stock trends, sentiment, or portfolio health!"
