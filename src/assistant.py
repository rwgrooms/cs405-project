"""LLM assistant: Gemini through LangChain.

The key is read from the sidebar box, the GOOGLE_API_KEY environment variable,
a .env file in the project folder, or .streamlit/secrets.toml. Without a key
the rest of the app still runs and the assistant is turned off.
"""
import os
from pathlib import Path

MODEL = "gemini-3.5-flash"

SYSTEM = (
    "You are a data science assistant for a student project on FEMA flood "
    "insurance (NFIP) building claims in North Carolina. Answer in plain "
    "language in under 200 words. Base the answer on the project results "
    "given. If the results cannot support a claim, say so."
)


def get_key(typed=""):
    """Return an API key, or None if there is none."""
    if typed:
        return typed.strip()
    if os.environ.get("GOOGLE_API_KEY"):
        return os.environ["GOOGLE_API_KEY"]
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GOOGLE_API_KEY":
                return value.strip().strip('"')
    try:
        import streamlit as st
        return st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


def build_chain(key):
    """Prompt template -> Gemini -> plain text."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Project results:\n{context}\n\nQuestion: {question}"),
    ])
    llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=key,
                                 temperature=0.2, max_retries=1, timeout=120)
    return prompt | llm | StrOutputParser()


def ask(question, context, key):
    """Send one question with the project context and return the answer."""
    return build_chain(key).invoke({"question": question, "context": context})
