"""Memory management module for chat history persistence."""
import os
import json
from langchain_core.messages import HumanMessage, AIMessage


def load_memory(filename="chat_memory.json"):
    """Load raw chat history from file.

    Args:
        filename: Path to memory file

    Returns:
        List of dicts with keys {'type': 'human'|'ai', 'content': str}
    """
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []


def save_memory(history, filename="chat_memory.json", max_turns=10):
    """Save chat history to file, trimmed to last N turns.

    Args:
        history: List of message dicts or LangChain message objects
        filename: Path to memory file
        max_turns: Maximum number of conversation turns to keep
    """
    # Convert LangChain messages to raw dicts if needed
    raw_history = to_raw_dicts(history)

    # Trim to last N turns (each turn is user + assistant = 2 messages)
    if len(raw_history) > max_turns * 2:
        raw_history = raw_history[-(max_turns * 2):]

    with open(filename, "w") as f:
        json.dump(raw_history, f)


def to_langchain_messages(raw_history):
    """Convert raw history dicts to LangChain message objects.

    Args:
        raw_history: List of dicts with keys {'type': 'human'|'ai', 'content': str}

    Returns:
        List of HumanMessage or AIMessage objects
    """
    messages = []
    for msg in raw_history:
        if msg.get('type') == 'human':
            messages.append(HumanMessage(content=msg.get('content', '')))
        else:
            messages.append(AIMessage(content=msg.get('content', '')))
    return messages


def to_raw_dicts(messages):
    """Convert LangChain message objects to raw history dicts.

    Args:
        messages: List of HumanMessage, AIMessage, or raw dicts

    Returns:
        List of dicts with keys {'type': 'human'|'ai', 'content': str}
    """
    raw = []
    for msg in messages:
        if isinstance(msg, dict):
            raw.append(msg)
        elif isinstance(msg, HumanMessage):
            raw.append({'type': 'human', 'content': msg.content})
        elif isinstance(msg, AIMessage):
            raw.append({'type': 'ai', 'content': msg.content})
    return raw


def trim_memory(memory, max_turns=6):
    """Trim memory to last N conversation turns.

    Args:
        memory: List of LangChain message objects
        max_turns: Maximum number of turns (each turn = user + assistant)

    Returns:
        Trimmed list of message objects
    """
    # Each turn is 2 messages (user + assistant)
    max_messages = max_turns * 2
    if len(memory) > max_messages:
        return memory[-max_messages:]
    return memory
