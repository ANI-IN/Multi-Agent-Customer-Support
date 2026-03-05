"""
State module defining the shared state schema for the multi-agent graph.
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps


class State(TypedDict):
    """Shared state schema for the multi-agent LangGraph system."""
    customer_id: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_memory: str
    remaining_steps: RemainingSteps
