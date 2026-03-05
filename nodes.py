"""
Nodes module containing all node functions for the multi-agent graph.

FIX #1:  Parameterized queries in verification lookup.
FIX #4:  Phone number normalization before matching.
FIX #5:  Case-insensitive email matching.
FIX #7:  customer_id injected via SystemMessage, not extracted from text.
FIX #11: Removed fragile ast.literal_eval parsing.
FIX #15: Structured logging throughout.
FIX #17: Memory writes only when new preferences exist.
"""

import logging
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import interrupt

from state import State
from models import UserInput, UserProfile
from prompts import (
    generate_music_assistant_prompt,
    STRUCTURED_EXTRACTION_PROMPT,
    VERIFICATION_PROMPT,
    CREATE_MEMORY_PROMPT,
)
from database import get_engine, normalize_phone

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def get_customer_id_from_identifier(identifier: str) -> Optional[int]:
    """
    Retrieve Customer ID using an identifier (customer ID, email, or phone).
    Returns the CustomerId if found, otherwise None.
    """
    if not identifier or not identifier.strip():
        return None

    identifier = identifier.strip()
    engine = get_engine()

    try:
        from sqlalchemy import text

        # 1. Email address (case-insensitive)
        if "@" in identifier:
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT CustomerId FROM Customer WHERE LOWER(Email) = LOWER(:email)"),
                    {"email": identifier},
                )
                row = result.fetchone()
                if row:
                    return int(row[0])

        # 2. Direct numeric customer ID
        if identifier.isdigit():
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT CustomerId FROM Customer WHERE CustomerId = :cid"),
                    {"cid": int(identifier)},
                )
                row = result.fetchone()
                if row:
                    return int(row[0])

        # 3. Phone number check (Executes if Email and CustomerId lookups failed)
        normalized_input = normalize_phone(identifier)
        if normalized_input and len(normalized_input) >= 5:
            with engine.connect() as conn:
                # Fetch all customers and compare normalized phone numbers
                result = conn.execute(text("SELECT CustomerId, Phone FROM Customer WHERE Phone IS NOT NULL"))
                for row in result:
                    db_phone_normalized = normalize_phone(str(row[1]))
                    if db_phone_normalized == normalized_input:
                        return int(row[0])

    except Exception as e:
        logger.error(f"Error looking up customer by identifier '{identifier}': {e}")

    return None


def format_user_memory(user_data: dict) -> str:
    """Formats music preferences from stored user data."""
    try:
        profile = user_data.get("memory")
        if profile and hasattr(profile, "music_preferences") and profile.music_preferences:
            return f"Music Preferences: {', '.join(profile.music_preferences)}"
    except Exception as e:
        logger.error(f"Error formatting user memory: {e}")
    return ""


# ─────────────────────────────────────────────
# Music Assistant Node
# ─────────────────────────────────────────────

def create_music_assistant_node(llm, music_tools):
    """Factory function to create the music assistant node with bound tools."""
    llm_with_tools = llm.bind_tools(music_tools)

    def music_assistant(state: State, config: RunnableConfig):
        memory = state.get("loaded_memory", "None") or "None"
        prompt = generate_music_assistant_prompt(memory)

        messages = [SystemMessage(content=prompt)]
        if state.get("customer_id"):
            messages.append(
                SystemMessage(content=f"The current verified customer ID is: {state['customer_id']}")
            )
        messages.extend(state["messages"])

        logger.info(f"Music assistant invoked with {len(state['messages'])} conversation messages")
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return music_assistant


# ─────────────────────────────────────────────
# Conditional Edge Functions
# ─────────────────────────────────────────────

def should_continue(state: State, config: RunnableConfig) -> str:
    """Determine if the music sub-agent should continue calling tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"


def should_interrupt(state: State, config: RunnableConfig) -> str:
    """Determine if customer verification is complete or needs input."""
    if state.get("customer_id") is not None:
        return "continue"
    return "interrupt"


# ─────────────────────────────────────────────
# Verification Nodes
# ─────────────────────────────────────────────

def create_verify_info_node(llm):
    """Factory function to create the verify_info node."""
    structured_llm = llm.with_structured_output(schema=UserInput)

    def verify_info(state: State, config: RunnableConfig):
        """Verify customer account by parsing input and matching with database."""
        # Already verified — pass through
        if state.get("customer_id") is not None:
            logger.info(f"Customer already verified: {state['customer_id']}")
            return None

        user_input = state["messages"][-1]
        logger.info(f"Verification attempt with message: {getattr(user_input, 'content', '')[:100]}")

        try:
            parsed_info = structured_llm.invoke(
                [SystemMessage(content=STRUCTURED_EXTRACTION_PROMPT)] + [user_input]
            )
            identifier = parsed_info.identifier
            logger.info(f"Extracted identifier: '{identifier}'")
        except Exception as e:
            logger.error(f"Error parsing user input for verification: {e}")
            identifier = ""

        customer_id = None
        if identifier:
            customer_id = get_customer_id_from_identifier(identifier)
            logger.info(f"DB lookup result: customer_id={customer_id}")

        if customer_id is not None:
            # Inject a system message that clearly states the customer_id
            # so downstream agents can see it in context (FIX #7)
            intent_message = SystemMessage(
                content=(
                    f"Customer verified successfully. "
                    f"The verified customer_id is {customer_id}. "
                    f"Use this customer_id for all invoice and purchase lookups."
                )
            )
            return {
                "customer_id": str(customer_id),
                "messages": [intent_message],
            }
        else:
            response = llm.invoke(
                [SystemMessage(content=VERIFICATION_PROMPT)] + state["messages"]
            )
            return {"messages": [response]}

    return verify_info


def human_input(state: State, config: RunnableConfig):
    """Node that triggers an interrupt to request human input."""
    user_input = interrupt("Please provide input.")
    return {"messages": [HumanMessage(content=user_input)]}


# ─────────────────────────────────────────────
# Memory Nodes
# ─────────────────────────────────────────────

def load_memory(state: State, config: RunnableConfig, store: BaseStore):
    """Load user's long-term memory (music preferences) into graph state."""
    user_id = str(state.get("customer_id", ""))
    if not user_id:
        return {"loaded_memory": ""}

    namespace = ("memory_profile", user_id)
    try:
        existing_memory = store.get(namespace, "user_memory")
        if existing_memory and existing_memory.value:
            formatted = format_user_memory(existing_memory.value)
            logger.info(f"Loaded memory for customer {user_id}: {formatted}")
            return {"loaded_memory": formatted}
    except Exception as e:
        logger.error(f"Error loading memory for user {user_id}: {e}")

    return {"loaded_memory": ""}


def create_memory_node(llm):
    """Factory function to create the create_memory node."""

    def create_memory(state: State, config: RunnableConfig, store: BaseStore):
        """Analyze conversation and save/update user music preferences."""
        user_id = str(state.get("customer_id", ""))
        if not user_id:
            return None

        namespace = ("memory_profile", user_id)

        try:
            # Load existing memory
            existing_preferences = []
            existing_memory = store.get(namespace, "user_memory")
            formatted_memory = ""
            if existing_memory and existing_memory.value:
                mem_dict = existing_memory.value
                profile = mem_dict.get("memory")
                if profile and hasattr(profile, "music_preferences"):
                    existing_preferences = list(profile.music_preferences or [])
                    formatted_memory = f"Music Preferences: {', '.join(existing_preferences)}"

            # Summarize conversation for the memory prompt
            recent_messages = state["messages"][-10:]
            conversation_summary = "\n".join(
                f"{getattr(msg, 'type', 'unknown')}: {getattr(msg, 'content', '')}"
                for msg in recent_messages
                if getattr(msg, "content", "")
            )

            formatted_prompt = CREATE_MEMORY_PROMPT.format(
                conversation=conversation_summary,
                memory_profile=formatted_memory or "Empty, no existing profile",
            )

            updated_memory = llm.with_structured_output(UserProfile).invoke(
                [SystemMessage(content=formatted_prompt)]
            )

            # FIX #17: Only write if new preferences are a superset of existing
            new_prefs = updated_memory.music_preferences or []
            if not new_prefs and existing_preferences:
                # LLM returned empty but we had existing preferences — don't overwrite
                logger.info(f"Memory unchanged for customer {user_id} (preserving existing preferences)")
                return None

            # Merge: ensure existing preferences are preserved
            merged_prefs = list(set(existing_preferences + new_prefs))
            updated_memory.music_preferences = merged_prefs
            updated_memory.customer_id = user_id

            store.put(namespace, "user_memory", {"memory": updated_memory})
            logger.info(f"Memory updated for customer {user_id}: {merged_prefs}")

        except Exception as e:
            logger.error(f"Error creating/updating memory for user {user_id}: {e}")

    return create_memory
