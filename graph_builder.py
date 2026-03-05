"""
Graph builder module.
Assembles the complete multi-agent LangGraph workflow with:
1. Customer verification (human in the loop)
2. Long term memory (load and save preferences)
3. Supervisor routing to sub agents
4. Music catalog sub agent (hand built ReAct)
5. Invoice information sub agent (pre-built ReAct)
"""

import logging
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from state import State
from tools import music_tools, invoice_tools
from prompts import INVOICE_SUBAGENT_PROMPT, SUPERVISOR_PROMPT
from nodes import (
    create_music_assistant_node,
    should_continue,
    should_interrupt,
    create_verify_info_node,
    human_input,
    load_memory,
    create_memory_node,
)

logger = logging.getLogger(__name__)


def build_graph(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
    openai_api_key: str = None,
    openai_api_base: str = None,
):
    """
    Build and compile the complete multi-agent graph.

    Returns:
        Tuple of (compiled_graph, checkpointer, in_memory_store)
    """
    # Initialize LLM
    llm_kwargs = {
        "model": model_name,
        "temperature": temperature,
    }
    if openai_api_key:
        llm_kwargs["api_key"] = openai_api_key
    if openai_api_base:
        llm_kwargs["base_url"] = openai_api_base

    llm = ChatOpenAI(**llm_kwargs)
    logger.info(f"LLM initialized: {model_name}, temperature={temperature}")

    # Initialize Memory
    in_memory_store = InMemoryStore()
    checkpointer = MemorySaver()

    # ── Build Music Catalog Sub Agent (from scratch) ──
    music_assistant_fn = create_music_assistant_node(llm, music_tools)
    music_tool_node = ToolNode(music_tools)

    music_workflow = StateGraph(State)
    music_workflow.add_node("music_assistant", music_assistant_fn)
    music_workflow.add_node("music_tool_node", music_tool_node)
    music_workflow.add_edge(START, "music_assistant")
    music_workflow.add_conditional_edges(
        "music_assistant",
        should_continue,
        {"continue": "music_tool_node", "end": END},
    )
    music_workflow.add_edge("music_tool_node", "music_assistant")

    music_catalog_subagent = music_workflow.compile(
        name="music_catalog_subagent",
        checkpointer=checkpointer,
        store=in_memory_store,
    )
    logger.info("Music catalog sub agent compiled.")

    # ── Build Invoice Information Sub Agent (pre-built) ──
    invoice_information_subagent = create_react_agent(
        llm,
        tools=invoice_tools,
        name="invoice_information_subagent",
        prompt=INVOICE_SUBAGENT_PROMPT,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
    )
    logger.info("Invoice information sub agent compiled.")

    # ── Build Supervisor ──
    try:
        from langgraph_supervisor import create_supervisor

        supervisor_workflow = create_supervisor(
            agents=[invoice_information_subagent, music_catalog_subagent],
            output_mode="last_message",
            model=llm,
            prompt=SUPERVISOR_PROMPT,
            state_schema=State,
        )
        supervisor_prebuilt = supervisor_workflow.compile(
            name="supervisor",
            checkpointer=checkpointer,
            store=in_memory_store,
        )
        logger.info("Supervisor compiled.")
    except ImportError:
        logger.error("langgraph_supervisor not installed. Install with: pip install langgraph-supervisor")
        raise

    # ── Build Final Multi Agent Graph ──
    verify_info_fn = create_verify_info_node(llm)
    create_memory_fn = create_memory_node(llm)

    multi_agent = StateGraph(State)
    multi_agent.add_node("verify_info", verify_info_fn)
    multi_agent.add_node("human_input", human_input)
    multi_agent.add_node("load_memory", load_memory)
    multi_agent.add_node("supervisor", supervisor_prebuilt)
    multi_agent.add_node("create_memory", create_memory_fn)

    multi_agent.add_edge(START, "verify_info")
    multi_agent.add_conditional_edges(
        "verify_info",
        should_interrupt,
        {"continue": "load_memory", "interrupt": "human_input"},
    )
    multi_agent.add_edge("human_input", "verify_info")
    multi_agent.add_edge("load_memory", "supervisor")
    multi_agent.add_edge("supervisor", "create_memory")
    multi_agent.add_edge("create_memory", END)

    compiled_graph = multi_agent.compile(
        name="multi_agent_final",
        checkpointer=checkpointer,
        store=in_memory_store,
    )
    logger.info("Final multi agent graph compiled successfully.")

    return compiled_graph, checkpointer, in_memory_store
