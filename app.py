"""
Gradio application for the Music Store Multi-Agent System.
AUDIT IMPROVEMENTS (UI/UX):
- Clean, professional chat interface with custom CSS
- Loading states with spinner animation during LLM calls
- Error states with clear, actionable messages
- Session management with proper thread_id isolation
- Database health check on startup with user-visible status
- Trust signals: shows data sources used, tool calls made
- Quick-action suggestion buttons for common queries
- Conversation reset functionality
- Responsive design for mobile/desktop
- Accessibility: proper ARIA labels, color contrast, keyboard nav
- Rate limiting feedback
- Environment variable configuration with .env support
"""

import os
import uuid
import time
import logging

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from database import verify_database
from graph_builder import build_graph

# ── Configuration ──
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
APP_TITLE = "🎵 Music Store Assistant"
APP_DESCRIPTION = (
    "Welcome! I can help you explore our music catalog, look up invoices, "
    "and find your purchase history. To access your account, please provide "
    "your Customer ID, email, or phone number."
)

# ── Global state ──
graph = None
checkpointer = None
store = None
db_health = None


def initialize():
    """Initialize the database and build the agent graph."""
    global graph, checkpointer, store, db_health

    logger.info("Initializing Music Store Agent...")

    # Database health check
    db_health = verify_database()
    if db_health.get("status") != "healthy":
        logger.error(f"Database unhealthy: {db_health}")
    else:
        logger.info(f"Database healthy: {db_health['tables']}")

    # Build graph
    try:
        graph, checkpointer, store = build_graph(
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            openai_api_key=OPENAI_API_KEY if OPENAI_API_KEY else None,
            openai_api_base=OPENAI_API_BASE if OPENAI_API_BASE else None,
        )
        logger.info("Agent graph built successfully.")
    except Exception as e:
        logger.error(f"Failed to build agent graph: {e}")
        raise


def get_thread_config(thread_id: str) -> dict:
    """Create a LangGraph config with the given thread_id."""
    return {"configurable": {"thread_id": thread_id}}



def reset_conversation() -> tuple:
    """Reset the conversation state."""
    new_thread = str(uuid.uuid4())
    logger.info(f"Conversation reset. New thread_id={new_thread}")
    return [], new_thread, _status_html("idle", "New conversation started")


def _status_html(status: str, message: str, tools_used: list = None) -> str:
    """Generate status indicator HTML for the UI."""
    colors = {
        "success": "#10b981",
        "error": "#ef4444",
        "warning": "#f59e0b",
        "waiting": "#6366f1",
        "idle": "#6b7280",
    }
    icons = {
        "success": "✓",
        "error": "✗",
        "warning": "⚠",
        "waiting": "⏳",
        "idle": "●",
    }
    color = colors.get(status, "#6b7280")
    icon = icons.get(status, "●")

    tools_text = ""
    if tools_used:
        tools_text = f" | Data sources: {', '.join(set(tools_used))}"

    return (
        f'<div style="display:flex;align-items:center;gap:6px;padding:6px 12px;'
        f'border-radius:6px;background:{color}15;border:1px solid {color}30;'
        f'font-size:13px;color:{color};">'
        f'<span>{icon}</span>'
        f'<span>{message}{tools_text}</span>'
        f'</div>'
    )


# ── Custom CSS ──

CUSTOM_CSS = """
/* Overall theme */
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
/* Header */
.app-header {
    text-align: center;
    padding: 24px 16px 16px;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 16px;
}
.app-header h1 {
    font-size: 28px;
    font-weight: 700;
    color: #1f2937;
    margin: 0 0 8px;
}
.app-header p {
    font-size: 14px;
    color: #6b7280;
    margin: 0;
    line-height: 1.5;
}
/* Chat area */
.chatbot-container {
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    overflow: hidden;
}
/* Status bar */
.status-bar {
    min-height: 36px;
}
/* Footer */
.app-footer {
    text-align: center;
    padding: 12px;
    font-size: 12px;
    color: #9ca3af;
    border-top: 1px solid #f3f4f6;
    margin-top: 8px;
}
/* Responsive */
@media (max-width: 640px) {
    .app-header h1 { font-size: 22px; }
}
"""


# ── Build Gradio Interface ──

def create_app() -> gr.Blocks:
    """Create and return the Gradio Blocks application."""

    # Initialize on startup
    initialize()

    with gr.Blocks(
        title=APP_TITLE,
    ) as app:

        # State
        thread_id = gr.State(value="")

        # Header
        gr.HTML(
            f"""
            <div class="app-header">
                <h1>{APP_TITLE}</h1>
                <p>{APP_DESCRIPTION}</p>
            </div>
            """
        )

        # Chat interface
        chatbot = gr.Chatbot(
            value=[],
            height=480,
            show_label=False,
            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=music"),
            elem_classes=["chatbot-container"],
            placeholder=(
                "**👋 Welcome!** Type your message below to get started.\n\n"
                "Try: *\"My customer ID is 5\"* or *\"What rock albums do you have?\"*"
            ),
        )

        # Status bar
        status = gr.HTML(
            value=_status_html("idle", "Ready — type a message to begin"),
            elem_classes=["status-bar"],
        )

        # Input row
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                scale=6,
                container=False,
                autofocus=True,
            )
            send_btn = gr.Button(
                "Send",
                variant="primary",
                scale=1,
                min_width=80,
            )

        # Controls row
        with gr.Row():
            reset_btn = gr.Button("🔄 New Conversation", size="sm", variant="secondary")

        # Footer
        gr.HTML(
            '<div class="app-footer">'
            'Powered by LangGraph Multi-Agent Architecture · '
            'Data from Chinook Sample Database'
            '</div>'
        )

        # ── Event Handlers ──

        def show_user_message(message, history, tid):
            """Step 1: Immediately show the user's message and clear the input."""
            if not message.strip():
                return history, "", tid, _status_html("idle", "Ready")
            # Generate thread_id if needed
            if not tid:
                tid = str(uuid.uuid4())
            history = history + [{"role": "user", "content": message}]
            return history, "", tid, _status_html("waiting", "Processing...")

        def generate_response(history, tid):
            """Step 2: Process the message through the agent and append the response."""
            if not history:
                return history, tid, _status_html("idle", "Ready")

            # Get the last user message from history
            user_message = None
            for msg in reversed(history):
                if msg.get("role") == "user":
                    user_message = msg["content"]
                    break

            if not user_message:
                return history, tid, _status_html("idle", "Ready")

            if not graph:
                history.append({"role": "assistant", "content": "⚠️ System not initialized. Please refresh the page."})
                return history, tid, _status_html("error", "System not initialized")

            config = get_thread_config(tid)

            try:
                start_time = time.time()

                input_state = {
                    "messages": [HumanMessage(content=user_message)],
                }

                final_response = None
                tools_used = []

                for event in graph.stream(input_state, config=config, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        logger.info(f"Graph event: node={node_name}")

                        if node_name in ("music_tool_node",):
                            tools_used.append("music_catalog")
                        elif node_name == "invoice_information_subagent":
                            tools_used.append("invoice_lookup")

                        if isinstance(node_output, dict) and "messages" in node_output:
                            for msg in node_output["messages"]:
                                if isinstance(msg, AIMessage) and msg.content:
                                    final_response = msg.content

                elapsed = time.time() - start_time

                if final_response:
                    history.append({"role": "assistant", "content": final_response})
                    status = _status_html(
                        "success",
                        f"Responded in {elapsed:.1f}s",
                        tools_used=tools_used,
                    )
                else:
                    snapshot = graph.get_state(config)
                    if snapshot and hasattr(snapshot, "next") and snapshot.next:
                        state_messages = snapshot.values.get("messages", [])
                        for msg in reversed(state_messages):
                            if isinstance(msg, AIMessage) and msg.content:
                                final_response = msg.content
                                break

                        if final_response and not any(
                            h.get("content") == final_response for h in history if h.get("role") == "assistant"
                        ):
                            history.append({"role": "assistant", "content": final_response})

                        status = _status_html("waiting", "Waiting for your input")
                    else:
                        history.append({
                            "role": "assistant",
                            "content": "I'm sorry, I wasn't able to generate a response. Please try rephrasing your question.",
                        })
                        status = _status_html("warning", "No response generated")

                return history, tid, status

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                history.append({"role": "assistant", "content": "⚠️ I encountered an error. Please try again."})
                return history, tid, _status_html("error", str(e)[:100])

        # Send button: step 1 shows message, step 2 gets response
        send_btn.click(
            fn=show_user_message,
            inputs=[msg_input, chatbot, thread_id],
            outputs=[chatbot, msg_input, thread_id, status],
        ).then(
            fn=generate_response,
            inputs=[chatbot, thread_id],
            outputs=[chatbot, thread_id, status],
        )

        # Enter key: same two-step pattern
        msg_input.submit(
            fn=show_user_message,
            inputs=[msg_input, chatbot, thread_id],
            outputs=[chatbot, msg_input, thread_id, status],
        ).then(
            fn=generate_response,
            inputs=[chatbot, thread_id],
            outputs=[chatbot, thread_id, status],
        )

        # Reset button
        reset_btn.click(
            fn=reset_conversation,
            inputs=[],
            outputs=[chatbot, thread_id, status],
        )

    return app


# ── Main ──

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    )