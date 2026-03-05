"""
Multi-Agent Customer Support Demo
Built with LangGraph, LangChain, and Gradio
FIX #3:  _extract_response now collects responses from ALL agents, not just the last one.
FIX #15: Added logging for response extraction debugging.
FIX #G6: Use messages format (dicts with role/content) for Chatbot history.
FIX #G7: Moved css and theme back into gr.Blocks() — launch() ignores them in this version.
FIX #G8: Added show_label=False to Chatbot to hide the "Chatbot" header label.
FIX #G9: Fixed examples CSS — force single-column layout, allow text wrapping, no truncation.
"""

import os
import uuid
import logging
import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────
_graph = None
_checkpointer = None
_store = None
_init_error = None


def initialize_system():
    global _graph, _checkpointer, _store, _init_error

    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    if not api_key:
        _init_error = (
            "OPENAI_API_KEY is not set. Please add it in your "
            "Hugging Face Space Settings under Repository secrets."
        )
        logger.error(_init_error)
        return False

    try:
        from database import verify_database

        if not verify_database():
            _init_error = "Failed to download or initialize the Chinook database."
            logger.error(_init_error)
            return False

        logger.info("Database initialized OK.")

        from graph_builder import build_graph

        _graph, _checkpointer, _store = build_graph(
            model_name=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base if api_base else None,
        )
        logger.info("Agent graph built OK.")
        return True

    except Exception as exc:
        _init_error = f"Initialization failed: {exc}"
        logger.error(_init_error, exc_info=True)
        return False


# ─────────────────────────────────────────────
# Conversation Manager
# ─────────────────────────────────────────────
class ConversationManager:
    def __init__(self, session=None):
        session = session or {}
        self.thread_id = session.get("thread_id", str(uuid.uuid4()))
        self.verified = session.get("verified", False)
        self.awaiting_verification = session.get("awaiting_verification", False)
        self.customer_id = session.get("customer_id", None)
        self.turn_count = session.get("turn_count", 0)

    def to_dict(self):
        return {
            "thread_id": self.thread_id,
            "verified": self.verified,
            "awaiting_verification": self.awaiting_verification,
            "customer_id": self.customer_id,
            "turn_count": self.turn_count,
        }

    def get_config(self):
        return {
            "configurable": {
                "thread_id": self.thread_id,
                "user_id": self.customer_id or "unknown",
            }
        }


# ─────────────────────────────────────────────
# Response Extraction (FIX #3)
# ─────────────────────────────────────────────

_SKIP_PREFIXES = (
    "Customer verified successfully",
    "The verified customer_id",
)
_SKIP_NAME_PATTERNS = ("transfer_to_",)


def _extract_response(result):
    """
    Extract the final user-facing response from the graph result.
    FIX #3: Collects substantive AI responses from all agents, not just the last one.
    """
    if not result or "messages" not in result:
        return "I could not generate a response. Please try again."

    messages = result["messages"]
    ai_responses = []

    for msg in messages:
        content = getattr(msg, "content", "")
        msg_type = getattr(msg, "type", "")
        name = getattr(msg, "name", "")
        tool_calls = getattr(msg, "tool_calls", None)

        if msg_type in ("tool", "system", "human"):
            continue
        if not content or not content.strip():
            continue
        if any(pattern in (name or "") for pattern in _SKIP_NAME_PATTERNS):
            continue
        if any(content.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue
        if tool_calls and not content.strip():
            continue

        ai_responses.append(content.strip())

    if not ai_responses:
        return "I have processed your request. Is there anything else I can help with?"

    last_response = ai_responses[-1]
    if len(last_response) >= 20:
        return last_response

    seen = set()
    combined = []
    for resp in ai_responses:
        if resp not in seen and len(resp) >= 20:
            seen.add(resp)
            combined.append(resp)

    if combined:
        return "\n\n".join(combined)

    return last_response


# ─────────────────────────────────────────────
# Chat Logic  (FIX #G6: messages format)
# ─────────────────────────────────────────────

def make_status(kind, text):
    icons = {"ok": "✅", "verify": "🔐", "chat": "💬", "error": "⚠️", "new": "🆕"}
    return f"{icons.get(kind, '')} {text}"


def add_user_message(user_msg, history):
    """Append the user message as a messages-format dict and clear input."""
    if not user_msg or not user_msg.strip():
        return history, ""
    history = history + [{"role": "user", "content": user_msg}]
    return history, ""


def run_agent(history, session_state):
    """Run the agent and append assistant reply as a messages-format dict."""
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    if not history:
        return history, session_state, make_status("new", "Ready. Send a message to begin.")

    user_message = history[-1]["content"]

    if not _graph:
        err = _init_error or "System not ready. Please check Space secrets."
        history.append({"role": "assistant", "content": f"⚠️ {err}"})
        return history, session_state, make_status("error", err)

    conv = ConversationManager(session_state)
    conv.turn_count += 1

    try:
        config = conv.get_config()

        if conv.awaiting_verification:
            logger.info("Resuming graph with verification input...")
            result = _graph.invoke(Command(resume=user_message), config=config)
            conv.awaiting_verification = False
        else:
            logger.info(f"New message (turn {conv.turn_count}): {user_message[:100]}")
            result = _graph.invoke(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
            )

        # Log all messages for debugging (FIX #15)
        if result and "messages" in result:
            for i, msg in enumerate(result["messages"]):
                msg_type = getattr(msg, "type", "?")
                name = getattr(msg, "name", "")
                content_preview = getattr(msg, "content", "")[:80]
                logger.info(f"  MSG[{i}] type={msg_type} name={name} content={content_preview}")

        response_text = _extract_response(result)

        try:
            gstate = _graph.get_state(config)
            if gstate and gstate.next and "human_input" in gstate.next:
                conv.awaiting_verification = True
        except Exception:
            pass

        if result and result.get("customer_id"):
            conv.customer_id = str(result["customer_id"])
            conv.verified = True

        if conv.awaiting_verification:
            status = make_status("verify", "Awaiting verification. Provide your Customer ID, email, or phone.")
        elif conv.verified:
            status = make_status("ok", f"Verified as Customer #{conv.customer_id}")
        else:
            status = make_status("chat", "Active conversation")

        history.append({"role": "assistant", "content": response_text})

    except Exception as exc:
        logger.error(f"Error: {exc}", exc_info=True)
        history.append({
            "role": "assistant",
            "content": (
                "I encountered an error processing your request. "
                "Please try again or start a new chat.\n\n"
                f"`{str(exc)[:300]}`"
            ),
        })
        status = make_status("error", "An error occurred")

    return history, conv.to_dict(), status


def reset_conversation(_s):
    return [], ConversationManager().to_dict(), make_status("new", "New conversation started")


# ─────────────────────────────────────────────
# Examples
# ─────────────────────────────────────────────
EXAMPLES = [
    "My customer ID is 1. What was my most recent purchase?",
    "My phone number is +55 (12) 3923-5555. How much was my most recent invoice?",
    "What albums do you have by the Rolling Stones?",
    "Do you have any songs by AC/DC? I love rock music!",
    "My email is luisg@embraer.com.br. Who helped me with my latest invoice?",
    "What songs do you have in the Jazz genre?",
    "My customer ID is 3. What is my most recent purchase? Also, what albums do you have by U2?",
]

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    font-family: 'DM Sans', system-ui, -apple-system, sans-serif !important;
}
.gradio-container {
    max-width: 820px !important;
    margin: 0 auto !important;
    padding: 0 1rem !important;
}
.hdr { text-align: center; padding: 1.75rem 0 1.25rem; }
.hdr h1 {
    font-size: 1.6rem; font-weight: 700;
    margin: 0 0 0.3rem; letter-spacing: -0.03em;
}
.hdr p {
    font-size: 0.84rem; margin: 0; opacity: 0.55; line-height: 1.5;
}
.pills {
    display: flex; justify-content: center; gap: 0.4rem;
    margin-top: 0.7rem; flex-wrap: wrap;
}
.pill {
    font-size: 0.68rem; font-weight: 500;
    padding: 0.18rem 0.55rem; border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.08);
    opacity: 0.6; letter-spacing: 0.01em;
    white-space: nowrap;
}
/* Hide the "Chatbot" label */
.chat-wrap > label,
.chat-wrap > .label-wrap,
.chat-wrap > span.svelte-1gfkn6j {
    display: none !important;
}
.chat-wrap .chatbot { border-radius: 12px !important; }
.chat-wrap .message-wrap { padding: 0.5rem !important; }
.status-strip textarea,
.status-strip input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    padding: 0.4rem 0.7rem !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    min-height: unset !important;
    height: 1.9rem !important;
    line-height: 1rem !important;
    opacity: 0.85;
}
.input-row { margin-top: 0.35rem; }
.input-row .textbox textarea {
    font-size: 0.92rem !important;
    padding: 0.65rem 0.85rem !important;
    border-radius: 10px !important;
}
.input-row button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    min-height: 42px !important;
}

/* ── FIX #G9: Examples — single column, full text, no truncation ── */
.examples-block {
    margin-top: 0.6rem;
    padding-top: 0.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.examples-block .label-wrap { margin-bottom: 0.3rem !important; }
.examples-block .label-wrap span {
    font-size: 0.72rem !important; opacity: 0.4 !important;
    text-transform: uppercase !important; letter-spacing: 0.05em !important;
    font-weight: 600 !important;
}

/* Force the examples gallery into a single column */
.examples-block .gallery,
.examples-block .gallery-item,
.examples-block .examples-table,
.examples-block .examples-table tbody,
.examples-block .examples-table tr,
.examples-block .examples-table td {
    display: block !important;
    width: 100% !important;
}

/* Force the examples grid wrapper to single column */
.examples-block .gallery {
    grid-template-columns: 1fr !important;
    display: grid !important;
    gap: 0.35rem !important;
}

/* Make sure example buttons show full text */
.examples-block button,
.examples-block button.gallery-item,
.examples-block .examples-table button {
    font-size: 0.82rem !important;
    padding: 0.5rem 0.85rem !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    transition: border-color 0.15s ease !important;
    white-space: normal !important;
    word-break: break-word !important;
    overflow: visible !important;
    text-overflow: unset !important;
    max-width: 100% !important;
    width: 100% !important;
    height: auto !important;
    min-height: unset !important;
    text-align: left !important;
    line-height: 1.4 !important;
}
.examples-block button:hover,
.examples-block button.gallery-item:hover,
.examples-block .examples-table button:hover {
    border-color: rgba(255,255,255,0.2) !important;
}

/* Also target the inner span/text inside example buttons */
.examples-block button span,
.examples-block button div,
.examples-block button p {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
}

.ftr {
    text-align: center; font-size: 0.68rem; opacity: 0.28;
    padding: 0.7rem 0 0.5rem; letter-spacing: 0.015em;
}
footer { display: none !important; }
@media (max-width: 640px) {
    .gradio-container { padding: 0 0.5rem !important; }
    .hdr h1 { font-size: 1.25rem; }
    .hdr p { font-size: 0.78rem; }
    .pills { gap: 0.3rem; }
    .pill { font-size: 0.6rem; padding: 0.15rem 0.4rem; }
    .chat-wrap .chatbot { height: 55vh !important; }
}
"""


# ─────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────
def build_demo():
    with gr.Blocks(
        title="Multi-Agent Customer Support",
        css=CSS,
        theme=gr.themes.Base(),
    ) as app:

        # Header
        gr.HTML("""
        <div class="hdr">
            <h1>🎵 Multi-Agent Customer Support</h1>
            <p>AI powered support for a digital music store, built with LangGraph multi-agent orchestration</p>
            <div class="pills">
                <span class="pill">🤖 LangGraph</span>
                <span class="pill">🔀 Supervisor Routing</span>
                <span class="pill">🔐 Identity Verification</span>
                <span class="pill">🧠 Long Term Memory</span>
                <span class="pill">🎸 Chinook DB</span>
            </div>
        </div>
        """)

        # State
        session_state = gr.State(value=ConversationManager().to_dict())

        # Chat
        chatbot = gr.Chatbot(
            value=[],
            height=480,
            show_label=False,
            label=None,
            elem_classes=["chat-wrap"],
        )

        # Status
        status_text = gr.Textbox(
            value=make_status("new", "Ready. Send a message to begin."),
            show_label=False,
            interactive=False,
            container=False,
            max_lines=1,
            elem_classes=["status-strip"],
        )

        # Input row
        with gr.Row(elem_classes=["input-row"]):
            msg_input = gr.Textbox(
                placeholder="Ask about the music catalog, invoices, or provide your customer ID…",
                show_label=False,
                container=False,
                scale=6,
                autofocus=True,
                elem_classes=["textbox"],
            )
            send_btn = gr.Button("Send", variant="primary", scale=1, min_width=72)
            clear_btn = gr.Button("New Chat", variant="secondary", scale=1, min_width=88)

        # Examples
        with gr.Group(elem_classes=["examples-block"]):
            gr.Examples(
                examples=EXAMPLES,
                inputs=msg_input,
                label="Suggested prompts",
                examples_per_page=7,
                cache_examples=False,
            )

        # Footer
        gr.HTML('<div class="ftr">Built with LangGraph · LangChain · Gradio · Chinook Database</div>')

        # ── Two step event chain ──
        send_btn.click(
            fn=add_user_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
            api_name=False,
            queue=False,
        ).then(
            fn=run_agent,
            inputs=[chatbot, session_state],
            outputs=[chatbot, session_state, status_text],
            api_name=False,
        )

        msg_input.submit(
            fn=add_user_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
            api_name=False,
            queue=False,
        ).then(
            fn=run_agent,
            inputs=[chatbot, session_state],
            outputs=[chatbot, session_state, status_text],
            api_name=False,
        )

        clear_btn.click(
            fn=reset_conversation,
            inputs=[session_state],
            outputs=[chatbot, session_state, status_text],
            api_name=False,
        )

    return app


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────
logger.info("=== Starting Multi-Agent Customer Support Demo ===")
_ok = initialize_system()
if not _ok:
    logger.warning(
        "System init incomplete. App will show an error banner "
        "until OPENAI_API_KEY is set in Space secrets."
    )

demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )