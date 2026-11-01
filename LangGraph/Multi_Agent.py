from typing import Annotated, Iterator
import logging
from pydantic import BaseModel
import operator
import os
from pathlib import Path

import gradio as gr
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LLM_LOCAL_URL = os.getenv("LLM_LOCAL_URL", "http://localhost:11434")

# 1. Define Shared Graph State


class AgentState(BaseModel):
    # Operator.add ensures new messages append to the list rather than overwriting
    messages: Annotated[list[BaseMessage], operator.add]
    next_step: str = ""
    code_generated: bool = False
    security_audited: bool = False


# 2. Configure Local Ollama LLM
llm = ChatOllama(
    model=LLM_MODEL_NAME.removeprefix("ollama/"),
    base_url=LLM_LOCAL_URL,
    temperature=0.1,
)

# 4. Supervisor Node


def supervisor_node(state: AgentState):
    if not state.code_generated:
        next_step = "code_generator"
    elif not state.security_audited:
        next_step = "security_auditor"
    else:
        next_step = "FINISH"

    get_stream_writer()({"status": f"Supervisor selected: {next_step}"})
    logger.info("Supervisor selected %s", next_step)
    return {"next_step": next_step}

# 5. Worker Agent 1: Code Generator


def code_generator_node(state: AgentState):
    get_stream_writer()({"status": "Code generator is drafting a response..."})
    generator_prompt = (
        "You are an expert Python engineer. Provide clean, functional Python code "
        "addressing the user's requirements. Keep explanations brief."
    )
    messages = [SystemMessage(content=generator_prompt)] + state.messages
    response = llm.invoke(messages)

    logger.info("Code generator completed draft")
    return {
        "messages": [HumanMessage(content=response.content, name="code_generator")],
        "code_generated": True,
    }

# 6. Worker Agent 2: Security Auditor


def security_auditor_node(state: AgentState):
    get_stream_writer()({"status": "Security auditor is reviewing the response..."})
    auditor_prompt = (
        "You are an Application Security Auditor. Inspect the generated Python code in the conversation. "
        "Check for SQL injection, insecure imports, or missing context managers. "
        "State whether it is secure or provide hardened, corrected code."
    )
    messages = [SystemMessage(content=auditor_prompt)] + state.messages
    response = llm.invoke(messages)

    logger.info("Security auditor completed review")
    return {
        "messages": [HumanMessage(content=response.content, name="security_auditor")],
        "security_audited": True,
    }


# 7. Assemble the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("code_generator", code_generator_node)
workflow.add_node("security_auditor", security_auditor_node)

# Add Edges
workflow.add_edge(START, "supervisor")

# Conditional Routing from Supervisor
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state.next_step,
    {
        "code_generator": "code_generator",
        "security_auditor": "security_auditor",
        "FINISH": END,
    },
)

# Workers loop back to Supervisor to evaluate next action
workflow.add_edge("code_generator", "supervisor")
workflow.add_edge("security_auditor", "supervisor")

# Compile graph
app = workflow.compile()

# 8. Gradio interface
def run_agent_graph(task: str) -> Iterator[tuple[str, str]]:
    """Run the agent graph and yield progress and accumulated agent responses."""
    if not task.strip():
        yield "Enter a task before running the agents.", ""
        return

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "next_step": "",
    }
    progress_lines = ["Starting multi-agent workflow..."]
    response_sections: list[str] = []
    yield "\n".join(progress_lines), ""

    try:
        for stream_mode, event in app.stream(
            initial_state,
            config={"recursion_limit": 12},
            stream_mode=["updates", "custom"],
        ):
            if stream_mode == "custom":
                progress_lines.append(event["status"])
                yield "\n".join(progress_lines), "\n\n---\n\n".join(response_sections)
                continue

            for node_name, state_update in event.items():
                progress_lines.append(f"Completed: {node_name}")

                messages = state_update.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    speaker = getattr(last_message, "name", None) or node_name
                    response_sections.append(
                        f"### {speaker.replace('_', ' ').title()}\n\n"
                        f"{last_message.content}"
                    )

                yield "\n".join(progress_lines), "\n\n---\n\n".join(response_sections)

        progress_lines.append("Workflow finished.")
        yield "\n".join(progress_lines), "\n\n---\n\n".join(response_sections)
    except Exception as error:
        logger.exception("Multi-agent workflow failed")
        progress_lines.append(f"Workflow failed: {error}")
        yield "\n".join(progress_lines), "\n\n---\n\n".join(response_sections)


with gr.Blocks(title="LangGraph Multi-Agent Code Assistant") as demo:
    gr.Markdown("# LangGraph Multi-Agent Code Assistant")
    gr.Markdown(
        "Describe the Python task. The supervisor will route it to the code "
        "generator and security auditor, showing each completed graph step below."
    )

    task_input = gr.Textbox(
        label="Task",
        lines=4,
        placeholder="For example: Write a Python function to query users by username from SQLite.",
    )
    run_button = gr.Button("Run Agents", variant="primary")

    progress_output = gr.Textbox(
        label="Workflow Progress",
        lines=8,
        interactive=False,
    )
    response_output = gr.Markdown(label="Agent Responses")

    run_button.click(
        fn=run_agent_graph,
        inputs=[task_input],
        outputs=[progress_output, response_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7861")),
        theme=gr.themes.Soft(),
    )
