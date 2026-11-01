
from typing import Annotated, Iterator
import logging
import operator
import os
from pathlib import Path

import gradio as gr
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_ollama import ChatOllama

from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LLM_LOCAL_URL = os.getenv(
    "LLM_LOCAL_URL",
    "http://localhost:11434"
)

if not LLM_MODEL_NAME:
    raise ValueError("LLM_MODEL_NAME is missing in .env")


# ============================================================
# 2. SHARED GRAPH STATE
# ============================================================

class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], operator.add]

    intent: str = ""
    intent_classified: bool = False

    next_step: str = ""

    solution_generated: bool = False
    security_audited: bool = False
    task_answered: bool = False


# ============================================================
# 3. LOCAL OLLAMA LLM
# ============================================================

llm = ChatOllama(
    model=LLM_MODEL_NAME.removeprefix("ollama/"),
    base_url=LLM_LOCAL_URL,
    temperature=0.1,
)


# ============================================================
# 4. INTENT CLASSIFIER AGENT
# ============================================================

def intent_classifier_node(state: AgentState):

    get_stream_writer()({
        "status": "Intent Classifier is analyzing the request..."
    })

    classifier_prompt = """
    You are an Intent Classification Agent.

    Analyze the user's request and classify it into
    exactly ONE of these categories:

    1. code_generation:
       User wants new Python code, functions, scripts,
       APIs, automation, or database implementations.

    2. explanation:
       User wants to understand a programming concept,
       technology, code, or technical topic.

    3. debugging:
       User provides existing code, an error, or a bug
       and wants it fixed, optimized, or troubleshooted.

    4. other:
       Greetings, general questions, or requests that
       do not belong to the above categories.

    Return ONLY one category name.

    Valid outputs:
    code_generation
    explanation
    debugging
    other
    """

    messages = [
        SystemMessage(content=classifier_prompt)
    ] + state.messages

    response = llm.invoke(messages)

    raw_intent = str(response.content).strip().lower()

    valid_intents = {
        "code_generation",
        "explanation",
        "debugging",
        "other",
    }

    intent = raw_intent.splitlines()[0].strip(
        " `\"'.,:;"
    )

    if intent not in valid_intents:
        intent = next(
            (
                item
                for item in valid_intents
                if item in raw_intent
            ),
            "other"
        )

    logger.info("Detected intent: %s", intent)

    return {
        "intent": intent,
        "intent_classified": True,
    }


# ============================================================
# 5. SUPERVISOR AGENT
# ============================================================

def supervisor_node(state: AgentState):

    if not state.intent_classified:
        next_step = "intent_classifier"

    elif state.intent == "code_generation":

        if not state.solution_generated:
            next_step = "code_generator"

        elif not state.security_audited:
            next_step = "security_auditor"

        else:
            next_step = "FINISH"

    elif state.intent == "debugging":

        if not state.solution_generated:
            next_step = "debugger"

        elif not state.security_audited:
            next_step = "security_auditor"

        else:
            next_step = "FINISH"

    elif state.intent in ["explanation", "other"]:

        if not state.task_answered:
            next_step = "general_response"

        else:
            next_step = "FINISH"

    else:
        next_step = "FINISH"

    get_stream_writer()({
        "status": f"Supervisor selected: {next_step}"
    })

    logger.info("Supervisor selected: %s", next_step)

    return {
        "next_step": next_step
    }


# ============================================================
# 6. CODE GENERATOR AGENT
# ============================================================

def code_generator_node(state: AgentState):

    get_stream_writer()({
        "status": "Code Generator is generating Python code..."
    })

    generator_prompt = """
    You are an expert Python developer.

    Understand the user's requirements and generate
    clean, functional, readable Python code.

    Requirements:

    1. Follow Python best practices.
    2. Write reusable and maintainable code.
    3. Include appropriate exception handling.
    4. Follow secure coding practices.
    5. Include necessary imports.
    6. Provide a simple explanation.
    7. Include an example of how to use the code.

    Return the complete solution.
    """

    messages = [
        SystemMessage(content=generator_prompt)
    ] + state.messages

    response = llm.invoke(messages)

    logger.info("Code generation completed")

    return {
        "messages": [
            AIMessage(
                content=response.content,
                name="code_generator"
            )
        ],
        "solution_generated": True,
    }


# ============================================================
# 7. EXPLANATION AGENT
# ============================================================

def explanation_node(state: AgentState):

    get_stream_writer()({
        "status": "Explanation Agent is preparing the explanation..."
    })

    explanation_prompt = """
    You are an expert Python programming instructor.

    Explain the user's requested concept in a simple,
    beginner-friendly and understandable manner.

    Follow this structure:

    1. What is it?
    2. Why do we use it?
    3. How does it work?
    4. Simple Python example.
    5. Real-world application.
    6. Important points to remember.

    Use practical examples.
    Avoid unnecessary complexity.
    """

    messages = [
        SystemMessage(content=explanation_prompt)
    ] + state.messages

    response = llm.invoke(messages)

    logger.info("Explanation completed")

    return {
        "messages": [
            AIMessage(
                content=response.content,
                name="explanation_agent"
            )
        ],
        "task_answered": True,
    }


# ============================================================
# 8. DEBUGGING AGENT
# ============================================================

def debugging_node(state: AgentState):

    get_stream_writer()({
        "status": "Debugging Agent is analyzing the code..."
    })

    debugger_prompt = """
    You are an expert Python debugging engineer.

    Analyze the user's code and reported error.

    Follow this structure:

    1. Identify the root cause.
    2. Explain why the error occurs.
    3. Provide corrected Python code.
    4. Explain the changes.
    5. Suggest ways to prevent the issue.

    If the user has not provided enough code or
    error details, explain what information is missing.

    Follow secure Python coding practices.
    """

    messages = [
        SystemMessage(content=debugger_prompt)
    ] + state.messages

    response = llm.invoke(messages)

    logger.info("Debugging completed")

    return {
        "messages": [
            AIMessage(
                content=response.content,
                name="debugger"
            )
        ],
        "solution_generated": True,
    }


# ============================================================
# 9. SECURITY AUDITOR AGENT
# ============================================================

def security_auditor_node(state: AgentState):

    get_stream_writer()({
        "status": "Security Auditor is reviewing the code..."
    })

    auditor_prompt = """
    You are an Application Security Auditor.

    Review the Python code generated or corrected
    by the previous agent.

    Check for:

    1. SQL injection vulnerabilities.
    2. Insecure imports and unsafe operations.
    3. Missing context managers.
    4. Improper exception handling.
    5. Hardcoded credentials or secrets.
    6. Unsafe file operations.
    7. Other common security vulnerabilities.

    If the code is secure, explain that it passed
    your review.

    If vulnerabilities exist, explain the issues
    and provide hardened, corrected code.

    Do not claim that code was executed or tested.
    """

    messages = [
        SystemMessage(content=auditor_prompt)
    ] + state.messages

    response = llm.invoke(messages)

    logger.info("Security audit completed")

    return {
        "messages": [
            AIMessage(
                content=response.content,
                name="security_auditor"
            )
        ],
        "security_audited": True,
    }


# ============================================================
# 10. GENERAL RESPONSE AGENT
# ============================================================

def general_response_node(state: AgentState):

    get_stream_writer()({
        "status": "General Response Agent is preparing a response..."
    })

    general_prompt = """
    You are a helpful AI assistant.

    Understand the user's request and provide
    a relevant, clear and useful response.

    You can answer greetings, general questions,
    and other requests.

    If the request is unclear, ask for clarification.

    Keep your response natural and helpful.
    """

    messages = [
        SystemMessage(content=general_prompt)
    ] + state.messages

    response = llm.invoke(messages)

    logger.info("General response completed")

    return {
        "messages": [
            AIMessage(
                content=response.content,
                name="general_response"
            )
        ],
        "task_answered": True,
    }


# ============================================================
# 11. BUILD LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(AgentState)

# Register nodes

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("intent_classifier", intent_classifier_node)
workflow.add_node("code_generator", code_generator_node)
workflow.add_node("explanation", explanation_node)
workflow.add_node("debugger", debugging_node)
workflow.add_node("security_auditor", security_auditor_node)
workflow.add_node("general_response", general_response_node)

# Starting point

workflow.add_edge(START, "supervisor")

# Conditional routing

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state.next_step,
    {
        "intent_classifier": "intent_classifier",
        "code_generator": "code_generator",
        "explanation": "explanation",
        "debugger": "debugger",
        "security_auditor": "security_auditor",
        "general_response": "general_response",
        "FINISH": END,
    }
)

# Workers return to supervisor

workflow.add_edge("intent_classifier", "supervisor")
workflow.add_edge("code_generator", "supervisor")
workflow.add_edge("explanation", "supervisor")
workflow.add_edge("debugger", "supervisor")
workflow.add_edge("security_auditor", "supervisor")
workflow.add_edge("general_response", "supervisor")

# Compile workflow

app = workflow.compile()


# ============================================================
# 12. GRADIO EXECUTION FUNCTION
# ============================================================

def run_agent_graph(task: str) -> Iterator[tuple[str, str]]:

    if not task.strip():
        yield "Enter a task before running agents.", ""
        return

    # Initialize shared state

    initial_state = {
        "messages": [
            HumanMessage(content=task)
        ]
    }

    progress_lines = [
        "Starting multi-agent workflow..."
    ]

    response_sections = []

    yield "\n".join(progress_lines), ""

    try:

        for stream_mode, event in app.stream(
            initial_state,
            config={"recursion_limit": 12},
            stream_mode=["updates", "custom"],
        ):

            # Custom progress events

            if stream_mode == "custom":

                progress_lines.append(
                    event["status"]
                )

                yield (
                    "\n".join(progress_lines),
                    "\n\n---\n\n".join(response_sections)
                )

                continue

            # Node updates

            for node_name, state_update in event.items():

                progress_lines.append(
                    f"Completed: {node_name}"
                )

                # Display classified intent

                if "intent" in state_update:

                    progress_lines.append(
                        f"Detected intent: {state_update['intent']}"
                    )

                # Display agent responses

                messages = state_update.get("messages", [])

                if messages:

                    last_message = messages[-1]

                    speaker = (
                        getattr(last_message, "name", None)
                        or node_name
                    )

                    response_sections.append(
                        f"### {speaker.replace('_', ' ').title()}\n\n"
                        f"{last_message.content}"
                    )

                yield (
                    "\n".join(progress_lines),
                    "\n\n---\n\n".join(response_sections)
                )

        progress_lines.append("Workflow finished successfully.")

        yield (
            "\n".join(progress_lines),
            "\n\n---\n\n".join(response_sections)
        )

    except Exception as error:

        logger.exception("Multi-agent workflow failed")

        progress_lines.append(
            f"Workflow failed: {error}"
        )

        yield (
            "\n".join(progress_lines),
            "\n\n---\n\n".join(response_sections)
        )


# ============================================================
# 13. GRADIO USER INTERFACE
# ============================================================

with gr.Blocks(
    title="LangGraph Multi-Agent AI Assistant"
) as demo:

    gr.Markdown("# LangGraph Multi-Agent AI Assistant")

    gr.Markdown(
        """
        Enter your request. The Intent Classifier identifies
        your requirement and the Supervisor routes it to
        the appropriate agent.

        Supported tasks:
        - Code Generation
        - Programming Explanation
        - Code Debugging
        - General Questions
        """
    )

    task_input = gr.Textbox(
        label="Enter your task",
        lines=5,
        placeholder=(
            "Example 1: Create a Python FastAPI endpoint.\n"
            "Example 2: Explain Python decorators.\n"
            "Example 3: Debug this TypeError.\n"
            "Example 4: What can you help me with?"
        )
    )

    run_button = gr.Button(
        "Run Agents",
        variant="primary"
    )

    progress_output = gr.Textbox(
        label="Workflow Progress",
        lines=12,
        interactive=False
    )

    response_output = gr.Markdown(
        label="Agent Responses"
    )

    run_button.click(
        fn=run_agent_graph,
        inputs=[task_input],
        outputs=[
            progress_output,
            response_output
        ]
    )


# ============================================================
# 14. LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name=os.getenv(
            "GRADIO_SERVER_NAME",
            "127.0.0.1"
        ),
        server_port=int(
            os.getenv(
                "GRADIO_SERVER_PORT",
                "7861"
            )
        ),
        theme=gr.themes.Soft()
    )
