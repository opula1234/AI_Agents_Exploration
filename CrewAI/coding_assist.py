"""Gradio application for reviewing Python code with local CrewAI agents."""

import logging
import os
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator

import gradio as gr
from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIRECTORY = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIRECTORY / "coding_assist.log"

load_dotenv(PROJECT_ROOT / ".env")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ollama/qwen2.5-coder:3b")
LLM_LOCAL_URL = os.getenv("LLM_LOCAL_URL", "http://localhost:11434")


def configure_logging() -> logging.Logger:
    """Configure console and rotating-file logs without recording submitted code."""
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    app_logger = logging.getLogger(__name__)
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False

    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename) == LOG_FILE.resolve()
        for handler in app_logger.handlers
    ):
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"
            )
        )
        app_logger.addHandler(file_handler)

    if not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in app_logger.handlers
    ):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        )
        app_logger.addHandler(console_handler)

    return app_logger


logger = configure_logging()

local_llm = LLM(
    model=LLM_MODEL_NAME,
    base_url=LLM_LOCAL_URL,
    temperature=0.2,
)


def run_crew_audit(code_snippet: str) -> Iterator[tuple[str, str]]:
    """Audit a submitted code snippet and yield process status and report."""
    audit_id = uuid.uuid4().hex[:8]
    if not code_snippet.strip():
        logger.warning(
            "Audit %s rejected because the submitted snippet was empty",
            audit_id,
        )
        yield "No code was provided.", ""
        return

    logger.info("Audit %s started", audit_id)
    started_at = time.perf_counter()
    yield f"Audit {audit_id} started. Waiting for the local model...", "Auditing in progress..."

    try:
        logger.info("Audit %s: preparing review and security agents", audit_id)
        performance_reviewer = Agent(
            role="Senior Python Engineer",
            goal="Identify bugs, anti-patterns, and computational inefficiencies in the code.",
            backstory=(
                "You are a seasoned software engineer who obsesses over clean architecture, "
                "PEP 8 standards, and optimal time/space complexity."
            ),
            llm=local_llm,
            verbose=True,
        )

        security_analyst = Agent(
            role="Application Security Auditor",
            goal=(
                "Detect security vulnerabilities (e.g., injection, unclosed resources, "
                "CWEs) and propose secure patches."
            ),
            backstory=(
                "You are an AppSec specialist trained in OWASP Top 10 vulnerabilities. "
                "You pinpoint exact exploits and provide hardened code alternatives."
            ),
            llm=local_llm,
            verbose=True,
        )

        review_task = Task(
            description=(
                "Analyze the following Python snippet for bugs, performance bottlenecks, "
                "and code style issues:\n\n"
                f"```python\n{code_snippet}\n```\n\n"
                "List clear bullet points for each issue found with recommended refactors."
            ),
            expected_output=(
                "A structured report detailing code smells, bugs, and suggested "
                "performance optimizations."
            ),
            agent=performance_reviewer,
        )

        security_task = Task(
            description=(
                "Review the original code and the findings from the Performance Reviewer. "
                "Focus specifically on security risks (e.g., SQL injection, resource exhaustion).\n\n"
                "Deliver a hardened version of the code that fixes all discovered issues."
            ),
            expected_output=(
                "A vulnerability summary with CWE classifications and the finalized, "
                "fully refactored, secure Python code."
            ),
            agent=security_analyst,
            context=[review_task],
        )

        code_audit_crew = Crew(
            agents=[performance_reviewer, security_analyst],
            tasks=[review_task, security_task],
            process=Process.sequential,
            verbose=True,
        )

        logger.info("Audit %s: running sequential review and security tasks", audit_id)
        result = code_audit_crew.kickoff()
        elapsed_seconds = time.perf_counter() - started_at
        logger.info("Audit %s completed in %.2f seconds", audit_id, elapsed_seconds)
        yield f"Audit {audit_id} completed in {elapsed_seconds:.2f} seconds.", str(result)
    except Exception as error:
        elapsed_seconds = time.perf_counter() - started_at
        logger.exception("Audit %s failed after %.2f seconds", audit_id, elapsed_seconds)
        yield (
            f"Audit {audit_id} failed after {elapsed_seconds:.2f} seconds. "
            f"See {LOG_FILE} for details.",
            f"Execution failed: {error}",
        )


default_snippet = """import sqlite3

def get_user(username):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    # SQL Injection flaw
    cursor.execute("SELECT * FROM users WHERE username = '" + username + "'")
    return cursor.fetchall()
"""

with gr.Blocks(title="CrewAI + Ollama Auditor") as demo:
    gr.Markdown("# Local CrewAI + Ollama Code Auditor")
    gr.Markdown(
        f"Reviews code using the configured Ollama model (`{LLM_MODEL_NAME}`)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            code_input = gr.Code(
                label="Source Code to Audit",
                language="python",
                value=default_snippet,
                lines=18,
            )
            submit_btn = gr.Button("Run Crew Audit", variant="primary")

        with gr.Column(scale=1):
            final_output = gr.Markdown(
                label="Hardened Audit Report",
                value="*Submit code on the left to see the final report here.*",
            )

    with gr.Accordion("Audit Process Log", open=True):
        terminal_logs = gr.Textbox(
            label="Audit status",
            lines=4,
            interactive=False,
            placeholder="Audit start, completion, and errors appear here.",
        )

    submit_btn.click(
        fn=run_crew_audit,
        inputs=[code_input],
        outputs=[terminal_logs, final_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
    )
