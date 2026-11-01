# Local AI Agent Applications

This repository contains three runnable Gradio applications that use an Ollama model:

- **CrewAI code auditor** (`CrewAI/coding_assist.py`): runs code-quality and security reviews.
- **LangGraph multi-agent assistant** (`LangGraph/Multi_Agent_Test.py`): classifies a request and routes it to code-generation, explanation, debugging, or general-response agents. It streams workflow progress and agent responses to the UI.
- **Basic LangGraph assistant** (`LangGraph/Multi_Agent.py`): generates a response and runs a security review.

> **Review with care:** Model-generated answers, findings, and patches can be incomplete or incorrect. Verify recommendations—especially security-related changes—before using them.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) installed and running
- Docker Desktop with its Linux engine running, if using Docker

## Configure Ollama

Start Ollama and pull the model used by the example configuration:

```powershell
ollama pull qwen2.5-coder:3b
```

The app reads these settings from the root `.env` file:

```dotenv
LLM_MODEL_NAME=ollama/qwen2.5-coder:3b
LLM_LOCAL_URL=http://localhost:11434
```

For local use, copy `.env.example` to `.env` if you do not already have a `.env`:

```powershell
Copy-Item .env.example .env
```

Edit `.env` to choose another Ollama model or URL if needed. Keep `.env` private; it is excluded from Git and Docker build contexts.

## Run locally with uv

From the repository root (`AI_Agents`):

```powershell
uv sync
uv run python .\CrewAI\coding_assist.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860), enter a Python snippet, and select **Run Crew Audit**. The first audit can take longer while Ollama loads the model. The two audit tasks run sequentially.

By default, the web app listens only on the local machine. To change the bind host or port, set `GRADIO_SERVER_NAME` or `GRADIO_SERVER_PORT` in `.env` before starting it.

### Run the LangGraph multi-agent assistant

The intent-aware application is `LangGraph/Multi_Agent_Test.py`. Despite the filename, it is a runnable Gradio application, not a pytest test module:

```powershell
uv run python .\LangGraph\Multi_Agent_Test.py
```

Open [http://127.0.0.1:7861](http://127.0.0.1:7861), enter a task, and select **Run Agents**. The intent classifier routes requests to the appropriate workflow:

- Code generation: generate code, then send it for security review.
- Debugging: ask the debugger agent to analyze the reported issue, then send its response for security review.
- Explanation: answer the programming or technical question.
- Other: provide a general response.

The progress panel reports classification, routing, and completed nodes. The response panel shows agent results as nodes finish; a model response itself is not streamed token by token. The graph limits an execution to 12 steps to prevent an unbounded workflow. The simpler `LangGraph/Multi_Agent.py` app is also available on port 7861; run only one LangGraph app at a time unless you configure different ports.

Both LangGraph apps load the root `.env`. They remove an optional `ollama/` prefix from `LLM_MODEL_NAME` before passing the tag to LangChain's `ChatOllama`; the Ollama tag itself should match a model shown by `ollama list`.

### Debug in VS Code

Open this repository as the VS Code workspace and select its `.venv\Scripts\python.exe` interpreter. Choose **Debug CrewAI Gradio app** in Run and Debug to debug the CrewAI app. To debug a LangGraph app, create/select a Python launch configuration for the desired file, for example `LangGraph/Multi_Agent_Test.py`, and set breakpoints in that file.

## Run with Docker

The current Docker image packages **only the CrewAI code auditor**; the LangGraph apps are not copied into the image. Ensure Ollama is running on the host and the desired model has been pulled. From the repository root, build the image:

```powershell
docker build -t crewai-auditor .
```

Run the app in a container. The URL override lets the app reach Ollama running on the Windows host. The port is published on loopback only, and the named volume preserves app logs:

```powershell
docker run --rm `
  -p 127.0.0.1:7860:7860 `
  --env-file .env `
  -e LLM_LOCAL_URL=http://host.docker.internal:11434 `
  -v crewai-auditor-logs:/app/CrewAI/logs `
  crewai-auditor
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). The container binds Gradio to `0.0.0.0` internally so Docker can forward the port; the published port above remains available only on the host's loopback interface.

To stop the container, press **Ctrl+C**. To start it again, repeat the `docker run` command. The Ollama model remains managed by Ollama on the host; it is not included in the Docker image.

## Logging

The app writes lifecycle events, audit IDs, elapsed time, and error tracebacks to:

```text
CrewAI/logs/coding_assist.log
```

The log file rotates when it reaches approximately 2 MB, keeping up to three backups. The app's own lifecycle logger does not write submitted snippets. CrewAI's verbose output may show details of task execution in the terminal/container logs, so avoid submitting sensitive code unless you are comfortable with your local logging setup.

When running in Docker, the `crewai-auditor-logs` named volume stores the app log directory. To inspect container output, run:

```powershell
docker logs <container-id>
```

## Troubleshooting

- **Cannot connect to Ollama:** Confirm Ollama is running and `LLM_LOCAL_URL` is correct. Local execution uses `http://localhost:11434`; the Docker command uses `http://host.docker.internal:11434`.
- **Model not found:** Run `ollama pull qwen2.5-coder:3b`, or pull the model named in `LLM_MODEL_NAME`.
- **Port 7860 is in use:** Set `GRADIO_SERVER_PORT` in `.env` for local execution. For Docker, change the host side of the port mapping, for example `-p 127.0.0.1:7861:7860`, and visit port 7861.
- **Port 7861 is in use:** Another LangGraph app may already be running. Stop it with **Ctrl+C**, or set `GRADIO_SERVER_PORT` to a free port before starting the other app.
- **Docker cannot connect to its engine:** Start Docker Desktop and wait for its Linux engine to become ready, then retry `docker build`.
- **Audit appears slow:** The two agents execute one after another, and the first request may need to load the Ollama model into memory.

## Privacy and network exposure

The app has no authentication configured. Keep it bound to localhost unless you deliberately add authentication and appropriate network protections. The Ollama model is separate from the app container; use a local Ollama endpoint if you want snippets to remain on your machine.

## Interview preparation

See [INTERVIEW_QUESTIONS.md](./INTERVIEW_QUESTIONS.md) for beginner-to-advanced Python, Gradio, Ollama, CrewAI, LangGraph, testing, and deployment questions with answers based on this project.