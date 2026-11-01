# Interview Questions and Answers

This guide progresses from Python fundamentals to application architecture, using the Gradio, CrewAI, LangGraph, and Ollama applications in this repository as examples.

## Beginner

### 1. What is Python, and why is it commonly used?

Python is a high-level, general-purpose programming language. Its readable syntax, standard library, and ecosystem of packages make it useful for automation, web applications, data processing, and AI integrations.

### 2. What is the difference between a list and a tuple?

A list is mutable, so its contents can be changed after creation. A tuple is immutable, so its item references cannot be reassigned after creation. Use a list for a collection that changes and a tuple for a fixed grouping of values.

### 3. What is a Python dictionary?

A dictionary stores key-value pairs. Keys must be hashable, and lookup by key is typically efficient. Application code can use a dictionary to map a graph node name to its state update.

### 4. What is a function, and why use one?

A function packages a named operation that can accept inputs and return a result. Functions make code easier to reuse, test, and understand. In this project, graph node functions each implement a distinct agent responsibility.

### 5. What is an exception?

An exception represents an error or unusual condition during execution. `try`/`except` can handle expected failure paths, while logging the exception with its traceback helps diagnose unexpected failures. Avoid catching errors and then pretending the operation succeeded.

### 6. What is a virtual environment?

A virtual environment isolates a project's Python packages from other projects and the system installation. This repository uses `uv` to synchronize its environment from `pyproject.toml` and `uv.lock`.

### 7. What is an environment variable?

An environment variable supplies configuration outside source code, such as a model name, host, or port. A `.env` file is a convenient local source for these values; it should not be committed if it contains private information.

## Intermediate

### 8. What is a generator in Python?

A generator is an iterator that produces values lazily, commonly using `yield`. A Gradio callback that yields repeated output updates can display progress while a longer workflow continues.

### 9. What is a type annotation?

A type annotation describes the expected type of a parameter, variable, or return value. It helps readers and static analysis tools understand code, but Python does not generally enforce annotations at runtime.

### 10. What is the purpose of `pathlib.Path`?

`Path` provides an object-oriented way to work with filesystem paths. It helps construct paths relative to a script instead of depending on the terminal's current working directory.

### 11. Why use parameterized SQL queries?

Parameterized queries keep SQL commands separate from user-provided values. This avoids constructing SQL through string concatenation and helps prevent SQL injection. With `sqlite3`, use placeholders such as `?` and pass values separately.

### 12. What is Gradio used for in this project?

Gradio creates a web interface around Python functions. The LangGraph app accepts a task in a textbox, invokes a generator callback, and updates progress and response components as that callback yields values.

### 13. What is Ollama, and how does this application use it?

Ollama runs and serves language models locally. The apps create a LangChain `ChatOllama` client configured with a model tag and server URL. The model must already be available to the Ollama server.

### 14. Why does the LangGraph app remove the `ollama/` prefix from the model name?

The shared environment configuration may use a provider-qualified name such as `ollama/qwen2.5-coder:3b`, while `ChatOllama` expects the Ollama model tag `qwen2.5-coder:3b`. Removing the optional prefix adapts the shared setting for that client.

### 15. What is the difference between CrewAI and LangGraph in this repository?

The CrewAI app defines agents and tasks and runs its code-quality review followed by a security review. The LangGraph app models routing and worker steps as graph nodes and edges, allowing it to choose different paths based on a classified request.

### 16. What is state in a graph workflow?

State is the data passed between graph nodes. The LangGraph app stores conversation messages and workflow flags such as detected intent, whether a solution has been generated, and whether it has been audited. Nodes return partial updates that the graph applies to the shared state.

### 17. How does a conditional edge work in LangGraph?

A conditional edge calls a routing function and selects the next node based on its return value. In this app, the supervisor's `next_step` selects an agent node or `FINISH`, which ends the graph.

### 18. What is the purpose of `operator.add` in an annotated state field?

When used as a LangGraph reducer, `operator.add` tells the graph to combine new message-list updates with the existing list rather than replace the entire list. This preserves conversation history as agents contribute messages.

### 19. What is the difference between logging and `print()`?

`print()` writes unstructured text to standard output. The `logging` module supports severity levels, timestamps, and routing to configured handlers. Logs should avoid secrets and sensitive user content.

## Advanced

### 20. Why should workflow transitions use explicit state rather than relying only on an LLM's routing decision?

If a model repeatedly selects the same worker, the graph can loop until it reaches its recursion limit. Explicit state flags make required stages and completion conditions deterministic—for example, classify once, generate or debug once, audit when needed, then finish. Model-based routing is useful when decisions are genuinely dynamic, but its choices should be validated and bounded.

### 21. What does reaching a graph recursion limit mean?

It means the graph did not reach an end condition within the configured maximum number of steps. This can indicate a routing loop, although a legitimate workflow could also require a higher bound. Inspect the routing and state transitions rather than simply increasing the limit.

### 22. How does graph progress streaming differ from token streaming?

Graph progress streaming sends events such as “security auditor started” or a completed node's response. Token streaming sends pieces of a model response as they are generated. An app can stream node-level progress while still waiting for a full response from each model call.

### 23. How can you prevent an LLM agent workflow from looping indefinitely?

Use explicit completion flags, validate routing choices against known nodes, set a finite graph recursion limit, and show failures to the user. Test all expected routes, including completion and error paths.

### 24. How would you test this workflow without calling a real model?

Inject or monkeypatch a fake model that returns predictable messages. Run the graph and assert that expected nodes ran in the correct order, outputs were accumulated, and the workflow completed. Keep routing tests separate from integration tests that require Ollama.

### 25. How would you make model calls more reliable?

Validate required configuration before serving requests, handle connection and model-not-found errors explicitly, set reasonable request timeouts, log a run identifier, and avoid returning success-shaped results when a call fails. Use bounded retries only for transient failures.

### 26. What security concerns apply to a local code-review assistant?

Treat submitted code as untrusted input. Do not execute it, avoid storing or logging it unless necessary, keep the UI on loopback unless authentication and network controls are added, and do not treat an LLM's security review as proof that code is safe.

### 27. What is the difference between Docker build-time and run-time configuration?

Build-time instructions create the image and install dependencies. Run-time configuration supplies values such as environment variables, port mappings, and volumes when a container starts. Keeping `.env` out of the image lets configuration vary without rebuilding and avoids embedding secrets.

### 28. Why does a Docker container use `host.docker.internal` to call Ollama on Windows?

Inside a container, `localhost` refers to that container. On Docker Desktop, `host.docker.internal` lets the container reach services running on the host, such as Ollama on port 11434.

### 29. How would you extend this project for production?

Add authentication and authorization, validate and limit request size, use structured logging with sensitive-data redaction, add health checks and request timeouts, pin and scan dependencies, expose metrics, and run behind a properly configured reverse proxy. Add automated unit and integration tests and define operational limits for model calls.

### 30. How would you decide between one agent and a multi-agent workflow?

Start with the simplest workflow that meets the quality requirement. Multiple agents can separate responsibilities and add review stages, but they also add latency, complexity, cost, and failure points. Measure whether the additional stage improves outcomes enough to justify those costs.
