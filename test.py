from typing import Annotated, Literal
from pydantic import BaseModel, Field
import operator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# 1. Define Shared Graph State


class AgentState(BaseModel):
    # Operator.add ensures new messages append to the list rather than overwriting
    messages: Annotated[list[BaseMessage], operator.add]
    next_step: str = ""


# 2. Configure Local Ollama LLM
llm = ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.1,
)

# 3. Router Schema for Structured Output


class RouterDecision(BaseModel):
    """Routing choice made by the supervisor."""
    next_agent: Literal["code_generator", "security_auditor", "FINISH"] = Field(
        ...,
        description="The next specialist agent to call, or FINISH if the user request is fulfilled and audited."
    )


# 4. Supervisor Node
supervisor_prompt = (
    "You are a supervisor managing two worker agents:\n"
    "1. `code_generator`: Writes functional Python code based on requirements.\n"
    "2. `security_auditor`: Inspects written code for SQL injections, resource leaks, and OWASP flaws.\n\n"
    "Workflow rules:\n"
    "- If no code has been generated yet, route to `code_generator`.\n"
    "- If code was generated but has not been audited by the security reviewer, route to `security_auditor`.\n"
    "- If the code is audited and clean, route to `FINISH`."
)


def supervisor_node(state: AgentState):
    structured_llm = llm.with_structured_output(RouterDecision)
    messages = [SystemMessage(content=supervisor_prompt)] + state.messages
    decision = structured_llm.invoke(messages)

    print(f"\n[Supervisor] -> Delegating to: {decision.next_agent}")
    return {"next_step": decision.next_agent}

# 5. Worker Agent 1: Code Generator


def code_generator_node(state: AgentState):
    generator_prompt = (
        "You are an expert Python engineer. Provide clean, functional Python code "
        "addressing the user's requirements. Keep explanations brief."
    )
    messages = [SystemMessage(content=generator_prompt)] + state.messages
    response = llm.invoke(messages)

    print("[Code Generator] Completed draft.")
    return {"messages": [HumanMessage(content=response.content, name="code_generator")]}

# 6. Worker Agent 2: Security Auditor


def security_auditor_node(state: AgentState):
    auditor_prompt = (
        "You are an Application Security Auditor. Inspect the generated Python code in the conversation. "
        "Check for SQL injection, insecure imports, or missing context managers. "
        "State whether it is secure or provide hardened, corrected code."
    )
    messages = [SystemMessage(content=auditor_prompt)] + state.messages
    response = llm.invoke(messages)

    print("[Security Auditor] Completed security review.")
    return {"messages": [HumanMessage(content=response.content, name="security_auditor")]}


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

# 8. Run the Graph
if __name__ == "__main__":
    task = "Write a Python function to query user records by username from a SQLite database."
    print(f"Task: {task}")
    print("=" * 60)

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "next_step": "",
    }

    # Stream the graph execution step by step
    for step in app.stream(initial_state):
        for node_name, state_update in step.items():
            print(f"\n--- Node '{node_name}' Finished ---")
            if "messages" in state_update and state_update["messages"]:
                last_msg = state_update["messages"][-1]
                print(
                    f"Output ({getattr(last_msg, 'name', 'LLM')}):\n{last_msg.content[:300]}...\n")
