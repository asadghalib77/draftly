from langchain.agents import create_agent
from datetime import datetime
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

#System Prompt
SYSTEM = """
You are a helpful AI email draft agent!
"""

#Agent Tool(s)
def get_today_date():
    """Return today's date."""
    return datetime.now().strftime("%Y-%m-%d")

#AI Agent
agent = create_agent(
    model="ollama:gemma4:31b-cloud",
    tools=[get_today_date],
    system_prompt=SYSTEM,
    checkpointer=InMemorySaver()
)

thread_config = {
    "configurable": {
        "thread_id": "draftly-thread-1"
    }
}

while True:
    q = input("Enter email draft idea: ")
    if q.lower() in {"bye", "exit", "quit"}:
        break

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config = thread_config
        )

        print("Agent: ", result["messages"][-1].text)
    
    except Exception as e:
        print("Agent connection failed: ", e)