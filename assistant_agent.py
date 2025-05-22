"""
============================
assistant_agent.py
----------------------------
Simple command‑line assistant that loads an open‑source LLM (Phi‑2 by Microsoft)
and reasons over the MCP tools exposed by `astronomy_mcp_server.py`.

Install extra deps:
    pip install transformers accelerate sentencepiece langchain mcp

Run the server in one terminal:
    mcp run astronomy_mcp_server.py

Then chat in another terminal:
    python assistant_agent.py
============================
"""

import asyncio
from typing import List

from mcp.client import Client
from langchain.agents import Tool, initialize_agent
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 1 — load small open‑source LLM (Phi‑2 is 2.7 B params; runs on CPU/GPU)
# ---------------------------------------------------------------------------
MODEL_NAME = "microsoft/phi-2"
print("Loading Phi‑2 model… (≈ 500 MB on first download)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

hf_llm = HuggingFacePipeline.from_model_id(
    model_id=MODEL_NAME,
    task="text-generation",
    model_kwargs={
        "torch_dtype": model.dtype,
        "device_map": "auto",
        "max_new_tokens": 512,
        "temperature": 0.2,
    },
)

# ---------------------------------------------------------------------------
# 2 — connect to the local MCP server and wrap each tool
# ---------------------------------------------------------------------------
async def build_tools() -> List[Tool]:
    client = await Client.connect("http://127.0.0.1:8000/sse")

    async def make_caller(tool_name: str):
        async def _caller(**kwargs):
            result = await client.call_tool(tool_name, kwargs)
            return result["output"]
        return _caller

    tools: List[Tool] = []
    for t in await client.list_tools():
        caller = await make_caller(t["name"])
        tools.append(
            Tool(
                name=t["name"],
                description=t["description"],
                func=None,          # sync fallback not implemented
                coroutine=caller,   # async implementation
            )
        )
    return tools

# ---------------------------------------------------------------------------
# 3 — chat loop
# ---------------------------------------------------------------------------
async def main():
    tools = await build_tools()

    agent = initialize_agent(
        tools,
        hf_llm,
        agent="chat-zero-shot-react-description",
        verbose=True,
    )

    print("Astronomy Assistant ready!  Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower().strip() in {"exit", "quit"}:
            break
        answer = agent.run(query)
        print(f"Assistant: {answer}\n")

if __name__ == "__main__":
    asyncio.run(main())
