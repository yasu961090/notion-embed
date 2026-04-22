#!/usr/bin/env python3
"""
FastAPI server for the Deep Research Tool.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000

Then embed in Notion via: http://your-server:8000
"""

import asyncio
import json
import os
from typing import AsyncGenerator

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from deep_research import search_web, SYSTEM_PROMPT, TOOLS

app = FastAPI(title="Deep Research API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    topic: str
    model: str = "claude-opus-4-7"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the frontend."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, encoding="utf-8") as f:
        return f.read()


@app.post("/research/stream")
async def research_stream(req: ResearchRequest):
    """
    Stream research progress as Server-Sent Events.
    Frontend receives events:
      - {type: "search", query: "..."}
      - {type: "chunk", text: "..."}
      - {type: "done"}
      - {type: "error", message: "..."}
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set on server")

    return StreamingResponse(
        _research_generator(req.topic, req.model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _research_generator(topic: str, model: str) -> AsyncGenerator[str, None]:
    """Run deep research and yield SSE events."""

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    client = anthropic.Anthropic()

    messages = [
        {
            "role": "user",
            "content": (
                f"Please conduct a comprehensive deep research report on the following topic:\n\n"
                f"**{topic}**\n\n"
                f"Search for information systematically, covering multiple angles. "
                f"After gathering enough information, produce a well-structured Markdown report."
            ),
        }
    ]

    max_iterations = 15
    iteration = 0

    try:
        while iteration < max_iterations:
            # Run blocking API call in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.messages.create(
                    model=model,
                    max_tokens=8192,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                ),
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text") and block.text.strip():
                        yield sse({"type": "chunk", "text": block.text})
                yield sse({"type": "done"})
                return

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use" and block.name == "search_web":
                        query = block.input.get("query", "")
                        yield sse({"type": "search", "query": query})

                        result = await asyncio.get_event_loop().run_in_executor(
                            None, lambda q=query: search_web(q)
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result, ensure_ascii=False),
                            }
                        )

                if tool_results:
                    messages.append({"role": "user", "content": tool_results})

            iteration += 1

        yield sse({"type": "error", "message": "Research reached iteration limit."})

    except Exception as e:
        yield sse({"type": "error", "message": str(e)})


@app.get("/health")
async def health():
    return {"status": "ok", "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY"))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
