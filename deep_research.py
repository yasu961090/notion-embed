#!/usr/bin/env python3
"""
Deep Research Tool using Claude API

Usage:
    python deep_research.py "研究したいトピック"
    python deep_research.py "AI in healthcare" --output report.md
    python deep_research.py "量子コンピュータ" --model claude-opus-4-7
"""

import anthropic
import json
import os
import sys
import argparse
import time
from typing import Optional


def search_web(query: str) -> dict:
    """
    Web search using Tavily API.
    Set TAVILY_API_KEY env var to enable real search.
    Falls back to a placeholder when key is not set.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {
            "results": [],
            "note": "TAVILY_API_KEY not set. Set this env var to enable real web search.",
            "query": query,
        }

    try:
        import requests
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "max_results": 5,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"results": [], "error": str(e), "query": query}


TOOLS = [
    {
        "name": "search_web",
        "description": (
            "Search the web for current information on a topic. "
            "Call this multiple times with different queries to gather comprehensive information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A specific search query to find relevant information",
                }
            },
            "required": ["query"],
        },
    }
]

SYSTEM_PROMPT = """You are an expert deep research assistant. Your goal is to produce comprehensive,
well-structured research reports on any given topic.

Research process:
1. Identify 3-5 key aspects or sub-questions to investigate
2. Search for each aspect using specific, targeted queries
3. Search for recent developments, statistics, and expert opinions
4. Cross-reference information from multiple searches
5. Synthesize all findings into a clear, structured report

Report format (Markdown):
- Executive Summary
- Background / Overview
- Key Findings (with subsections)
- Current Trends / Recent Developments
- Challenges and Limitations (if applicable)
- Conclusion
- Sources (list all search queries used)

Be thorough, objective, and always base claims on the search results."""


def deep_research(
    topic: str,
    model: str = "claude-opus-4-7",
    max_iterations: int = 15,
    verbose: bool = True,
) -> str:
    """
    Perform deep research on a topic using Claude API with iterative web search.

    Args:
        topic: The research topic or question
        model: Claude model ID to use
        max_iterations: Maximum number of tool-use iterations
        verbose: Print progress to stdout

    Returns:
        Research report as a Markdown string
    """
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

    if verbose:
        print(f"[Deep Research] Topic: {topic}")
        print(f"[Deep Research] Model: {model}")
        print("-" * 60)

    iteration = 0
    search_count = 0

    while iteration < max_iterations:
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text") and block.text.strip():
                    if verbose:
                        print(f"\n[Deep Research] Done. {search_count} searches performed.")
                    return block.text
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "search_web":
                    query = block.input.get("query", "")
                    search_count += 1
                    if verbose:
                        print(f"[Search {search_count}] {query}")

                    result = search_web(query)
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

    return "# Research Incomplete\n\nThe research process reached its iteration limit without producing a final report."


def main():
    parser = argparse.ArgumentParser(
        description="Deep Research Tool powered by Claude API"
    )
    parser.add_argument("topic", help="Research topic or question")
    parser.add_argument(
        "--model",
        default="claude-opus-4-7",
        help="Claude model ID (default: claude-opus-4-7)",
    )
    parser.add_argument(
        "--output", "-o", help="Save report to file (default: print to stdout)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    report = deep_research(
        topic=args.topic,
        model=args.model,
        verbose=not args.quiet,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print(report)


if __name__ == "__main__":
    main()
