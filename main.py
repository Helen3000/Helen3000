#!/usr/bin/env python3
"""CLI for the flight and hotel price search agent."""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.live import Live
from rich import print as rprint

from agent import FlightHotelAgent

load_dotenv()

console = Console()

BANNER = """[bold cyan]
 ✈  Flight & Hotel Price Agent
 Powered by Claude — searches Google Flights, Kayak, Skyscanner,
 Booking.com, Expedia, Hotels.com and more.[/bold cyan]
"""

EXAMPLES = """\
[dim]Examples:
  • Find the cheapest round-trip flights from New York to Tokyo in July
  • Cheapest hotels in Paris for 3 nights starting Dec 20
  • Compare flights from SFO to London next weekend, economy class
  • Hotels near Times Square NYC under $150/night for next Friday[/dim]
"""


def run_interactive():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY is not set.")
        console.print("Copy [bold].env.example[/bold] to [bold].env[/bold] and add your key.")
        sys.exit(1)

    agent = FlightHotelAgent(api_key=api_key)

    console.print(Panel(BANNER, expand=False))
    console.print(EXAMPLES)
    console.print("[dim]Type 'new' to start a new search, 'quit' to exit.[/dim]\n")

    while True:
        try:
            query = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if query.lower() in ("new", "reset", "clear"):
            agent.reset()
            console.print("[dim]Started a new search session.[/dim]\n")
            continue

        console.print()

        # Track tool usage for status display
        tools_used: list[str] = []
        response_parts: list[str] = []

        TOOL_LABELS = {
            "web_search": "Searching the web",
            "web_fetch": "Fetching page",
        }

        with Live(console=console, refresh_per_second=10) as live:
            status_text = Text("Thinking...", style="dim yellow")
            live.update(status_text)

            def on_tool_use(tool_name: str):
                label = TOOL_LABELS.get(tool_name, tool_name)
                tools_used.append(label)
                live.update(Text(f"{label}...", style="dim yellow"))

            def on_text(chunk: str):
                response_parts.append(chunk)
                # Show a rolling preview of the last 120 chars while streaming
                preview = "".join(response_parts)[-120:]
                live.update(Text(preview, style="white"))

            agent.search(query, on_text=on_text, on_tool_use=on_tool_use)

        full_response = "".join(response_parts)
        console.print(
            Panel(
                full_response,
                title="[bold cyan]Agent[/bold cyan]",
                border_style="cyan",
                expand=False,
            )
        )

        if tools_used:
            console.print(f"[dim]Used: {', '.join(dict.fromkeys(tools_used))}[/dim]")

        console.print()


def run_single(query: str):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    agent = FlightHotelAgent(api_key=api_key)

    print(f"Searching: {query}\n", flush=True)

    def on_text(chunk: str):
        print(chunk, end="", flush=True)

    def on_tool_use(tool_name: str):
        labels = {"web_search": "[searching]", "web_fetch": "[fetching]"}
        print(f"\n{labels.get(tool_name, tool_name)} ", end="", flush=True)

    agent.search(query, on_text=on_text, on_tool_use=on_tool_use)
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_single(" ".join(sys.argv[1:]))
    else:
        run_interactive()
