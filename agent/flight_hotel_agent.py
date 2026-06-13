import os
from typing import Optional, Callable
import anthropic

SYSTEM_PROMPT = """You are a smart travel assistant specializing in finding the cheapest flights and hotel rooms. \
When the user asks about flights or hotels, search across multiple major platforms to find the best deals.

For flights, search:
- Google Flights (google.com/travel/flights)
- Kayak (kayak.com)
- Skyscanner (skyscanner.com)
- Expedia (expedia.com)
- Momondo (momondo.com)

For hotels, search:
- Booking.com
- Hotels.com
- Expedia
- Airbnb (airbnb.com)
- Trivago (trivago.com)
- Agoda (agoda.com)

Always:
1. Search multiple platforms and compare prices
2. Present results sorted by price (cheapest first)
3. Include key details: price, airline/hotel name, departure/return times for flights, \
check-in/check-out for hotels, direct vs. layover for flights, amenities for hotels
4. Highlight the best deal with a clear recommendation
5. Note any important caveats (baggage fees, cancellation policy, etc.)

Be thorough but concise. Format results clearly with prices prominently displayed."""


class FlightHotelAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = "claude-opus-4-8"
        # Server-side tools: Anthropic executes these automatically — no client loop needed
        self.tools = [
            {"type": "web_search_20260209", "name": "web_search"},
            {"type": "web_fetch_20260209", "name": "web_fetch"},
        ]
        self.conversation_history: list[dict] = []

    def search(
        self,
        query: str,
        on_text: Optional[Callable[[str], None]] = None,
        on_tool_use: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Search for cheap flights or hotels using a natural language query.

        Server-side tools (web_search, web_fetch) are executed by Anthropic's infrastructure,
        so a single API call handles all searching internally.

        Args:
            query: Natural language query, e.g. "Find cheapest flights from NYC to London next week"
            on_text: Called with each streamed text token
            on_tool_use: Called with the tool name each time Claude invokes a tool

        Returns:
            The complete text response.
        """
        self.conversation_history.append({"role": "user", "content": query})

        response_text = ""
        current_tool: Optional[str] = None

        with self.client.messages.stream(
            model=self.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=self.tools,
            messages=self.conversation_history,
            betas=["interleaved-thinking-2025-05-14"],
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool = block.name
                        if on_tool_use:
                            on_tool_use(block.name)

                elif event.type == "content_block_stop":
                    current_tool = None

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        response_text += event.delta.text
                        if on_text:
                            on_text(event.delta.text)

        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
        })

        return response_text

    def follow_up(
        self,
        query: str,
        on_text: Optional[Callable[[str], None]] = None,
        on_tool_use: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Send a follow-up question in the same search session."""
        return self.search(query, on_text=on_text, on_tool_use=on_tool_use)

    def reset(self):
        """Clear conversation history to start a fresh session."""
        self.conversation_history = []
