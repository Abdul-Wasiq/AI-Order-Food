"""
groq_order_agent.py — Phase 3

Takes the raw signal Gemini emits (tool name + args) and turns it into a
structured action dict that the backend can safely execute against Postgres.

Groq's job is ONLY to understand and structure — it never writes SQL and
never touches the database directly. It matches loose text like
"5 cheese burgers, 1 large fries" against the real product list you give it,
and resolves things like "change from 5 to 4 cheeseburgers" into a clean
list of {product_name, quantity} pairs. The backend (main.py) is the only
thing that runs real queries, using this output as validated input.

Usage:
    from groq_order_agent import generate_order_action

    action = generate_order_action(
        tool_name="place_order",
        tool_args={"items_summary": "...", "phone": "...", ...},
        available_products=[{"name": "Cheese Burger", "price": 650.00}, ...],
    )
    # action is a dict like:
    # {"action": "create_order", "status": "ok", "customer_phone": "...",
    #  "address": "...", "items": [{"product_name": "Cheese Burger", "quantity": 5}, ...]}
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a translation layer between a voice ordering assistant and a
restaurant's database. You receive a signal describing what the customer wants to do, along
with the restaurant's actual current product list. Your ONLY job is to turn that signal into a
clean, structured JSON action — you never write SQL and you never invent data that wasn't in
the signal.

You will be given:
1. The name of the action the voice assistant decided on: place_order, cancel_order,
   update_order, or check_order_status.
2. The raw arguments that came with it (these are loose natural language, not already clean).
3. The restaurant's current product list (name, price, availability) — this is the ONLY valid
   set of products. Never invent a product name that isn't in this list.

YOUR JOB PER ACTION TYPE:

place_order:
- Parse items_summary into a list of {"product_name": ..., "quantity": ...} pairs.
- Match each mentioned item to the CLOSEST real product name from the given list. If something
  clearly doesn't match any real product, include it in "unmatched_items" instead of guessing.
- Do not invent quantities — if a quantity wasn't stated for an item, default to 1.
- Pass through phone and address exactly as given, only trimming obvious whitespace.

cancel_order / update_order / check_order_status:
- Pass through order_reference as-is (it's either an order number or a phone number — you don't
  need to know which, the backend will figure that out).
- For update_order, parse change_description into a structured list of changes where possible,
  e.g. {"product_name": "Cheese Burger", "new_quantity": 4} for a quantity change, or
  {"field": "address", "new_value": "..."} for a non-item change. If a change can't be cleanly
  parsed, include it under "unparsed_changes" as the original text instead of guessing.

ALWAYS respond with ONLY a JSON object, no other text, in this shape:
{
  "action": "create_order | cancel_order | update_order | check_status",
  "status": "ok | needs_clarification",
  "customer_phone": "... or null",
  "address": "... or null",
  "order_reference": "... or null",
  "items": [{"product_name": "...", "quantity": N}, ...],
  "unmatched_items": ["..."],
  "changes": [{"product_name": "...", "new_quantity": N}, {"field": "...", "new_value": "..."}],
  "unparsed_changes": ["..."],
  "notes": "brief note if status is needs_clarification, explaining what's ambiguous"
}
Omit fields that don't apply to this action type by leaving them as null or empty lists — always
include every key from the shape above regardless of action type, just leave the irrelevant ones
empty/null so the backend can rely on a consistent shape.
"""


def generate_order_action(tool_name: str, tool_args: dict, available_products: list[dict]) -> dict:
    """
    tool_name: one of "place_order", "cancel_order", "update_order", "check_order_status"
        (the exact Gemini function-call name).
    tool_args: the raw args dict Gemini passed with that call.
    available_products: list of dicts like {"name": ..., "price": ..., "available": ...},
        pulled fresh from the products table right before calling this.

    Returns a structured action dict (see SYSTEM_PROMPT shape above).
    Raises on failure (bad JSON, API error) — caller decides how to handle
    (e.g. tell the customer something went wrong and ask them to repeat).
    """
    user_content = json.dumps({
        "tool_name": tool_name,
        "tool_args": tool_args,
        "available_products": available_products,
    }, indent=2)

    completion = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,  # low — we want consistent structured parsing, not creativity
        max_completion_tokens=800,
        reasoning_effort="low",
    )
    raw = completion.choices[0].message.content
    return json.loads(raw)


if __name__ == "__main__":
    # Quick smoke test — mirrors a real place_order call from the transcript
    # you already tested, so you can sanity-check this in isolation before
    # wiring it into main.py.
    test_products = [
        {"name": "Cheese Burger", "price": 650.00, "available": True},
        {"name": "Beef Burger", "price": 750.00, "available": True},
        {"name": "Chicken Pizza", "price": 1200.00, "available": True},
        {"name": "Cheese Pizza", "price": 1100.00, "available": True},
        {"name": "Large Fries", "price": 350.00, "available": True},
        {"name": "Soft Drink", "price": 150.00, "available": True},
    ]

    result = generate_order_action(
        tool_name="place_order",
        tool_args={
            "items_summary": "5 cheese burgers, 1 large fries, 5 soft drinks",
            "phone": "03182182142",
            "address": "Clifton Do Talwar",
            "total_price": "Rs. 4350",
        },
        available_products=test_products,
    )
    print(json.dumps(result, indent=2))