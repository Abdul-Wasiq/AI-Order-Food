import os
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from dotenv import load_dotenv

from groq_Order_Agent import generate_order_action
import database as db

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if (GEMINI_API_KEY):
    print("working")
else:
    print("Not working")

client = genai.Client()

MODEL = "gemini-3.1-flash-live-preview"


# Each tool is a "bucket" for one specific customer intent. Gemini decides
# which bucket applies based on what the customer is asking for, then fills
# it in over the course of the conversation before calling it. Field VALUES
# are kept as loose natural-language strings (not strict enums/numbers) —
# they still get enforced as REQUIRED by the tool schema, but their content
# stays easy for Groq to read in Phase 3, since Groq is the one that will
# actually parse "2 cheese burgers, 1 large fries" into real DB rows.

PLACE_ORDER_TOOL = {
    "function_declarations": [
        {
            "name": "place_order",
            "description": (
                "Call this ONLY after you have collected the customer's full order "
                "(items + quantities), their phone number, their delivery address, "
                "and they have explicitly confirmed the order and total price out loud. "
                "Do NOT call this the moment they name an item — gather everything first, "
                "read it back to them, and only call this after they say yes/confirm."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items_summary": {
                        "type": "string",
                        "description": (
                            "Plain-language list of everything ordered with quantities, "
                            "e.g. '2 cheese burgers, 1 large fries, 1 soft drink'."
                        ),
                    },
                    "phone": {
                        "type": "string",
                        "description": "Customer's phone number, exactly as they gave it.",
                    },
                    "address": {
                        "type": "string",
                        "description": "Customer's delivery address, exactly as they gave it.",
                    },
                    "total_price": {
                        "type": "string",
                        "description": "The total price you read back and they confirmed, e.g. 'Rs. 1650'.",
                    },
                },
                "required": ["items_summary", "phone", "address", "total_price"],
            },
        }
    ]
}

CANCEL_ORDER_TOOL = {
    "function_declarations": [
        {
            "name": "cancel_order",
            "description": (
                "Call this when the customer clearly asks to cancel an existing order. "
                "Confirm with them which order before calling this if there's any ambiguity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_reference": {
                        "type": "string",
                        "description": (
                            "However the customer identified the order — an order number if "
                            "they gave one, otherwise their phone number to look it up by."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why they're cancelling, if they mentioned one. Empty string if not given.",
                    },
                },
                "required": ["order_reference"],
            },
        }
    ]
}

UPDATE_ORDER_TOOL = {
    "function_declarations": [
        {
            "name": "update_order",
            "description": (
                "Call this when the customer wants to change an order that's already been "
                "placed (add/remove/change items or quantities, change address, etc). Confirm "
                "the change with them before calling this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_reference": {
                        "type": "string",
                        "description": "Order number if given, otherwise their phone number to look it up by.",
                    },
                    "change_description": {
                        "type": "string",
                        "description": (
                            "Plain-language description of exactly what should change, e.g. "
                            "'add 1 large fries' or 'remove the soft drink' or "
                            "'change address to 45 Main Street'."
                        ),
                    },
                },
                "required": ["order_reference", "change_description"],
            },
        }
    ]
}

CHECK_ORDER_STATUS_TOOL = {
    "function_declarations": [
        {
            "name": "check_order_status",
            "description": (
                "Call this when the customer asks where their order is, or wants to know its status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_reference": {
                        "type": "string",
                        "description": "Order number if given, otherwise their phone number to look it up by.",
                    },
                },
                "required": ["order_reference"],
            },
        }
    ]
}

ALL_TOOLS = (
    PLACE_ORDER_TOOL["function_declarations"]
    + CANCEL_ORDER_TOOL["function_declarations"]
    + UPDATE_ORDER_TOOL["function_declarations"]
    + CHECK_ORDER_STATUS_TOOL["function_declarations"]
)


SYSTEM_PROMPT = """You are a friendly phone order-taker at "Kababjees Restaurant". You are speaking
live with a customer who just called in to place an order — behave exactly like a real
restaurant employee answering the phone, not like a generic AI assistant.

YOUR MENU (this is the ONLY info you have right now — you do NOT have a live database yet,
so just use this fixed list for this phase):
- Cheese Burger — Rs. 650
- Beef Burger — Rs. 750
- Chicken Pizza — Rs. 1200
- Cheese Pizza — Rs. 1100
- Large Fries — Rs. 350
- Soft Drink — Rs. 150

CONVERSATION FLOW FOR A NEW ORDER:
1. Greet the caller briefly and naturally, e.g. "Hi, thanks for calling Kababjees, what can I get
   started for you?" Do not over-explain who you are.
2. Listen to what they want. If they name something not on the menu, apologize briefly and
   mention 2-3 similar items from the menu instead.
3. Confirm quantities if unclear ("just to confirm, that's two cheese burgers?").
4. Once you have their full order, ask if they'd like anything else.
5. Once they say that's everything, ask for their phone number, then their delivery address.
6. Read back the full order (items, quantities, total price) and ask them to confirm.
7. Once — and only once — they explicitly confirm, call the place_order function with the
   complete details. Do NOT call it before they've confirmed.

HANDLING CANCELLATIONS, CHANGES, AND STATUS CHECKS:
- If the customer wants to cancel an existing order, confirm which order (order number or the
  phone number it was placed under), then call cancel_order.
- If the customer wants to change an order that's already placed (add/remove items, change
  address, etc), confirm the exact change with them out loud, then call update_order.
- If the customer asks about the status of an existing order, call check_order_status.
- These are different situations from placing a brand new order — listen for which one the
  customer actually means before deciding which function to use.

FUNCTION-CALLING RULES (IMPORTANT):
- You MUST actually call the relevant function when taking one of these actions — saying words
  like "okay, placing your order now" out loud does NOT do anything by itself. The function call
  is what actually processes the request. Always call the function in the same turn where you
  tell the customer the action is happening.
- For place_order specifically: never call it with missing or guessed information. If the
  customer hasn't given you a phone number or address yet, ask for it — do not call the function
  with a blank or placeholder value in those fields.
- After you call a function, you get a real result back from the restaurant's database. React to
  it naturally and specifically:
  - status "ok" on place_order -> confirm the order is placed, mention the order number if given.
  - status "unavailable" -> tell the customer which item(s) couldn't be added and why (out of
    stock, not on the menu, etc), and offer to adjust the order instead of just failing silently.
  - status "not_found" (cancel/update/status) -> let the customer know you couldn't find an order
    under that reference, and ask them to double check the order number or phone number.
  - status "already_cancelled" / "already_delivered" -> explain that plainly, don't re-attempt.
  - status "cannot_modify" -> explain the order is already cancelled or delivered so it can't be
    changed.
  - status "error" -> apologize briefly, say there was a system issue, and offer to try again.
  Never claim something succeeded if the result says otherwise.

BEHAVIOR RULES:
- Speak naturally, 1-3 sentences per turn. Real phone staff don't monologue.
- Don't recite the whole menu unprompted — only mention items when relevant.
- Stay warm but efficient — this is a phone order, not a long chat.
- Do the math correctly when reading back the total price.
- If the caller changes their mind mid-order ("actually, no fries"), handle it gracefully and
  update the order out loud so they know it registered, before any function call happens.
- Stay on topic. If the customer goes off-topic (jokes, songs, unrelated requests), redirect
  warmly but firmly back to taking their order — don't play along indefinitely.
"""


def build_config(system_prompt: str, resumption_handle=None):
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(parts=[types.Part(text=system_prompt)]),
        tools=[{"function_declarations": ALL_TOOLS}],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
            )
        ),
        session_resumption=types.SessionResumptionConfig(handle=resumption_handle),
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        ),
    )


def build_order_update_payload(tool_name: str, tool_result: dict) -> dict | None:
    """
    Turns a raw db.py result into the small, frontend-friendly payload the
    browser needs to update the live order panel + fire a notification.
    Returns None if this result isn't something the UI should react to
    (e.g. an error, or a status the frontend doesn't need to show).
    """
    status = tool_result.get("status")

    if tool_name == "place_order":
        if status == "ok":
            return {
                "type": "order_update",
                "event": "placed",
                "order": {
                    "order_id": tool_result.get("order_id"),
                    "status": "confirmed",
                    "items": [
                        {"name": i["name"], "quantity": i["quantity"], "price": i["price"]}
                        for i in tool_result.get("items", [])
                    ],
                    "total_price": tool_result.get("total_price"),
                },
            }
        elif status == "unavailable":
            return {
                "type": "order_update",
                "event": "unavailable",
                "unavailable_items": tool_result.get("unavailable_items", []),
            }
        return None

    elif tool_name == "cancel_order":
        if status == "ok":
            return {
                "type": "order_update",
                "event": "cancelled",
                "order": {
                    "order_id": tool_result.get("order_id"),
                    "status": "cancelled",
                    "items": tool_result.get("items", []),
                    "total_price": tool_result.get("total_price"),
                },
            }
        return None

    elif tool_name == "update_order":
        if status == "ok":
            return {
                "type": "order_update",
                "event": "updated",
                "order": {
                    "order_id": tool_result.get("order_id"),
                    "status": "confirmed",
                    "applied": tool_result.get("applied", []),
                    "items": tool_result.get("items", []),
                    "total_price": tool_result.get("new_total_price"),
                },
            }
        return None

    elif tool_name == "check_order_status":
        if status == "ok":
            return {
                "type": "order_update",
                "event": "status",
                "order": {
                    "order_id": tool_result.get("order_id"),
                    "status": tool_result.get("order_status"),
                    "total_price": tool_result.get("total_price"),
                },
            }
        return None

    return None


def handle_tool_call(tool_name: str, tool_args: dict) -> dict:
    """
    The Groq -> DB pipeline for a single Gemini tool call. Runs in a worker
    thread (via asyncio.to_thread in the caller) since psycopg2 and the
    Groq SDK are both synchronous/blocking.

    Groq only produces structured JSON describing WHAT to do — every actual
    database write happens in db.py using parameterized queries, never raw
    SQL from Groq.
    """
    try:
        available_products = db.get_available_products()
    except Exception as e:
        return {"status": "error", "reason": f"could not read product list: {e}"}

    try:
        parsed = generate_order_action(tool_name, tool_args, available_products)
    except Exception as e:
        return {"status": "error", "reason": f"could not understand the request: {e}"}

    if parsed.get("status") == "needs_clarification":
        return {
            "status": "needs_clarification",
            "notes": parsed.get("notes", "Some details were unclear."),
        }

    try:
        if tool_name == "place_order":
            if parsed.get("unmatched_items"):
                return {
                    "status": "unavailable",
                    "unmatched_items": parsed["unmatched_items"],
                }
            return db.create_order(
                phone=parsed.get("customer_phone") or tool_args.get("phone", ""),
                address=parsed.get("address") or tool_args.get("address", ""),
                items=parsed.get("items", []),
            )

        elif tool_name == "cancel_order":
            return db.cancel_order(
                order_reference=parsed.get("order_reference") or tool_args.get("order_reference", ""),
                reason=tool_args.get("reason"),
            )

        elif tool_name == "update_order":
            return db.update_order(
                order_reference=parsed.get("order_reference") or tool_args.get("order_reference", ""),
                changes=parsed.get("changes", []),
            )

        elif tool_name == "check_order_status":
            return db.check_order_status(
                order_reference=parsed.get("order_reference") or tool_args.get("order_reference", ""),
            )

        else:
            return {"status": "error", "reason": f"unknown tool: {tool_name}"}

    except Exception as e:
        return {"status": "error", "reason": f"database error: {e}"}


@app.websocket("/media-stream")
async def handle_order_call(websocket: WebSocket):
    await websocket.accept()
    print("🎙️ Client connected to local server websocket!")

    system_prompt = SYSTEM_PROMPT
    resumption_handle = None

    candidate_buffer = []
    ai_buffer = []
    transcript_log = []

    def flush_candidate():
        if candidate_buffer:
            line = "".join(candidate_buffer).strip()
            if line:
                print(f"🧑 Customer: {line}")
                transcript_log.append(("customer", line))
            candidate_buffer.clear()

    def flush_ai():
        if ai_buffer:
            line = "".join(ai_buffer).strip()
            if line:
                print(f"🤖 Order-taker: {line}")
                transcript_log.append(("order-taker", line))
            ai_buffer.clear()

    while True:
        go_away_triggered = False
        try:
            async with client.aio.live.connect(
                model=MODEL, config=build_config(system_prompt, resumption_handle)
            ) as session:
                print("⚡ Connected to Gemini Live!")

                async def stream_ai_to_browser():
                    nonlocal resumption_handle, go_away_triggered
                    while True:
                        async for response in session.receive():
                            if response.session_resumption_update:
                                update = response.session_resumption_update
                                if update.resumable and update.new_handle:
                                    resumption_handle = update.new_handle

                            if response.go_away:
                                print(f"⚠️ GoAway received, {response.go_away.time_left} left. Reconnecting...")
                                go_away_triggered = True
                                return

                            # Detect tool calls — this is the "signal" Gemini emits once it has
                            # gathered everything needed for that specific intent bucket.
                            if response.tool_call:
                                for fc in response.tool_call.function_calls:
                                    args = dict(fc.args)
                                    print(f"\n🔔 Gemini called: {fc.name}")
                                    print(f"   args: {json.dumps(args, indent=2)}\n")

                                    tool_result = await asyncio.to_thread(
                                        handle_tool_call, fc.name, args
                                    )
                                    print(f"📦 Result: {json.dumps(tool_result, indent=2, default=str)}\n")

                                    # Piggyback on the same websocket used for audio to push a
                                    # live update to the browser UI — no polling, no reload.
                                    ui_payload = build_order_update_payload(fc.name, tool_result)
                                    if ui_payload:
                                        try:
                                            await websocket.send_text(
                                                json.dumps(ui_payload, default=str)
                                            )
                                        except Exception as e:
                                            print(f"⚠️ Failed to send order_update to browser: {e}")

                                    await session.send_tool_response(
                                        function_responses=[
                                            types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response=tool_result,
                                            )
                                        ]
                                    )

                            server_content = response.server_content

                            if server_content:
                                if server_content.input_transcription and server_content.input_transcription.text:
                                    candidate_buffer.append(server_content.input_transcription.text)
                                if server_content.output_transcription and server_content.output_transcription.text:
                                    frag = server_content.output_transcription.text
                                    if not ai_buffer or "".join(ai_buffer).strip() != frag.strip():
                                        ai_buffer.append(frag)

                                if server_content.turn_complete:
                                    flush_ai()
                                    flush_candidate()

                                if server_content.interrupted:
                                    flush_ai()

                            if server_content and server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if part.inline_data:
                                        await websocket.send_bytes(part.inline_data.data)

                async def stream_browser_to_ai():
                    while True:
                        message = await websocket.receive()
                        if "bytes" in message and message["bytes"] is not None:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=message["bytes"], mime_type="audio/pcm;rate=16000"
                                )
                            )

                ai_task = asyncio.create_task(stream_ai_to_browser())
                mic_task = asyncio.create_task(stream_browser_to_ai())

                done, pending = await asyncio.wait(
                    [ai_task, mic_task], return_when=asyncio.FIRST_COMPLETED
                )

                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                for task in done:
                    if task.exception():
                        raise task.exception()

            if go_away_triggered:
                continue
            else:
                break

        except WebSocketDisconnect:
            print("👋 Customer disconnected or closed tab.")
            break
        except Exception as e:
            print("Gemini voice connection closed:", e)
            break

    flush_ai()
    flush_candidate()

    if transcript_log:
        print("\n" + "=" * 60)
        print("FULL CALL TRANSCRIPT")
        print("=" * 60) 
        for speaker, line in transcript_log:
            label = "Customer" if speaker == "customer" else "Order-taker"
            print(f"[{label}] {line}")
        print("=" * 60 + "\n")

    try:
        await websocket.close()
    except Exception:
        pass

    print("Session fully closed.")


@app.get("/")
def health_check():
    return {"status": "ok", "phase": "3 - Groq + real Postgres DB wired in"}