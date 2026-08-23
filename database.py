"""
db.py — Phase 3

The ONLY place raw SQL gets written and executed in this project. Every
function here takes already-structured input (from groq_order_agent.py's
output) and runs a parameterized psycopg2 query — never string-built SQL,
never anything Groq wrote directly.

Each function returns a plain dict describing what happened, in a shape
main.py can hand straight back to Gemini as a tool response.
"""

import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "food_ordering"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def get_available_products() -> list[dict]:
    """Fresh product list for Groq to match items against. Called right
    before every generate_order_action() so Groq always sees current
    prices/stock, not stale data from earlier in the conversation."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, name, price, stock, available FROM products ORDER BY name;"
            )
            rows = [dict(row) for row in cur.fetchall()]
            # psycopg2 returns NUMERIC columns as Decimal, which json.dumps()
            # can't serialize on its own — convert to float here, once, so
            # every downstream consumer (Groq's JSON payload, tool results
            # sent back to Gemini) can just json.dumps() this list directly.
            for row in rows:
                row["price"] = float(row["price"])
            return rows


def _get_or_create_customer(cur, phone: str, address: str | None) -> int:
    """Upsert-by-phone: reuse the existing customer row if this phone
    number has ordered before, otherwise create one. Keeps address fresh
    on every order in case they've moved."""
    cur.execute("SELECT id FROM customers WHERE phone = %s;", (phone,))
    row = cur.fetchone()
    if row:
        customer_id = row["id"]
        if address:
            cur.execute(
                "UPDATE customers SET address = %s WHERE id = %s;",
                (address, customer_id),
            )
        return customer_id

    cur.execute(
        "INSERT INTO customers (phone, address) VALUES (%s, %s) RETURNING id;",
        (phone, address),
    )
    return cur.fetchone()["id"]


def create_order(phone: str, address: str, items: list[dict]) -> dict:
    """
    items: [{"product_name": "Cheese Burger", "quantity": 5}, ...]
    (already matched against real product names by Groq — but we still
    re-validate against the DB here rather than trust that blindly).
    """
    if not phone or not address:
        return {"status": "error", "reason": "missing_phone_or_address"}
    if not items:
        return {"status": "error", "reason": "no_items"}

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            resolved_items = []
            unavailable = []

            for item in items:
                cur.execute(
                    "SELECT id, name, price, stock, available FROM products WHERE name = %s;",
                    (item["product_name"],),
                )
                product = cur.fetchone()

                if not product:
                    unavailable.append({"name": item["product_name"], "reason": "not_found"})
                    continue
                if not product["available"]:
                    unavailable.append({"name": product["name"], "reason": "unavailable"})
                    continue
                if product["stock"] < item["quantity"]:
                    unavailable.append({
                        "name": product["name"],
                        "reason": "insufficient_stock",
                        "requested": item["quantity"],
                        "in_stock": product["stock"],
                    })
                    continue

                resolved_items.append({
                    "product_id": product["id"],
                    "name": product["name"],
                    "price": float(product["price"]),
                    "quantity": item["quantity"],
                })

            if unavailable:
                # Fail the whole order rather than partially place it — the
                # caller (main.py) tells Gemini exactly what's wrong so it
                # can offer alternatives, same as your original menu design.
                return {"status": "unavailable", "unavailable_items": unavailable}

            total_price = sum(i["price"] * i["quantity"] for i in resolved_items)

            customer_id = _get_or_create_customer(cur, phone, address)

            cur.execute(
                """
                INSERT INTO orders (customer_id, status, total_price)
                VALUES (%s, 'confirmed', %s)
                RETURNING id;
                """,
                (customer_id, total_price),
            )
            order_id = cur.fetchone()["id"]

            for item in resolved_items:
                cur.execute(
                    """
                    INSERT INTO order_items (order_id, product_id, quantity)
                    VALUES (%s, %s, %s);
                    """,
                    (order_id, item["product_id"], item["quantity"]),
                )
                cur.execute(
                    "UPDATE products SET stock = stock - %s WHERE id = %s;",
                    (item["quantity"], item["product_id"]),
                )

            conn.commit()

            return {
                "status": "ok",
                "order_id": order_id,
                "total_price": total_price,
                "items": resolved_items,
            }


def _find_order_by_reference(cur, order_reference: str) -> dict | None:
    """order_reference is either a numeric order id or a phone number —
    try order id first, fall back to most recent order for that phone."""
    if order_reference.isdigit():
        cur.execute(
            "SELECT id, status, total_price FROM orders WHERE id = %s;",
            (int(order_reference),),
        )
        row = cur.fetchone()
        if row:
            return dict(row)

    cur.execute(
        """
        SELECT o.id, o.status, o.total_price
        FROM orders o
        JOIN customers c ON c.id = o.customer_id
        WHERE c.phone = %s
        ORDER BY o.created_at DESC
        LIMIT 1;
        """,
        (order_reference,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def _get_order_items(cur, order_id: int) -> list[dict]:
    """Full current line-item list for an order, joined against products
    so callers (main.py -> frontend) can render the complete cart instead
    of just the delta that was applied in this call."""
    cur.execute(
        """
        SELECT p.name, p.price, oi.quantity
        FROM order_items oi
        JOIN products p ON p.id = oi.product_id
        WHERE oi.order_id = %s
        ORDER BY p.name;
        """,
        (order_id,),
    )
    return [
        {"name": row["name"], "price": float(row["price"]), "quantity": row["quantity"]}
        for row in cur.fetchall()
    ]


def cancel_order(order_reference: str, reason: str | None = None) -> dict:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            order = _find_order_by_reference(cur, order_reference)
            if not order:
                return {"status": "not_found"}
            if order["status"] == "cancelled":
                return {"status": "already_cancelled", "order_id": order["id"]}
            if order["status"] == "delivered":
                return {"status": "already_delivered", "order_id": order["id"]}

            items = _get_order_items(cur, order["id"])

            cur.execute(
                "UPDATE orders SET status = 'cancelled', updated_at = NOW() WHERE id = %s;",
                (order["id"],),
            )
            conn.commit()
            return {
                "status": "ok",
                "order_id": order["id"],
                "items": items,
                "total_price": float(order["total_price"]),
            }


def update_order(order_reference: str, changes: list[dict]) -> dict:
    """
    changes: list of either
        {"product_name": "...", "new_quantity": N}  -> update/remove a line item
        {"field": "address", "new_value": "..."}     -> update a non-item field
    Recomputes total_price after applying item changes.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            order = _find_order_by_reference(cur, order_reference)
            if not order:
                return {"status": "not_found"}
            if order["status"] in ("cancelled", "delivered"):
                return {"status": "cannot_modify", "order_status": order["status"]}

            applied = []
            failed = []

            for change in changes:
                if "product_name" in change:
                    cur.execute(
                        "SELECT id, price FROM products WHERE name = %s;",
                        (change["product_name"],),
                    )
                    product = cur.fetchone()
                    if not product:
                        failed.append({"reason": "product_not_found", "change": change})
                        continue

                    new_qty = change.get("new_quantity", 0)
                    if new_qty <= 0:
                        cur.execute(
                            "DELETE FROM order_items WHERE order_id = %s AND product_id = %s;",
                            (order["id"], product["id"]),
                        )
                    else:
                        # True upsert in one statement. This relies on the
                        # UNIQUE (order_id, product_id) constraint added to
                        # order_items — without it, ON CONFLICT has nothing
                        # to match against and silently falls through to
                        # inserting a duplicate row every time an existing
                        # item's quantity is changed (this was the bug that
                        # caused doubled line items and inflated totals).
                        cur.execute(
                            """
                            INSERT INTO order_items (order_id, product_id, quantity)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (order_id, product_id)
                            DO UPDATE SET quantity = EXCLUDED.quantity;
                            """,
                            (order["id"], product["id"], new_qty),
                        )
                    applied.append(change)

                elif change.get("field") == "address" and "new_value" in change:
                    cur.execute(
                        """
                        UPDATE customers SET address = %s
                        WHERE id = (SELECT customer_id FROM orders WHERE id = %s);
                        """,
                        (change["new_value"], order["id"]),
                    )
                    applied.append(change)
                else:
                    failed.append({"reason": "unrecognized_change", "change": change})

            # Recompute total from current order_items
            cur.execute(
                """
                SELECT COALESCE(SUM(oi.quantity * p.price), 0) AS new_total
                FROM order_items oi
                JOIN products p ON p.id = oi.product_id
                WHERE oi.order_id = %s;
                """,
                (order["id"],),
            )
            new_total = cur.fetchone()["new_total"]

            cur.execute(
                "UPDATE orders SET total_price = %s, updated_at = NOW() WHERE id = %s;",
                (new_total, order["id"]),
            )

            # Full current cart post-update, not just the delta that was
            # applied this call — the frontend needs the whole picture to
            # render the live order panel correctly.
            items = _get_order_items(cur, order["id"])

            conn.commit()

            return {
                "status": "ok",
                "order_id": order["id"],
                "applied": applied,
                "failed": failed,
                "items": items,
                "new_total_price": float(new_total),
            }


def check_order_status(order_reference: str) -> dict:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            order = _find_order_by_reference(cur, order_reference)
            if not order:
                return {"status": "not_found"}
            return {
                "status": "ok",
                "order_id": order["id"],
                "order_status": order["status"],
                "total_price": float(order["total_price"]),
            }