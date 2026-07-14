"""
AI Event Planner — app.py
Flask backend using IBM watsonx.ai (Meta Llama 3.3 70B Instruct)
"""

import os
import sqlite3
import json
import hashlib
import secrets
from datetime import datetime
from functools import wraps
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, g
)
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

# ── watsonx.ai credentials ────────────────────────────────────────────────────
WATSONX_API_KEY    = os.getenv("IBM_WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("IBM_WATSONX_PROJECT_ID")
WATSONX_URL        = os.getenv("IBM_WATSONX_URL", "https://au-syd.ml.cloud.ibm.com")
MODEL_ID           = "meta-llama/llama-3-3-70b-instruct"

# ── SQLite database path ──────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "events.db")


# ─────────────────────────────────────────────────────────────────────────────
# Database — connection, init, teardown
# ─────────────────────────────────────────────────────────────────────────────

def get_db():
    """Return (and cache) a per-request SQLite connection on Flask's g object."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db


@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db:
        db.close()


def init_db():
    """
    Create tables if they do not already exist.
    Called once at startup — safe to call repeatedly.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name     TEXT    NOT NULL,
                username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                email         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT    NOT NULL,
                salt          TEXT    NOT NULL,
                created_at    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                event_type  TEXT    NOT NULL,
                date        TEXT    NOT NULL,
                location    TEXT    NOT NULL,
                budget      REAL    NOT NULL DEFAULT 0,
                guests      INTEGER NOT NULL DEFAULT 0,
                theme       TEXT    NOT NULL,
                plan        TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_user
                ON events(user_id);
            CREATE INDEX IF NOT EXISTS idx_events_date
                ON events(date);
        """)
        conn.commit()


# Run once at import time
init_db()


# ─────────────────────────────────────────────────────────────────────────────
# Password helpers  (stdlib only — hashlib + secrets)
# ─────────────────────────────────────────────────────────────────────────────

def _make_salt() -> str:
    return secrets.token_hex(32)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# User helpers
# ─────────────────────────────────────────────────────────────────────────────

def db_register_user(full_name: str, username: str, email: str, password: str):
    """Insert a new user. Returns (True, user_id) or (False, error_message)."""
    salt    = _make_salt()
    pw_hash = _hash_password(password, salt)
    try:
        db  = get_db()
        cur = db.execute(
            "INSERT INTO users (full_name, username, email, password_hash, salt, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (full_name, username.strip(), email.strip().lower(),
             pw_hash, salt, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        db.commit()
        return True, cur.lastrowid
    except sqlite3.IntegrityError as e:
        msg = str(e).lower()
        if "username" in msg:
            return False, "Username already taken. Please choose another."
        if "email" in msg:
            return False, "An account with that email already exists."
        return False, "Registration failed — please try again."


def db_verify_user(username: str, password: str):
    """
    Verify credentials against the DB.
    Returns the sqlite3.Row on success, None on failure.
    """
    row = get_db().execute(
        "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username.strip(),)
    ).fetchone()
    if row is None:
        return None
    if secrets.compare_digest(_hash_password(password, row["salt"]), row["password_hash"]):
        return row
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Event helpers
# ─────────────────────────────────────────────────────────────────────────────

def db_save_event(user_id, event_type, date, location, budget, guests, theme, plan):
    """Insert a generated event plan. Returns the new row id."""
    db  = get_db()
    cur = db.execute(
        "INSERT INTO events "
        "(user_id, event_type, date, location, budget, guests, theme, plan, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, event_type, date, location,
         float(budget), int(guests), theme, plan,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    db.commit()
    return cur.lastrowid


def db_get_events(user_id, q: str = ""):
    """Return all events for a user, optionally filtered by search term."""
    if q:
        pat = f"%{q}%"
        return get_db().execute(
            "SELECT * FROM events "
            "WHERE user_id = ? AND (event_type LIKE ? OR location LIKE ? OR theme LIKE ?) "
            "ORDER BY created_at DESC",
            (user_id, pat, pat, pat)
        ).fetchall()
    return get_db().execute(
        "SELECT * FROM events WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()


def db_get_event(event_id, user_id):
    """Return a single event, owned by user_id (prevents cross-user access)."""
    return get_db().execute(
        "SELECT * FROM events WHERE id = ? AND user_id = ?", (event_id, user_id)
    ).fetchone()


def db_delete_event(event_id, user_id):
    db = get_db()
    db.execute("DELETE FROM events WHERE id = ? AND user_id = ?", (event_id, user_id))
    db.commit()


def db_dashboard_stats(user_id):
    """Return (stats_dict, recent_5_events, upcoming_5_events)."""
    db     = get_db()
    stats  = dict(db.execute(
        "SELECT COUNT(*) AS total, COALESCE(SUM(budget),0) AS total_budget, "
        "COALESCE(AVG(guests),0) AS avg_guests, COALESCE(SUM(guests),0) AS total_guests "
        "FROM events WHERE user_id = ?", (user_id,)
    ).fetchone())
    recent = db.execute(
        "SELECT * FROM events WHERE user_id = ? ORDER BY created_at DESC LIMIT 5",
        (user_id,)
    ).fetchall()
    today  = datetime.today().strftime("%Y-%m-%d")
    upcoming = db.execute(
        "SELECT * FROM events WHERE user_id = ? AND date >= ? ORDER BY date ASC LIMIT 5",
        (user_id, today)
    ).fetchall()
    return stats, recent, upcoming


def db_analytics_data(user_id):
    """Return aggregated data for the Analytics page."""
    db   = get_db()
    year = str(datetime.today().year)

    # Totals
    totals = dict(db.execute(
        "SELECT COUNT(*) AS total, COALESCE(SUM(budget),0) AS total_budget, "
        "COALESCE(AVG(guests),0) AS avg_guests "
        "FROM events WHERE user_id = ?", (user_id,)
    ).fetchone())

    # Events per calendar-month (current year)
    monthly = db.execute(
        "SELECT strftime('%m', date) AS month, COUNT(*) AS cnt "
        "FROM events WHERE user_id = ? AND strftime('%Y', date) = ? "
        "GROUP BY month ORDER BY month",
        (user_id, year)
    ).fetchall()

    # Count + total budget per event type
    by_type = db.execute(
        "SELECT event_type, COUNT(*) AS cnt, COALESCE(SUM(budget),0) AS total_budget "
        "FROM events WHERE user_id = ? GROUP BY event_type ORDER BY cnt DESC",
        (user_id,)
    ).fetchall()

    # 5 most recent events (for "recent plans" table)
    recent = db.execute(
        "SELECT * FROM events WHERE user_id = ? ORDER BY created_at DESC LIMIT 5",
        (user_id,)
    ).fetchall()

    return totals, monthly, by_type, recent


# ─────────────────────────────────────────────────────────────────────────────
# Auth decorator
# ─────────────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in") or not session.get("user_id"):
            session.clear()
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────────────────────
# watsonx.ai helpers  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

def get_model():
    if not WATSONX_API_KEY:
        raise RuntimeError(
            "IBM_WATSONX_API_KEY is not set. "
            "Add it to your .env file and restart the server."
        )
    if not WATSONX_PROJECT_ID:
        raise RuntimeError(
            "IBM_WATSONX_PROJECT_ID is not set. "
            "Add it to your .env file and restart the server."
        )
    credentials = Credentials(api_key=WATSONX_API_KEY, url=WATSONX_URL)
    model = ModelInference(
        model_id=MODEL_ID,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID,
        params={
            "decoding_method":    DecodingMethods.SAMPLE.value,
            "max_new_tokens":     1200,
            "min_new_tokens":     100,
            "temperature":        0.7,
            "top_p":              0.9,
            "top_k":              50,
            "repetition_penalty": 1.1,
        },
    )
    return model


def build_prompt(event_type, date, location, budget, guests, theme):
    system_msg = """\
You are an expert event planner with 20+ years of experience organising \
weddings, corporate galas, birthday celebrations, and special occasions. \
You create detailed, practical, and creative event plans tailored to the \
client's budget, guest count, and chosen theme. \
Always respond using exactly these eight numbered sections — nothing more, \
nothing less:

1. EVENT SUMMARY
2. VENUE
3. CATERING
4. DECORATIONS
5. ENTERTAINMENT
6. BUDGET BREAKDOWN
7. TIMELINE
8. CHECKLIST

Use plain text with clear headings for each section. \
Be specific, actionable, and stay within the stated budget."""

    user_msg = (
        f"Please create a comprehensive event plan for the following event:\n\n"
        f"  Event Type     : {event_type}\n"
        f"  Date           : {date}\n"
        f"  Location       : {location}\n"
        f"  Total Budget   : ${budget}\n"
        f"  Number of Guests: {guests}\n"
        f"  Theme          : {theme}\n\n"
        "Include specific vendor suggestions, realistic cost estimates that fit "
        "within the total budget, a detailed day-of timeline with times, and a "
        "numbered checklist of action items the client must complete before the event."
    )
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def generate_event_plan(event_type, date, location, budget, guests, theme):
    try:
        model    = get_model()
        prompt   = build_prompt(event_type, date, location, budget, guests, theme)
        response = model.generate_text(prompt=prompt)
        if not response or not response.strip():
            return "The model returned an empty response. Please try again."
        return response.strip()
    except RuntimeError as config_err:
        return f"⚠ Configuration Error\n\n{config_err}"
    except Exception as err:
        return (
            f"⚠ An error occurred while contacting IBM watsonx.ai.\n\n"
            f"Details: {err}\n\n"
            "Please check your API key, Project ID, and network connection."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes — Auth
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    form  = {}
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username  = request.form.get("username",  "").strip()
        email     = request.form.get("email",     "").strip()
        password  = request.form.get("password",  "")
        confirm   = request.form.get("confirm",   "")
        form = dict(full_name=full_name, username=username, email=email)

        if not all([full_name, username, email, password, confirm]):
            error = "All fields are required."
        elif len(username) < 3:
            error = "Username must be at least 3 characters."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            ok, result = db_register_user(full_name, username, email, password)
            if ok:
                return redirect(url_for("login", registered="1"))
            error = result

    return render_template("signup.html", error=error, form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    error      = None
    registered = request.args.get("registered") == "1"
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user     = db_verify_user(username, password)
        if user:
            session.clear()
            session["logged_in"] = True
            session["user_id"]   = user["id"]
            session["username"]  = user["username"]
            session["full_name"] = user["full_name"]
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error, registered=registered)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes — Protected pages
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
@login_required
def dashboard():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    if request.method == "POST":
        event_type = request.form.get("event_type", "").strip()
        date = request.form.get("date", "").strip()
        location = request.form.get("location", "").strip()
        budget = request.form.get("budget", "0").strip()
        guests = request.form.get("guests", "0").strip()
        theme = request.form.get("theme", "").strip()

        form_data = {
            "event_type": event_type,
            "date": date,
            "location": location,
            "budget": budget,
            "guests": guests,
            "theme": theme,
        }

        plan = None
        saved_id = None

        if all([event_type, date, location, budget, guests, theme]):

            plan = generate_event_plan(
                event_type,
                date,
                location,
                budget,
                guests,
                theme
            )

            if plan and not plan.startswith("⚠"):
                saved_id = db_save_event(
                    user_id,
                    event_type,
                    date,
                    location,
                    budget,
                    guests,
                    theme,
                    plan
                )

        else:
            plan = "⚠ Please fill in all six fields before generating a plan."

        return render_template(
            "planner.html",
            plan=plan,
            saved_id=saved_id,
            form_data=form_data,
        )

    stats, recent, upcoming = db_dashboard_stats(user_id)

    return render_template(
        "dashboard.html",
        stats=stats,
        recent=recent,
        upcoming=upcoming,
    )

@app.route("/planner", methods=["GET", "POST"])
@login_required
def planner():
    user_id   = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    plan      = None
    saved_id  = None
    form_data = {}

    if request.method == "POST":
        event_type = request.form.get("event_type", "").strip()
        date       = request.form.get("date",       "").strip()
        location   = request.form.get("location",   "").strip()
        budget     = request.form.get("budget",     "0").strip()
        guests     = request.form.get("guests",     "0").strip()
        theme      = request.form.get("theme",      "").strip()
        form_data  = dict(event_type=event_type, date=date, location=location,
                          budget=budget, guests=guests, theme=theme)

        if all([event_type, date, location, budget, guests, theme]):
            plan = generate_event_plan(event_type, date, location, budget, guests, theme)
            # Save every valid plan to the database
            if plan and not plan.startswith("⚠"):
                saved_id = db_save_event(
                    user_id,
                    event_type, date, location,
                    budget, guests, theme, plan
                )
        else:
            plan = "⚠ Please fill in all six fields before generating a plan."

    return render_template("planner.html", plan=plan, saved_id=saved_id, form_data=form_data)


@app.route("/analytics")
@login_required
def analytics():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    totals, monthly, by_type, recent = db_analytics_data(user_id)

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    months_labels = json.dumps([month_names[int(r["month"]) - 1] for r in monthly])
    months_data   = json.dumps([r["cnt"] for r in monthly])
    type_labels   = json.dumps([r["event_type"] for r in by_type])
    type_counts   = json.dumps([r["cnt"] for r in by_type])
    type_budgets  = json.dumps([r["total_budget"] for r in by_type])

    return render_template(
        "analytics.html",
        totals=totals,
        months_labels=months_labels,
        months_data=months_data,
        type_labels=type_labels,
        type_counts=type_counts,
        type_budgets=type_budgets,
        recent=recent,
        current_year=datetime.today().year,
    )


@app.route("/history")
@login_required
def history():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    q      = request.args.get("q", "").strip()
    events = db_get_events(user_id, q)
    return render_template("history.html", events=events, q=q)


@app.route("/history/<int:event_id>")
@login_required
def history_view(event_id):
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    event = db_get_event(event_id, user_id)
    if not event:
        return redirect(url_for("history"))
    return render_template("history_view.html", event=event)


@app.route("/history/<int:event_id>/delete", methods=["POST"])
@login_required
def history_delete(event_id):
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    db_delete_event(event_id, user_id)
    return redirect(url_for("history"))


@app.route("/settings")
@login_required
def settings():
    return render_template(
        "settings.html",
        api_key_set=bool(WATSONX_API_KEY),
        project_id_set=bool(WATSONX_PROJECT_ID),
        watsonx_url=WATSONX_URL,
        model_id=MODEL_ID,
    )


@app.route("/api/test-connection", methods=["POST"])
@login_required
def api_test_connection():
    """Quick ping to IBM watsonx.ai — returns JSON status."""
    try:
        model    = get_model()
        response = model.generate_text(prompt="Say: OK")
        if response and response.strip():
            return jsonify({"status": "ok", "message": "Connection successful."})
        return jsonify({"status": "error", "message": "Empty response from model."})
    except RuntimeError as e:
        return jsonify({"status": "error", "message": str(e)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
