from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.agent import agent, get_llm_status
from app.memory import get_recent_history, load_memory
from app.models import ChatRequest, DailyLogInput
from app.semantic_memory import add_semantic_memory, list_semantic_memories, query_semantic_memory
from database.db import (
    get_recent_daily_logs,
    get_recent_recommendations,
    get_user_habits,
    init_db,
    insert_recommendation,
    is_database_configured,
    upsert_daily_log,
    upsert_user,
)

app = FastAPI(title="Life Optimizer AI")

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Life Optimizer AI</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

    :root {
      --bg-top: #f7fbf8;
      --bg-bottom: #edf4ff;
      --surface: #ffffff;
      --surface-soft: #f6f9fd;
      --ink: #1a2537;
      --ink-dim: #4c5c74;
      --line: #d8e2ef;
      --accent: #0f766e;
      --accent-2: #ea580c;
      --accent-soft: #ddf3f0;
      --danger: #b91c1c;
      --ok: #166534;
      --shadow: 0 18px 30px rgba(18, 41, 72, 0.12);
      --radius: 16px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
      min-height: 100vh;
    }

    .bg-shape {
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: -1;
      overflow: hidden;
    }

    .blob {
      position: absolute;
      border-radius: 999px;
      filter: blur(48px);
      opacity: 0.55;
    }

    .blob.one {
      width: 340px;
      height: 340px;
      top: -100px;
      left: -90px;
      background: #c2f0e9;
    }

    .blob.two {
      width: 320px;
      height: 320px;
      right: -80px;
      top: 120px;
      background: #ffd9bf;
    }

    .blob.three {
      width: 300px;
      height: 300px;
      right: 20%;
      bottom: -140px;
      background: #d7e8ff;
    }

    .shell {
      max-width: 1140px;
      margin: 30px auto;
      padding: 0 18px 28px;
    }

    .hero {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 22px;
      display: grid;
      gap: 14px;
    }

    .hero-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    h1 {
      margin: 0;
      font-size: clamp(1.4rem, 1.8vw, 2rem);
      letter-spacing: -0.02em;
    }

    .subtitle {
      margin: 0;
      color: var(--ink-dim);
      font-size: 0.98rem;
      line-height: 1.4;
    }

    .badge {
      font-size: 0.84rem;
      border-radius: 999px;
      padding: 6px 12px;
      border: 1px solid var(--line);
      background: var(--surface-soft);
      color: var(--ink-dim);
      white-space: nowrap;
    }

    .badge.ok {
      background: #e7f7ea;
      border-color: #a7e1b4;
      color: var(--ok);
    }

    .badge.warn {
      background: #fff3e6;
      border-color: #ffcf9f;
      color: var(--accent-2);
    }

    .quick-links {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .quick-links a {
      text-decoration: none;
      color: var(--accent);
      font-weight: 600;
      background: var(--accent-soft);
      border: 1px solid #bae6e1;
      border-radius: 10px;
      padding: 8px 12px;
    }

    .layout {
      margin-top: 18px;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 18px;
      align-items: start;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 18px;
    }

    .panel h2 {
      margin: 0 0 12px;
      font-size: 1.1rem;
    }

    .step-list {
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 12px;
    }

    .step-item {
      display: grid;
      grid-template-columns: 30px 1fr;
      gap: 12px;
      background: var(--surface-soft);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      transition: border-color 0.2s ease, transform 0.2s ease, background 0.2s ease;
    }

    .step-item.done {
      border-color: #86e0d6;
      background: #ecfbf9;
      transform: translateY(-1px);
    }

    .step-num {
      width: 30px;
      height: 30px;
      border-radius: 999px;
      background: #dce7f5;
      display: grid;
      place-items: center;
      font-weight: 700;
      color: #234567;
      font-size: 0.92rem;
    }

    .step-item.done .step-num {
      background: var(--accent);
      color: #fff;
    }

    .step-title {
      margin: 1px 0 4px;
      font-weight: 700;
      font-size: 0.95rem;
    }

    .step-copy {
      margin: 0;
      color: var(--ink-dim);
      font-size: 0.88rem;
      line-height: 1.3;
    }

    .input-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }

    label {
      display: grid;
      gap: 6px;
      font-size: 0.86rem;
      color: var(--ink-dim);
      font-weight: 600;
    }

    input, select, textarea {
      font: inherit;
      color: var(--ink);
      padding: 10px 11px;
      border-radius: 11px;
      border: 1px solid #c8d6e6;
      background: #fff;
      outline: none;
    }

    input:focus, select:focus, textarea:focus {
      border-color: #65b6ac;
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14);
    }

    .message-wrap {
      margin-top: 6px;
      margin-bottom: 12px;
    }

    textarea {
      width: 100%;
      min-height: 94px;
      resize: vertical;
    }

    .buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 10px;
    }

    button {
      font: inherit;
      font-weight: 700;
      border: none;
      border-radius: 11px;
      padding: 10px 14px;
      cursor: pointer;
      transition: transform 0.15s ease, filter 0.15s ease;
    }

    button:hover { transform: translateY(-1px); filter: saturate(1.08); }
    button:active { transform: translateY(0); }

    .btn-main { background: var(--accent); color: #fff; }
    .btn-soft { background: #eef6ff; color: #114774; border: 1px solid #bfd9ff; }
    .btn-alt { background: #fff4eb; color: #9a3f03; border: 1px solid #ffd8bd; }

    .notice {
      border-radius: 10px;
      border: 1px solid var(--line);
      background: var(--surface-soft);
      color: var(--ink-dim);
      padding: 9px 11px;
      font-size: 0.9rem;
      margin-bottom: 12px;
    }

    .notice.error {
      color: var(--danger);
      border-color: #efb6b6;
      background: #fff1f1;
    }

    .result-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .result-card {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--surface-soft);
      padding: 12px;
      min-height: 225px;
    }

    .result-card h3 {
      margin: 0 0 9px;
      font-size: 0.98rem;
    }

    .response {
      margin: 0 0 12px;
      line-height: 1.45;
      font-size: 0.95rem;
      color: var(--ink);
    }

    .meta {
      margin: 0;
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 6px 10px;
      font-size: 0.86rem;
    }

    .meta dt { color: var(--ink-dim); }
    .meta dd { margin: 0; font-weight: 600; color: var(--ink); }

    .summary-list {
      margin: 0;
      padding-left: 18px;
      display: grid;
      gap: 8px;
      font-size: 0.9rem;
      color: var(--ink-dim);
    }

    .summary-list b {
      color: var(--ink);
      font-weight: 700;
    }

    .reveal {
      opacity: 0;
      transform: translateY(14px);
      animation: rise 0.55s ease forwards;
    }

    .delay-1 { animation-delay: 0.08s; }
    .delay-2 { animation-delay: 0.16s; }

    @keyframes rise {
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @media (max-width: 980px) {
      .layout {
        grid-template-columns: 1fr;
      }

      .result-grid {
        grid-template-columns: 1fr;
      }

      .input-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="bg-shape">
    <div class="blob one"></div>
    <div class="blob two"></div>
    <div class="blob three"></div>
  </div>

  <div class="shell">
    <section class="hero reveal">
      <div class="hero-top">
        <div>
          <h1>Life Optimizer AI Assistant</h1>
          <p class="subtitle">Follow the 3 guided steps below. You can chat with the coach, save check-ins, and review progress in one place.</p>
        </div>
        <span id="healthBadge" class="badge">Checking API status...</span>
      </div>
      <div class="quick-links">
        <a href="/about">About This AI</a>
        <a href="/docs" target="_blank" rel="noreferrer">Open API Docs</a>
        <a href="/health" target="_blank" rel="noreferrer">Open Health Endpoint</a>
      </div>
    </section>

    <main class="layout">
      <section class="panel reveal delay-1">
        <h2>How To Use This Page</h2>
        <ol class="step-list">
          <li id="step-1" class="step-item">
            <div class="step-num">1</div>
            <div>
              <p class="step-title">Add your profile basics</p>
              <p class="step-copy">Set user ID and optional health metrics like sleep, steps, mood, calories, or screen time.</p>
            </div>
          </li>
          <li id="step-2" class="step-item">
            <div class="step-num">2</div>
            <div>
              <p class="step-title">Ask for guidance</p>
              <p class="step-copy">Write your current challenge and click Get Recommendation.</p>
            </div>
          </li>
          <li id="step-3" class="step-item">
            <div class="step-num">3</div>
            <div>
              <p class="step-title">Review response and save progress</p>
              <p class="step-copy">Read the action plan, save daily check-in, then load your summary.</p>
            </div>
          </li>
        </ol>
      </section>

      <section class="panel reveal delay-2">
        <h2>Guided Assistant Panel</h2>
        <div class="input-grid">
          <label>User ID
            <input id="userId" value="demo-user" placeholder="e.g. rayyan-01" />
          </label>
          <label>Mood
            <select id="mood">
              <option value="">Select mood (optional)</option>
              <option value="great">Great</option>
              <option value="good">Good</option>
              <option value="okay">Okay</option>
              <option value="stressed">Stressed</option>
              <option value="anxious">Anxious</option>
              <option value="low">Low</option>
            </select>
          </label>
          <label>Sleep Hours
            <input id="sleepHours" type="number" min="0" step="0.1" placeholder="e.g. 6.5" />
          </label>
          <label>Steps
            <input id="steps" type="number" min="0" step="1" placeholder="e.g. 4500" />
          </label>
          <label>Exercise Minutes
            <input id="exerciseMinutes" type="number" min="0" step="1" placeholder="e.g. 20" />
          </label>
          <label>Calories
            <input id="calories" type="number" min="0" step="1" placeholder="e.g. 2100" />
          </label>
          <label>Screen Time (hours)
            <input id="screenTime" type="number" min="0" step="0.1" placeholder="e.g. 4.5" />
          </label>
          <label>Water (liters)
            <input id="waterLiters" type="number" min="0" step="0.1" placeholder="e.g. 2.1" />
          </label>
        </div>

        <div class="message-wrap">
          <label>Your Question
            <textarea id="message" placeholder="Example: I slept only 5 hours and feel tired on Mondays. What should I do first?"></textarea>
          </label>
        </div>

        <div class="buttons">
          <button class="btn-main" onclick="sendChat()">Get Recommendation</button>
          <button class="btn-soft" onclick="saveDailyLog()">Save Daily Check-In</button>
          <button class="btn-alt" onclick="loadSummary()">Load My Summary</button>
        </div>

        <div id="notice" class="notice">Ready. Fill step 1 and continue.</div>

        <div class="result-grid">
          <article class="result-card">
            <h3>Recommendation Output</h3>
            <p id="responseText" class="response">No response yet. Ask a question to begin.</p>
            <dl class="meta">
              <dt>Action</dt><dd id="metaAction">-</dd>
              <dt>Specialist</dt><dd id="metaAgent">-</dd>
              <dt>Goal Plan</dt><dd id="metaGoal">-</dd>
              <dt>Memory Context</dt><dd id="metaMemory">-</dd>
            </dl>
          </article>

          <article class="result-card">
            <h3>Stored Summary Snapshot</h3>
            <ul id="summaryList" class="summary-list">
              <li><b>Habits:</b> -</li>
              <li><b>Recent Logs:</b> -</li>
              <li><b>Recommendations:</b> -</li>
              <li><b>Semantic Memories:</b> -</li>
            </ul>
          </article>
        </div>
      </section>
    </main>
  </div>

  <script>
    function setNotice(message, isError = false) {
      const notice = document.getElementById("notice");
      notice.textContent = message;
      notice.classList.toggle("error", isError);
    }

    function markStep(stepId) {
      const step = document.getElementById(stepId);
      if (step) {
        step.classList.add("done");
      }
    }

    function parseNumber(value) {
      if (value === "" || value === null || value === undefined) return null;
      const num = Number(value);
      return Number.isFinite(num) ? num : null;
    }

    function buildProfile() {
      const profile = {};
      const map = {
        sleep_hours: parseNumber(document.getElementById("sleepHours").value),
        steps: parseNumber(document.getElementById("steps").value),
        exercise_minutes: parseNumber(document.getElementById("exerciseMinutes").value),
        calories: parseNumber(document.getElementById("calories").value),
        screen_time: parseNumber(document.getElementById("screenTime").value),
        water_liters: parseNumber(document.getElementById("waterLiters").value),
      };
      const mood = document.getElementById("mood").value;
      if (mood) map.mood = mood;

      for (const key in map) {
        if (map[key] !== null && map[key] !== "") {
          profile[key] = map[key];
        }
      }
      return profile;
    }

    async function sendChat() {
      const userId = document.getElementById("userId").value.trim() || "demo-user";
      const message = document.getElementById("message").value.trim();

      if (!message) {
        setNotice("Please type a question before requesting advice.", true);
        return;
      }

      markStep("step-1");
      markStep("step-2");
      setNotice("Requesting recommendation...");

      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: userId,
            user_input: message,
            user_profile: buildProfile()
          })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Failed to get recommendation.");
        }

        document.getElementById("responseText").textContent = data.response || "-";
        document.getElementById("metaAction").textContent = data.action || "-";
        document.getElementById("metaAgent").textContent = data.specialist_agent || "-";
        document.getElementById("metaGoal").textContent = data.goal_plan || "-";
        document.getElementById("metaMemory").textContent = data.memory_context_used || "-";

        markStep("step-3");
        setNotice("Recommendation ready. You can now save a daily check-in or load summary.");
      } catch (error) {
        setNotice("Chat request failed: " + error.message, true);
      }
    }

    async function saveDailyLog() {
      const userId = document.getElementById("userId").value.trim() || "demo-user";
      const profile = buildProfile();

      setNotice("Saving daily check-in...");
      markStep("step-1");

      try {
        const response = await fetch("/logs/daily", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: userId,
            sleep_hours: profile.sleep_hours,
            steps: profile.steps,
            exercise_minutes: profile.exercise_minutes,
            mood: profile.mood,
            calories: profile.calories,
            screen_time: profile.screen_time,
            water_liters: profile.water_liters
          })
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Failed to save daily log.");
        }
        setNotice("Daily check-in saved. Load summary to review trends.");
        markStep("step-3");
      } catch (error) {
        setNotice("Daily check-in failed: " + error.message, true);
      }
    }

    async function loadSummary() {
      const userId = document.getElementById("userId").value.trim() || "demo-user";
      setNotice("Loading summary...");

      try {
        const response = await fetch("/users/" + encodeURIComponent(userId) + "/summary");
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Failed to load summary.");
        }

        const habitsCount = Array.isArray(data.habits) ? data.habits.length : 0;
        const logsCount = Array.isArray(data.recent_daily_logs) ? data.recent_daily_logs.length : 0;
        const recCount = Array.isArray(data.recent_recommendations) ? data.recent_recommendations.length : 0;
        const memCount = Array.isArray(data.semantic_memories) ? data.semantic_memories.length : 0;

        const list = document.getElementById("summaryList");
        list.innerHTML = ""
          + "<li><b>Habits:</b> " + habitsCount + "</li>"
          + "<li><b>Recent Logs:</b> " + logsCount + "</li>"
          + "<li><b>Recommendations:</b> " + recCount + "</li>"
          + "<li><b>Semantic Memories:</b> " + memCount + "</li>";

        setNotice("Summary loaded for " + userId + ".");
        markStep("step-3");
      } catch (error) {
        setNotice("Could not load summary: " + error.message, true);
      }
    }

    async function checkHealth() {
      const badge = document.getElementById("healthBadge");
      try {
        const response = await fetch("/health");
        const data = await response.json();
        if (!response.ok) {
          throw new Error();
        }

        const dbState = data.database_configured ? "DB connected" : "DB not connected";
        badge.textContent = "API online | " + dbState;
        badge.classList.add(data.database_configured ? "ok" : "warn");
      } catch {
        badge.textContent = "API unreachable";
        badge.classList.add("warn");
      }
    }

    checkHealth();
  </script>
</body>
</html>
"""

ABOUT_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>About Life Optimizer AI</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

    :root {
      --bg: #f4f8ff;
      --surface: #ffffff;
      --line: #d7e2f1;
      --ink: #17243a;
      --ink-dim: #4b5d79;
      --accent: #0f766e;
      --accent-soft: #ddf3f0;
      --shadow: 0 16px 28px rgba(16, 41, 76, 0.12);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f8fcff, var(--bg));
      color: var(--ink);
    }

    .wrap {
      max-width: 980px;
      margin: 32px auto;
      padding: 0 18px 24px;
      display: grid;
      gap: 16px;
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow);
      padding: 20px;
    }

    h1, h2 {
      margin: 0 0 10px;
      letter-spacing: -0.02em;
    }

    p {
      margin: 0 0 10px;
      color: var(--ink-dim);
      line-height: 1.5;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .feature {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #f8fbff;
      padding: 14px;
    }

    .tag {
      display: inline-block;
      font-size: 0.82rem;
      font-weight: 700;
      color: #0a5f58;
      background: var(--accent-soft);
      border: 1px solid #bde6e1;
      border-radius: 999px;
      padding: 5px 10px;
      margin-bottom: 8px;
    }

    .steps {
      margin: 0;
      padding-left: 22px;
      color: var(--ink-dim);
      line-height: 1.5;
    }

    .steps li { margin-bottom: 8px; }

    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 8px;
    }

    a.btn {
      text-decoration: none;
      font-weight: 700;
      border-radius: 10px;
      padding: 9px 12px;
      border: 1px solid var(--line);
    }

    .btn-main {
      background: var(--accent);
      color: #fff;
      border-color: #0e6d65;
    }

    .btn-soft {
      background: #eef5ff;
      color: #123c6b;
      border-color: #bfd7ff;
    }

    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="card">
      <h1>About Life Optimizer AI</h1>
      <p>
        Life Optimizer AI is a friendly coaching assistant that helps users improve sleep, activity, focus, mood, and daily habits through simple step-by-step guidance.
      </p>
      <div class="actions">
        <a class="btn btn-main" href="/">Open Assistant</a>
        <a class="btn btn-soft" href="/docs" target="_blank" rel="noreferrer">API Docs</a>
      </div>
    </section>

    <section class="card">
      <h2>What It Can Do</h2>
      <div class="grid">
        <article class="feature">
          <span class="tag">Smart Recommendations</span>
          <p>Analyzes your input and health signals to provide focused, actionable advice.</p>
        </article>
        <article class="feature">
          <span class="tag">Personal Memory</span>
          <p>Remembers previous interactions and behavior patterns to personalize future guidance.</p>
        </article>
        <article class="feature">
          <span class="tag">Daily Check-ins</span>
          <p>Tracks sleep, steps, mood, calories, exercise, and screen time to build consistent habits.</p>
        </article>
        <article class="feature">
          <span class="tag">Goal Planning</span>
          <p>Turns broad advice into clear, short goals that are easier to follow every day.</p>
        </article>
      </div>
    </section>

    <section class="card">
      <h2>Why It Is Useful</h2>
      <ol class="steps">
        <li>It keeps recommendations practical and easy to apply, not overwhelming.</li>
        <li>It gives continuity across sessions by using long-term memory and trend tracking.</li>
        <li>It helps users understand progress through summaries of habits, logs, and recommendations.</li>
        <li>It supports better consistency by combining coaching + daily reflection in one place.</li>
      </ol>
    </section>
  </main>
</body>
</html>
"""


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/about", response_class=HTMLResponse)
def about_page() -> HTMLResponse:
    return HTMLResponse(ABOUT_HTML)


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "database_configured": is_database_configured(),
        "llm": get_llm_status(),
    }


@app.post("/chat")
def chat(data: dict[str, Any]) -> dict[str, Any]:
    user_input = str(data.get("user_input") or data.get("message") or "").strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="user_input is required")

    user_id = str(data.get("user_id", "anonymous"))
    user_profile = data.get("user_profile", {})
    if not isinstance(user_profile, dict):
        raise HTTPException(status_code=400, detail="user_profile must be an object")

    request_model = ChatRequest(
        user_id=user_id,
        user_input=user_input,
        user_profile=user_profile,
        history=data.get("history", []) if isinstance(data.get("history"), list) else [],
    )
    history = request_model.history or get_recent_history(request_model.user_id, limit=8)

    state = {
        "user_id": request_model.user_id,
        "user_input": request_model.user_input,
        "history": history,
        "user_profile": request_model.user_profile,
    }
    try:
        result = agent.invoke(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent invocation failed: {exc}") from exc
    response = result.get("recommendation", "Focus on balanced habits.")

    upsert_user(request_model.user_id)
    insert_recommendation(
        user_id=request_model.user_id,
        recommendation=response,
        source_agent=result.get("specialist_agent", "Supervisor Agent"),
        context={
            "action": result.get("action"),
            "route": result.get("route"),
            "goal_plan": result.get("goal_plan"),
        },
    )
    add_semantic_memory(
        user_id=request_model.user_id,
        text=(
            f"User asked: {request_model.user_input}. "
            f"Agent ({result.get('specialist_agent', 'Supervisor Agent')}) responded: {response}"
        ),
        metadata={
            "type": "chat_exchange",
            "action": result.get("action"),
        },
    )

    return {
        "response": response,
        "action": result.get("action"),
        "specialist_agent": result.get("specialist_agent"),
        "goal_plan": result.get("goal_plan"),
        "memory_context_used": result.get("memory_context", ""),
    }


@app.post("/logs/daily")
def add_daily_log(log: DailyLogInput) -> dict[str, Any]:
    upsert_user(log.user_id)
    payload = log.model_dump(exclude_none=True)
    stored = upsert_daily_log(payload)
    add_semantic_memory(
        user_id=log.user_id,
        text=(
            f"Daily check-in: sleep={payload.get('sleep_hours')}, steps={payload.get('steps')}, "
            f"exercise_minutes={payload.get('exercise_minutes')}, mood={payload.get('mood')}, "
            f"calories={payload.get('calories')}, screen_time={payload.get('screen_time')}."
        ),
        metadata={"type": "daily_log"},
    )
    return {
        "status": "saved",
        "database_stored": stored is not None,
        "log": stored or payload,
    }


@app.get("/users/{user_id}/summary")
def get_user_summary(user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "memory": load_memory(user_id),
        "habits": get_user_habits(user_id, limit=10),
        "recent_daily_logs": get_recent_daily_logs(user_id, limit=7),
        "recent_recommendations": get_recent_recommendations(user_id, limit=10),
        "semantic_memories": list_semantic_memories(user_id, limit=10),
    }


@app.get("/users/{user_id}/memory/search")
def search_user_memory(user_id: str, q: str) -> dict[str, Any]:
    if not q.strip():
        raise HTTPException(status_code=400, detail="q is required")
    return {"user_id": user_id, "query": q, "matches": query_semantic_memory(user_id, q, n_results=5)}
