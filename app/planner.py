from app.models import AgentState


def planner(state: AgentState) -> dict[str, str]:
    message = state["user_input"].lower()

    if any(token in message for token in ("sleep", "slept", "rest", "tired")):
        return {"action": "sleep_analysis"}

    if any(token in message for token in ("exercise", "workout", "steps", "run", "fitness")):
        return {"action": "fitness_analysis"}

    if any(token in message for token in ("diet", "food", "nutrition", "meal", "water")):
        return {"action": "diet_analysis"}

    if any(token in message for token in ("mood", "stress", "anxious", "sad", "overwhelmed")):
        return {"action": "mood_analysis"}

    if any(token in message for token in ("goal", "plan", "focus", "productivity", "procrast")):
        return {"action": "productivity_analysis"}

    return {"action": "general_advice"}
