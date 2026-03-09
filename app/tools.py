from typing import Any, Callable


def analyze_sleep(data: dict[str, Any]) -> str:
    sleep_hours = float(data.get("sleep_hours", 0) or 0)

    if sleep_hours == 0:
        return "Log your sleep daily so recommendations can be more accurate."

    if sleep_hours < 6:
        return "You should aim for 7-8 hours of sleep."

    if sleep_hours < 7:
        return "You are close. Try adding 30-45 minutes of extra sleep."

    return "Your sleep looks healthy."


def recommend_steps(data: dict[str, Any]) -> str:
    steps = int(data.get("steps", 0) or 0)
    exercise_minutes = int(data.get("exercise_minutes", 0) or 0)

    if steps < 5000:
        return "Try walking at least 7000 steps daily."

    if exercise_minutes < 20:
        return "Add a 20-minute workout to improve your activity baseline."

    return "Great activity level!"


def analyze_diet(data: dict[str, Any]) -> str:
    diet_notes = str(data.get("diet_notes", "") or "").lower()
    water_liters = float(data.get("water_liters", 0) or 0)

    if any(token in diet_notes for token in ("fast food", "sugar", "fried")):
        return "Reduce processed foods and add one whole-food meal today."

    if water_liters and water_liters < 2:
        return "Increase hydration to around 2 liters of water daily."

    return "Your diet habits look stable. Keep meals balanced with protein and fiber."


def analyze_mood(data: dict[str, Any]) -> str:
    mood = str(data.get("mood", "") or "").lower()

    if mood in {"stressed", "anxious", "sad", "low"}:
        return "Take a 10-minute break for breathing or a short walk before your next task."

    if mood in {"okay", "fine"}:
        return "Add one intentional recovery block today to protect your energy."

    return "Mood looks steady. Keep your current routine and sleep schedule consistent."


def productivity_advice(data: dict[str, Any], user_input: str) -> str:
    mood = str(data.get("mood", "") or "").lower()
    if mood in {"stressed", "anxious", "low"}:
        return "Use a light workload block first, then one 25-minute focus sprint."

    if "procrast" in user_input.lower():
        return "Pick one top priority and commit to 25 focused minutes right now."

    return "Set your top 3 tasks and finish the hardest one in your first work block."


def get_tools() -> dict[str, Callable[..., str]]:
    return {
        "sleep_analysis": analyze_sleep,
        "fitness_analysis": recommend_steps,
        "diet_analysis": analyze_diet,
        "mood_analysis": analyze_mood,
        "productivity_analysis": productivity_advice,
    }
