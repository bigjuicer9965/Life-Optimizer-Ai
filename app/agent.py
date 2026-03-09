import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.memory import get_recent_history, save_memory
from app.models import AgentState
from app.planner import planner
from app.semantic_memory import add_semantic_memory, query_semantic_memory
from app.tools import (
    analyze_diet,
    analyze_mood,
    analyze_sleep,
    productivity_advice,
    recommend_steps,
)
from database.db import get_user_habits, upsert_habit

load_dotenv()

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY", "").strip()
    or os.getenv("OPENAI_API_KEY", "").strip()
)
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "").strip() or os.getenv("APP_URL", "").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Life Optimizer AI")


def _build_llm() -> ChatOpenAI | None:
    if not OPENROUTER_API_KEY:
        return None

    headers: dict[str, str] = {"X-Title": OPENROUTER_APP_NAME}
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER

    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        temperature=0.3,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers=headers,
    )


llm = _build_llm()


def get_llm_status() -> dict[str, Any]:
    return {
        "provider": "openrouter",
        "configured": llm is not None,
        "model": OPENROUTER_MODEL,
        "base_url": OPENROUTER_BASE_URL,
        "has_http_referer": bool(OPENROUTER_HTTP_REFERER),
        "using_key_env": "OPENROUTER_API_KEY" if os.getenv("OPENROUTER_API_KEY", "").strip() else ("OPENAI_API_KEY" if os.getenv("OPENAI_API_KEY", "").strip() else "none"),
    }


def memory_context_agent(state: AgentState) -> dict[str, Any]:
    user_id = state.get("user_id", "anonymous")
    if user_id == "anonymous":
        return {"semantic_memories": [], "habit_context": [], "memory_context": ""}

    semantic_hits = query_semantic_memory(user_id=user_id, query=state["user_input"], n_results=4)
    semantic_memories = [hit["text"] for hit in semantic_hits if hit.get("text")]
    habits = get_user_habits(user_id=user_id, limit=5)
    habit_context = [f"{habit['habit_key']}: {habit['habit_value']}" for habit in habits]

    context_lines: list[str] = []
    if habit_context:
        context_lines.append("Known habits -> " + "; ".join(habit_context))
    if semantic_memories:
        context_lines.append("Relevant behavioral history -> " + " | ".join(semantic_memories))

    return {
        "semantic_memories": semantic_memories,
        "habit_context": habit_context,
        "memory_context": "\n".join(context_lines),
    }


def planner_agent(state: AgentState) -> dict[str, str]:
    return planner(state)


def supervisor_agent(state: AgentState) -> dict[str, str]:
    action = state.get("action", "general_advice")
    route_map = {
        "sleep_analysis": "health_agent",
        "mood_analysis": "health_agent",
        "fitness_analysis": "fitness_agent",
        "diet_analysis": "diet_agent",
        "productivity_analysis": "productivity_agent",
        "general_advice": "health_agent",
    }
    return {"route": route_map.get(action, "health_agent")}


def health_agent(state: AgentState) -> dict[str, str]:
    profile = state.get("user_profile", {})
    action = state.get("action", "")
    if action == "mood_analysis":
        recommendation = analyze_mood(profile)
    elif action == "general_advice" and llm is not None:
        prompt = (
            "You are a practical life coach. Give a short, specific recommendation in 2-4 sentences.\n"
            f"User input: {state['user_input']}\n"
            f"User profile: {profile}\n"
            f"Memory context: {state.get('memory_context', 'none')}"
        )
        try:
            response = llm.invoke(prompt)
            content = response.content if isinstance(response.content, str) else str(response.content)
            recommendation = content.strip() or "Focus on balanced habits."
        except Exception:
            recommendation = "Focus on balanced habits."
    else:
        recommendation = analyze_sleep(profile)
    return {
        "specialist_agent": "Health Agent",
        "specialist_recommendation": recommendation,
    }


def fitness_agent(state: AgentState) -> dict[str, str]:
    recommendation = recommend_steps(state.get("user_profile", {}))
    return {
        "specialist_agent": "Fitness Agent",
        "specialist_recommendation": recommendation,
    }


def diet_agent(state: AgentState) -> dict[str, str]:
    recommendation = analyze_diet(state.get("user_profile", {}))
    return {
        "specialist_agent": "Diet Agent",
        "specialist_recommendation": recommendation,
    }


def productivity_agent(state: AgentState) -> dict[str, str]:
    recommendation = productivity_advice(
        state.get("user_profile", {}),
        state["user_input"],
    )
    return {
        "specialist_agent": "Productivity Agent",
        "specialist_recommendation": recommendation,
    }


def goal_planning_agent(state: AgentState) -> dict[str, str]:
    base_recommendation = state.get("specialist_recommendation", "Focus on balanced habits.")
    memory_context = state.get("memory_context", "")
    prompt = (
        "You are a personal life optimization coach.\n"
        "Use the memory context and recommendation to produce one concrete goal for today.\n\n"
        f"User input: {state['user_input']}\n"
        f"Memory context: {memory_context or 'No long-term context available'}\n"
        f"Recommendation: {base_recommendation}"
    )
    try:
        if llm is None:
            raise ValueError("OpenRouter API key is not configured.")
        response = llm.invoke(prompt)
        goal_plan = response.content if isinstance(response.content, str) else str(response.content)
    except Exception:
        goal_plan = f"For the next 7 days: {base_recommendation}"

    return {
        "goal_plan": goal_plan,
        "recommendation": f"{base_recommendation} Goal: {goal_plan}",
    }


def memory_agent(state: AgentState) -> dict[str, list[str]]:
    history = list(state.get("history", []))
    recommendation = state.get("recommendation", "Focus on balanced habits.")
    base_recommendation = state.get("specialist_recommendation", recommendation)
    goal_plan = state.get("goal_plan", "")
    history.append(f"User: {state['user_input']}")
    history.append(f"Agent: {recommendation}")

    user_id = state.get("user_id", "anonymous")
    if user_id != "anonymous":
        save_memory(
            user_id=user_id,
            user_input=state["user_input"],
            recommendation=recommendation,
            metadata={
                "action": state.get("action"),
                "route": state.get("route"),
                "specialist_agent": state.get("specialist_agent"),
            },
        )
        semantic_payload = (
            f"User input: {state['user_input']}. "
            f"Detected action: {state.get('action', 'unknown')}. "
            f"Recommendation: {base_recommendation}. "
            f"Goal: {goal_plan}. "
            f"Habit context: {'; '.join(state.get('habit_context', []))}."
        )
        add_semantic_memory(
            user_id=user_id,
            text=semantic_payload,
            metadata={
                "action": state.get("action", "general_advice"),
                "specialist_agent": state.get("specialist_agent", "Unknown Agent"),
            },
        )

        profile = state.get("user_profile", {})
        preferred_workout_time = str(profile.get("preferred_workout_time", "") or "").strip().lower()
        if preferred_workout_time:
            upsert_habit(
                user_id=user_id,
                habit_key="preferred_workout_time",
                habit_value=preferred_workout_time,
                confidence=0.7,
                source="user_profile",
            )
    return {"history": history}


def _route_specialist(state: AgentState) -> str:
    return state.get("route", "health_agent")


workflow = StateGraph(AgentState)
workflow.add_node("memory_context_agent", memory_context_agent)
workflow.add_node("planner_agent", planner_agent)
workflow.add_node("supervisor_agent", supervisor_agent)
workflow.add_node("health_agent", health_agent)
workflow.add_node("fitness_agent", fitness_agent)
workflow.add_node("diet_agent", diet_agent)
workflow.add_node("productivity_agent", productivity_agent)
workflow.add_node("goal_planning_agent", goal_planning_agent)
workflow.add_node("memory_agent", memory_agent)

workflow.set_entry_point("memory_context_agent")
workflow.add_edge("memory_context_agent", "planner_agent")
workflow.add_edge("planner_agent", "supervisor_agent")
workflow.add_conditional_edges(
    "supervisor_agent",
    _route_specialist,
    {
        "health_agent": "health_agent",
        "fitness_agent": "fitness_agent",
        "diet_agent": "diet_agent",
        "productivity_agent": "productivity_agent",
    },
)
workflow.add_edge("health_agent", "goal_planning_agent")
workflow.add_edge("fitness_agent", "goal_planning_agent")
workflow.add_edge("diet_agent", "goal_planning_agent")
workflow.add_edge("productivity_agent", "goal_planning_agent")
workflow.add_edge("goal_planning_agent", "memory_agent")
workflow.add_edge("memory_agent", END)

agent = workflow.compile()


def run_agent(
    user_input: str,
    user_profile: dict[str, Any] | None = None,
    history: list[str] | None = None,
    user_id: str = "anonymous",
) -> AgentState:
    existing_history = history or get_recent_history(user_id=user_id, limit=8)
    state: AgentState = {
        "user_id": user_id,
        "user_input": user_input,
        "history": existing_history,
        "user_profile": user_profile or {},
    }
    return agent.invoke(state)
