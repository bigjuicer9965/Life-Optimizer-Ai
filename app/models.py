from datetime import date
from typing import Any, NotRequired, TypedDict

from pydantic import BaseModel, Field


class AgentState(TypedDict):
    user_input: str
    history: list[str]
    user_profile: dict[str, Any]
    user_id: NotRequired[str]
    semantic_memories: NotRequired[list[str]]
    habit_context: NotRequired[list[str]]
    memory_context: NotRequired[str]
    action: NotRequired[str]
    route: NotRequired[str]
    specialist_agent: NotRequired[str]
    specialist_recommendation: NotRequired[str]
    goal_plan: NotRequired[str]
    recommendation: NotRequired[str]


class UserQuery(BaseModel):
    user_id: str
    message: str


class ChatRequest(BaseModel):
    user_id: str = "anonymous"
    user_input: str
    user_profile: dict[str, Any] = Field(default_factory=dict)
    history: list[str] = Field(default_factory=list)


class DailyLogInput(BaseModel):
    user_id: str
    log_date: date | None = None
    sleep_hours: float | None = None
    steps: int | None = None
    exercise_minutes: int | None = None
    mood: str | None = None
    calories: float | None = None
    screen_time: float | None = None
    diet_notes: str | None = None
    water_liters: float | None = None
