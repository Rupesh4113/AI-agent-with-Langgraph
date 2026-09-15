"""
Global configuration for the AI Research Agent.
Loads settings from environment variables with safe defaults.
"""
import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Attempt to load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

class Settings(BaseModel):
    """Application configuration."""

    # Provider & Model Settings
    MODEL_PROVIDER: Literal["openai", "google", "mock"] = Field(
        default=os.getenv("MODEL_PROVIDER", "mock"),  # type: ignore
        description="LLM provider: 'openai', 'google', or 'mock'"
    )
    MODEL_NAME: str = Field(
        default=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        description="Model identifier, e.g. gpt-4o-mini, gemini-1.5-pro, etc."
    )
    TEMPERATURE: float = Field(
        default=float(os.getenv("TEMPERATURE", "0.2")),
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature"
    )

    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=os.getenv("OPENAI_API_KEY"))
    GOOGLE_API_KEY: Optional[str] = Field(default=os.getenv("GOOGLE_API_KEY"))

    # Agent Parameters
    MAX_RESEARCH_ITERATIONS: int = Field(
        default=int(os.getenv("MAX_RESEARCH_ITERATIONS", "3")),
        ge=1,
        le=10,
        description="Maximum cycles of research refinement before proceeding"
    )
    MIN_CONFIDENCE_THRESHOLD: float = Field(
        default=float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.80")),
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required to skip re-research"
    )

    # Storage & Persistence
    CHECKPOINT_DB_PATH: Path = Field(
        default=PROJECT_ROOT / "data" / "checkpoints.db",
        description="SQLite database path for durable conversation checkpointing"
    )
    SAMPLE_DOCS_DIR: Path = Field(
        default=PROJECT_ROOT / "data" / "sample_documents",
        description="Directory for local document search"
    )

    # LangSmith Observability
    LANGCHAIN_TRACING_V2: bool = Field(
        default=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1")
    )
    LANGCHAIN_API_KEY: Optional[str] = Field(default=os.getenv("LANGCHAIN_API_KEY"))
    LANGCHAIN_PROJECT: str = Field(
        default=os.getenv("LANGCHAIN_PROJECT", "ai-research-agent")
    )
    LANGCHAIN_ENDPOINT: str = Field(
        default=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    )

settings = Settings()
