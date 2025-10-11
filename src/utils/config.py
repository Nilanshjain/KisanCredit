"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = Field(default="KisanCredit", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:pass@localhost:5432/kisancredit",
        alias="DATABASE_URL"
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        alias="MLFLOW_TRACKING_URI"
    )

    # Model
    model_path: str = Field(default="models/profitability_model.pkl", alias="MODEL_PATH")

    # Rate Limiting
    rate_limit_requests: int = Field(default=100, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=900, alias="RATE_LIMIT_WINDOW")

    # Feature Weights (from PDF - Profitability Score Matrix)
    income_weight: float = Field(default=0.40, alias="INCOME_WEIGHT")
    expense_weight: float = Field(default=0.25, alias="EXPENSE_WEIGHT")
    social_weight: float = Field(default=0.15, alias="SOCIAL_WEIGHT")
    discipline_weight: float = Field(default=0.10, alias="DISCIPLINE_WEIGHT")
    behavioral_weight: float = Field(default=0.10, alias="BEHAVIORAL_WEIGHT")

    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
