"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator

# Sentinel: the placeholder secret shipped in defaults. Production refuses to start with this.
_DEFAULT_SECRET_KEY = "kisan-credit-jwt-secret-key-change-in-production-at-least-32-chars-long"


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = Field(default="KisanCredit", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Security / Authentication
    secret_key: str = Field(
        default=_DEFAULT_SECRET_KEY,
        alias="SECRET_KEY"
    )
    access_token_expire_minutes: int = Field(default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, alias="REFRESH_TOKEN_EXPIRE_DAYS")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:pass@localhost:5432/kisancredit",
        alias="DATABASE_URL"
    )
    database_url_sync: str = Field(
        default="postgresql://user:pass@localhost:5432/kisancredit",
        alias="DATABASE_URL_SYNC"
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        alias="MLFLOW_TRACKING_URI"
    )

    # Model
    model_path: str = Field(default="models/home_credit_v2.pkl", alias="MODEL_PATH")

    # Rate Limiting
    rate_limit_requests: int = Field(default=100, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=900, alias="RATE_LIMIT_WINDOW")

    # Feature Weights (from PDF - Profitability Score Matrix)
    income_weight: float = Field(default=0.40, alias="INCOME_WEIGHT")
    expense_weight: float = Field(default=0.25, alias="EXPENSE_WEIGHT")
    social_weight: float = Field(default=0.15, alias="SOCIAL_WEIGHT")
    discipline_weight: float = Field(default=0.10, alias="DISCIPLINE_WEIGHT")
    behavioral_weight: float = Field(default=0.10, alias="BEHAVIORAL_WEIGHT")

    # CORS — comma-separated list of allowed origins. In dev, default is "*".
    # In production, must be set explicitly to a comma-separated whitelist.
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # LLM — optional in dev (falls back to template explanation), required in prod
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # Email OTP via Resend — optional in dev (falls back to console-logged OTP)
    resend_api_key: str = Field(default="", alias="RESEND_API_KEY")
    resend_from_email: str = Field(
        default="KisanCredit <onboarding@resend.dev>", alias="RESEND_FROM_EMAIL"
    )

    # Demo mode — for the public portfolio deployment, so recruiters can use
    # the full app (incl. the operator dashboard) without inbox access.
    # When demo_mode is on, /auth/send-otp echoes the OTP in its response so
    # the login UI can surface it. A real production deploy leaves this off.
    demo_mode: bool = Field(default=False, alias="DEMO_MODE")
    # If set, this account is auto-seeded with role=admin on every startup so
    # the operator dashboard is always reachable on the deployed demo.
    demo_admin_email: str = Field(default="", alias="DEMO_ADMIN_EMAIL")

    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = False

    @model_validator(mode="after")
    def _guard_production_defaults(self):
        """Refuse to start in production with development placeholders."""
        if self.environment.lower() == "production":
            if self.secret_key == _DEFAULT_SECRET_KEY:
                raise ValueError(
                    "SECRET_KEY is the default placeholder. Set a real SECRET_KEY "
                    "(>=32 random chars) in the environment before running in production."
                )
            if len(self.secret_key) < 32:
                raise ValueError(
                    f"SECRET_KEY is too short ({len(self.secret_key)} chars). "
                    "Use at least 32 random characters in production."
                )
            if self.cors_origins.strip() == "*":
                raise ValueError(
                    "CORS_ORIGINS is wildcard. Set a comma-separated whitelist of "
                    "allowed origins before running in production."
                )
        return self

    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS env var into a list."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


# Global settings instance
settings = Settings()
