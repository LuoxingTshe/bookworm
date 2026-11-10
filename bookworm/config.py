"""Configuration with an explicit workspace and backwards-compatible environment names."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore', populate_by_name=True)
    deepseek_api_key: str = Field('', repr=False)
    deepseek_base_url: str = 'https://api.deepseek.com'
    deepseek_model: str = 'deepseek-chat'
    deepseek_temperature: float = Field(0.2, ge=0, le=2)
    request_timeout: float = Field(60, gt=0, validation_alias='BOOKWORM_REQUEST_TIMEOUT')
    wos_api_key: str = Field('', repr=False)
    wos_base_url: str = 'https://api.clarivate.com/apis/wos-starter/v1'
    wos_max_requests: int = Field(60, ge=0)
    wos_min_interval: float = Field(1.1, ge=0)
    openalex_email: str = ''
    openalex_api_key: str = Field('', repr=False)
    openalex_max_requests: int = Field(200, ge=1, validation_alias='BOOKWORM_OPENALEX_MAX_REQUESTS')
    # Python attribute names remain compatible; new environment names take priority.
    scib_cache_dir: Path = Field(Path('artifacts/cache'), validation_alias=AliasChoices('BOOKWORM_CACHE_DIR', 'SCIB_CACHE_DIR'))
    scib_corpora_dir: Path = Field(Path('artifacts/runs'), validation_alias=AliasChoices('BOOKWORM_RUNS_DIR', 'SCIB_CORPORA_DIR'))
    scib_keep_history: bool = True  # legacy setting: run history is now always retained
    scib_keep_cache: bool = Field(True, validation_alias=AliasChoices('BOOKWORM_KEEP_CACHE', 'SCIB_KEEP_CACHE'))
    scib_obsidian_vault: Path | None = Field(None, validation_alias=AliasChoices('BOOKWORM_OBSIDIAN_VAULT', 'SCIB_OBSIDIAN_VAULT'))

    @field_validator('scib_obsidian_vault', mode='before')
    @classmethod
    def empty_vault(cls, value):
        return None if value == '' else value


_home: Path | None = None


def configure(home: Path | None = None) -> None:
    """Select a workspace without changing process cwd. Environment overrides its .env."""
    global _home
    _home = Path(home).expanduser().resolve() if home is not None else None
    get_settings.cache_clear()


def workspace() -> Path:
    return (_home or Path(os.environ.get('BOOKWORM_HOME') or os.environ.get('SCIBOOSTER_HOME') or '.')).expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    root = workspace()
    settings = Settings(_env_file=root / '.env')
    for name in ('scib_cache_dir', 'scib_corpora_dir', 'scib_obsidian_vault'):
        path = getattr(settings, name)
        if path is not None:
            path = path.expanduser()
            setattr(settings, name, (root / path).resolve() if not path.is_absolute() else path.resolve())
    return settings


def resolve_source(source: str, settings: Settings) -> str:
    from .errors import InputError
    if source not in ('auto', 'wos', 'openalex'):
        raise InputError('source must be auto, wos or openalex')
    return ('wos' if settings.wos_api_key else 'openalex') if source == 'auto' else source

