from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import PromptConfig


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system: str
    user: str
    path: Path


class PromptLoader:
    def __init__(self, config: PromptConfig | None = None):
        self.config = config or PromptConfig()

    def load(self, version: str | None = None) -> PromptTemplate:
        prompt_version = version or self.config.version
        path = self._path_for_version(prompt_version)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found for {prompt_version}: {path}")

        raw = path.read_text(encoding="utf-8")
        metadata, system, user = self._parse(raw)
        file_version = metadata.get("version", prompt_version)
        return PromptTemplate(version=file_version, system=system, user=user, path=path)

    def _path_for_version(self, version: str) -> Path:
        if version == self.config.version:
            return self.config.prompt_dir / self.config.filename
        suffix = version.replace(".", "_")
        return self.config.prompt_dir / f"esg_scoring_prompt_{suffix}.txt"

    def _parse(self, raw: str) -> tuple[dict[str, str], str, str]:
        metadata: dict[str, str] = {}
        lines = raw.splitlines()
        body_start = 0
        for idx, line in enumerate(lines):
            if line.strip() == "---":
                body_start = idx + 1
                break
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip().lower()] = value.strip()

        body = "\n".join(lines[body_start:])
        if "SYSTEM:" not in body or "USER:" not in body:
            raise ValueError("Prompt file must contain SYSTEM: and USER: sections.")
        system_part = body.split("SYSTEM:", 1)[1].split("USER:", 1)[0].strip()
        user_part = body.split("USER:", 1)[1].strip()
        return metadata, system_part, user_part
