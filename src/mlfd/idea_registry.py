from __future__ import annotations

from typing import Any

from .config import ProjectPaths
from .utils import read_json, utc_timestamp, write_json


DEFAULT_ITEMS: tuple[dict[str, Any], ...] = (
    {
        "idea_key": "pixelshuffle_film_groupnorm_v1",
        "family": "decoder",
        "status": "blocked",
        "source_worktree": "ML_FluidDynamics_decoder_combo",
        "source_experiments": ["exp-0001"],
        "summary": "The current PixelShuffle/FiLM/GroupNorm decoder line is negative evidence in its tested form.",
        "revisit_only_if": "Only if the decoder formulation changes materially instead of retuning the same stack.",
    },
    {
        "idea_key": "ffl_wake_decoder_ft_v1",
        "family": "loss_recipe",
        "status": "blocked",
        "source_worktree": "ML_FluidDynamics_loss_combo",
        "source_experiments": ["exp-0001"],
        "summary": "The tested FFL/wake/decoder-FT recipe is negative evidence in its current form.",
        "revisit_only_if": "Only if the loss contract changes materially.",
    },
    {
        "idea_key": "phase_refine_phase_residual_v1",
        "family": "phase_amplitude",
        "status": "blocked",
        "source_worktree": "ML_FluidDynamics_phase_amplitude",
        "source_experiments": ["exp-0080", "exp-0082"],
        "summary": "Explicit phase splitting alone was not enough; the v1 phase_refine + phase_residual line is negative evidence.",
        "revisit_only_if": "Only if the formulation changes materially, such as decoder-side phase conditioning or a different phase objective.",
    },
    {
        "idea_key": "joint_v2_decoder_curriculum_v1",
        "family": "joint_training",
        "status": "blocked",
        "source_worktree": "ML_FluidDynamics_joint_v2",
        "source_experiments": ["exp-0068"],
        "summary": "Decoder-only curriculum 8 -> 16 -> 32 regressed badly enough to close the current formulation.",
        "revisit_only_if": "Only if the curriculum objective changes materially.",
    },
    {
        "idea_key": "joint_v2_teacher_anchor_v1",
        "family": "joint_training",
        "status": "blocked",
        "source_worktree": "ML_FluidDynamics_joint_v2",
        "source_experiments": ["exp-0073"],
        "summary": "The first teacher-anchor setting collapsed and should not be retried as a near miss.",
        "revisit_only_if": "Only if the teacher-anchor formulation changes materially.",
    },
    {
        "idea_key": "joint_v2_alternating_refresh_v1",
        "family": "joint_training",
        "status": "blocked",
        "source_worktree": "ML_FluidDynamics_joint_v2",
        "source_experiments": ["exp-0075"],
        "summary": "The first linear-only alternating-refresh setting is negative evidence in its current form.",
        "revisit_only_if": "Only if refresh mechanics change materially.",
    },
    {
        "idea_key": "joint_v2_ema_latent_stats_v1",
        "family": "joint_training",
        "status": "blocked",
        "source_worktree": "ML_FluidDynamics_joint_v2",
        "source_experiments": ["exp-0077"],
        "summary": "EMA latent-stat stabilization regressed relative to the corrected control.",
        "revisit_only_if": "Only if the EMA formulation changes materially.",
    },
    {
        "idea_key": "residual_target_supervision_v2",
        "family": "decoder_supervision",
        "status": "blocked",
        "source_worktree": "ML_FluidDynamics_residual_target_v2",
        "source_experiments": ["exp-0060"],
        "summary": "Residual-target supervision v2 cleared the AE gate but collapsed on promoted full rollout, so it is negative evidence.",
        "revisit_only_if": "Only if the supervision target or decoder contract changes materially.",
    },
)


def _default_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": utc_timestamp(),
        "items": list(DEFAULT_ITEMS),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Idea Registry",
        "",
        "This file is the compressed long-horizon memory for idea families.",
        "Read this before opening a new candidate line.",
        "",
    ]
    for item in payload["items"]:
        lines.extend(
            [
                f"## {item['idea_key']}",
                "",
                f"- Status: {item['status']}",
                f"- Family: {item['family']}",
                f"- Source Worktree: {item['source_worktree']}",
                f"- Source Experiments: {', '.join(item['source_experiments'])}",
                f"- Summary: {item['summary']}",
                f"- Revisit Only If: {item['revisit_only_if']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def ensure_registry_files(paths: ProjectPaths) -> None:
    if not paths.idea_registry_json.exists():
        payload = _default_payload()
        write_json(paths.idea_registry_json, payload)
    else:
        payload = read_json(paths.idea_registry_json)
    paths.idea_registry_md.write_text(_render_markdown(payload), encoding="utf-8")


def load_registry(paths: ProjectPaths) -> dict[str, Any]:
    ensure_registry_files(paths)
    return read_json(paths.idea_registry_json)


def find_idea(payload: dict[str, Any], idea_key: str) -> dict[str, Any] | None:
    for item in payload.get("items", []):
        if item.get("idea_key") == idea_key:
            return item
    return None


def blocked_lines(payload: dict[str, Any], limit: int = 5) -> list[str]:
    items = [item for item in payload.get("items", []) if item.get("status") in {"blocked", "paused"}]
    lines = []
    for item in items[:limit]:
        lines.append(
            f"{item['idea_key']} ({item['status']}): {item['summary']} "
            f"[{item['source_worktree']}: {', '.join(item['source_experiments'])}]"
        )
    return lines


def check_idea_allowed(
    payload: dict[str, Any],
    idea_key: str,
    *,
    force_revisit_reason: str = "",
) -> dict[str, Any] | None:
    item = find_idea(payload, idea_key)
    if item is None:
        return None
    status = item.get("status")
    if status == "blocked":
        raise RuntimeError(
            f"Idea '{idea_key}' is blocked. {item['summary']} "
            f"Revisit only if: {item['revisit_only_if']}"
        )
    if status == "paused" and not force_revisit_reason.strip():
        raise RuntimeError(
            f"Idea '{idea_key}' is paused. Provide --force-revisit-reason to override. "
            f"Revisit only if: {item['revisit_only_if']}"
        )
    return item
