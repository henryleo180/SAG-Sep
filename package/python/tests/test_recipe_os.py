from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sagsep.config import PRESET_SIZES, PRESETS, from_toml, preset

_TOP = """\
policy_decode_mode = "sweep_viol"
policy_sweep_coexist = {coexist}
policy_device = "cpu"
mbp_onion_mode = "mean"
"""

_TTS_FULL = """\
[tts]
enabled = true
samples_per_tau = 0
threshold_eps = []
pl_enabled = false
jitter_enabled = false
max_prefix = 90
lp_grow_seeds = 16
lp_grow_native_rules = true
compress_mode = "full"
fcapfix_enabled = true
fcapfix_seed_limit = 32
fcapfix_seed_mode = "model"
old_repair_enabled = true
old_repair_bank_size = 100000
adddrop_enabled = true
adddrop_seed_old_bank = true
adddrop_seed_limit = 0
adddrop_seed_sources = []
score_mode = "efficacy"
top_b = 120
top_l = 24
max_flips = 40
ls_swap = true
max_size_frac = 1.0
seed = 19
rng_mode = "derived"
flow_enabled = true
source_quotas = {{ fcapfix = 4, old_repair = 40, lp_grow = 80, adddrop = 40 }}
per_vehicle_quota = 200
global_quota = 200
near_duplicate_hamming = 3
diversity_weight = 0.15
vehicle_tighten_start = 8
vehicle_tighten_power = 0.75
vehicle_tighten_min_scale = 0.4
"""

_INSTANCES = """
[[instances]]
name = "random-0-X-n50"
vehicles = 8
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cfg.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("size", PRESET_SIZES)
def test_os_preset_differs_from_the_released_one_only_in_tts_enabled(size: str) -> None:
    full = preset(size)
    assert full is PRESETS[size]
    assert full.tts.enabled is True
    os_cfg = preset(size, recipe="os")
    assert os_cfg.tts.enabled is False
    assert os_cfg == replace(full, tts=replace(full.tts, enabled=False))


def test_unknown_recipe_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown recipe"):
        preset("n50", recipe="oxps")


def test_toml_tts_disabled_needs_no_other_tts_key(tmp_path: Path) -> None:
    text = _TOP.format(coexist="false") + "\n[tts]\nenabled = false\n" + _INSTANCES
    cfg = from_toml(_write(tmp_path, text))
    assert cfg.tts.enabled is False


def test_toml_full_recipe_still_parses_and_keeps_tts_on(tmp_path: Path) -> None:
    text = _TOP.format(coexist="true") + "\n" + _TTS_FULL.format() + _INSTANCES
    cfg = from_toml(_write(tmp_path, text))
    assert cfg.tts.enabled is True


def test_toml_coexist_false_is_refused_while_tts_is_on(tmp_path: Path) -> None:
    text = _TOP.format(coexist="false") + "\n" + _TTS_FULL.format() + _INSTANCES
    with pytest.raises(ValueError, match="policy_sweep_coexist"):
        from_toml(_write(tmp_path, text))


def test_toml_tts_on_still_requires_every_key(tmp_path: Path) -> None:
    text = _TOP.format(coexist="true") + "\n[tts]\nenabled = true\n" + _INSTANCES
    with pytest.raises(ValueError, match="missing required key"):
        from_toml(_write(tmp_path, text))


def test_toml_without_enabled_is_still_the_required_key_error(tmp_path: Path) -> None:
    text = _TOP.format(coexist="true") + "\n[tts]\nseed = 19\n" + _INSTANCES
    with pytest.raises(ValueError, match="'enabled'"):
        from_toml(_write(tmp_path, text))


def test_toml_legacy_pipeline_is_still_refused(tmp_path: Path) -> None:
    text = (
        'decode_pipeline = "legacy"\n'
        + _TOP.format(coexist="false")
        + "\n[tts]\nenabled = false\n"
        + _INSTANCES
    )
    with pytest.raises(ValueError, match="decode_pipeline"):
        from_toml(_write(tmp_path, text))
