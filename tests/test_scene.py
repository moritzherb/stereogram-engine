from pathlib import Path

import pytest
import yaml

from engine.formats import LONG, SHORT
from engine.project import Project
from engine.scene import (
    GeneratedPattern,
    SceneError,
    contains_emoji,
    depth_hash,
    load_compilation,
    load_scene,
    render_hash,
    reveal_enabled,
)

REPO = Path(__file__).resolve().parents[1]


def write(project, name, data, folder="scenes"):
    p = project.root / folder / name
    p.write_text(yaml.safe_dump(data, allow_unicode=True))
    return p


def test_project_paths(tmp_project):
    assert tmp_project.music == tmp_project.root / "assets" / "music"
    assert tmp_project.font_path.name == "Montserrat-ExtraBold.ttf"


def test_minimal_scene_defaults(tmp_project):
    s = load_scene(write(tmp_project, "heart.yaml", {"id": "heart", "shape": "heart"}), tmp_project)
    assert s.duration == 30 and s.formats == ["short", "long"]
    assert s.text.top == "This is a stereogram"
    assert s.text.bottom == "What do you see?"
    assert s.text.end == "Comment what you see"
    assert isinstance(s.pattern, GeneratedPattern) and s.pattern.seed is not None
    assert s.pattern.generate == "swirl"  # busy patterns hide pattern seams best
    assert s.title == "Heart"
    assert s.music == "auto" and s.reveal == "auto"
    assert s.advanced.depth_strength == 0.33 and s.advanced.viewing == "parallel"
    assert s.platform == "tiktok" and s.text.bottom_y is None


def test_new_motions_and_settings(tmp_project):
    for motion in ("sway", "tilt", "rock", "breathe"):
        s = load_scene(write(tmp_project, f"{motion}.yaml", {"id": motion, "shape": "heart", "motion": motion,
                                                              "motion_settings": {"amount": 15, "seconds": 6}}),
                       tmp_project)
        assert s.motion_settings.amount == 15 and s.motion_settings.seconds == 6
    p = write(tmp_project, "bad.yaml", {"id": "bad", "shape": "heart", "motion_settings": {"seconds": 0}})
    with pytest.raises(SceneError, match="seconds"):
        load_scene(p, tmp_project)


def test_edge_softness_default_and_bounds(tmp_project):
    s = load_scene(write(tmp_project, "e.yaml", {"id": "e", "shape": "heart"}), tmp_project)
    assert s.advanced.edge_softness == 1.0 and s.advanced.lift == 0.35
    p = write(tmp_project, "f.yaml", {"id": "f", "shape": "heart", "advanced": {"edge_softness": 9}})
    with pytest.raises(SceneError, match="edge_softness"):
        load_scene(p, tmp_project)


def test_depth_hash_changes_with_motion_settings(tmp_project):
    a = load_scene(write(tmp_project, "a.yaml", {"id": "a", "shape": "heart", "motion": "sway"}), tmp_project)
    b = load_scene(write(tmp_project, "a.yaml", {"id": "a", "shape": "heart", "motion": "sway",
                                                  "motion_settings": {"amount": 30}}), tmp_project)
    assert depth_hash(a, SHORT) != depth_hash(b, SHORT)


def test_platform_and_bottom_y_validation(tmp_project):
    p = write(tmp_project, "p.yaml", {"id": "p", "shape": "heart", "platform": "myspace"})
    with pytest.raises(SceneError, match="tiktok"):
        load_scene(p, tmp_project)
    p = write(tmp_project, "q.yaml", {"id": "q", "shape": "heart", "text": {"bottom_y": 1.2}})
    with pytest.raises(SceneError, match="bottom_y"):
        load_scene(p, tmp_project)


def test_seed_stable_per_id(tmp_project):
    a = load_scene(write(tmp_project, "a.yaml", {"id": "dolphin-jump", "shape": "heart"}), tmp_project)
    b = load_scene(write(tmp_project, "b.yaml", {"id": "dolphin-jump", "shape": "sphere"}), tmp_project)
    assert a.pattern.seed == b.pattern.seed


def test_relative_paths_resolve_against_assets_with_spaces(tmp_project):
    pat = tmp_project.patterns / "my ocean pic.jpg"
    pat.write_bytes(b"x")
    model = tmp_project.models / "dolphin" / "dolphin model.glb"
    model.parent.mkdir()
    model.write_bytes(b"x")
    s = load_scene(write(tmp_project, "d.yaml", {
        "id": "dolphin", "model": "models/dolphin/dolphin model.glb", "pattern": "patterns/my ocean pic.jpg"}),
        tmp_project)
    assert s.pattern == pat and s.model == model


def test_emoji_rejected_with_field_name(tmp_project):
    p = write(tmp_project, "e.yaml", {"id": "e", "shape": "heart", "text": {"top": "Look \U0001F440"}})
    with pytest.raises(SceneError, match=r"text\.top: .*emoji"):
        load_scene(p, tmp_project)


def test_emoji_rejected_in_title(tmp_project):
    p = write(tmp_project, "e.yaml", {"id": "e", "shape": "heart", "title": "Wow ❤️"})
    with pytest.raises(SceneError, match=r"title: .*emoji"):
        load_scene(p, tmp_project)


def test_contains_emoji():
    assert contains_emoji("hi \U0001F600") and contains_emoji("❤️")
    assert not contains_emoji("What do you see?") and not contains_emoji("Größe: 3 / 15")


def test_exactly_one_source(tmp_project):
    p = write(tmp_project, "x.yaml", {"id": "x", "shape": "heart", "text3d": "HI"})
    with pytest.raises(SceneError, match="model, text3d or shape"):
        load_scene(p, tmp_project)
    p = write(tmp_project, "y.yaml", {"id": "y"})
    with pytest.raises(SceneError, match="model, text3d or shape"):
        load_scene(p, tmp_project)


def test_missing_model_file_message(tmp_project):
    p = write(tmp_project, "m.yaml", {"id": "m", "model": "models/nope.glb"})
    with pytest.raises(SceneError) as e:
        load_scene(p, tmp_project)
    msg = str(e.value)
    assert "m.yaml" in msg and "model" in msg and "nope.glb" in msg and "assets/models" in msg


def test_bad_id(tmp_project):
    p = write(tmp_project, "b.yaml", {"id": "Dolphin Jump", "shape": "heart"})
    with pytest.raises(SceneError, match="dolphin-jump"):
        load_scene(p, tmp_project)


def test_unknown_motion_lists_presets(tmp_project):
    p = write(tmp_project, "u.yaml", {"id": "u", "shape": "heart", "motion": "dance"})
    with pytest.raises(SceneError, match="spin, float, bob"):
        load_scene(p, tmp_project)


def test_unknown_generated_style(tmp_project):
    p = write(tmp_project, "g.yaml", {"id": "g", "shape": "heart", "pattern": {"generate": "plaid"}})
    with pytest.raises(SceneError, match="noise, swirl"):
        load_scene(p, tmp_project)


def test_yaml_syntax_error_is_scene_error(tmp_project):
    p = tmp_project.scenes / "broken.yaml"
    p.write_text("id: [unclosed\n")
    with pytest.raises(SceneError, match="broken.yaml"):
        load_scene(p, tmp_project)


def test_reveal_auto(tmp_project):
    s = load_scene(write(tmp_project, "r.yaml", {"id": "r", "shape": "heart"}), tmp_project)
    assert reveal_enabled(s, "short") is False and reveal_enabled(s, "long") is True
    s2 = load_scene(write(tmp_project, "r2.yaml", {"id": "r2", "shape": "heart", "reveal": "on"}), tmp_project)
    assert reveal_enabled(s2, "short") is True


def test_depth_hash_ignores_text_and_pattern(tmp_project):
    a = load_scene(write(tmp_project, "a.yaml", {"id": "a", "shape": "heart"}), tmp_project)
    b = load_scene(write(tmp_project, "a.yaml", {"id": "a", "shape": "heart", "text": {"top": "Hi"},
                                                  "pattern": {"generate": "cells"}}), tmp_project)
    assert depth_hash(a, SHORT) == depth_hash(b, SHORT)
    assert depth_hash(a, SHORT) != depth_hash(a, LONG)
    assert render_hash(a) != render_hash(b)


def test_depth_hash_changes_with_motion(tmp_project):
    a = load_scene(write(tmp_project, "a.yaml", {"id": "a", "shape": "heart", "motion": "spin"}), tmp_project)
    b = load_scene(write(tmp_project, "a.yaml", {"id": "a", "shape": "heart", "motion": "float"}), tmp_project)
    assert depth_hash(a, SHORT) != depth_hash(b, SHORT)


def test_compilation_defaults_and_validation(tmp_project):
    c = load_compilation(write(tmp_project, "c.yaml", {"id": "c", "title": "Hidden things", "scenes": ["a"]},
                               "compilations"), tmp_project)
    assert c.segment_seconds == 35 and c.scenes == ["a"]
    with pytest.raises(SceneError, match="scenes"):
        load_compilation(write(tmp_project, "d.yaml", {"id": "d", "title": "T", "scenes": []}, "compilations"),
                         tmp_project)


def test_examples_load():
    project = Project(REPO)
    files = sorted((REPO / "scenes").glob("*.yaml")) + sorted((REPO / "compilations").glob("*.yaml"))
    assert len(files) >= 3
    for f in (REPO / "scenes").glob("*.yaml"):
        load_scene(f, project)
    for f in (REPO / "compilations").glob("*.yaml"):
        load_compilation(f, project)


def test_pattern_drift_option(tmp_project):
    s = load_scene(write(tmp_project, "d.yaml", {"id": "d", "shape": "heart"}), tmp_project)
    assert s.advanced.pattern_drift == 0
    s = load_scene(write(tmp_project, "d.yaml", {"id": "d", "shape": "heart", "advanced": {"pattern_drift": 15}}),
                   tmp_project)
    assert s.advanced.pattern_drift == 15
