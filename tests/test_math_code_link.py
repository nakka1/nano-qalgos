"""The math <-> code link must never drift. Run: pytest -q"""
import os
import re

import pytest

import nanoqalgos_cli
import nanoqalgos_math as m

ROOT = os.path.join(os.path.dirname(__file__), "..")
base = lambda tag: re.sub(r"[a-z]$", "", tag)  # noqa: E731  "3.2a" -> "3.2"


@pytest.mark.parametrize("eq", m.EQS, ids=lambda e: e.id)
@pytest.mark.parametrize("seed", range(5))
def test_formula_matches_code(eq, seed):
    r = m.run(eq, seed)
    assert r["ok"], r


def test_every_tag_in_the_code_has_an_equation():
    orphans = [t for t in m.all_tags() if base(t) not in m.BY_ID]
    assert not orphans, f"tags in nanoqalgos.py with no equation: {orphans}"


@pytest.mark.parametrize("eq", m.EQS, ids=lambda e: e.id)
def test_every_term_points_at_lines_inside_its_function(eq):
    in_func = {t for _, _, tags in m.source(eq) for t in tags}
    everywhere = {t for t in m.all_tags() if base(t) == eq.id}
    assert set(eq.terms) == in_func == everywhere, (eq.terms.keys(), in_func, everywhere)


def test_every_term_appears_in_the_latex():
    for eq in m.EQS:
        for tag in eq.terms:
            assert "t" + tag.replace(".", "_") in eq.tex, (eq.id, tag)


def test_math_md_is_up_to_date():
    """Regenerate MATH.md; everything but the numeric result lines must match the committed file."""
    strip = lambda text: [ln for ln in text.splitlines() if not ln.startswith(("✅", "❌"))]  # noqa: E731
    committed = open(os.path.join(ROOT, "MATH.md"), encoding="utf-8").read()
    assert strip(committed) == strip(m.markdown()), "run: nanoqalgos verify --markdown > MATH.md"


def test_verify_cli(capsys):
    with pytest.raises(SystemExit) as done:
        nanoqalgos_cli.main(["verify"])
    assert done.value.code == 0
    assert "29/29" in capsys.readouterr().out
    nanoqalgos_cli.main(["verify", "3.2"])
    out = capsys.readouterr().out
    assert "def diffusion" in out and "max |formula − code|" in out


def test_readme_verify_snippet_is_live_output(capsys, monkeypatch):
    monkeypatch.setattr(nanoqalgos_cli, "COLOR", False)
    nanoqalgos_cli.main(["verify", "3.2"])
    live = capsys.readouterr().out.rstrip().splitlines()
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    snippet = readme.split("$ nanoqalgos verify 3.2\n")[1].split("```")[0].rstrip().splitlines()
    assert snippet == live[:len(snippet)]


def test_readme_section_links_point_at_the_section_headers():
    src = open(os.path.join(ROOT, "nanoqalgos.py"), encoding="utf-8").read().splitlines()
    headers = [i for i, ln in enumerate(src) if ln.startswith("# Part ")]  # 0-based index = 1-based line - 1
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    anchors = [int(a) for a in re.findall(r"\(nanoqalgos\.py#L(\d+)\) \| \d+–\d+ \|", readme)]
    assert anchors == headers  # each link lands on the "# =====" line just above "# Part N"
