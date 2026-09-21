import pytest
import yaml

from geo_mini_rag import settings
from geo_mini_rag.appraisal import pass2

POLICY = yaml.safe_load((settings.CONFIG_DIR / "policy.yaml").read_text())
PATTERNS = pass2.compile_patterns(POLICY)


def norm(stem: str) -> str:
    return pass2.normalize_stem(stem, PATTERNS)


@pytest.mark.parametrize(("stem", "expected"), [
    ("report (1)", "report"),
    ("report (copy 2)", "report"),
    ("report - copy 2", "report"),
    ("report copy", "report"),
    ("Copy of report", "report"),
    ("copy_of_GOLDEN", "golden"),
    ("copy 2 report", "report"),
])
def test_explicit_copy_markers_are_stripped(stem, expected):
    assert norm(stem) == expected


@pytest.mark.parametrize("stem", [
    "49005256730000",                    # API number
    "us49025107010000_0_00133h490614",
    "N-10-35_2000",                      # map tile
    "GOLDEN",
    "2WCFaults",
    "report_v2",                         # version markers are NOT stripped any more
    "report rev 3",
    "report_final",
    "survey_2017-07-13",                 # a timestamp is part of the identity
    "dailyreport_7-13-2017",
    "Copy Creek 1-2H",                   # a well name, not a copy marker
    "Copyright notice",
    "49-025-10294",                      # dashed API number
    "100-04-11-082-04W5-00",             # Canadian DLS UWI
    "200-A-016-K-094-A-11",              # Canadian NTS UWI
    "15_9-F-11 B",                       # Norwegian well
])
def test_names_are_left_alone_unless_an_explicit_copy_marker(stem):
    assert norm(stem) == stem.lower()


def rows(*paths):
    return [{"path": p, "verdict": "PENDING"} for p in paths]


def keys(*paths):
    return [pass2.family_key(p, POLICY, PATTERNS) for p in paths]


def test_same_directory_same_extension_is_one_family():
    a, b, c = keys("wells/report.doc", "wells/Copy of report.doc", "wells/report (1).doc")
    assert a == b == c


def test_different_directory_is_not_a_family_by_default():
    a, b = keys("wells/report.doc", "backup/Copy of report.doc")
    assert a != b, "scope: directory keeps distant copies apart"


def test_different_extension_is_not_a_family_by_default():
    a, b = keys("wells/report.doc", "wells/report.pdf")
    assert a != b, "match_extension: true"


def test_tree_scope_crosses_directories():
    policy = {**POLICY, "version_family": {**POLICY["version_family"], "scope": "tree"}}
    a = pass2.family_key("wells/report.doc", policy, PATTERNS)
    b = pass2.family_key("backup/Copy of report.doc", policy, PATTERNS)
    assert a == b
