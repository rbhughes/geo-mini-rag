import warnings

import pytest

from geo_mini_rag import settings
from geo_mini_rag.ep.las import LasHandler
from geo_mini_rag.rag.store import metadata_rows
from geo_mini_rag.rag.trace import OFF

LAS = """~VERSION INFORMATION
 VERS.                  2.0:   CWLS LOG ASCII STANDARD -VERSION 2.0
 WRAP.                   NO:   ONE LINE PER DEPTH STEP
~WELL INFORMATION
 STRT .F        30.0                    :START DEPTH
 STOP .F       570.0                    :STOP DEPTH
 STEP .F         0.5                    :STEP LENGTH
 NULL .      -999.25                     :NO VALUE
 WELL .     NPR #3 #13SX11-11            :WELL
 FLD  .     TEAPOT                       :FIELD
 CNTY .     NATRONA                      :COUNTY
 STAT .     WYOMING                      :STATE
 SRVC .     Schlumberger                 :SERVICE COMPANY
 DATE .     27-FEB-1977                  :LOGDATE
 API  .     490251029400                 :API NUMBER
~CURVE INFORMATION
 DEPT .F                                 :
 GRD  .GAPI                              :GAMMA RAY FROM DENSITY LOG
 RHOB .G/C3                              :BULK DENSITY
~ASCII
   30.0    12.0   2.45
"""


@pytest.fixture
def parsed(tmp_path):
    path = tmp_path / "49025102940000.las"
    path.write_text(LAS)
    warnings.filterwarnings("ignore")
    return LasHandler().parse(path, settings.load_rag_config(), OFF)


def test_each_curve_is_its_own_fact(parsed):
    rows = metadata_rows("d", parsed.metadata)
    curves = sorted(v for _, k, v, _ in rows if k == "curve")
    assert curves == ["DEPT", "GRD", "RHOB"], "one row per curve, not one joined string"
    descriptions = [v for _, k, v, _ in rows if k == "curve_description"]
    assert "GAMMA RAY FROM DENSITY LOG" in descriptions


def test_depths_are_numbers(parsed):
    numeric = {k: n for _, k, _, n in metadata_rows("d", parsed.metadata) if n is not None}
    assert numeric["depth_min"] == 30.0
    assert numeric["depth_max"] == 570.0
    assert numeric["depth_step"] == 0.5
    assert parsed.metadata["depth_units"] == "F"


def test_log_year_comes_out_of_the_date(parsed):
    assert parsed.metadata["log_year"] == 1977


def test_header_card_still_reads_for_a_person(parsed):
    header = parsed.segments[0][1]
    assert "Well: NPR #3 #13SX11-11" in header
    assert "API number: 490251029400" in header
    assert "Logged interval: 30 to 570 F, step 0.5" in header
