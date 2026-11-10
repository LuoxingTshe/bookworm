import csv
import io

from conftest import UNIVERSE
from bookworm.evaluate import recall_report
from bookworm.export import to_bibtex, to_csv, to_ris
from bookworm.models import Paper, Relevance
from bookworm.store import CorpusStore


def test_recall_against_survey(small_corpus):
    survey = UNIVERSE["W5"].model_copy(deep=True)  # references W1, W2, W4, W9; W5 is also the corpus seed
    c = CorpusStore(small_corpus).finalize()
    rep = recall_report(c, [survey], ks=(2,))
    assert rep.gold_size == 4 and rep.found == 3 and rep.recall == 0.75
    assert rep.missed == ["W9"] and rep.corpus_size == 6  # the survey itself is not counted
    assert rep.gold_share_of_corpus == 0.5 and 2 in rep.recall_at
    assert any("seed" in w for w in rep.warnings)
    other = Paper(id="W100", title="Another survey", referenced_works=["W7", "W8", "W404"])
    rep = recall_report(c, [other])
    assert rep.found == 2 and rep.missed == ["W404"] and not rep.warnings


def _papers():
    a = Paper(id="W1", doi="10.1/a", title="Graphs & molecules: 50% better_models", authors=["José Müller", "B Lee"],
              year=2020, venue="J. Chem", abstract="Line one.\nLine two.", keywords=["gnn"],
              relevance=Relevance(score=8, reason="core"), is_seed=True)
    b = Paper(id="W2", title="Graphs everywhere", authors=["José Müller"], year=2020, retracted=True)
    c = Paper(id="WOS:123", title="Graphs everywhere again", authors=["José Müller"], year=2020)
    return [a, b, c]


def test_bibtex():
    bib = to_bibtex(_papers())
    assert bib.count("@article{") == 1 and bib.count("@misc{") == 2
    assert "@article{muller2020graphs," in bib and "@misc{muller2020graphsa," in bib  # ascii key, deduplicated
    assert r"Graphs \& molecules: 50\% better\_models" in bib
    assert "author = {José Müller and B Lee}" in bib and "doi = {10.1/a}" in bib
    assert "RETRACTED" in bib and "url = {https://openalex.org/W2}" in bib


def test_ris():
    ris = to_ris(_papers())
    assert ris.endswith("ER  - \n")
    recs = ris.rstrip("\n").split("\n\n")
    assert len(recs) == 3 and all(r.startswith("TY  - ") and r.endswith("\nER  - ") for r in recs)
    assert "AU  - José Müller\nAU  - B Lee" in recs[0] and "AB  - Line one. Line two." in recs[0]
    assert "TY  - GEN" in recs[1] and "UR  - " not in recs[2] and "N1  - WOS:123" in recs[2]


def test_csv():
    rows = list(csv.DictReader(io.StringIO(to_csv(_papers()))))
    assert len(rows) == 3 and rows[0]["relevance"] == "8.0" and rows[0]["is_seed"] == "True"
    assert rows[1]["retracted"] == "True" and rows[1]["relevance"] == ""
