import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import FakeLLM, FakeOA
from bookworm.agent.tools import AgentContext, t_add
from bookworm.cli import app
from bookworm.dedup import deduplicate, match_rule
from bookworm.evaluate import recall_report
from bookworm.models import Corpus, Edge, EdgeRelation, Paper, Relevance, ResearchIntent
from bookworm.obsidian import export_vault
from bookworm.pipeline.build import BuildParams, build_corpus
from bookworm.pipeline.gaps import top_missing_refs
from bookworm.pipeline.relations import build_edges
from bookworm.store import CorpusStore
from bookworm.trace import Tracer


@pytest.fixture
def records():
    data = json.loads((Path(__file__).parent / 'fixtures/version_records.json').read_text())
    return {p['id']: Paper.model_validate(p) for p in data['papers']}


def pair(records):
    return records['W910001'], records['W910002']


def test_real_versions_and_same_title_reviews(records):
    store = CorpusStore(Corpus(papers=list(records.values())))
    report = deduplicate(store)
    assert report['last_pass'] == {'before': 10, 'after': 7, 'merged': 3}
    assert {m['removed_id'] for m in report['merges']} == {
        'W910002', 'W910004', 'WOS:000445498700005'}
    # Same title, different authors (book/review); distant reprint with different DOI.
    assert all(store.get(pid).id == pid for pid in ['W2113556984', 'W1563830325', 'W4237302105', 'W2160240057'])
    assert len(report['review_candidates']) >= 2


def test_order_independence_raw_versions_seed_and_roundtrip(records, tmp_path):
    pub, pre = pair(records)
    pre.is_seed, pre.origin, pre.relevance = True, 'seed', Relevance(score=10, reason='seed')
    expected = [p.model_dump() for p in sorted([pub, pre], key=lambda p: p.id)]
    results = []
    for papers in ([pre, pub], [pub, pre]):
        store = CorpusStore(Corpus(papers=[p.model_copy(deep=True) for p in papers]))
        deduplicate(store)
        main = store.get(pre.id)
        assert main.id == pub.id and main.doi == pub.doi and main.year == pub.year
        assert main.is_seed and main.origin == 'seed' and main.relevance.score == 10
        assert [v.model_dump() for v in main.versions] == [{k: v for k, v in p.items() if k != 'versions'} for p in expected]
        store.save(tmp_path / 'corpus.json')
        loaded = CorpusStore.load(tmp_path / 'corpus.json')
        assert loaded.find_by_doi(pre.doi).id == pub.id
        assert loaded.get(pre.id).id == pub.id
        deduplicate(loaded)
        assert len(loaded.corpus.meta.deduplication['merges']) == 1
        assert loaded.corpus.meta.deduplication['last_pass']['merged'] == 0
        assert loaded.finalize().stats['n_source_records'] == 2
        results.append(loaded.papers[0].model_dump())
    assert results[0] == results[1]


@pytest.mark.parametrize('change', [
    {'authors': []}, {'authors': ['Another Person', 'Other Author']}, {'year': None}, {'year': 1990},
    {'doi': '10.5555/other-journal', 'venue': 'A journal'},
    {'title': 'Correction: A multi-level quantitative analysis method on the scale, shape and quantity of rockeries'},
])
def test_ambiguous_or_conflicting_metadata_not_merged(records, change):
    pub, pre = pair(records)
    pre = pre.model_copy(update=change)
    assert match_rule(pub, pre) is None


def test_same_title_without_authors_is_not_enough():
    a = Paper(id='W1', title='A long title shared by different works', year=2020)
    b = a.model_copy(update={'id': 'W2'})
    assert match_rule(a, b) is None


def test_transitive_chain_does_not_merge():
    kwargs = dict(title='A long title shared by different works', authors=['Ann Author'], year=2020)
    a = Paper(id='W1', doi='10.1/one', **kwargs)
    b = Paper(id='W2', **kwargs)
    c = Paper(id='W3', doi='10.1/three', **kwargs)
    store = CorpusStore(Corpus(papers=[a, b, c]))
    assert match_rule(a, b) and match_rule(b, c) and not match_rule(a, c)
    assert deduplicate(store)['last_pass']['merged'] == 0
    assert len(store) == 3


def test_citation_aliases_no_self_loops_no_lost_labels(records, tmp_path):
    pub, pre = pair(records)
    pre.title = pub.title
    pub.referenced_works = [pre.id, 'W9']
    pre.referenced_works = ['W9']
    citing = Paper(id='W8', referenced_works=[pre.id, pub.id])
    target = Paper(id='W9')
    relation = EdgeRelation(label='uses_method', rationale='original')
    store = CorpusStore(Corpus(papers=[pub, pre, citing, target], edges=[
        Edge(source='W8', target=pre.id, relation=relation), Edge(source=pub.id, target=pre.id)]))
    deduplicate(store)
    store.set_edges(build_edges(store))
    assert {(e.source, e.target) for e in store.corpus.edges} == {('W8', pub.id), (pub.id, 'W9')}
    edge = next(e for e in store.corpus.edges if e.source == 'W8')
    assert edge.relation == relation
    assert edge.provenance == 'openalex_version_mapped'
    assert set(edge.record_pairs) == {('W8', pre.id), ('W8', pub.id)}
    assert pub.external_refs_count == 0
    assert top_missing_refs([citing], store, min_count=1) == []
    store.save(tmp_path / 'c.json')
    loaded = CorpusStore.load(tmp_path / 'c.json')
    assert build_edges(loaded) == store.corpus.edges
    loaded.remove(pre.id)
    assert loaded.get(pub.id) is None and loaded.find_by_doi(pre.doi) is None
    assert not loaded.corpus.edges


def test_recall_counts_aliases_once(records):
    pub, pre = pair(records)
    store = CorpusStore(Corpus(papers=[pub, pre]))
    deduplicate(store)
    report = recall_report(store.finalize(), [Paper(id='SURVEY', referenced_works=[pre.id, pub.id])])
    assert report.gold_size == report.found == report.corpus_size == 1
    assert report.recall == report.gold_share_of_corpus == 1


def test_agent_version_merge_at_cap_preserves_seen_and_seed(records):
    pub, pre = pair(records)
    pre.is_seed = True
    pre.relevance = Relevance(score=10)
    store = CorpusStore(Corpus(papers=[pre]))
    ctx = AgentContext(store, FakeLLM(), FakeOA(), None, ResearchIntent(topic='rockery'), Tracer(), max_papers=1)
    ctx.seen.add(pub)
    before = pub.model_dump()
    result = t_add(ctx, {'ids': [pub.id]})
    assert result['added'] == [pub.id] and len(store) == 1
    assert store.get(pre.id).is_seed and ctx.n_added == 0
    assert pub.model_dump() == before
    other = Paper(id='W999', title='Different study', relevance=Relevance(score=9))
    ctx.seen.add(other)
    assert t_add(ctx, {'ids': [other.id]})['rejected_corpus_full'] == [other.id]


def test_build_filters_versions_before_size_cap(records, tmp_path):
    pub, pre = pair(records)
    pub.title = pre.title = 'Graph neural networks for quantitative analysis of Chinese classical garden rockeries'
    other = Paper(id='W999', title='Graph models for other molecules', doi='10.1/other', authors=['Other'], year=2024)
    oa = FakeOA({p.id: p for p in [pub, pre, other]})
    params = BuildParams(prompt='graph', source='openalex', hops=0, gap_fill=False, max_papers=2)
    store = build_corpus(params, FakeLLM(), oa, None, Tracer(), tmp_path / 'c.json', log=lambda _: None,
                         frozen_intent=ResearchIntent(topic='graph', keywords_en=['graph']))
    assert {p.id for p in store.papers} == {pub.id, other.id}
    assert store.corpus.meta.prisma.included == 2
    assert store.corpus.meta.deduplication['last_pass']['merged'] == 1


def test_cli_preserves_source_and_exports_versions(records, tmp_path):
    pub, pre = pair(records)
    source = tmp_path / 'input.json'
    source.write_text(Corpus(papers=[pub, pre]).model_dump_json())
    before = source.read_bytes()
    out = tmp_path / 'deduplicated'
    runner = CliRunner()
    result = runner.invoke(app, ['deduplicate', str(source), '--out', str(out), '--no-obsidian'])
    assert result.exit_code == 0, result.output
    assert source.read_bytes() == before
    assert all((out / name).exists() for name in ['corpus.json', 'deduplication.json', 'corpus.bib', 'corpus.ris', 'corpus.csv'])
    corpus = CorpusStore.load(out / 'corpus.json').finalize()
    export_vault(corpus, out / 'obsidian')
    notes = list((out / 'obsidian' / 'papers').glob('*.md'))
    assert len(notes) == 1 and pre.doi in notes[0].read_text() and '## 文献版本' in notes[0].read_text()
    assert runner.invoke(app, ['deduplicate', str(source), '--out', str(out)]).exit_code != 0


def test_similar_titles_with_different_subject_are_not_merged(records):
    pub, pre = pair(records)
    pre.title = pub.title.replace('Example Site', 'Different Site')
    pre.referenced_works = list(pub.referenced_works)
    assert match_rule(pub, pre) is None


def test_retraction_and_notes_survive(records):
    pub, pre = pair(records)
    pre.retracted = True
    pre.notes = ['Preprint annotation']
    pub.notes = ['Journal annotation']
    store = CorpusStore(Corpus(papers=[pub, pre]))
    deduplicate(store)
    assert store.papers[0].retracted
    assert store.papers[0].notes == ['Journal annotation', 'Preprint annotation']
    assert any(v.retracted for v in store.papers[0].versions)


def test_identifier_upgrade_preserves_incoming_edges_after_reload(tmp_path):
    first = Paper(id='WOS:one', doi='10.1/one', referenced_works=['W3'])
    oa = Paper(id='W1', doi='10.1/one', referenced_works=['W4'])
    citing = Paper(id='W2', referenced_works=['WOS:one'])
    store = CorpusStore(Corpus(papers=[first, oa, citing], edges=[Edge(source='W2', target='WOS:one')]))
    assert len(store) == 2
    assert store.get('WOS:one').id == 'W1'
    assert store.corpus.edges[0].target == 'W1'
    assert store.get('W1').referenced_works == ['W3', 'W4']
    store.save(tmp_path / 'c.json')
    assert CorpusStore.load(tmp_path / 'c.json').get('WOS:one').id == 'W1'
