"""Service layer and the --json contract a calling agent relies on: one JSON object on stdout, typed errors."""

import json
from pathlib import Path

from typer.testing import CliRunner

from conftest import FakeLLM, FakeOA
from bookworm import cli, service
from bookworm.config import Settings
from bookworm.pipeline.build import BuildParams
from bookworm.runtime import RetrievalRun

runner = CliRunner()


def settings(tmp_path, **extra):
    return Settings(**{'_env_file': None, 'deepseek_api_key': 'x', 'scib_corpora_dir': tmp_path / 'runs',
                       'scib_cache_dir': tmp_path / 'cache', **extra})


def fake(monkeypatch, s):
    monkeypatch.setattr(service, 'get_settings', lambda: s)
    monkeypatch.setattr(service, 'make_clients', lambda settings, tracer, source, run=None: (FakeLLM(), FakeOA(), None))


def invoke(*args):
    result = runner.invoke(cli.app, ['--json', *args])
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1, result.output  # progress must not leak onto stdout
    return result, json.loads(lines[0])


def test_build_json_is_a_single_result_object_with_compact_summary(tmp_path, monkeypatch):
    s = settings(tmp_path); fake(monkeypatch, s)
    result, data = invoke('build', 'graph neural networks', '--source', 'openalex', '--no-obsidian', '--top', '3')
    assert result.exit_code == 0, result.output
    r = service.RunResult.model_validate(data)
    assert r.ok and r.complete and r.command == 'build'
    assert r.summary.n_papers > 0 and len(r.summary.top_papers) == 3
    assert r.summary.top_papers[0].corpus_cites >= r.summary.top_papers[-1].corpus_cites
    assert (tmp_path / 'runs').resolve() in [p.resolve() for p in Path(r.corpus).parents]
    assert r.cleanup is not None and r.usage is not None
    assert 'Parsing research intent' in result.stderr


def test_quiet_drops_progress(tmp_path, monkeypatch):
    s = settings(tmp_path); fake(monkeypatch, s)
    result = runner.invoke(cli.app, ['--json', '--quiet', 'build', 'graph', '--source', 'openalex', '--no-obsidian'])
    assert result.exit_code == 0 and result.stderr == ''


def test_python_api_build_matches_cli(tmp_path, monkeypatch):
    s = settings(tmp_path); fake(monkeypatch, s)
    logs = []
    r = service.build(BuildParams(prompt='graph', source='openalex', hops=0), obsidian=False, log=logs.append)
    assert r.summary.n_papers > 0 and logs


def test_unfinished_agent_reports_partial_result(tmp_path, monkeypatch):
    from bookworm.agent import loop
    s = settings(tmp_path); fake(monkeypatch, s)
    monkeypatch.setattr(loop, 'run_agent', lambda *args, **kwargs: None)
    result, data = invoke('agent', 'graph', '--no-obsidian')
    assert result.exit_code == 3, result.output
    assert data['complete'] is False and data['cleanup'] is None
    assert any('partial result' in w for w in data['warnings'])


def test_errors_are_typed_json(tmp_path, monkeypatch):
    s = settings(tmp_path, deepseek_api_key='')
    monkeypatch.setattr(service, 'get_settings', lambda: s)
    result, data = invoke('build', 'graph', '--source', 'openalex')
    assert result.exit_code == 2 and data == {'ok': False, 'command': 'build', 'protocol_version': '1', 'error': {
        'code': 'config', 'message': data['error']['message']}}
    assert 'DEEPSEEK_API_KEY' in data['error']['message']
    assert not s.scib_corpora_dir.exists()  # no empty run directory left behind

    s = settings(tmp_path); fake(monkeypatch, s)
    with RetrievalRun(s, s.scib_corpora_dir / 'holder', kind='build'):
        result, data = invoke('build', 'graph', '--source', 'openalex')
    assert result.exit_code == 75 and data['error']['code'] == 'busy'

    def boom(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(service, 'write_obsidian', boom)
    result, data = invoke('build', 'graph', '--source', 'openalex')
    assert result.exit_code == 3 and data['error']['code'] == 'internal'
    assert data['stop_reason'] == 'output_failed' and Path(data['corpus']).exists()


def test_read_only_commands(tmp_path, small_corpus):
    path = tmp_path / 'corpus.json'
    path.write_text(small_corpus.model_dump_json())
    result, data = invoke('stats', str(path), '--top', '2')
    assert result.exit_code == 0 and data['summary']['n_papers'] == 7 and len(data['summary']['top_papers']) == 2
    result, data = invoke('export', str(path), '-f', 'ris')
    assert result.exit_code == 0 and data['n_papers'] == 7 and (tmp_path / 'corpus.ris').exists()
    result, data = invoke('traverse', str(path), '--start', 'W5', '--direction', 'refs')
    assert result.exit_code == 0 and data['visits'][0]['id'] == 'W5'
    result, data = invoke('traverse', str(path), '--start', 'W404')
    assert result.exit_code == 2 and data['error']['code'] == 'invalid_input'


def test_schema_lists_every_result():
    result = runner.invoke(cli.app, ['schema'])
    schemas = json.loads(result.stdout)
    assert set(schemas['results']) == set(service.RESULT_MODELS) and 'error' in schemas
