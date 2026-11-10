import json

import pytest
from typer.testing import CliRunner

from conftest import FakeLLM, FakeOA
from bookworm import cli, service
from bookworm.config import Settings
from bookworm.models import Corpus
from bookworm.runtime import MARKER, RetrievalRun
from bookworm.sources.cache import Cache


def settings(tmp_path):
    return Settings(_env_file=None, deepseek_api_key='x', scib_corpora_dir=tmp_path / 'runs', scib_cache_dir=tmp_path / 'cache',
                    scib_keep_history=False, scib_keep_cache=False)


def corpus(directory):
    path = directory / 'corpus.json'
    path.write_text(Corpus().model_dump_json())
    return path


def finished(s, name, **options):
    folder = s.scib_corpora_dir / name
    with RetrievalRun(s, folder, kind='build', **options) as run:
        run.complete([corpus(folder)])
    return folder


def test_success_preserves_all_results_and_can_clear_cache(tmp_path):
    s = settings(tmp_path)
    old = finished(s, 'old')
    unknown = s.scib_corpora_dir / 'manual-notes'
    unknown.mkdir(); (unknown / 'note.txt').write_text('Keep me')
    outside = tmp_path / 'outside'; outside.mkdir(); corpus(outside)
    (s.scib_corpora_dir / 'linked').symlink_to(outside, target_is_directory=True)
    cache = Cache(s.scib_cache_dir / 'http.sqlite'); cache.set('key', 'x' * 1000000); cache.close()
    size = (s.scib_cache_dir / 'http.sqlite').stat().st_size
    new = finished(s, 'new')
    assert old.exists() and new.exists() and unknown.exists() and outside.exists()
    cache = Cache(s.scib_cache_dir / 'http.sqlite')
    assert cache.get('key') is None
    assert cache.path.stat().st_size < size
    cache.close()
    report = json.loads((new / 'cleanup.json').read_text())
    assert report['removed_runs'] == [] and not report['errors']


@pytest.mark.parametrize('error', [RuntimeError('export failed'), KeyboardInterrupt()])
def test_failure_and_interrupt_preserve_previous_result_and_cache(tmp_path, error):
    s = settings(tmp_path); old = finished(s, 'old')
    target = s.scib_corpora_dir / 'failed'
    with pytest.raises(type(error)):
        with RetrievalRun(s, target, kind='build') as run:
            cache = run.cache(Cache(s.scib_cache_dir / 'http.sqlite'))
            cache.set('key', 'retained')
            corpus(target)  # even a saved corpus must not trigger cleanup before export succeeds
            raise error
    assert old.exists()
    assert json.loads((target / MARKER).read_text())['status'] in {'incomplete', 'interrupted'}
    cache = Cache(s.scib_cache_dir / 'http.sqlite'); assert cache.get('key') == 'retained'; cache.close()
    finished(s, 'next')
    assert target.exists()  # keep failure diagnostics


def test_validation_and_keep_options(tmp_path):
    s = settings(tmp_path); old = finished(s, 'old')
    folder = s.scib_corpora_dir / 'kept'
    with RetrievalRun(s, folder, kind='build', keep_history=True, keep_cache=True) as run:
        cache = run.cache(Cache(folder / 'http.sqlite')); cache.set('k', 'v')
        run.complete([corpus(folder)])
    assert old.exists()
    cache = Cache(folder / 'http.sqlite'); assert cache.get('k') == 'v'; cache.close()
    bad = s.scib_corpora_dir / 'bad'
    with pytest.raises(ValueError):
        with RetrievalRun(s, bad, kind='build') as run:
            (bad / 'corpus.json').write_text('not json')
            run.complete([bad / 'corpus.json'])
    assert old.exists() and folder.exists()


def test_custom_output_does_not_prune_managed_root_and_lock_blocks_concurrent_runs(tmp_path):
    s = settings(tmp_path); old = finished(s, 'old')
    with RetrievalRun(s, tmp_path / 'custom', kind='build') as run:
        with pytest.raises(RuntimeError, match='Another retrieval'):
            with RetrievalRun(s, s.scib_corpora_dir / 'concurrent', kind='build'):
                pytest.fail('must not enter')
        run.complete([corpus(tmp_path / 'custom')])
    assert old.exists()
    with pytest.raises(ValueError, match='empty directory'):
        with RetrievalRun(s, old, kind='build'):
            pytest.fail('must not overwrite a completed result')
    assert json.loads((old / MARKER).read_text())['status'] == 'completed'


def fake_clients(monkeypatch, s):
    monkeypatch.setattr(service, 'get_settings', lambda: s)
    def clients(settings, tracer, source, run=None):
        cache = Cache(s.scib_cache_dir / 'http.sqlite')
        cache.set('sentinel', 'cached')
        if run:
            run.cache(cache)
        return FakeLLM(), FakeOA(), None
    monkeypatch.setattr(service, 'make_clients', clients)


def test_build_cli_cleanup_waits_for_obsidian_and_keeps_complete_output(tmp_path, monkeypatch):
    s = settings(tmp_path); old = finished(s, 'old'); fake_clients(monkeypatch, s)
    runner = CliRunner(); out = s.scib_corpora_dir / 'new' / 'corpus.json'
    def export(*args, **kwargs):
        assert old.exists()
        (out.parent / 'obsidian').mkdir()
        (out.parent / 'obsidian' / '总览.md').write_text('complete export')
        return service.ObsidianInfo(dir=str(out.parent / 'obsidian'), overview='', vault_root='')
    monkeypatch.setattr(service, 'write_obsidian', export)
    result = runner.invoke(cli.app, ['build', 'graph', '--source', 'openalex', '--out', str(out)])
    assert result.exit_code == 0, result.output
    assert old.exists() and (out.parent / 'obsidian' / '总览.md').exists()
    cache = Cache(s.scib_cache_dir / 'http.sqlite'); assert cache.get('sentinel') is None; cache.close()
    def failing_export(*args, **kwargs):
        raise OSError('export failed')
    monkeypatch.setattr(service, 'write_obsidian', failing_export)
    result = runner.invoke(cli.app, ['build', 'graph', '--source', 'openalex', '--out', str(s.scib_corpora_dir / 'bad' / 'corpus.json')])
    assert result.exit_code != 0 and out.exists()


def test_agent_continuation_uses_new_directory_and_unfinished_agent_does_not_prune(tmp_path, monkeypatch):
    from bookworm.agent import loop
    s = settings(tmp_path); old = finished(s, 'old'); fake_clients(monkeypatch, s)
    before = (old / 'corpus.json').read_bytes()
    runner = CliRunner()
    monkeypatch.setattr(loop, 'run_agent', lambda *args, **kwargs: None)
    result = runner.invoke(cli.app, ['agent', 'graph', '--corpus', str(old / 'corpus.json'), '--no-obsidian'])
    assert result.exit_code == 3, result.output
    assert (old / 'corpus.json').read_bytes() == before
    def finish(ctx, *args, **kwargs):
        ctx.finished = True
    monkeypatch.setattr(loop, 'run_agent', finish)
    result = runner.invoke(cli.app, ['agent', 'graph', '--corpus', str(old / 'corpus.json'), '--no-obsidian'])
    assert result.exit_code == 0, result.output
    assert old.exists()
    complete = [p for p in s.scib_corpora_dir.glob('*/'+MARKER) if json.loads(p.read_text())['status']=='completed']
    assert len(complete) == 2


def test_invalid_or_linked_ownership_marker_is_never_pruned(tmp_path):
    s = settings(tmp_path)
    root = s.scib_corpora_dir
    root.mkdir()
    invalid = root / 'invalid'; invalid.mkdir()
    (invalid / MARKER).write_text('[]')
    linked = root / 'linked-marker'; linked.mkdir()
    outside_marker = tmp_path / 'outside-marker.json'
    outside_marker.write_text(json.dumps({'owner': 'bookworm', 'status': 'completed'}))
    (linked / MARKER).symlink_to(outside_marker)
    finished(s, 'new')
    assert invalid.exists() and linked.exists()
