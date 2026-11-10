"""Cross-entry-point contracts and failure recovery; no external service is called."""
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from bookworm import cli, service
from bookworm.agent.loop import run_agent
from bookworm.agent.tools import AgentContext, dispatch
from bookworm.config import Settings, configure, get_settings
from bookworm.errors import BudgetError
from bookworm.models import Corpus, ResearchIntent, Relevance
from bookworm.requests import ResearchRequest
from bookworm.sources.cache import Cache
from bookworm.sources.openalex import OpenAlexClient
from bookworm.store import CorpusStore
from bookworm.trace import Tracer
from conftest import FakeLLM, FakeOA, tool_msg

runner = CliRunner()


@pytest.fixture
def local(monkeypatch, tmp_path):
    s = Settings(_env_file=None, deepseek_api_key='fake', wos_api_key='',
                 scib_cache_dir=tmp_path/'cache', scib_corpora_dir=tmp_path/'runs')
    monkeypatch.setattr(service, 'get_settings', lambda: s)
    return s


def install_fakes(monkeypatch, llm=None):
    llm = llm or FakeLLM()
    monkeypatch.setattr(service, 'make_clients', lambda settings, tracer, source, run=None: (llm, FakeOA(), None))
    return llm


def output(args, *, code=0, input=None):
    result = runner.invoke(cli.app, args, input=input)
    assert result.exit_code == code, result.output
    assert len(result.stdout.strip().splitlines()) == 1, result.output
    return json.loads(result.stdout)


@pytest.mark.parametrize('args', [
    ['--json', 'bogus'], ['--json', 'run'], ['--json', 'run', 'x', '--bad'],
    ['--json', 'run', 'x', '--max-papers', '-1'], ['--json', 'run', 'x', '--years', '2026-2020'],
    ['--json', 'build', 'x', '--source', 'invalid'], ['--json', 'stats', '/nonexistent/bookworm.json'],
    ['--json', 'run', 'x', '--mode', 'invalid'], ['--json', 'run', 'x', '--years', '0000'],
])
def test_every_invalid_cli_request_is_one_json(args):
    data = output(args, code=2)
    assert data['ok'] is False and data['error']['code'] == 'invalid_input'


@pytest.mark.parametrize('raw', ['{', '[]', '{"prompt":"x","max_steps":0}',
                                  '{"prompt":"x","unknown":1}', '{"prompt":"   "}',
                                  '{"prompt":"x","years":[2026,2020]}'])
def test_task_rejects_bad_input_before_creating_a_run(local, raw):
    data = output(['task'], code=2, input=raw)
    assert not data['ok'] and not local.scib_corpora_dir.exists()


def test_task_runs_real_orchestration_and_exports(local, monkeypatch):
    install_fakes(monkeypatch, FakeLLM(tool_script=[
        tool_msg(('search_openalex', '{"query":"molecular"}')),
        tool_msg(('screen', '{"ids":["W6","W7","W8"]}')),
        tool_msg(('add_to_corpus', '{"ids":["W6","W7","W8"]}')),
        tool_msg(('finish', '{"summary":"Three relevant papers"}')),
    ]))
    result = output(['--quiet', 'task'], input=json.dumps({'prompt':'molecular graphs', 'years':[2022,2024]}))
    assert result['complete'] and result['source'] == 'openalex'
    assert result['summary']['n_papers'] == 3 and result['summary']['n_edges'] == 2
    assert result['command'] == 'task'
    for path in result['outputs'].values():
        assert Path(path).is_absolute() and Path(path).is_file()
    assert json.loads(Path(result['outputs']['result']).read_text()) == result
    assert json.loads(Path(result['outputs']['request']).read_text())['mode'] == 'agent'


def test_pipeline_entry_point_and_source_auto(local, monkeypatch):
    install_fakes(monkeypatch)
    result = output(['--json', '-q', 'run', 'graph neural networks', '--mode', 'pipeline'])
    assert result['complete'] and result['summary']['n_papers'] > 0
    assert result['source'] == 'openalex' and 'bibtex' in result['outputs']


def test_continuation_preserves_input_and_refreshes_intent_and_budget(local, monkeypatch, tmp_path):
    old = Corpus()
    old.meta.usage.deepseek_prompt_tokens = 999999
    old.meta.intent = ResearchIntent(topic='old', year_range=(2010,2020))
    path = tmp_path/'old.json'; path.write_text(old.model_dump_json())
    before = path.read_bytes()
    install_fakes(monkeypatch)
    result = service.run(ResearchRequest(prompt='new molecular graphs', corpus=path, years=(2022,2024), max_tokens=100))
    assert result.complete and result.usage.deepseek_prompt_tokens == 0
    new = service.load_corpus(Path(result.corpus))
    assert new.meta.intent.year_range == (2022,2024)
    assert new.meta.params['previous_runs'][0]['usage']['deepseek_prompt_tokens'] == 999999
    assert path.read_bytes() == before


def test_interrupted_agent_returns_recoverable_result(local, monkeypatch):
    class Interrupted(FakeLLM):
        def chat_tools(self, *args):
            raise KeyboardInterrupt
    install_fakes(monkeypatch, Interrupted())
    result = output(['--json', '-q', 'run', 'graph', '--seed', 'W5'], code=130)
    assert not result['complete'] and result['stop_reason'] == 'interrupted'
    assert result['summary']['n_seeds'] == 1
    assert Path(result['outputs']['checkpoint']).exists()
    assert service.load_corpus(Path(result['corpus'])).meta.complete is False


def test_pipeline_upstream_failure_is_not_empty_success(local, monkeypatch):
    class Broken(FakeOA):
        def search(self, *args, **kwargs):
            raise httpx.ConnectError('https://upstream/?api_key=secret-value')
    monkeypatch.setattr(service, 'make_clients', lambda *args: (FakeLLM(), Broken(), None))
    result = output(['--json', '-q', 'run', 'graph', '--mode', 'pipeline'], code=3)
    assert not result['complete'] and result['error']['code'] == 'upstream'
    assert 'secret-value' not in json.dumps(result)
    assert Path(result['outputs']['checkpoint']).is_file()


def test_year_cap_and_finish_are_enforced_independently_of_model():
    tracer = Tracer()
    ctx = AgentContext(CorpusStore(), FakeLLM(), FakeOA(), None,
                       ResearchIntent(topic='graph', year_range=(2023,2024)), tracer, max_papers=1)
    for pid in ('W6','W7','W8'):
        p = ctx.oa.by_id(pid); p.relevance = Relevance(score=9); ctx.remember([p])
    response = json.loads(dispatch(ctx, 'add_to_corpus', '{"ids":["W6","W7","W8"]}'))
    assert response['rejected_out_of_year'] == ['W6'] and response['rejected_corpus_full'] == ['W8']
    assert [p.id for p in ctx.corpus.papers] == ['W7']
    ctx.llm = FakeLLM([tool_msg(('finish', '{"summary":"done"}'), ('remove_from_corpus', '{"ids":["W7"]}'))])
    run_agent(ctx, 'graph')
    assert len(ctx.corpus) == 1 and ctx.finished


def test_openalex_query_timeout_budget_and_credentials_are_isolated(tmp_path):
    settings = Settings(_env_file=None, openalex_api_key='secret-value', openalex_max_requests=1)
    tracer = Tracer(tmp_path/'trace.jsonl')
    client = OpenAlexClient(settings, Cache(tmp_path/'cache.sqlite'), tracer)
    requests = []
    def handle(request):
        requests.append(request)
        assert request.url.params['search'] == 'graph'
        assert request.url.params['per-page'] == '5'
        assert request.url.params['api_key'] == 'secret-value'
        assert request.extensions['timeout']['read'] == settings.request_timeout
        return httpx.Response(200, json={'results':[]})
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handle), timeout=settings.request_timeout)
    try:
        client.search('graph', limit=5)
        client.search('graph', limit=5)
        assert len(requests) == 1
        with pytest.raises(BudgetError):
            client.search('other')
        assert 'secret-value' not in tracer.path.read_text()
    finally:
        client.close(); client.cache.close()


def test_home_resolves_env_and_outputs_without_chdir(tmp_path, monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.delenv('BOOKWORM_RUNS_DIR', raising=False)
    monkeypatch.delenv('SCIB_CORPORA_DIR', raising=False)
    (tmp_path/'.env').write_text('DEEPSEEK_API_KEY=fake\nBOOKWORM_RUNS_DIR=results\nBOOKWORM_OBSIDIAN_VAULT=\n')
    before = Path.cwd()
    configure(tmp_path)
    try:
        settings = get_settings()
        assert settings.scib_corpora_dir == tmp_path/'results' and settings.scib_obsidian_vault is None
        assert Path.cwd() == before
    finally:
        configure()


def test_doctor_is_offline_and_redacts_keys(local):
    data = output(['--json', 'doctor'])
    assert data['ready'] and data['source'] == 'openalex'
    assert 'fake' not in json.dumps(data) and not local.scib_corpora_dir.exists()


def test_schema_is_one_line_in_json_mode():
    data = output(['--json','schema'])
    assert 'request' in data and 'task' in data['results']


def test_offline_export_cannot_overwrite_its_input(tmp_path):
    corpus = tmp_path/'corpus.json'
    corpus.write_text(Corpus().model_dump_json())
    before = corpus.read_bytes()
    result = output(['--json','export',str(corpus),'--out',str(corpus)], code=2)
    assert result['error']['code'] == 'invalid_input' and corpus.read_bytes() == before


def test_cli_restores_signal_handler(local, monkeypatch):
    import signal
    previous = signal.getsignal(signal.SIGTERM)
    install_fakes(monkeypatch)
    output(['--json','-q','run','graph'])
    assert signal.getsignal(signal.SIGTERM) == previous


def test_legacy_obsidian_user_notes_are_preserved(tmp_path, small_corpus):
    from bookworm.obsidian import export_vault, NOTES_MARKER
    export_vault(small_corpus, tmp_path)
    note = next((tmp_path/'papers').glob('*.md'))
    text = note.read_text().replace('bookworm', 'scibooster')
    note.write_text(text + '\nMy original research notes\n')
    export_vault(small_corpus, tmp_path)
    assert 'My original research notes' in note.read_text() and NOTES_MARKER in note.read_text()


def test_clients_do_not_share_default_usage():
    settings = Settings(_env_file=None)
    first, second = OpenAlexClient(settings), OpenAlexClient(settings)
    try:
        first.tracer.usage.openalex_requests = 42
        assert second.tracer.usage.openalex_requests == 0
    finally:
        first.close(); second.close()
