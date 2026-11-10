"""Safe subprocess delegation. Usage: python examples/delegate.py /workspace 'research need'."""
import json
import subprocess
import sys


def delegate(workspace: str, prompt: str) -> dict:
    result = subprocess.run(
        [sys.executable, '-m', 'bookworm', '--home', workspace, '--quiet', 'task'],
        input=json.dumps({'prompt': prompt, 'tier': 'quick'}),
        text=True, capture_output=True, check=False,
    )
    payload = json.loads(result.stdout)
    # Exit 3/130 can contain usable partial results. Never replace them with an empty corpus.
    return {'exit_code': result.returncode, 'result': payload}


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit("Usage: delegate.py WORKSPACE PROMPT")
    print(json.dumps(delegate(sys.argv[1], sys.argv[2]), ensure_ascii=False))
