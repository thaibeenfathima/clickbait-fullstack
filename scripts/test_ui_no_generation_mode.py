from pathlib import Path
p = Path(__file__).resolve().parents[1]
app_path = p / 'app.py'
content = app_path.read_text()
assert 'Generation mode' not in content, 'UI must not show generation mode toggles'
assert 'distilgpt2' not in content, 'UI must not show model names'
assert 'AI-based' not in content, 'UI must not show AI vs Template labels'
print('UI generation-mode removal test passed ✅')
