import time
import sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]
if str(p) not in sys.path:
    sys.path.insert(0,str(p))
import app

print('st.session_state type:', type(app.st.session_state))
# Monkeypatch session_state with a plain dict to simulate streamlit runtime
orig = app.st.session_state
try:
    fake = {}
    app.st.session_state = fake
    print('Using fake session_state:', type(app.st.session_state))
    call1 = app._allow_generation('unittest', cooldown=2.5)
    print('First call (should be True):', call1)
    assert call1 is True, 'First call must return True'
    print('After first call:', app.st.session_state)
    call2 = app._allow_generation('unittest', cooldown=2.5)
    print('Second call immediately (should be False):', call2)
    assert call2 is False, 'Second call within cooldown must return False'
    print('After second call:', app.st.session_state)
    print('Sleeping 3s...')
    time.sleep(3)
    call3 = app._allow_generation('unittest', cooldown=2.5)
    print('Third call after cooldown (should be True):', call3)
    assert call3 is True, 'Third call after cooldown must return True'
    print('After third call:', app.st.session_state)
    print('Debounce test passed ✅')
    sys.exit(0)
finally:
    app.st.session_state = orig
