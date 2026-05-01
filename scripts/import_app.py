import sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]
if str(p) not in sys.path:
    sys.path.insert(0,str(p))
try:
    import app
    print('Imported app OK')
except Exception as e:
    import traceback; traceback.print_exc()
    print('Failed to import app')
