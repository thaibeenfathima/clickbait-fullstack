import re
p='app.py'
with open(p,'r',encoding='utf-8') as f:
    s=f.read()
count = s.count('rewrite_only=True')
print('rewrite_only occurrences:', count)
assert count>=3, 'Expected rewrite_only=True used in Single/URL/Batch generate calls'
print('UI calls use rewrite_only flag ✅')
