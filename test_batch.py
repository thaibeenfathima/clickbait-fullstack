import requests
import json

with open('data/clickbait_data.csv', 'rb') as f:
    response = requests.post('http://localhost:5000/api/batch', files={'file': f}, data={'column': 'headline'})
    
results = response.json()
print(f'\n{"="*110}')
print(f'BATCH ANALYSIS - {results["total"]} Headlines Analyzed (100% Results)')
print(f'{"="*110}\n')
print(f'{"#":<3} {"Headline":<45} {"CB":<5} {"Conf":<7} {"Sentiment":<12} {"Keywords"}')
print('-' * 110)

for i, r in enumerate(results['results'], 1):
    headline = r['headline'][:42] + "..." if len(r['headline']) > 45 else r['headline']
    cb = 'YES' if r['is_clickbait'] else 'NO'
    conf = f"{r['confidence']*100:.0f}%"
    sent = r.get('sentiment', 'N/A')
    kw = ', '.join(r.get('highlighted_words', [])[:2])
    if len(r.get('highlighted_words', [])) > 2:
        kw += f" (+{len(r['highlighted_words'])-2})"
    
    print(f'{i:<3} {headline:<45} {cb:<5} {conf:<7} {sent:<12} {kw}')

print(f'\n{"="*110}')
clickbait_count = sum(1 for r in results["results"] if r["is_clickbait"])
legit_count = sum(1 for r in results["results"] if not r["is_clickbait"])
print(f'SUMMARY: {clickbait_count} Clickbait | {legit_count} Legitimate | {results["total"]} Total')
print(f'Detection Rate: {(clickbait_count/results["total"]*100):.0f}% Clickbait')
print(f'{"="*110}\n')
