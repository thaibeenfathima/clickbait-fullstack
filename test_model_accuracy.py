import requests
import json

tests = [
    ("This one trick will change your life", True),
    ("Scientists discover new species", False),
    ("You won't believe what happened next", True),
    ("This photo will make you cry", True),
    ("How to save money this year", False),
    ("Doctors hate this one weird hack", True),
    ("Local man helps neighbor with groceries", False),
    ("What happened next shocked everyone", True),
]

correct = 0
print("\n" + "="*80)
print("MODEL ACCURACY TEST - BiLSTM Prediction Validation")
print("="*80 + "\n")

for headline, expected in tests:
    try:
        r = requests.post('http://localhost:5000/api/analyze', json={'headline': headline})
        result = r.json()
        pred = result['is_clickbait']
        conf = result['confidence'] * 100
        match = "PASS" if pred == expected else "FAIL"
        if match == "PASS":
            correct += 1
        
        print(f"{match} | Expected: {str(expected):<5} | Predicted: {str(pred):<5} | Confidence: {conf:>3.0f}%")
        print(f"     Headline: {headline}")
        print()
    except Exception as e:
        print(f"ERROR: {e}")

print("="*80)
accuracy = (correct / len(tests)) * 100
print(f"RESULTS: {correct}/{len(tests)} correct predictions = {accuracy:.0f}% accuracy")
print("="*80 + "\n")
