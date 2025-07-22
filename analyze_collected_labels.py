import os
import json
from collections import Counter

folder = os.path.dirname(os.path.abspath(__file__))

stats = []

for fname in os.listdir(folder):
    if fname.endswith('.json'):
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Try to find issues list (either top-level or under 'issues')
            if isinstance(data, dict) and 'issues' in data:
                issues = data['issues']
            elif isinstance(data, list):
                issues = data
            else:
                continue
            total = len(issues)
            empty_labels = sum(1 for issue in issues if not issue.get('labels'))
            nonempty_labels = total - empty_labels
            label_counts = Counter()
            for issue in issues:
                for label in issue.get('labels', []):
                    label_counts[label] += 1
            stats.append({
                'file': fname,
                'total_issues': total,
                'empty_labels': empty_labels,
                'nonempty_labels': nonempty_labels,
                'percent_with_labels': 100 * nonempty_labels / total if total else 0,
                'unique_labels': len(label_counts),
                'top_labels': label_counts.most_common(5)
            })
        except Exception as e:
            print(f"Error reading {fname}: {e}")

# Print summary
print(f"{'File':40} {'Total':>6} {'Empty':>6} {'WithLabels':>10} {'%WithLabels':>12} {'UniqueLabels':>12} TopLabels")
for s in stats:
    print(f"{s['file'][:40]:40} {s['total_issues']:6} {s['empty_labels']:6} {s['nonempty_labels']:10} {s['percent_with_labels']:12.1f} {s['unique_labels']:12} {s['top_labels']}")

# Optionally, save to CSV
with open(os.path.join(folder, 'label_stats.csv'), 'w', encoding='utf-8') as f:
    f.write('file,total_issues,empty_labels,nonempty_labels,percent_with_labels,unique_labels,top_labels\n')
    for s in stats:
        f.write(f"{s['file']},{s['total_issues']},{s['empty_labels']},{s['nonempty_labels']},{s['percent_with_labels']:.1f},{s['unique_labels']},\"{s['top_labels']}\"\n")

print("\nStats saved to label_stats.csv")
