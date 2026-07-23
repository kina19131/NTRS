# ...existing code...
import json

def half_max_sigma(results):
    target = max(r['density'] for r in results) / 2
    last = results[0]['sigma']
    for r in results:
        if r['density'] >= target:
            last = r['sigma']
    return last

paths = [
    ('outputs_0619_h2/gpt2/seq_agnews_to_sst2/phase1_task_a_density_acc.json', 'GPT-2'),
]
for path, label in paths:
    with open(path) as f:
        d = json.load(f)
    print(f'{label}: acc-based σ½_A = {half_max_sigma(d["sigma_results"]):.6f}  task_a_acc={d["task_a_acc"]:.4f}')
