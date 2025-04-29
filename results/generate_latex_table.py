import json
from collections import defaultdict

table_data = defaultdict(lambda: defaultdict(dict))

with open("results/results.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        emb = entry["embedder"]
        metric = entry["metric"]
        mode = entry["mode"]
        table_data[emb][metric][mode] = {
            "code_em": entry.get("code_em", "null"),
            "code_es": entry.get("code_es", "null"),
            "id_em": entry.get("id_em", "null"),
            "id_f1": entry.get("id_f1", "null"),
        }

modes = ["token_token", "token_embed", "embed_token", "embed_embed"]
metrics = ["cosine", "dot", "l2", "l1"]
embedders = ["codet5", "codebert", "graphcodebert"]

def render_row(val):
    return " & " + " & ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in val.values()) + " \\\\"

print("\\begin{tabular}{|l|c|c|c|c|}")
print("\\hline")
print("\\textbf{Method} & \\textbf{Code Match EM} & \\textbf{Code Match ES} & \\textbf{Identifier Match EM} & \\textbf{Identifier Match F1} \\\\")
print("\\hline")
print("\\textbf{Without retrieval} & 0.39 & 0.6813 & 0.39 & 0.6649 \\\\")
print("\\hline")

for embedder in embedders:
    print(f"\\textbf{{{embedder}}} & & & & \\\\")
    print("\\hline")
    for metric in metrics:
        print(f"\\quad \\textbf{{{metric}}} & & & & \\\\")
        for mode in modes:
            data = table_data.get(embedder, {}).get(metric, {}).get(mode)
            if data:
                print(f"\\quad\\quad {mode.replace('_', '\\_')}" + render_row(data))
            else:
                print(f"\\quad\\quad {mode.replace('_', '\\_')} & null & null & null & null \\\\")
        print("\\hline")
print("\\end{tabular}")
