import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def read_data_from_jsonl(jsonl_file):
    table_data = defaultdict(lambda: defaultdict(dict))
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            emb = entry["embedder"]
            metric = entry["metric"]
            mode = entry["mode"]
            
            table_data[emb][metric][mode] = {
                "retrieval_time": entry.get("retrieval_time", 0),
                "generation_time": entry.get("generation_time", 0)
            }
    
    return table_data

def plot_times(data):
    modes = ['token_token', 'embed_token', 'token_embed', 'embed_embed']
    metrics = ['cosine', 'dot', 'l2', 'l1']
    embedder_names = list(data.keys())
    
    max_time = 0
    for embedder in embedder_names:
        for metric in metrics:
            for mode in modes:
                if mode in data[embedder][metric]:
                    total = data[embedder][metric][mode]['retrieval_time'] + \
                            data[embedder][metric][mode]['generation_time']
                    max_time = max(max_time, total)

    fig, axes = plt.subplots(1, len(embedder_names), figsize=(18, 6))
    if len(embedder_names) == 1:
        axes = [axes]

    plt.style.use('seaborn-v0_8')
    
    for idx, embedder in enumerate(embedder_names):
        ax = axes[idx]
        
        retrieval_avgs = []
        generation_avgs = []
        
        for mode in modes:
            retrieval_times = []
            generation_times = []
            
            for metric in metrics:
                if mode in data[embedder][metric]:
                    retrieval_times.append(data[embedder][metric][mode]['retrieval_time'])
                    generation_times.append(data[embedder][metric][mode]['generation_time'])
            
            retrieval_avgs.append(np.mean(retrieval_times) if retrieval_times else 0)
            generation_avgs.append(np.mean(generation_times) if generation_times else 0)
        
        x = np.arange(len(modes))
        width = 0.6
        
        p1 = ax.bar(x, retrieval_avgs, width, label='Retrieval Time', color='#1f77b4')
        p2 = ax.bar(x, generation_avgs, width, bottom=retrieval_avgs, 
                   label='Generation Time', color='#ff7f0e')
        
        ax.set_title(f'Embedder: {embedder}', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=45, ha='right')
        ax.set_ylim(0, max_time * 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        if idx == 0:
            ax.set_ylabel('Time (seconds)')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.suptitle('Average Retrieval and Generation Time by Mode', y=1.05)
    plt.tight_layout()
    plt.show()

jsonl_file = "results/results.jsonl"
data = read_data_from_jsonl(jsonl_file)
plot_times(data)