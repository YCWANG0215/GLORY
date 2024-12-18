import csv
from pathlib import Path


def user_statistic(cfg, mode, user_history_map):
    user_stats = []
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    print(f"[{mode}] process user statistic")
    for user_id, history in user_history_map.items():
        topic_count = len(history[0])
        subtopic_count = len(history[1])

        user_stats.append({
            'user_id': user_id,
            'topic_count': topic_count,
            'subtopic_count': subtopic_count
        })

    with open((Path(data_dir[mode]) / 'user_stats.tsv'), mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['user_id', 'topic_count', 'subtopic_count'], delimiter='\t')
        writer.writeheader()
        for stat in user_stats:
            writer.writerow(stat)