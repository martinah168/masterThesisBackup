import csv
from pathlib import Path
from typing import Callable


def read_csv_labels(csv_fn: Path, label_str: str,
                    patient_id_fn: Callable[[dict], str]) -> dict[str, int]:
    labels: dict[str, int] = {}
    with open(csv_fn, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, fieldnames=f.readline().strip().split(','))
        for row in reader:
            label = row[label_str]
            patient_id = patient_id_fn(row)
            labels[patient_id] = label
    return labels
