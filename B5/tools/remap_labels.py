"""
remap_labels.py — Utilitário genérico para remapear ou filtrar IDs de classes em datasets formato YOLO.

Exemplo de uso (caso EPIs 6 classes -> 3 classes):
  python B5/tools/remap_labels.py \
      --dataset-dir B5/datasets/epi \
      --map "0:0, 1:0, 2:0, 3:0, 4:1, 5:2"
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_mapping(map_str: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for pair in map_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        old_id, new_id = pair.split(":")
        mapping[int(old_id.strip())] = int(new_id.strip())
    return mapping


def remap_dataset(dataset_root: Path, mapping: dict[int, int]) -> None:
    total_files = 0
    for split in ["train", "valid", "val", "test"]:
        labels_dir = dataset_root / split / "labels"
        if not labels_dir.exists():
            continue

        for txt_file in labels_dir.glob("*.txt"):
            new_lines = []
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_cls = int(parts[0])
                    if old_cls in mapping:
                        parts[0] = str(mapping[old_cls])
                        new_lines.append(" ".join(parts))

            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
            total_files += 1

    print(f"[B5] Remapeamento concluído: {total_files} arquivos atualizados em {dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remapeador genérico de classes YOLO")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Raiz do dataset (contendo train/valid/test)")
    parser.add_argument("--map", type=str, required=True, help="Mapeamento no formato 'antigo:novo, antigo:novo'")
    args = parser.parse_args()

    remap_dataset(Path(args.dataset_dir).resolve(), parse_mapping(args.map))