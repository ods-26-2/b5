"""
package_artifact.py — Empacotador e gerador de evidências generalizado da Frente B5 (S2 + S3 6c).
Avalia no split de teste, exporta para ONNX e gera os 5 arquivos JSON oficiais em B5/artifacts/<id>/<ver>/.

Uso standalone (sem retreinar):
  python B5/tools/package_artifact.py \
      --data B5/datasets/epi/data.yaml \
      --weights B5/runs/app-epi/1.0.0/weights/best.pt \
      --artifact-id app-epi \
      --artifact-version 1.0.0 \
      --dataset-id dataset-epi \
      --dataset-version 1.0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

# Diretório base B5 resolvido automaticamente a partir de B5/tools/package_artifact.py
B5_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = B5_ROOT / "artifacts"


def sha256_file(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze_dataset_and_hash(data_yaml_path: Path) -> tuple[list[str], dict[str, dict[str, int]], int, str, dict[str, Any]]:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names", [])
    if isinstance(names, dict):
        classes = [str(names[k]) for k in sorted(names.keys())]
    else:
        classes = [str(n) for n in names]

    dataset_root = data_yaml_path.parent
    hasher = hashlib.sha256()

    split_counts: dict[str, dict[str, int]] = {}
    total_counts = {cls_name: 0 for cls_name in classes}
    total_images = 0

    for split in ["train", "valid", "val", "test"]:
        canonical_split = "val" if split in ("val", "valid") else split
        if canonical_split in split_counts:
            continue

        split_dir = dataset_root / split / "labels"
        if not split_dir.exists():
            rel_key = "val" if split == "valid" and "val" in cfg else split
            if rel_key in cfg:
                candidate = (dataset_root / str(cfg[rel_key])).resolve().parent / "labels"
                if candidate.exists():
                    split_dir = candidate

        if not split_dir.exists():
            continue

        split_counts[canonical_split] = {cls_name: 0 for cls_name in classes}
        txt_files = sorted(split_dir.glob("*.txt"))
        total_images += len(txt_files)

        for txt_file in txt_files:
            content = txt_file.read_bytes()
            hasher.update(txt_file.name.encode("utf-8"))
            hasher.update(content)
            for line in content.decode("utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls_idx = int(parts[0])
                if 0 <= cls_idx < len(classes):
                    cls_name = classes[cls_idx]
                    split_counts[canonical_split][cls_name] += 1
                    total_counts[cls_name] += 1

    split_counts["total"] = total_counts
    return classes, split_counts, total_images, hasher.hexdigest(), cfg


def package_model_artifact(
    data_yaml: Path,
    weights_pt: Path,
    artifact_id: str,
    artifact_version: str,
    dataset_id: str,
    dataset_version: str,
    imgsz: int = 640,
    batch: int = 8,
    origin_description: str | None = None,
    hyperparams_override: dict[str, Any] | None = None,
) -> Path:
    data_yaml_path = data_yaml.resolve()
    best_pt_path = weights_pt.resolve()

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml não encontrado: {data_yaml_path}")
    if not best_pt_path.exists():
        raise FileNotFoundError(f"Peso .pt não encontrado: {best_pt_path}")

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    package_dir = DEFAULT_ARTIFACTS_DIR / artifact_id / artifact_version
    package_dir.mkdir(parents=True, exist_ok=True)

    # 1. Analisar dataset e gerar manifest.json (Genérico para qualquer domínio)
    print(f"[B5] Analisando dataset ({data_yaml_path.name}) e calculando SHA-256...")
    classes, class_counts, total_images, dataset_sha256, yaml_cfg = analyze_dataset_and_hash(data_yaml_path)

    default_rules = [
        f"Classes oficiais mapeadas: {', '.join(classes)}.",
        "Caixas delimitadoras (bboxes) ajustadas aos limites visíveis de cada objeto.",
        "Ignorar oclusões severas (>80%) ou instâncias ilegíveis (<10x10 pixels).",
    ]

    manifest_data = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "created_at": now_iso,
        "owner": "MOD-1",
        "origin": origin_description or f"Dataset do domínio '{artifact_id}' ({ data_yaml_path.parent.name })",
        "labeling_criteria": {
            "tool": "CVAT / Label Studio",
            "classes": classes,
            "rules": yaml_cfg.get("labeling_rules", default_rules),
        },
        "total_images": total_images,
        "class_distribution": class_counts,
        "hash_sha256": dataset_sha256,
    }

    manifest_path = package_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4, ensure_ascii=False)

    # 2. Avaliação no conjunto de teste isolado (ou val como fallback)
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(best_pt_path))
    eval_split = "test" if "test" in yaml_cfg else "val"

    print(f"[B5] Avaliando modelo no split '{eval_split}'...")
    metrics = model.val(data=str(data_yaml_path), imgsz=imgsz, batch=batch, split=eval_split, device=device)

    # 3. Exportação ONNX (Padrão B5 -> P2)
    print("[B5] Exportando modelo para ONNX (opset=12, nms=False)...")
    exported_onnx = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        simplify=True,
        opset=12,
        nms=False,
    )

    final_onnx_path = package_dir / f"{artifact_id}.onnx"
    shutil.copy2(exported_onnx, final_onnx_path)
    shutil.copy2(best_pt_path, package_dir / f"{artifact_id}_best.pt")

    onnx_sha256 = sha256_file(final_onnx_path)

    metrics_dict = {
        "evaluation_split": eval_split,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }

    # 4. Gerar run.json (Lendo automaticamente o args.yaml do treino se existir)
    default_hyperparams = {
        "base_weights": "yolov8s.pt",
        "batch_size": batch,
        "imgsz": imgsz,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "cos_lr": True,
        "amp": True,
    }

    # Tenta resgatar os parâmetros reais do treino gerados pelo Ultralytics
    train_args_yaml = weights_pt.parent.parent / "args.yaml"
    if train_args_yaml.exists():
        with open(train_args_yaml, "r", encoding="utf-8") as f:
            yolo_args = yaml.safe_load(f)
        default_hyperparams.update({
            "base_weights": str(yolo_args.get("model", default_hyperparams["base_weights"])),
            "epochs_configured": yolo_args.get("epochs", 100),
            "patience": yolo_args.get("patience", 20),
            "batch_size": yolo_args.get("batch", batch),
            "imgsz": yolo_args.get("imgsz", imgsz),
            "optimizer": yolo_args.get("optimizer", "AdamW"),
            "lr0": yolo_args.get("lr0", 0.001),
            "lrf": yolo_args.get("lrf", 0.01),
            "cos_lr": yolo_args.get("cos_lr", True),
            "amp": yolo_args.get("amp", True),
            "fliplr": yolo_args.get("fliplr", 0.0),
            "mixup": yolo_args.get("mixup", 0.0),
        })

    if hyperparams_override:
        default_hyperparams.update(hyperparams_override)

    run_data = {
        "run_id": f"run-{artifact_id}-v{artifact_version}",
        "executed_at": now_iso,
        "script": "train_pipeline.py",
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "dataset_hash_sha256": dataset_sha256,
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "artifact_hash_sha256": onnx_sha256,
        "hyperparameters": default_hyperparams,
        "metrics": metrics_dict,
        "hardware_benchmark_p2": {
            "target_device": "Jetson Orin Nano 8GB",
            "status": "pending_p2_compilation",
            "latency_ms": None,
            "gpu_memory_mb": None,
        },
    }

    run_path = package_dir / "run.json"
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=4, ensure_ascii=False)

    # 5. Gerar metadata.json (Pacote MODEL - Especificação 1.0 Arquitetura 6c)
    model_package_meta = {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "classes": classes,
        "input_shape": [1, 3, imgsz, imgsz],
        "precision": "FP32",
        "metrics": metrics_dict,
        "dataset_version": dataset_version,
        "onnx_file": final_onnx_path.name,
        "hash_sha256": onnx_sha256,
    }

    meta_path = package_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_package_meta, f, indent=4, ensure_ascii=False)

    # 6. Gerar descriptor.json e ods_artifact_published.json (Registro Geral B5 6c)
    relative_uri = f"B5/artifacts/{artifact_id}/{artifact_version}/{final_onnx_path.name}"
    descriptor_data = {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "kind": "MODEL",
        "owner": "MOD-1",
        "uri": relative_uri,
        "hash_sha256": onnx_sha256,
        "valid_from": now_iso,
    }

    descriptor_path = package_dir / "descriptor.json"
    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump(descriptor_data, f, indent=4, ensure_ascii=False)

    published_event = {
        "message_type": "event",
        "message_id": str(uuid.uuid4()),
        "schema": "ods.artifact.published",
        "schema_version": "1.0",
        "producer": "B5",
        "published_at": now_iso,
        "payload": descriptor_data,
    }

    event_path = package_dir / "ods_artifact_published.json"
    with open(event_path, "w", encoding="utf-8") as f:
        json.dump(published_event, f, indent=4, ensure_ascii=False)

    print(f"\n[B5] Artefato empacotado com sucesso em: {package_dir}")
    return package_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empacotador e gerador de evidências B5 (qualquer domínio)")
    parser.add_argument("--data", type=str, required=True, help="Caminho para o data.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Caminho para o best.pt treinado")
    parser.add_argument("--artifact-id", type=str, required=True, help="ID do artefato (ex: app-epi, app-volei)")
    parser.add_argument("--artifact-version", type=str, default="1.0.0", help="Versão do modelo (ex: 1.0.0)")
    parser.add_argument("--dataset-id", type=str, default=None, help="ID do dataset (default: dataset-<artifact-id>)")
    parser.add_argument("--dataset-version", type=str, default="1.0.0", help="Versão do dataset")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolução de entrada")
    parser.add_argument("--batch", type=int, default=8, help="Batch size para validação")
    parser.add_argument("--origin", type=str, default=None, help="Descrição opcional da origem do dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds_id = args.dataset_id or f"dataset-{args.artifact_id}"
    package_model_artifact(
        data_yaml=Path(args.data),
        weights_pt=Path(args.weights),
        artifact_id=args.artifact_id,
        artifact_version=args.artifact_version,
        dataset_id=ds_id,
        dataset_version=args.dataset_version,
        imgsz=args.imgsz,
        batch=args.batch,
        origin_description=args.origin,
    )