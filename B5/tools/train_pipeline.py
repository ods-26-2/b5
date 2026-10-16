"""
train_pipeline.py — Pipeline unificado de treinamento, validação e empacotamento ONNX (Frente B5).
Salva os treinos em B5/runs/<artifact_id>/<version> e gera o pacote final em B5/artifacts/<artifact_id>/<version>.

Uso para qualquer novo modelo:
  python B5/tools/train_pipeline.py \
      --data B5/datasets/epi/data.yaml \
      --artifact-id app-epi \
      --artifact-version 1.0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from package_artifact import package_model_artifact

B5_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = B5_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B5 - Treinamento e Empacotamento Automatizado")
    parser.add_argument("--data", type=str, required=True, help="Caminho para o data.yaml do dataset")
    parser.add_argument("--artifact-id", type=str, required=True, help="ID do artefato (ex: app-epi, app-placas)")
    parser.add_argument("--artifact-version", type=str, default="1.0.0", help="Versão do artefato (ex: 1.0.0)")
    parser.add_argument("--dataset-id", type=str, default=None, help="ID do dataset (default: dataset-<artifact-id>)")
    parser.add_argument("--dataset-version", type=str, default="1.0.0", help="Versão do dataset")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Peso base (yolo11n.pt, yolo11s.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Número máximo de épocas")
    parser.add_argument("--patience", type=int, default=20, help="Épocas sem melhoria para early stopping")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolução de entrada")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (8 é ideal para 6GB VRAM)")
    parser.add_argument("--origin", type=str, default=None, help="Descrição da origem do dataset para o manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml_path = Path(args.data).resolve()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device: int | str = 0
        print(f"[B5] GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[B5] AVISO: GPU não detectada. Utilizando CPU.")

    # Caminho absoluto estrito: B5/runs/<artifact_id>/<artifact_version>
    project_dir = (DEFAULT_RUNS_DIR / args.artifact_id).resolve()
    run_name = args.artifact_version

    model = YOLO(args.model)

    print(f"[B5] Iniciando treinamento: {args.artifact_id} (v{args.artifact_version})...")
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=4,
        amp=True,
        cache="disk",
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
    )

    # Recupera o diretório real direto do trainer do Ultralytics
    save_dir = Path(model.trainer.save_dir)
    best_pt_path = save_dir / "weights" / "best.pt"

    hyperparams_info = {
        "base_weights": args.model,
        "epochs_configured": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch,
        "imgsz": args.imgsz,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "cos_lr": True,
        "amp": True,
    }

    ds_id = args.dataset_id or f"dataset-{args.artifact_id}"

    # Executa automaticamente a validação no split test, exportação ONNX e geração dos 5 JSONs
    package_model_artifact(
        data_yaml=data_yaml_path,
        weights_pt=best_pt_path,
        artifact_id=args.artifact_id,
        artifact_version=args.artifact_version,
        dataset_id=ds_id,
        dataset_version=args.dataset_version,
        imgsz=args.imgsz,
        batch=args.batch,
        origin_description=args.origin,
        hyperparams_override=hyperparams_info,
    )


if __name__ == "__main__":
    main()