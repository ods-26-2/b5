# Squad MOD-1 — Frente Transversal B5 (Fábrica de Modelos)
**Disciplina:** Oficina de Desenvolvimento de Software (ODS 2026/2 — UFSJ)  
**Especificação:** Interfaces 1.0 (Arquitetura 6c Oficial)  
**Integrantes:** Felipe Girardi Siqueira (Líder), Thiago Mateus de Almeida Dias Orsini, Eduardo Monteiro Costa Pires

Ferramental padronizado de preparação de dados, treinamento (`YOLOv8`), avaliação em partição de teste estritamente
disjunta (`test`), análise comparativa de precisão/desempenho (`FP32`, `FP16`, `INT8`) e empacotamento imutável de
modelos `ONNX` acompanhados dos 5 artefatos JSON oficiais (`manifest.json`, `run.json`, `metadata.json`,
`descriptor.json` e `ods_artifact_published.json`).

Os pacotes publicados em `B5/artifacts/` são consumidos pelo Componente I1, no repositório
[`ods-26-2/i1`](https://github.com/ods-26-2/i1), que mantém uma cópia do pacote em uso em `I1/artifacts/`.
Ao publicar uma nova versão aqui, atualize também a cópia no I1 (o `hash_sha256` do `metadata.json` deve coincidir).

---

## 1. Estrutura do Repositório

```text
.
├── B5/
│   ├── tools/
│   │   ├── train_pipeline.py        # Treino automatizado + chamada ao empacotador
│   │   ├── package_artifact.py      # Exportação ONNX + geração dos 5 JSONs oficiais
│   │   ├── evaluate_quantization.py # Laudo técnico FP32 vs FP16 vs INT8 (T13)
│   │   └── remap_labels.py          # Utilitário de remapeamento de classes YOLO
│   ├── artifacts/
│   │   ├── app-epi/1.0.0/           # Pacote MODEL Aplicação 1 (EPIs: pessoa, capacete, colete) — .onnx pendente
│   │   └── app-plates/1.0.0/        # Pacote MODEL Aplicação 2 (VER-5: 4 classes de placas)
│   ├── datasets/                    # (local, não versionado)
│   └── runs/                        # (local, não versionado)
└── docs/diagrams/
```

---

## 2. Operação do Ferramental B5

Os scripts abaixo devem ser executados no ambiente local (`.venv` com suporte a GPU/CUDA) para treinar, empacotar e avaliar modelos:

### 2.1. Treinamento e Empacotamento Integrados (`train_pipeline.py`)
Executa o treinamento supervisionado, salva os pesos em `B5/runs/<artifact-id>/<version>/` e aciona automaticamente o `package_artifact.py` ao final:

```bash
python3 B5/tools/train_pipeline.py \
  --data B5/datasets/plates/global/data.yaml \
  --artifact-id app-plates \
  --artifact-version 1.0.0 \
  --dataset-id dataset-plates \
  --dataset-version 1.0.0 \
  --model yolov8s.pt \
  --epochs 50 \
  --patience 12 \
  --batch 16 \
  --origin "Dataset 20k Placas Balanceado (4 classes) - VER-5"
```

### 2.2. Empacotamento Standalone sem Retreinar (`package_artifact.py`)
Gera ou atualiza o arquivo `.onnx` (`opset=12`, `nms=False`, `simplify=True`) e os 5 arquivos JSON oficiais em `B5/artifacts/<artifact-id>/<version>/` a partir de um peso `best.pt` existente (lendo automaticamente o `args.yaml` do treino):

```bash
python3 B5/tools/package_artifact.py \
  --data B5/datasets/plates/global/data.yaml \
  --weights B5/runs/app-plates/1.0.0/weights/best.pt \
  --artifact-id app-plates \
  --artifact-version 1.0.0 \
  --dataset-id dataset-plates \
  --dataset-version 1.0.0 \
  --origin "Dataset 20k Placas Balanceado (4 classes) - VER-5"
```

### 2.3. Laudo de Acurácia e Desempenho por Precisão (`evaluate_quantization.py` — T13)
Avalia `FP32`, `FP16` e `INT8` na partição `test` isolada, aferindo `mAP@0.5`, `mAP@0.5:0.95`, `precision`, `recall`, tamanho em disco (MB), latência por quadro (ms), throughput (FPS) e pico de VRAM (MB). Gera `quantization_report.md` e `quantization_report.json` para subsidiar a compilação TensorRT pelo squad **SIS-1 (P2)**:

```bash
python3 B5/tools/evaluate_quantization.py \
  --data B5/datasets/plates/global/data.yaml \
  --weights B5/runs/app-plates/1.0.0/weights/best.pt \
  --artifact-id app-plates \
  --artifact-version 1.0.0 \
  --dataset-version 1.0.0
```
