# Laudo Técnico de Acurácia e Desempenho Computacional (T13 — B5 -> SIS-1)

- **Artefato:** `app-plates` (v`1.0.0`)
- **Dataset Version:** `1.0.0` (Partição avaliada: `test`, estritamente disjunta do treino)
- **Hardware de Aferição (Degrau A):** `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (Resolução `640x640`)
- **Data de Geração:** `2026-10-20T14:12:17Z`

---

## 1. Comparativo de Qualidade de Detecção (Acurácia)

| Precisão | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Variação mAP@0.5 (vs FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Baseline)** | 0.9950 | 0.9581 | 0.9990 | 1.0000 | 0.0000 |
| **FP16 (Half)** | 0.9950 | 0.9565 | 0.9990 | 1.0000 | +0.0000 |
| **INT8 (Quantizado)** | 0.9950 | 0.9477 | 0.9993 | 1.0000 | +0.0000 |

---

## 2. Comparativo de Desempenho Computacional (Latência, FPS, Memória e Tamanho)

| Precisão | Footprint Pesos | Latência Média (1 quadro) | Latência P95 | Throughput (FPS) | Pico VRAM Alocada | Speedup (vs FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Baseline)** | 42.68 MB | 14.02 ms | 14.44 ms | 71.3 FPS | 122.0 MB | 1.00x |
| **FP16 (Half)** | 21.34 MB | 9.18 ms | 11.08 ms | 108.9 FPS | 107.1 MB | 1.53x |
| **INT8 (ONNX CPU)** | 10.96 MB | 177.24 ms | 223.52 ms | 5.6 FPS | 0.0 MB | 0.08x |

> **Detalhamento do Pipeline Interno (em lote / batch=8):**
> - **FP32:** Pré-proc `0.28 ms` | Inferência `7.37 ms` | Pós-proc `0.31 ms`
> - **FP16:** Pré-proc `0.26 ms` | Inferência `4.15 ms` | Pós-proc `0.4 ms`

---

## 3. Parecer Técnico para Compilação TensorRT em P2 (Squad SIS-1)
Compilar engine TensorRT em FP16 na Jetson Orin Nano (P2): reduz o footprint dos pesos pela metade (42.68 MB -> 21.34 MB), atinge 108.9 FPS (latência média de 9.18 ms/quadro, speedup de 1.53x sobre FP32) e mantém variação de mAP@0.5 desprezível (+0.0000).
