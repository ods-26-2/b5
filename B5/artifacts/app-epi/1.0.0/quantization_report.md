# Laudo Técnico de Acurácia e Desempenho Computacional (T13 — B5 -> SIS-1)

- **Artefato:** `app-epi` (v`1.0.0`)
- **Dataset Version:** `1.0.0` (Partição avaliada: `test`, estritamente disjunta do treino)
- **Hardware de Aferição (Degrau A):** `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (Resolução `640x640`)
- **Data de Geração:** `2026-10-20T14:18:49Z`

---

## 1. Comparativo de Qualidade de Detecção (Acurácia)

| Precisão | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Variação mAP@0.5 (vs FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Baseline)** | 0.9517 | 0.5605 | 0.9397 | 0.9227 | 0.0000 |
| **FP16 (Half)** | 0.9512 | 0.5611 | 0.9380 | 0.9234 | -0.0005 |
| **INT8 (Quantizado)** | 0.9520 | 0.5611 | 0.9348 | 0.9179 | +0.0003 |

---

## 2. Comparativo de Desempenho Computacional (Latência, FPS, Memória e Tamanho)

| Precisão | Footprint Pesos | Latência Média (1 quadro) | Latência P95 | Throughput (FPS) | Pico VRAM Alocada | Speedup (vs FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 (Baseline)** | 42.68 MB | 13.82 ms | 14.31 ms | 72.3 FPS | 122.0 MB | 1.00x |
| **FP16 (Half)** | 21.34 MB | 8.45 ms | 8.95 ms | 118.3 FPS | 108.2 MB | 1.64x |
| **INT8 (ONNX CPU)** | 10.96 MB | 196.96 ms | 222.62 ms | 5.1 FPS | 0.0 MB | 0.07x |

> **Detalhamento do Pipeline Interno (em lote / batch=8):**
> - **FP32:** Pré-proc `0.41 ms` | Inferência `9.71 ms` | Pós-proc `0.45 ms`
> - **FP16:** Pré-proc `0.44 ms` | Inferência `5.41 ms` | Pós-proc `1.12 ms`

---

## 3. Parecer Técnico para Compilação TensorRT em P2 (Squad SIS-1)
Compilar engine TensorRT em FP16 na Jetson Orin Nano (P2): reduz o footprint dos pesos pela metade (42.68 MB -> 21.34 MB), atinge 118.3 FPS (latência média de 8.45 ms/quadro, speedup de 1.64x sobre FP32) e mantém variação de mAP@0.5 desprezível (-0.0005).
