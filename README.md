# Updated Sydney System Grading from Gastric Biopsy WSI

위 생검 H&E WSI의 patch feature에서 **Updated Sydney System 5개 항목**(H. pylori · 호중구 · 단핵구 · 위축 · 장상피화생)의 등급(0~3)을 **동시에** 예측하는 weakly-supervised multi-task MIL 파이프라인. patch-level label 없이 **사례(case) 단위 라벨만** 사용하고, [TRIDENT](https://github.com/mahmoodlab/TRIDENT)로 추출한 patch feature(`.h5`)에서 학습을 시작한다. 모델은 논문 **SydneyMTL** (Kim et al., *Histopathology* 2026)을 재구현했다. 구현은 `src/`, 실행 도구는 `scripts/`.

파이프라인은 5개 요소로 나뉘고, 각 요소를 설정으로 골라 조합한다. 기본 구성은 논문 재현 설정(`site_mode=none` + `la` + `code` split)이다.

| 요소 | 지정 방법 | default | 지원 옵션 |
|---|---|---|---|
| A. patch encoder | `ENCODERS` | `hibou_l` | `hibou_l` (1024) · `hoptimus0` (1536) · `virchow` (2560) · `midnight12k` (1536) |
| B. 배율 | `MAG` | `20` | `10` · `20` · `40` |
| C. 생검 부위 입력 | `--site_mode` | `none` (논문) | `head` (logit 보정) · `film` (FiLM) |
| D. loss | `--loss` | `la` (logit-adjusted CE) | `ce` · `wce` |
| E. fold 분할 | `--split` | `code` (논문) | `multilabel` (iterative stratification) |

모델·옵션 선택은 §5, 출력·실험 관리는 §6, 학습 편의 기능은 §7 참조. 아래는 환경 → 데이터 준비 → 분할 → 학습·평가 순서다.

## 1. 환경

```bash
pip install -r requirements.txt
git clone https://github.com/mahmoodlab/TRIDENT && git -C TRIDENT checkout e4c98b2
pip install --no-deps -e TRIDENT            # 수정 없이 사용
```

- Hibou-L · H-Optimus-0 · Virchow는 gated 모델이다 → HuggingFace 접근 승인 + 토큰.
- 경로와 인터프리터는 환경변수로 준다: `WSI_DIR` · `EMB_ROOT` · `PYTHON` · `MLFLOW_PYTHON` · `MLFLOW_TRACKING_URI`.

## 2. 데이터셋 준비

라벨 메타데이터 CSV 한 개와 slide마다 feature `.h5` 한 개가 필요하다.

```text
<EMB_ROOT>/<fm>/
├── contours/  contours_geojson/  thumbnails/        조직 분할 (모든 배율 공용)
├── 20x_224px_0px_overlap/
│   ├── patches/<slide_id>_patches.h5                패치 좌표
│   └── features_<fm>/<slide_id>.h5                  패치 feature
└── 40x_224px_0px_overlap/ …                         배율마다 하위 폴더가 따로라 섞이지 않는다
```

| dataset | shape | dtype | 의미 |
|---|---|---|---|
| `features` | `(N, D)` | float32 | 패치 N개의 feature (D = 인코더 차원) |
| `coords` | `(N, 2)` | int | 패치 좌표 |

**메타데이터 CSV** (`metadata/sydney_meta.csv`, slide 1줄). 사례별 정보라 이 저장소에는 포함하지 않는다.

| 열 | 의미 |
|---|---|
| `slide_id`, `case_id` | slide·사례 식별자 |
| `code_review`, `code_init` | Sydney CODE 5자리 = **HP – NEU – MONO – ATR – IM**, 각 0~3. 학습 라벨은 재검토 점수(`code_review`) |
| `region` | 생검 부위 (antrum / body / angle / other) — `--site_mode` 입력 |
| `exclude_recommended` | 학습 제외 여부 |

**데이터 규모.** 파일럿 1,000사례 / WSI 1,023장 (Philips iSyntax, 40x 스캔). 제외 2사례를 뺀 **998사례 / 1,002장**으로 학습한다. 등급별 사례 수 (0 / 1 / 2 / 3):

| HP | 호중구 | 단핵구 | 위축 | 장상피화생 |
|---|---|---|---|---|
| 773 / 184 / 31 / 10 | 244 / 453 / 285 / 16 | 19 / 421 / 425 / 133 | 532 / 208 / 142 / 116 | 504 / 125 / 288 / 81 |

**feature 추출.** 패치 좌표는 배율마다 한 번 만들어 나머지 FM 폴더로 복사하므로 **4개 FM이 같은 패치**를 쓴다.

```bash
python scripts/convert_isyntax_to_tiff.py <in.isyntax> <out.tiff>     # iSyntax → TIFF (무손실)
bash scripts/extract_embeddings.sh                                    # 최초 1회: 조직 분할 → 좌표 → feature
MAG=40 bash scripts/make_coords.sh                                    # 배율별 좌표 생성·공유 (CPU)
ENC="hoptimus0 virchow" MAG=40 GPUS="0 1 2 3" bash scripts/extract_features_fm.sh   # FM × 배율 feature (백그라운드 + 모니터)
SLIDES="<slide_id>" bash scripts/add_slides.sh                        # 신규·재스캔 slide만 증분 처리
```

| 배율 | 패치 수 (1,023장) | slide당 중앙값 (최소~최대) | FM 4개 feature 용량 |
|---|---:|---|---:|
| 10x | 243,099 | 200 (40~1,063) | 6.5 GB |
| 20x | 812,400 | 653 (115~3,801) | 22 GB |
| 40x | 2,909,668 | 2,337 (376~14,244) | 78.7 GB |

## 3. bag 구성과 fold 분할

**bag 1개 = 사례 1개.** 라벨이 사례 단위이므로 slide가 여러 장인 사례는 feature를 이어 붙인다. 패치는 **샘플링 없이 전부** 쓴다 (bag당 115~5,803개). 모든 feature를 처음에 한 번 메모리로 읽어 fold·epoch마다 스토리지를 다시 읽지 않는다.

| `--split` | 방식 | 특징 |
|---|---|---|
| `code` (기본) | 5자리 CODE 문자열로 층화한 5-fold | 논문 방식. CODE가 182종이라 드문 등급이 fold마다 흔들린다 (HP 3등급 test 0·3·3·3·1건) |
| `multilabel` | 항목별 등급 20칸 + 생검 부위 4칸을 함께 맞추는 iterative stratification (Sechidis et al. 2011) | 칸별 평균 어긋남 test 2.03 → 0.41건, HP 3등급 test 2·2·2·2·2 |

fold당 train 698 / val 100 / test 200 (70 / 10 / 20).

**고정 공유 split (paired 비교).** 분할은 라벨로만 정해지므로 FM·배율이 달라도 같은 fold다. 결과 폴더의 `splits.json`으로 고정되어, method A vs B를 fold별로 짝지어 비교할 수 있다. `multilabel`은 `random(seed)`만 써서 서버·numpy 버전이 달라도 같은 분할이 나온다.

```bash
python -m src.training.split_report --run_dir outputs/<run>      # fold × train/val/test × 항목 등급·부위 건수
```

## 4. 학습 / 평가

```bash
# ---- 학습 (5-fold CV → fold별 test → MLflow 업로드) ----
VERSION=0.1.0 ENCODERS=hoptimus0 GPU=0 CPUS=0-15 bash scripts/run_sydney_mtl.sh
VERSION=0.1.0 ENCODERS="hibou_l hoptimus0 virchow midnight12k" GPU=0 CPUS=0-15 bash scripts/run_sydney_mtl.sh   # FM 비교
VERSION=0.1.0 ENCODERS=hoptimus0 MAG=40 GPU=0 CPUS=0-15 bash scripts/run_sydney_mtl.sh                          # 배율 비교
VERSION=0.2.2 ENCODERS=hoptimus0 GPU=0 CPUS=0-15 EXTRA_ARGS="--site_mode head --site_lr 1e-2" bash scripts/run_sydney_mtl.sh

bash scripts/watch_sydney.sh                 # 실시간 모니터 (Ctrl+C는 모니터만 닫는다)

# ---- 직접 실행 ----
python -m src.training.train_sydney_mtl --version 0.1.0 \
  --meta_csv metadata/sydney_meta.csv --label_col code_review --loss la --gpu 0 \
  --feat_dir <EMB_ROOT>/hibou_l/20x_224px_0px_overlap/features_hibou_l

# ---- 비교 · 해석 ----
python -m src.training.compare_sydney_mtl --version 0.1.0 --mag 20          # FM 비교 그림 + pooled 지표 + 앙상블
python -m src.training.compare_site_sydney --versions 0.1.0 0.2.2 0.2.3     # 생검 부위 입력 비교
python -m src.training.heatmap_sydney_mtl --run_dir outputs/<run>           # 항목별 attention heatmap
```

**Loss 선택.** 기본은 항목별 **logit-adjusted CE** (`la`, τ = 0.75)의 합이다: `CE(logit + τ·log prior)`. prior는 그 fold 학습 라벨의 등급 비율이고, 예측은 보정 전 logit으로 해서 드문 등급이 덜 묻힌다. 학습 fold에 없는 등급은 예측에서 제외한다. 비교용 `ce`, `wce`(batch 1에서도 가중치가 약분되지 않게 표본별로 곱함).

**학습 설정.** Adam lr 1e-4, batch 1 × gradient accumulation 8, 최대 100 epoch. validation total loss가 10 epoch 연속 안 줄면 멈추고 최저 epoch 가중치로 test를 평가한다.

**지표.** 주지표는 **QWK**(quadratic weighted kappa)와 **lenient accuracy**(0등급은 정확히, 1~3등급은 이웃 등급까지 정답). 보조로 accuracy · MAE · balanced accuracy · macro F1 · AUC · `code_exact`(CODE 5자리 완전 일치율).

**생검 부위 입력 (논문에 없는 확장).** `head`는 항목별 logit에 부위·등급별 보정값을 더하고, `film`은 attention 전에 패치 feature를 `h·(1+γ)+β`로 조정한다. 보정값은 0에서 시작하고 "부위 모름"은 0으로 고정한다 → 부위 없이도 같은 모델로 추론된다. `--site_dropout`으로 학습 사례 일부를 부위 모름으로 넣고, `--site_lr`로 부위 파라미터만 학습률을 따로 준다. test는 부위를 넣은 결과와 부위 모름 결과를 둘 다 남긴다.

### 실험 결과 (Results)

998사례 / 1,002장, 5-fold CV(seed 42, 전 실험 공유 `splits.json`), test fold별 mean ± std. (**bold** = 열 최고)

**FM 비교 — v0.1.0 (논문 재현 설정), 20x.** 항목별 QWK.

| Encoder | HP | 호중구 | 단핵구 | 위축 | 장상피화생 | 평균 QWK | lenient acc | code_exact |
|---|---|---|---|---|---|---|---|---:|
| Hibou-L | 0.568 ± 0.024 | 0.548 ± 0.028 | 0.627 ± 0.013 | 0.291 ± 0.038 | 0.879 ± 0.015 | 0.582 ± 0.010 | 0.786 ± 0.017 | 0.075 |
| H-Optimus-0 | 0.573 ± 0.049 | **0.558** ± 0.054 | 0.642 ± 0.011 | **0.335** ± 0.058 | **0.902** ± 0.010 | **0.602** ± 0.016 | **0.802** ± 0.012 | **0.098** |
| Virchow | 0.566 ± 0.062 | 0.544 ± 0.042 | 0.637 ± 0.020 | 0.271 ± 0.023 | 0.892 ± 0.009 | 0.582 ± 0.012 | 0.786 ± 0.009 | 0.089 |
| Midnight-12k | **0.592** ± 0.051 | 0.524 ± 0.041 | **0.666** ± 0.019 | 0.333 ± 0.077 | 0.880 ± 0.004 | 0.599 ± 0.021 | 0.793 ± 0.013 | 0.097 |

**배율 비교 — v0.1.0, 평균 QWK.** `[n/5]` = 40x가 20x보다 높았던 fold 수.

| Encoder | 10x | 20x | 40x | 40x − 20x (fold 짝 비교) |
|---|---|---|---|---|
| H-Optimus-0 | **0.608** ± 0.020 | **0.602** ± 0.016 | **0.600** ± 0.014 | −0.002 [2/5] |
| Midnight-12k | 0.596 ± 0.022 | 0.599 ± 0.021 | 0.583 ± 0.015 | −0.015 [0/5] |
| Hibou-L | 0.575 ± 0.006 | 0.582 ± 0.010 | 0.594 ± 0.018 | +0.012 [4/5] |
| Virchow | 0.579 ± 0.018 | 0.582 ± 0.012 | 0.590 ± 0.018 | +0.008 [4/5] |

**생검 부위 입력 — 20x.** v0.1.0 → v0.2.2 (`head`) / v0.2.3 (`film`), `--site_lr 1e-2`. `[n/5]` = v0.1.0보다 높았던 fold 수.

| Encoder | 평균 QWK | 위축 QWK | 위축 fold 짝 비교 (head / film) |
|---|---|---|---|
| Hibou-L | 0.582 → 0.575 / 0.571 | 0.291 → 0.288 / 0.267 | −0.003 [3/5] / −0.024 [1/5] |
| H-Optimus-0 | 0.602 → 0.601 / 0.595 | 0.335 → 0.318 / 0.302 | −0.017 [2/5] / −0.033 [0/5] |
| Virchow | 0.582 → 0.594 / 0.601 | 0.271 → 0.315 / 0.318 | +0.044 [4/5] / +0.047 [4/5] |
| Midnight-12k | 0.599 → 0.593 / 0.582 | 0.333 → 0.344 / 0.321 | +0.011 [3/5] / −0.012 [2/5] |

- **배율은 결과를 거의 바꾸지 않는다.** 세 배율의 차이는 ±0.02 안이고 방향이 FM마다 다르다. 10x는 패치가 20x의 30%(학습 FM당 약 10분), 40x는 3.6배(FM당 25~40분)이므로 성능이 같다면 10x·20x가 실용적이다.
- **FM 차이도 작다.** H-Optimus-0이 20x에서 Hibou-L·Virchow보다 5개 fold 모두 높지만(+0.02) Midnight-12k와는 사실상 동률이다. 4개 FM 확률 평균 앙상블(998사례 pooled 평균 QWK 10x 0.609 / 20x 0.608 / 40x 0.606)도 단일 최고 모델 수준이다.
- **항목 간 난이도 차이가 FM·배율 차이보다 훨씬 크다.** 장상피화생 0.88~0.90 / 단핵구 0.63~0.67 / HP·호중구 0.52~0.59 / 위축 0.27~0.34.
- **부위 입력은 일관된 개선을 만들지 못했다.** 모델의 위축 QWK가 부위별로 크게 달라(전정부 0.28 · 체부 0.50 · 각부 0.10) 부위를 입력에 넣었지만, Virchow만 5개 중 4개 fold에서 좋아졌다. 첫 시도(v0.2.0 / v0.2.1)에서는 0에서 시작한 보정값이 lr 1e-4로 거의 움직이지 않아(최대 0.04) 부위 파라미터의 학습률을 분리했고, 그 값은 test가 아니라 fold 1의 validation으로 골랐다.
- **위축의 실제 문제는 2·3등급 경계다.** 부위와 상관없이 2등급을 거의 내지 않고 3등급으로 간다 (H-Optimus-0 예측 분포 [558, 170, 56, 214] vs 정답 [532, 208, 142, 116]) → 다음 후보는 순서형(ordinal) 손실.
- **logit adjustment는 항목마다 효과가 다르다.** 학습 없이 예측에 train prior를 되돌려 보면 위축 QWK는 0.338 → 0.175로 나빠지고 HP는 0.575 → 0.612로 오른다 → 항목별 τ 조정 후보.
- 0.01~0.02 수준의 차이는 시드 반복 없이는 개선으로 판단하지 않는다.

## 5. 모델 · 옵션 선택

**SydneyMTL** (`src/models/sydney_mtl.py`)

```text
bag: patch feature [N, D]
   │
   shared adaptor   Linear(D→256) → GELU → Linear(256→256)              5개 항목 공유
   │   └─ (film) 부위별 h·(1+γ)+β
   ├─ HP    attention Linear(256→1) → softmax → 가중합 → Linear(256→4)   항목마다 독립
   ├─ NEU   …
   ├─ MONO  …
   ├─ ATR   …
   └─ IM    …   └─ (head) 부위·등급별 logit 보정
   예측: 항목별 argmax → 5자리 Sydney CODE
```

| 옵션 | 값 | 설명 |
|---|---|---|
| `--site_mode` | `none` · `head` · `film` | 생검 부위 입력 방식 (§4) |
| `--site_dropout` · `--site_lr` | 예: `0.2` · `1e-2` | 부위 모름 비율 / 부위 파라미터 전용 학습률 |
| `--loss` · `--tau` | `la` · `0.75` | logit adjustment 온도 |
| `--adaptor_dim` | `256` | shared adaptor 크기 |
| `--split` | `code` · `multilabel` | fold 분할 방식 (§3) |
| `--folds` | 예: `1` | 일부 fold만 (빠른 확인용, MLflow 업로드 안 함) |

**patch encoder** (모두 TRIDENT `--patch_size 224`)

| Encoder | `ENCODERS` | HF 모델 | 차원 | feature |
|---|---|---|---:|---|
| Hibou-L | `hibou_l` | `histai/hibou-L` | 1024 | pooler 출력 |
| H-Optimus-0 | `hoptimus0` | `bioptimus/H-optimus-0` | 1536 | CLS |
| Virchow | `virchow` | `paige-ai/Virchow` | 2560 | CLS + 패치 평균 |
| Midnight-12k | `midnight12k` | `kaiko-ai/midnight` | 1536 | CLS |

**출처 (references)**

| 구성 | 논문 | 연도 | venue |
|---|---|---|---|
| SydneyMTL (모델 · 손실 · 학습 설정) | Kim et al., [doi:10.1111/his.70258](https://doi.org/10.1111/his.70258) · [SMF-AI/SydneyMTL](https://github.com/SMF-AI/SydneyMTL) | 2026 | Histopathology |
| logit adjustment | Menon et al., Long-tail learning via logit adjustment | 2021 | ICLR |
| iterative stratification | Sechidis et al., On the stratification of multi-label data | 2011 | ECML PKDD |
| FiLM | Perez et al., FiLM: Visual Reasoning with a General Conditioning Layer | 2018 | AAAI |
| 조직 분할 · 패치 feature | [mahmoodlab/TRIDENT](https://github.com/mahmoodlab/TRIDENT) (commit `e4c98b2`) | — | — |

## 6. 출력 레이아웃 · 실험 관리

```text
outputs/
  sydney_mtl_<fm>_<mag>x_v<x.y.z>/
    config.json                 실행 인자 + FM · 배율 · in_dim · 사례 수 · meta CSV / 코드 md5
    splits.json                 fold별 train / val / test case_id
    train.log                   등급 분포, epoch별 val 지표, fold별 test, 최종 요약
    code/                       학습에 쓴 코드 사본
    fold{k}/history.csv  curves.png  confusion.png  metrics.json  test_predictions.csv  best_model.pt
    summary.csv / summary.json  fold별 test 지표 + 평균 · 표준편차
    summary_metrics.png  confusion_all.png
    heatmaps/                   attention heatmap (별도 실행)
  compare_v<ver>_<mag>x/        FM 비교: metrics · grade_recall · error_size · confusion_compare · pooled_metrics.csv
  launch_logs/                  실행 기록
```

- **버전 규칙**: 같은 버전 번호 = 같은 실험 설정. FM·배율만 바꿔 같은 번호로 돌리면 그대로 FM·배율 비교가 된다.
- **설정 불일치 차단**: 이미 있는 버전 폴더에 설정·라벨(meta CSV md5)·코드(md5)가 다른 실행을 하려 하면 학습 스크립트가 멈춘다 → 버전을 올린다. 같은 설정이면 끝난 fold는 건너뛰고 이어서 학습한다.
- **MLflow**: experiment `sydney`. parent run(설정 · 요약 지표 · 결과 폴더 전체) + fold별 child run. run 설명 맨 위에 기준 버전과 다른 점 / 이유 / 비교 방법이 들어간다 (`mlflow_sydney.py`의 `VERSION_NOTES`).

## 7. 학습 편의 기능

- **백그라운드 실행 + 모니터**: 학습·추출을 `setsid nohup`으로 띄우고 실시간 모니터를 연다. 시작할 때 실행 기록 파일과 중단 명령(`kill -- -<pgid>`)을 출력한다.
- **중복 실행 차단**: 같은 FM·버전(학습) 또는 같은 FM·배율(추출)이 이미 돌고 있으면 띄우지 않는다.
- **resume**: 학습은 끝난 fold를, 추출은 끝난 slide를 건너뛴다.
- **사전 점검**: 임베딩 폴더와 학습 패키지를 띄우기 전에 확인한다 (백그라운드에서는 실패가 화면에 보이지 않기 때문).
- **서버 프로필**: `HOST_PROFILE=large|small`로 GPU 메모리에 맞는 기본 GPU·CPU·배치를 고른다. 10GB GPU에서 ViT-g(H-Optimus-0, Midnight)는 배치가 크면 slide마다 OOM이 나는데 `--skip_errors` 때문에 멈추지 않고 실패만 쌓인다.
- **GPU 번호 고정**: `CUDA_DEVICE_ORDER=PCI_BUS_ID` (GPU 종류가 섞인 서버). CPU는 `taskset`으로 묶는다.
- **reproducibility**: seed 42 고정, 학습 코드 사본과 md5를 결과에 보존.
- **분석 도구**: FM 비교(등급별 재현율 · 오차 크기 · 혼동행렬 · 확률 평균 앙상블), 부위별 비교, attention heatmap + 상위 패치, fold 구성 리포트.

## 테스트

```bash
bash tests/smoke_test.sh            # 가짜 임베딩: split + 5개 MIL × (이진, 3클래스) 3 epoch
bash tests/smoke_test_sydney.sh     # 가짜 Sydney CODE: split + multi-task ABMIL (ce, coral) 3 epoch
# SydneyMTL 빠른 확인 (fold 1개, 2 epoch, MLflow 없이)
VERSION=0.0.1 OUTPUT_ROOT=/tmp/sydney_smoke WATCH=0 GPU=0 CPUS=0-7 EXTRA_ARGS="--epochs 2 --folds 1 --no_mlflow" \
    bash scripts/run_sydney_mtl.sh
```

`src/training/train.py` · `train_sydney.py` · `src/models/multitask_abmil.py`는 라벨 확정 전에 만든 범용 MIL / multi-task ABMIL(gated attention, `ce`·`coral` head) 코드로, 참고용으로 남겼다.

> WSI · feature · 라벨 메타데이터 · 학습 결과 폴더는 이 저장소에 포함하지 않는다. 경로는 모두 `/path/to/...` 플레이스홀더다.
