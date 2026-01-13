## RDI-Net (Residual Defect Insertion) – MVTec AD (Few-Defect, Imbalanced) Experiments

이 프로젝트는 **극단적 클래스 불균형(정상 다수, 결함 극소수 K-shot)** 상황에서, 정상 이미지에 결함을 **삽입/조화(harmonize)**하는 생성 모델(**RDI-Net**)을 제안하고,
합성 결함 증강이 **분류기 기반 anomaly detection(AUROC)** 성능을 얼마나 개선하는지 실험합니다.

### 고정 설정 (논문 프로토콜)
- **Dataset**: MVTec AD
- **Categories (5)**: `bottle, cable, capsule, hazelnut, metal_nut`
- **Resolution**: 256
- **Few-defect**: 결함 K-shot = {1, 5, 10}
- **Downstream**: ResNet-18 binary classifier → image-level AUROC
- **Extra baselines**: Copy-Paste / Poisson blending / PatchCore(정상만)
- **Synthetic bias check**: real defect patch vs synthetic defect patch 구분 가능성 측정

### 환경
macOS에서 **PyTorch(MPS)**로 돌아가도록 구성합니다.
현재 워크스페이스는 로컬 패키지 디렉토리(`.pydeps312`)를 사용합니다.

> 주의: 이 레포는 Homebrew의 `python3.12`를 기준으로 동작합니다. (현재 `python3`는 3.14라서 torch 실행이 불안정할 수 있음)

### 다음 단계
코드/스크립트가 추가되면, 아래 커맨드로 재현 가능하게 할 예정입니다.
- 데이터 다운로드/전처리
- re-split 생성(K-shot)
- RDI-Net 학습 → 합성 생성
- ResNet-18 학습/평가(AUROC)
- PatchCore 비교

### 데이터셋 다운로드 (중요)
MVTec AD는 공식 사이트에서 **다운로드 폼(외부 임베드)**을 통해 제공되는 경우가 있어, 스크립트로 직접 URL 다운로드가 실패할 수 있습니다.

- **권장 방식**: MVTec 공식 페이지에서 `MVTec AD`를 다운로드한 뒤, 아래 경로에 파일을 놓습니다.
  - `data/raw/mvtec_anomaly_detection.tar.xz`
- 그 다음, 스크립트로 압축 해제/구조 검증을 수행합니다.

선택적으로 **15개 전체를 다 풀지 않고**, 논문에서 쓰는 5개 카테고리만 추출할 수 있습니다(디스크/시간 절약):

```bash
/opt/homebrew/bin/python3.12 /Users/levit/paper3/scripts/extract_mvtec.py \
  --tar /Users/levit/paper3/data/raw/mvtec_anomaly_detection.tar.xz \
  --out /Users/levit/paper3/data/processed/mvtec_ad \
  --categories bottle cable capsule hazelnut metal_nut
```



