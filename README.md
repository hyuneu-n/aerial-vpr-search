# Aerial VPR Search (DINOv3 + FAISS)

GNSS-denied 환경에서 항공 영상 기반 위치 인식(Visual Place Recognition)을 수행하는
검색·매칭(search) 코드입니다. 아래 논문에서 제안한 파이프라인의 **검색 단계 구현**이며,
C#(.NET)과 Python 두 가지로 제공됩니다.

> 서현은, 장인성, "GNSS-Denied 환경의 UAM 위치 인식을 위한 항공 영상 검색·매칭 시스템,"
> 대한공간정보학회 2026 춘계학술대회, pp. 171-174.

## 이 저장소에 포함된 것 / 안 된 것

**포함:** DINOv3 특징 추출(추론만) → FAISS 인덱스 로드·검색(Top-k) → 위경도 기반
Hit@1/Hit@5 검증 로직. 논문 Fig.1 파이프라인의 오른쪽 절반(레퍼런스 인덱스가
이미 구축되어 있다고 가정한 상태에서의 쿼리 검색)입니다.

**미포함 (직접 준비해야 함):**
- 갤러리 이미지로부터 FAISS 인덱스를 만드는 **빌드 코드**
- 학습 데이터, 항공 영상, JSON 메타데이터
- DINOv3 ONNX 모델 가중치(`model.onnx`)
- FAISS 네이티브 라이브러리(`faiss_c.dll` / `libfaiss_c.so`)
- 사전 빌드된 `.index` / `.txt` 인덱스 파일

## 구조

```
csharp/                  # .NET 9, Windows, ONNX Runtime GPU
  Program.cs             # check / search 두 모드만 존재 (build 없음)
  FeatureExtractor.cs    # 단일 이미지 -> DINOv3 CLS 1024차원 벡터 (추론 전용)
  FaissNative.cs         # FAISS C API 중 인덱스 로드/검색/파라미터 설정만 래핑
  VprSearch.csproj / .sln

python/                  # Windows / Linux / Jetson 겸용
  main.py                # 실험 러너: 인덱스 타입·공간 필터링 전략 비교 (논문 표 1 재현)
  models/
    dino_extractor.py    # DINOv3 ONNX 추론 (Jetson/CUDA/CPU 자동 분기)
    faiss_index.py        # FAISS 인덱스 로드·검색 (Jetson C-API / Python 바인딩 자동 분기)
  requirements.txt
```

## 두 구현의 차이

Python(`main.py`)에는 논문의 핵심 기법인 **이중 필터링(고도+방향)** 전략 4종
(필터 없음 / 고도 필터 / 방향 필터 / 이중 필터, `select_engines_for_query`)이
그대로 구현되어 있어 표 1의 실험을 재현할 수 있습니다.

C#(`Program.cs`)은 사용자가 지정한 **단일 인덱스 파일**을 검색하는 baseline만
제공합니다(필터 없음 경우에 해당). 고도/방향별로 이미 분리된 인덱스를 만들어두고
그중 하나를 골라 실행하는 방식으로 표 1의 각 행을 재현할 수 있지만, 필터링
선택 로직 자체는 자동화되어 있지 않습니다.

## 필요한 준비물

1. **모델**: DINOv3 ViT-L/16을 SAT-493M 정규화(mean `[0.430, 0.411, 0.296]`,
   std `[0.213, 0.156, 0.143]`)로 ONNX 내보내기한 `model.onnx`. CLS 토큰(1024차원)을
   사용합니다.
2. **인덱스**: 위 모델로 레퍼런스 갤러리를 인코딩해 만든 FAISS `Flat` 인덱스
   (`.index`)와, 인덱스 순서와 1:1 대응하는 파일명 목록(`.txt`).
3. **네이티브 라이브러리**: C# 쪽은 `faiss_c.dll`을 실행 파일 옆에 둬야 합니다.
   Python 쪽은 `libs/faiss_c.dll`(Windows) 또는 `/app/libs/libfaiss_c.so`(Jetson)가
   있으면 C-API 경로를, 없으면 `pip install faiss-gpu`/`faiss-cpu` 바인딩을 씁니다.
4. **메타데이터 JSON** (`--ref-json`, `--query-json` 또는 Python의 쿼리 폴더 내 JSON):

```json
{
  "images": [
    {
      "imageref": "파일명.jpg",
      "position": { "latitude": 0.0, "longitude": 0.0, "altitude": 300.0 },
      "attitude": { "yaw": 0.0 }
    }
  ]
}
```

## 사용법

### C#

```powershell
dotnet build -c Release
dotnet run -c Release -- check
dotnet run -c Release -- search `
    --query "TestData\query" `
    --query-json "TestData\query\query_labels.json" `
    --ref-json "dino_temp\ref_master.json" `
    --index "dino_temp\600_Flat.index"
```

### Python

```bash
pip install -r requirements.txt   # + faiss-gpu 또는 faiss-cpu (환경에 맞게)
python main.py
# 실행 후 프롬프트에서 실험 번호(1~14) 선택 — 10~14번이 논문 표 1의 공간 필터링 전략
```

`database/ref/`에 `.index`/`.txt`(및 분할 검색용 `Flat_<고도>_<yaw>.index`),
`test_images/`에 쿼리 이미지+JSON, `model/model.onnx`에 모델을 넣어두면 됩니다.

## 결과 지표

Hit@1 / Hit@k: FAISS Top-1(또는 Top-k) 검색 결과가 실제 위경도(Haversine) 기준
Top-5 정답 집합에 포함되는 비율. 결과는 `result/`(C#) 또는 `output/`(Python) 아래
CSV로 저장됩니다.
