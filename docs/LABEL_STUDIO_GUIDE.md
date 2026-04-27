# Label Studio 초기 수동 라벨링 가이드

## 왜 Label Studio로 수동 seed부터 시작?

이전 파이프라인의 heuristic은 다음을 오탐했습니다:
- 마른 잔디를 패널로 (HSV gray 범위 겹침)
- 방수포를 패널로 (blue 범위 겹침)
- String 중간의 module gap에서 분리

이런 오류는 어떤 자동 pre-labeling 방법으로도 완전히 해결되지 않습니다.
**가장 빠른 해결법: 5-20장을 정확히 수동 라벨링 → YOLO11n 부트스트랩 → 자동으로 확산.**

## Label Studio 설정

### 1. 설치 및 실행

```bash
pip install label-studio
label-studio start --port 8080
```

### 2. 프로젝트 생성

새 프로젝트 → Labeling Setup → **Custom Template** 선택 후 아래 XML 붙여넣기:

```xml
<View>
  <Header value="PV String &amp; Module Labeling"/>
  <Text name="instructions" value="
    1. 먼저 STRING 전체(세로 긴 줄 1개)를 큰 bbox로 감싸세요.
    2. 그 다음 MODULE(string 내부의 개별 패널)을 작은 bbox로 구분하세요.
    3. 잔디/방수포/건물은 절대 라벨링하지 마세요.
    4. String 경계는 프레임(금속 테두리) 기준으로 tight하게.
  "/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" 
         rotateControl="false"/>
  <RectangleLabels name="label" toName="image">
    <Label value="pv_string" background="#0066CC"
           hotkey="1"/>
    <Label value="pv_module" background="#00CC66"
           hotkey="2"/>
  </RectangleLabels>
</View>
```

### 3. 로컬 이미지 연결

```bash
# 환경변수로 로컬 파일 접근 허용
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/absolute/path/to/images
label-studio start
```

프로젝트 Settings → Cloud Storage → Add Source Storage → Local Files.

## 수동 라벨링 베스트 프랙티스 (업로드 이미지 분석 기반)

### ✅ 올바른 라벨링

**String 라벨**:
- 패널 프레임(금속 테두리) 기준, 지면/그림자 제외
- 세로로 연속된 패널은 **1개 string**으로 통합
- Aspect ratio 약 5:1 ~ 15:1 (매우 길쭉해야 정상)

**Module 라벨**:
- String 내부에서 어두운 가로 프레임 선을 경계로 분할
- 보통 string 하나에 3-10개 module
- 너무 작게(cell 단위) 쪼개지 말 것 — module은 60/72 cell 전체

### ❌ 피해야 할 패턴 (이전 오탐 사례)

업로드 이미지에서 관찰된 문제:
- ❌ 갈색 잔디 영역 → 라벨 안 함
- ❌ 검은 방수포(tarp) → 라벨 안 함
- ❌ String 중간에 억지로 경계 만들기 → 1개 string으로 통합
- ❌ 끝이 잘린 패널 → 잘린 부분까지만 bbox (extrapolate 금지)

## Seed 목표

| 데이터 규모 | Seed 수량 | 예상 소요 시간 | 초기 mAP |
|------------|----------|---------------|----------|
| 100장 | 10-15장 | 1-2시간 | 0.5-0.6 |
| 500장 | 20-30장 | 3-4시간 | 0.6-0.7 |
| 2000장+ | 50장 | 6-8시간 | 0.7+ |

## Export → active_learning.py seed 사용

Label Studio에서 작업 완료 후:
1. Export → **YOLO** 형식 선택
2. 압축 해제 시 `labels/` 디렉토리에 `.txt` 파일들이 있음
3. 다음 명령으로 부트스트랩:

```bash
python active_learning.py seed \
    --images /path/to/drone_images \
    --seed-labels /path/to/exported/labels \
    --output workspace/round_0 \
    --classes pv_string pv_module \
    --epochs 50 \
    --imgsz 1920
```

## 이후 Active Learning 루프

```bash
# Round 1: 현재 모델로 unlabeled 예측 + top-20 uncertain 선별
python active_learning.py iterate \
    --images /path/to/drone_images \
    --model workspace/round_0/weights/best.pt \
    --output workspace/round_1 \
    --select-top 20

# → workspace/round_1/human_review/ 에 20장의 예측 라벨 생성됨
# → Label Studio로 가져와 수정
# → 수정된 라벨 + seed 라벨을 합쳐 다시 seed
```

일반적으로 3-5 round 반복하면 mAP 0.85+ 달성 가능합니다.
