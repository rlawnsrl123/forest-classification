# 작업 노트 — Colab / GEE 조작 레퍼런스

산림 유형 분류 프로젝트 진행 중 반복해서 쓰는 조작법과 개념 정리.
`PROJECT.md`가 "무엇을 하는가"라면 이 문서는 "어떻게 다루는가".

---

## 1. geemap 지도 조작

| 하고 싶은 것 | 방법 |
|---|---|
| 레이어 켜기/끄기 | 지도 우측 상단 **겹친 사각형 아이콘에 마우스 올리기**(hover) → 체크박스 |
| 투명도 조절 | 같은 패널의 슬라이더 |
| 줌 | 마우스 휠, 또는 좌측 `+` / `-` |
| AOI에 자동 맞춤 | 코드에서 `m.centerObject(AOI, 12)` — 숫자가 줌 레벨 |
| 거리 재기 | 좌측 자 아이콘 |
| 좌표 확인 | 지도 위에서 마우스 이동 시 하단 표시 |

레이어는 **나중에 추가한 것이 위로** 쌓인다. `addLayer` 순서 = 아래에서 위.

### 시각화 파라미터

```python
rgb = {'bands': ['B4','B3','B2'], 'min': 0.02, 'max': 0.25}   # 사람 눈
nir = {'bands': ['B8','B4','B3'], 'min': 0.02, 'max': 0.40}   # 근적외선 강조
```

- `min`을 올리면 어두운 부분이 걷히고, `max`를 내리면 전체가 밝아진다
- 겨울 영상은 태양고도가 낮아 어둡다 → `max`를 0.30 정도로 낮출 것

### NIR 합성 판독법

| 색 | 정체 |
|---|---|
| 선홍색 | 건강한 식생 (여름엔 대부분의 산림) |
| 회청색 | 시가지, 도로, 나지 |
| 검정 | 물 (대청호, 갑천) — NIR 반사 거의 없음 |
| 흰색 점 | 잔설, 잔여 구름 |

여름과 낙엽기를 번갈아 켜면 **붉은 영역이 줄어드는 곳 = 활엽수림**,
**붉게 남는 곳 = 침엽수림**.

---

## 2. Colab 실행 규칙

### 단축키

| 키 | 동작 |
|---|---|
| `Shift + Enter` | 실행 후 다음 셀로 |
| `Ctrl + Enter` | 실행 후 제자리 |
| `Ctrl + M` → `B` | 아래에 셀 추가 |
| `Ctrl + M` → `D` | 셀 삭제 |

### 변수는 런타임에 살아있다

이미 실행한 셀을 다시 돌릴 필요 없음. 런타임이 유지되는 동안
`ee`, `gdf`, `summer_dj` 등은 메모리에 남아있다.

**`NameError: name 'X' is not defined`**
→ 문법 오류가 아니라 **그 변수를 만든 셀을 안 돌렸다**는 뜻.
런타임 재시작(주로 `pip install` 직후)이 원인인 경우가 대부분.
셀 A(셋업 셀)를 다시 실행하면 복구된다.

### 상태 점검 한 줄

```python
for n in ['ee', 'gdf', 'cloud_mask', 'composite_aoi', 'AOI_DJ']:
    print(f'{n:15s}', '있음' if n in globals() else '없음')
```

### 하지 말 것

- **`런타임 → 모두 실행` 금지.** export 셀이 다시 돌아 같은 작업이 중복 제출되고
  EECU 할당량을 두 배로 소모한다.
- Gemini 자동 수정 받지 말 것. 실행 순서 문제를 코드 문제로 오진한다.

---

## 3. GEE 동작 원리

### 지연 계산 (lazy evaluation)

`composite_aoi(...)` 를 호출해도 **그 자리에서 계산되지 않는다.**
"이렇게 계산하라"는 설계도만 만들어질 뿐.
실제 계산은 다음 순간에만 일어난다:

- `.getInfo()` 호출 시
- 지도에 그릴 때 (화면에 보이는 타일만)
- export 실행 시

그래서 무거워 보이는 셀이 1초에 끝나는 게 정상이다.

### export는 서버에서 돈다

```
task.start()   →  READY  →  RUNNING  →  COMPLETED
   (즉시 반환)      (대기)     (처리)      (Drive에 파일)
```

- 코드는 제출만 하고 바로 끝난다. 20~30분은 서버 처리 시간
- 노트북을 닫아도 작업은 계속된다
- 진행 상황: `code.earthengine.google.com` → 우측 **Tasks** 탭
- 파일은 `COMPLETED` 이후에만 Drive에 나타난다

한 번 찍기:

```python
print(task.status()['state'])
```

계속 지켜보기 (별도 셀):

```python
import time
while True:
    s = task.status(); st = s['state']
    print(time.strftime('%H:%M:%S'), st)
    if st in ('COMPLETED','FAILED','CANCELLED'):
        if st == 'FAILED': print('원인:', s.get('error_message'))
        break
    time.sleep(30)
```

취소:

```python
task.cancel()
```

### export 주의사항

- `folder=` 는 **Drive 최상위에만** 생성된다. 하위 경로 지정 불가.
  나중에 수동으로 `forest/data/s2/` 로 옮길 것
- 파일이 크면 자동으로 타일 분할된다
  (`...-0000000000-0000000000.tif`) — 정상
- `crs='EPSG:5179'` 를 지정해 임상도와 좌표계를 미리 맞춘다.
  이렇게 하면 나중에 재투영이 필요 없다

---

## 4. 할당량 (커뮤니티 등급)

- EECU 150시간
- 무거운 학습은 전부 Colab GPU에서 → GEE는 합성·export만 담당하므로 여유
- 같은 export를 중복 제출하지 않는 것이 절약의 핵심
- 부족해지면 참여자 등급(1,000시간)으로 변경 가능. 단 결제 계정 필요

---

## 5. 자주 만나는 에러

| 증상 | 원인 / 해결 |
|---|---|
| `NameError: 'ee' is not defined` | 셋업 셀 미실행 → 셀 A 재실행 |
| `EEException: not signed up` | `ee.Initialize(project=...)` 에 Project ID 누락 |
| export `FAILED` | `maxPixels` 초과 → 값을 키우거나 AOI 축소 |
| 지도에 아무것도 안 뜸 | `min`/`max` 범위가 데이터와 안 맞음 → 히스토그램 확인 |
| shapefile 컬럼이 비어있음 | `.dbf` 파일 누락. shp/shx/dbf/prj 는 한 세트 |
| 합성 영상에 구멍 | 해당 시기 영상 부족 → 기간 확대 또는 `cloud_pct` 완화 |

---

## 6. 파일 위치

```
Google Drive/
└── forest/
    ├── notebooks/          Colab 노트북
    ├── data/
    │   ├── imsang/2025_daejeon/   30.shp .dbf .shx .prj .sbn .sbx
    │   └── s2/                    GEE export 결과 (수동 이동)
    └── outputs/

D:\cnu\forest-classification\      git repo, Claude Code CLI 작업 위치
├── PROJECT.md                     프로젝트 정의·결정사항
└── NOTES.md                       이 문서
```
