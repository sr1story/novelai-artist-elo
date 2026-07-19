# 개인용 모바일 배포

이 구성은 PC를 계속 켜 두지 않고 휴대폰 브라우저에서 Artist ELO를 사용하기 위한 것입니다. 앱은 한 명이 사용하는 것을 전제로 하며, 로그인 계정과 모든 랭킹 데이터는 하나만 존재합니다.

## 준비할 것

- NovelAI 구독과 Persistent API Token
- GitHub 계정
- Docker 웹 서비스를 실행할 클라우드 계정

토큰과 로그인 암호는 저장소나 채팅에 입력하지 않습니다. 배포 서비스가 제공하는 **Secret environment variable** 입력란에만 등록하세요.

## 휴대폰에서 Render로 배포

저장소 루트의 `render.yaml`은 Docker 웹 서비스, 싱가포르 리전, 5GB 영구 디스크를 한 번에 생성합니다. 영구 디스크는 재배포 후에도 ELO 기록과 생성 이미지를 보존하지만 Render의 유료 웹 서비스가 필요합니다.

1. 이 저장소를 본인의 GitHub 저장소로 Fork합니다.
2. [Render Dashboard](https://dashboard.render.com/)에 로그인합니다.
3. **New → Blueprint**를 선택하고 Fork한 저장소를 연결합니다.
4. Render가 요청하는 비밀값을 입력합니다.
   - `NOVELAI_API_KEY`: NovelAI Persistent API Token
   - `APP_PASSWORD`: 모바일 웹 로그인 암호. 8자 이상을 사용합니다.
5. 생성될 유료 서비스와 디스크 요금을 Render 화면에서 확인한 다음 배포를 승인합니다.
6. 배포가 완료되면 제공되는 `onrender.com` 주소를 엽니다.
7. 사용자 이름 `artist`와 위에서 설정한 `APP_PASSWORD`로 로그인합니다.

첫 화면에는 이미지 두 장을 생성하므로 NovelAI 사용량이 발생합니다. 같은 비교 화면에서 단순 새로고침하는 경우에는 저장된 이미지를 다시 불러오며 새 이미지를 만들지 않습니다.

## 홈 화면에 추가

- Android Chrome: 메뉴 → **홈 화면에 추가**
- iPhone Safari: 공유 버튼 → **홈 화면에 추가**

설치 후에는 주소창이 없는 PWA 형태로 실행되며, 앱 기능과 데이터는 서버에서 처리됩니다.

## 필수 환경변수

| 이름 | 값 | 용도 |
|---|---|---|
| `NOVELAI_API_KEY` | 비밀값 | NovelAI 이미지 생성 |
| `APP_PASSWORD` | 비밀값, 8자 이상 | 개인 웹 로그인 |
| `APP_USERNAME` | `artist` | 로그인 사용자 이름 |
| `DATA_DIR` | `/data` | 영구 디스크 저장 경로 |
| `SERVER_HOST` | `0.0.0.0` | 클라우드 외부 접속 |

`NOVELAI_API_KEY`와 `APP_PASSWORD`는 GitHub에 커밋하지 마세요.

## 데이터 보존

다음 항목은 `/data` 아래에 저장됩니다.

- ELO 랭킹
- 비교 기록
- 활성 작가 풀
- 최근 비교 상태
- 프롬프트·이미지 설정 프리셋 10개
- 생성 이미지와 CSV 내보내기

영구 디스크가 없는 호스트에서는 재시작이나 재배포 시 이 데이터가 사라질 수 있습니다.
