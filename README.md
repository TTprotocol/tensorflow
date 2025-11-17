# 가상환경 사용법 요약

## 사전 준비

- Python 3.10–3.12 권장. 프로젝트 루트에서 실행합니다.
- 가상환경 폴더 이름은 `.venv`로 통일합니다.

## 1) 가상환경 생성

```bash
# 모든 OS 공통 (프로젝트 루트에서)
python -m venv .venv
```

## 2) 가상환경 활성화 및 종료

### Windows PowerShell

```powershell
# 활성화
.\.venv-tf\Scripts\Activate.ps1
# 종료
deactivate
```

실행 정책 오류 시:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Windows CMD

```cmd
:: 활성화
.\.venv-tf\Scriptsctivate.bat
:: 종료
deactivate
```

### macOS / Linux

```bash
# 활성화
source .venv/bin/activate
# 종료
deactivate
```

## 3) 의존성 설치

```bash
# 가상환경 활성화된 상태에서
pip install --upgrade pip
pip install -r requirements.txt   # 파일이 있을 때
# 없으면 필요한 패키지 직접 설치
# 예) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 예) pip install tensorflow
```

## 4) 코드 실행

```bash
# 예시: src/main.py 실행
python src/main.py

# Jupyter 노트북
pip install notebook
jupyter notebook
```

VS Code에서 인터프리터 지정: Command Palette → “Python: Select Interpreter” → `.venv` 선택

## 5) 패키지 잠금/공유

```bash
# 현재 환경을 requirements.txt로 저장
pip freeze > requirements.txt
```

## 6) 가상환경 삭제

```bash
# 비활성화 후 폴더 삭제
deactivate
# Windows
rmdir /s /q .venv
# macOS/Linux
rm -rf .venv
```

## 7) 자주 나는 오류 해결

- `ModuleNotFoundError: No module named '...'`

  - 가상환경이 활성화되었는지 확인합니다.
  - `pip install -r requirements.txt`로 설치합니다.
  - VS Code에서 Python 인터프리터를 `.venv`로 선택합니다.

- PowerShell에서 활성화가 막힘

  - `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 실행 후 새 탭에서 다시 시도합니다.

- CUDA 관련 PyTorch/TensorFlow 설치 문제
  - CPU만 필요하면 기본 `pip install torch` 또는 `pip install tensorflow`로 충분합니다.
  - GPU가 필요하면 공식 설치 가이드에 맞춘 CUDA 호환 버전을 지정합니다.

## 8) 권장 폴더 구조 예시

```
your-project/
├─ README.md
├─ requirements.txt
├─ .venv/
└─ src/
   └─ main.py
```

## 9) 빠른 시작

```bash
python -m venv .venv
# OS에 맞게 활성화
pip install --upgrade pip
pip install -r requirements.txt
python src/main.py
deactivate
```
