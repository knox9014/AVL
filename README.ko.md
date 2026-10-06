# AVL — AI Vector Language

[English](README.md) | [한국어](README.ko.md)

AVL은 AI 유닛 사이에서 의미를 보존하며 전달하는 학습된 벡터 통신을 연구합니다. 장기 목표는 유용한 AI 전용 통신 언어이며, 현재 버전은 실행 가능한 제한된 범위의 실험입니다.

**현재 상태: v1 연구 프로토타입.** 파라미터 총 98,511개인 모델이 정해진 영어 구간 하나를 16차원 float32 벡터로 바꾸고, 그 벡터만으로 의미 항목 9개를 예측합니다. 여러 구간의 순서를 유지하는 패킷, 미지원 입력 처리, 학습된 체크포인트 3개와 재검증 가능한 평가 기록을 포함합니다. **사전 성공 기준은 통과하지 못했으며, 새로운 문장 표현에서 오류가 남았습니다.** 범용 영어 이해와 무손실 의미 압축은 아직 입증하지 못했습니다. [측정 결과](docs/AVL_V1_RESULTS.md)를 참고하세요.

## 실행

Python 3.10 이상과 PyTorch를 사용하며 저장소 루트에서 실행합니다.

```bash
python -m pip install -r requirements.txt
python -m antlab.semantic_demo --text "When it rains, it is possible that the door is not closed." --text "Please make sure that the budget limit is at most 20 USD."
python -m unittest discover -s antlab/tests
python -m antlab.semantic_run --output antlab/runs/semantic-v1-20261006 --verify
```

수신자는 사실/요청·대상·술어·부정·확실성·조건·금액·비교 조건·단위를 예측합니다. 원문과 정답 없이 실제 직렬화된 벡터를 받습니다. 구간당 벡터는 64바이트이고 패킷 헤더는 12바이트입니다. 두 구간의 전송량은 네트워크 부가 비용을 제외한 140바이트이며, 짧은 원문보다 클 수 있습니다.

데모 기본 시드 11은 평가 전에 정했습니다. 결과는 `model_prediction`으로 표시하며, 지원 문법 안에서도 틀릴 수 있습니다. 미지원 입력은 원문 그대로 로컬에 남기고 `unsupported`로 표시합니다. 문법 검사 통과가 의미 예측의 정확성을 보장하지 않습니다.

## 범위와 문서

- [한국어 사용법](docs/USAGE.ko.md), [영어 사용법과 Python API](docs/USAGE.md).
- [학습 전 고정한 v1 명세](docs/AVL_V1_PROTOCOL.md), [v1 측정 결과](docs/AVL_V1_RESULTS.md).
- [의미 입력 연구 방향](docs/SEMANTIC_INPUT.md).

저장된 영어 템플릿만 지원합니다. 대상은 램프·히터·선풍기·문·예산이고 금액은 10/20/50/100 USD입니다. 구간 경계는 사용자가 제공합니다. 임의의 글·새 대상이나 금액·중첩 조건·자동 요약은 아직 미지원입니다. 벡터는 익명화·인증·네트워크 전달 기능을 제공하지 않습니다.

학습을 반복할 때는 새 출력 경로를 사용합니다.

```bash
python -m antlab.semantic_run --output antlab/runs/my-new-study
```

고정 실험은 시드 3개를 각각 1,500스텝 학습하고 마지막 체크포인트와 사전 성공 기준을 사용합니다. 검증은 데이터를 다시 만들고 소스·기록 해시와 학습 메타데이터를 확인한 뒤 체크포인트에서 바이트 송수신 평가를 재실행합니다. 재학습은 하지 않습니다. 엄격한 수치 재현은 [기록된 환경](antlab/runs/semantic-v1-environment-20261006.json)을 기준으로 합니다.

## 이전 v0 실험

기존 파라미터 98,928개인 연결형 문자 모델은 항상 `YES`라고 답해 YES/NO 시험 정확도가 50%였습니다. 기존 소스·체크포인트·실패 기록은 유지합니다: [몸체 설계](docs/research/2026-10-05-english-connected-body.md), [학습 결과](docs/research/2026-10-05-english-body-results.md), [긴 입력 크기 측정](docs/research/2026-10-06-long-vector-size.md).

이전 64바이트 벡터가 긴 원문보다 작다는 사실은 의미 보존의 증거가 아니었습니다. 이 저장소는 유닛 수 증가에 따른 범용 지능 향상을 입증하지 않았습니다.

`antlab`은 기록된 해시를 유지하기 위한 기존 Python 모듈 이름입니다. 프로젝트 이름은 AVL이며, 더 넓은 ACT 실험은 이 저장소 밖에서 다룹니다.

## 데이터와 라이선스

예제는 합성 데이터이며 공개 체크포인트는 텐서와 실험 메타데이터를 포함합니다. 개인 대화·인증 정보·로컬 사용자 설정은 이 공개본에 포함하지 않습니다.

Apache-2.0 라이선스입니다. [LICENSE](LICENSE)를 참고하세요.
