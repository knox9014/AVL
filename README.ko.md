# AVL — AI Vector Language

[English](README.md) | [한국어](README.ko.md)

[개발 기록](docs/DEVELOPMENT_LOG.md), [공개 전 개인정보 검사](docs/PUBLICATION_PRIVACY.md), [실험용 int8 전송 결과](docs/INT8_WIRE_RESULTS.md)를 남깁니다. 공개 예시는 직접 검토한 합성 데이터이며 실제 사용자 대화는 포함하지 않습니다.

선택 기능으로 [로컬 다국어 번역 계층](docs/MULTILINGUAL_GATEWAY.ko.md)을 추가했습니다. 첫 번역기는 100개 언어를 선언하지만, 실제 시험한 외국어 24문장의 번역 결과는 모두 현재 AVL 문법 밖이었습니다. [측정 결과와 한계](docs/MULTILINGUAL_RESULTS.md), [AI 통신 연구 참고 자료](docs/AI_COMMUNICATION_REFERENCES.md)를 확인하세요. 번역기의 자원과 의미 보존은 AVL 모델과 따로 평가해야 합니다.

독립 학습한 모델 사이의 [교차 통신 실험](docs/AI_COMMUNICATION_REFERENCES.md#avl-cross-play-diagnostic)에서는 같은 체크포인트끼리 의미 복원 100%, 다른 체크포인트 6쌍의 평균은 0.1650%였습니다. 현재 AVL은 독립 모델 모두가 공유하는 벡터 언어가 아니며, 공통 의미 공간을 정렬하는 방법이 필요합니다.

[번역 효율 개선](docs/TRANSLATION_EFFICIENCY.md): 요청 안의 중복을 재사용하고 문장을 묶어서 처리해 이번 합성 입력에서는 약 94초에서 38초로 줄었습니다. 번역 결과는 모두 같았으며 최대 메모리는 약 2.1% 증가했습니다. 의미 지원 범위가 넓어진 결과는 아닙니다.

AVL은 AI 유닛 사이에서 의미를 보존하며 전달하는 학습된 벡터 통신을 연구합니다. 장기 목표는 유용한 AI 전용 통신 언어이며, 현재 버전은 범위를 정하고 측정한 구현입니다.

**현재 상태: v2 연구 프로토타입.** 파라미터 99,977개인 모델이 정해진 영어 문장을 16차원 float32 벡터로 바꿉니다. 수신자는 벡터만으로 의미 항목 9개를 예측하고, 규칙 기반 표시 기능으로 그 의미를 영어 문장으로 다시 보여 줄 수 있습니다. 세 체크포인트 모두 사전 성공 기준을 통과했으며, 문장 표현과 의미 조합을 동시에 바꾼 시험도 포함합니다. **이 결과는 제한된 문법의 성공이며 범용 영어 이해를 뜻하지 않습니다.** [v2 결과](docs/AVL_V2_RESULTS.md)를 참고하세요.

## 실행

Python 3.10 이상과 PyTorch를 사용하며 저장소 루트에서 실행합니다.

~~~bash
python -m pip install -r requirements.txt
python -m antlab.semantic_v2_demo --text "When it rains, that the door is not closed is now possible." --text "Please, now ensure that the budget limit is at most 20 USD."
python -m unittest discover -s antlab/tests
python -m antlab.semantic_v2_run --output antlab/runs/semantic-v2-20261006 --verify
~~~

수신자는 사실/요청·대상·술어·부정·확실성·조건·금액·비교 조건·단위를 예측합니다. 원문과 정답 없이 실제 직렬화된 벡터를 받습니다. 구간당 벡터는 64바이트이고 패킷 헤더는 12바이트입니다. 두 구간은 네트워크 부가 비용을 제외한 140바이트이며 짧은 원문보다 클 수 있습니다.

데모 기본 시드 44는 평가 전에 정했습니다. 출력은 model_prediction으로 표시하고, 미지원 입력은 정확한 원문과 unsupported 상태로 로컬에 남깁니다. 문법 검사 통과가 예측의 정확성을 독립적으로 보장하는 것은 아닙니다.

## 범위와 문서

- [한국어 사용법](docs/USAGE_V2.ko.md), [영어 사용법과 Python API](docs/USAGE_V2.md).
- [학습 전 고정한 v2 명세](docs/AVL_V2_PROTOCOL.md), [v2 측정 결과](docs/AVL_V2_RESULTS.md).
- [문법 전체 사후 점검](antlab/runs/semantic-v2-coverage-20261006.json), [실제 벡터와 영어 복원 예제](antlab/runs/semantic-v2-demo-20261006.json).
- [의미 입력 연구 방향](docs/SEMANTIC_INPUT.md).

저장된 영어 템플릿만 지원합니다. 대상은 램프·히터·선풍기·문·예산이고 금액은 10/20/50/100 USD입니다. 인코더의 단어 41개는 학습 입력에서만 얻습니다. 구간 경계는 사용자가 제공합니다. 임의의 글·새 대상이나 금액·중첩 조건·자동 요약은 아직 미지원입니다.

영어 표시 기능은 예측 항목으로 부정·조건·불확실성을 표현하고 모순된 조합은 거부합니다. 원래 문장 표현을 복원하거나 모델 정답 여부를 보장하지는 않습니다. 송수신자는 같은 체크포인트를 써야 하며 패킷 버전은 모델 식별자가 아닙니다. 벡터는 익명화·인증·네트워크 전달 기능을 제공하지 않습니다.

재학습에는 새 출력 경로를 사용합니다.

~~~bash
python -m antlab.semantic_v2_run --output antlab/runs/my-new-v2-study
~~~

고정 실험은 시드 3개를 각각 1,800스텝 학습합니다. 같은 뜻의 문장 쌍은 학습 데이터 안에서만 뽑고 마지막 체크포인트를 사전 성공 기준으로 평가합니다. 검증은 데이터·소스·기록 해시와 학습 메타데이터를 확인하고 실제 바이트 송수신을 재현합니다. 재학습은 하지 않습니다. 엄격한 수치 재현은 [기록된 환경](antlab/runs/semantic-v2-environment-20261006.json)을 기준으로 합니다.

## 이전 실패 기록

v0의 파라미터 98,928개인 연결형 문자 모델은 항상 YES라고 답해 정확도가 50%였습니다: [학습 결과](docs/research/2026-10-05-english-body-results.md), [긴 입력 측정](docs/research/2026-10-06-long-vector-size.md).

v1의 파라미터 98,511개인 모델은 새로운 의미 조합을 복원했지만 표현 변화 시험의 성공 기준은 통과하지 못했습니다: [v1 결과](docs/AVL_V1_RESULTS.md), [v1 사용법](docs/USAGE.ko.md). 기존 소스와 기록은 그대로 유지합니다. v2는 구조·데이터·목적 함수를 함께 바꿨으므로 버전 간 점수 비교로 개별 변경의 효과를 입증할 수는 없습니다.

실용적인 압축 효율과 유닛 수 증가에 따른 범용 지능 향상은 아직 입증하지 않았습니다. 정답 의미 항목이 이미 알려진 이 작은 체계는 훨씬 적은 비트로 표시할 수 있습니다.

antlab은 기록된 해시를 유지하기 위한 기존 Python 모듈 이름입니다. 프로젝트 이름은 AVL이며 더 넓은 ACT 실험은 이 저장소 밖에서 다룹니다.

## 데이터와 라이선스

예제는 합성 데이터이며 공개 체크포인트는 텐서와 실험 메타데이터를 포함합니다. 개인 대화·인증 정보·로컬 사용자 설정은 이 공개본에 포함하지 않습니다.

Apache-2.0 라이선스입니다. [LICENSE](LICENSE)를 참고하세요.
