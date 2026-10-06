# AVL v2 사용법

[English](USAGE_V2.md) | [한국어](USAGE_V2.ko.md)

v2는 지원 범위를 정한 영어 의미 송수신 모델입니다. 파라미터 99,977개로
단어 순서를 읽고, 수신자는 벡터만 받아 의미 항목을 예측합니다.
문법 검사 통과가 예측의 정확성을 보장하지는 않습니다.
[명세](AVL_V2_PROTOCOL.md)에 범위와 학습·평가 조건을 고정했습니다.

## 실행

PyTorch를 설치한 뒤 저장소 루트에서 실행합니다.

```bash
python -m antlab.semantic_v2_demo --text "When it rains, that the door is not closed is now possible." --text "Please, now ensure that the budget limit is at most 20 USD."
python -m antlab.semantic_v2_run --output antlab/runs/semantic-v2-20261006 --verify
python -m unittest discover -s antlab/tests
```

각 `--text`는 사용자가 직접 나눈 독립 구간입니다. 원문은 로컬에 보존하고
정해진 문법에 맞는지만 검사합니다. 인코더는 학습 입력에서 얻은 단어 41개의
순서를 읽으며 대소문자·공백을 정규화하고 구두점은 무시합니다.
수신자는 직렬화된 벡터만 받아 사실/요청·대상·술어·부정·확실성·조건·금액·
비교 조건·단위의 9개 항목을 예측합니다.

출력은 모두 `model_prediction`입니다. 새 대상이나 17 USD 같은 미지원
입력은 정확한 원문과 `unsupported` 상태로 남기며 벡터를 만들지 않습니다.
자동 구간 분할·요약·임의의 영어 이해는 아직 미지원입니다. 조건은 구간 하나
전체에 적용하며, 부정을 반대 상태로 바꾸어 추론하지 않습니다.

## Python API와 전송

[영어 문서](USAGE_V2.md)에 `load_checkpoint`, `transmit_segments`,
`receive_packet` 예제를 제공합니다. 송수신에는 패킷만 넘기고 원문·지원 여부·
평가 정답은 보내지 않습니다. 송신자와 수신자는 같은 v2 체크포인트를 써야
합니다. 패킷의 AVL1 표기는 수치 형식 버전이며 모델 식별자가 아닙니다.
다른 체크포인트 사용이나 위조 송신자를 자동으로 검출하지 않습니다.
벡터 자체는 익명화가 아니며 네트워크 전달 보장도 제공하지 않습니다.

벡터는 구간당 64바이트이고 패킷 헤더는 12바이트입니다. 두 구간은 네트워크
부가 비용을 제외한 140바이트이며 짧은 원문보다 클 수 있습니다.
의미 보존과 압축 효율은 별도로 판단해야 합니다.

## 실험 반복

```bash
python -m antlab.semantic_v2_run --output antlab/runs/my-new-v2-study
```

새 출력 경로를 사용합니다. 시드 44/55/66을 각각 1,800스텝 학습하고 마지막
체크포인트를 평가합니다. 같은 뜻의 문장 쌍은 학습 데이터 안에서만 뽑습니다.
데모 기본 시드 44는 평가 전에 정했습니다. 검증은 데이터와 기록을 확인한 뒤
모든 체크포인트의 바이트 송수신 평가를 재현하며 재학습은 하지 않습니다.
엄격한 수치 재현은 기록된 환경을 기준으로 합니다. 이 제한된 시험을 통과해도
범용 영어의 의미 보존을 입증한 것은 아닙니다.
