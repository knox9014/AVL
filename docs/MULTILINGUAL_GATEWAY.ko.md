# 선택 기능: 로컬 다국어 번역 계층

[English](MULTILINGUAL_GATEWAY.md) | [한국어](MULTILINGUAL_GATEWAY.ko.md)

`원문 → 영어 번역 → AVL 벡터 → 수신 모델의 의미 예측 → 영어 표현 → 목표 언어 번역`
순서로 연결합니다. 첫 번역기는 [M2M100](https://huggingface.co/facebook/m2m100_418M)이며
한국어를 포함한 100개 언어를 선언합니다. 모든 언어 지원이나 정확한 의미
보존을 보장하는 것은 아닙니다. 번역기는 약 1.94 GB의 별도 모델 파일과 추론
메모리를 사용하며, 99,977개 파라미터의 AVL 모델에 포함되지 않습니다.

기본 AVL 환경과 PyTorch 2.6 이상을 준비한 뒤 저장소 루트에서 실행합니다.

```bash
python -m pip install -r requirements.txt
python -m pip install --target .translation-deps --only-binary=:all: -r requirements-translation.txt
python -m antlab.download_translation --revision 55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636
python -m antlab.multilingual_demo --list-languages
python -m antlab.multilingual_demo --translation-only --source-lang ko --target-lang en --text "램프가 켜져 있다."
python -m antlab.multilingual_demo --source-lang en --target-lang ko --text "It is certain that the lamp is on."
```

설치와 명시적인 다운로드는 공개 서버를 이용합니다. 추론은 로컬 파일만
사용하고 원문을 외부 번역 서비스에 보내지 않습니다. 의존성과 모델 파일은
Git에서 제외하며, 외부 모델의 MIT 라이선스와 출처를 유지합니다.

`--translation-only`는 일반 문장을 번역합니다. 일반 모드는 번역된 영어가
현재 AVL의 제한된 문법에 들어맞아야 벡터화합니다. 원문과 번역문은 로컬에
보존하며, 문법 밖이면 `out_of_scope`, 미지원 언어는 `unsupported_language`,
번역 자원이나 길이 제한 문제는 `translation_unavailable`로 표시합니다.
수신 측 목표 언어 번역이 실패하면 영어를 남기고
`target_translation_unavailable`로 표시합니다. 결과는 검증된 사실이 아닌
모델 예측입니다.

기본적으로 고유 문장을 최대 4개씩 묶어서 처리합니다. `--batch-size 1`은
개별 생성, `--batch-size 4`는 묶음 생성입니다. 같은 요청의 중복 문장은
번역 결과를 재사용하되 원문 순서와 결과 개수는 유지합니다. 원문을 영구
캐시에 저장하지 않습니다. 송수신 계층 모두 묶음 처리를 사용할 수 있으며
기존 단건 번역기 인터페이스도 지원합니다.

이 제한은 한 번에 토큰화하고 생성하는 문장 수를 제한합니다. 매우 긴
문장 하나는 토큰 제한을 확인하기 전에 토큰화 비용이 들 수 있으며 전체
요청 바이트 크기를 제한하는 것은 아닙니다. 명시적인 메모리 부족 오류는
해당 문장의 번역 실패로 처리해 원문을 남깁니다. 실제 시간과 메모리는
[효율 측정 문서](TRANSLATION_EFFICIENCY.md)에 따로 기록합니다.

언어는 `ko`, `en`, `ja` 같은 코드를 직접 지정합니다. 자동 언어 감지는 없고,
긴 문장의 자동 분할도 구현하지 않았습니다. 기본 제한은 입력 256토큰,
생성 128토큰이며 입력을 조용히 잘라내지 않습니다. `--text`를 반복해서
사용자가 구분한 여러 문장을 전달할 수 있습니다.

수신 모델은 실제 벡터 패킷만 받습니다. 원문이나 정답 의미를 받지 않습니다.
송수신 모델은 같은 AVL 체크포인트를 사용해야 합니다. 독립 학습한 모델
사이의 공통 벡터 해석은 아직 해결되지 않았습니다.

[실제 시험과 한계](MULTILINGUAL_RESULTS.md): 8개 외국어 24문장 모두 영어
번역을 생성했지만 AVL 문법 밖이어서 전부 변환을 보류했습니다. 일부에는
번역 의미 오류도 관찰됐습니다. 번역기를 붙이는 것만으로 다국어 의미
전달이 완성되지는 않습니다.

정확한 API, 상태 목록과 재현 환경은 [영문 사용법](MULTILINGUAL_GATEWAY.md),
관련 논문과 모델 간 교차 시험은 [연구 자료](AI_COMMUNICATION_REFERENCES.md)에 있습니다.
