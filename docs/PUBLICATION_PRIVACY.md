# Publication privacy checks

Before publication, run `python -m antlab.publication_guard --staged` (the default).
Run `--tracked` to inspect all tracked Git index files. The scanner reads Git blobs,
so editing a working file after staging does not change the checked snapshot.
It does not read ignored translation weights, dependencies, or untracked files.

The scanner blocks recognizable local user paths, the current OS account identifier,
email addresses, common API token formats, and private key headers. Only file, line,
and category are reported; matched values are never printed. Sensitive filenames
are replaced with a redacted label. Line 0 means a filename finding.
The project's neutral contributor email and reserved example domains are allowed.
Author emails in research material require review; do not bypass a finding silently.
Exit 0 means no detected text pattern, and exit 1 blocks publication (including Git errors).

Binary files are explicitly listed with their byte size and `binary_not_inspected`.
This includes PyTorch checkpoints: a skipped checkpoint is **not privacy cleared**.
Review checkpoint metadata and tensors separately with safe loading; do not load
untrusted pickle objects. This scanner does not inspect Git commit identities or
earlier history, detect every secret format, or understand semantic personal details.

Publish manually reviewed synthetic examples, reproducible code, and measurements.
Do not publish chat logs, real user messages, personal documents, credentials, machine
paths, or raw exception logs. Check commit author identity separately. The current
local translation backend does not send input text to an external service; explicit
model downloads access the model host. Download caches and model weights stay ignored.

## 한국어

공개 전 `python -m antlab.publication_guard --staged`를 실행한다. `--tracked`는
추적되는 전체 Git 인덱스 파일을 검사한다. 개인 경로·현재 OS 계정 이름·이메일·일부
비밀 키 형식을 탐지하면 공개를 차단하며, 발견된 원문 값은 출력하지 않는다.
이 검사는 모든 개인정보의 부재를 보장하지 않는다. 바이너리 체크포인트는 크기와
검사 제외 사실을 명시하며 메타데이터·텐서는 별도로 검토해야 한다. 커밋 작성자와
과거 Git 기록도 별도 확인한다. 공개 예시는 직접 검토한 합성 데이터만 사용하고
대화 기록·실제 사용자 입력·자격 증명·개인 문서를 넣지 않는다. 현재 로컬 번역기는
입력을 외부 서비스로 보내지 않으며 모델 다운로드만 호스트에 접속한다.
