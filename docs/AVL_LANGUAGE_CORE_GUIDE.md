# AVL-L1 language core

AVL-L1 is an experimental executable language for bounded spatial assertions and questions. It defines truth conditions independently of a particular neural checkpoint. It does not yet integrate negation with learned vector messages.

## Run
```sh
python -m antlab.language_core --names a b c --source "assert a left b; assert b left c; ask a left c;"
```
The status is `entailed`. A satisfying coordinate witness is returned.

```python
from antlab.language_core import parse, dumps, loads, evaluate, execute

message = dumps(parse("assert not a left b; ask a right b;", ["a", "b"]))
received = loads(message)
assert evaluate(received)["status"] == "undetermined"
assert execute("assert a left b; assert b left a; ask a above b;",
               ["a", "b"])["status"] == "inconsistent"
```

## Contract
Declare 2–4 distinct ASCII names. Use lowercase `assert [not ]SUBJECT RELATION OBJECT;` followed by exactly one `ask [not ]SUBJECT RELATION OBJECT;`. At most 32 assertions. Relation words: `left`, `right`, `above`, `below`. Keywords and arguments are separated by one ASCII space. Whitespace around statements is allowed. Names are case sensitive; reserved words are excluded case insensitively. Self-relations are unsupported.

Left/right compare the independent x axis; above/below compare y. Equal coordinates are possible. Thus NOT left means greater than **or equal**, not strictly right. All assertions must hold together.

- `entailed`: every satisfying world makes the query true.
- `refuted`: every satisfying world makes the query false.
- `undetermined`: both true and false satisfying worlds exist; both witnesses are returned.
- `inconsistent`: no world satisfies all assertions; neither witness is returned.
- Unsupported grammar or invalid typed/JSON messages raise `ValueError`; CLI exits 2. Unsupported input is an admission error, not a fifth semantic truth value.

Canonical JSON uses version AVL-L1 and exact typed fields. Extra/duplicate fields, undeclared names and non-Boolean negation are rejected. `execute` passes through parse → canonical JSON → receive validation → interpretation. This symbolic JSON contract is separate from existing AVL1/AVC2/ACQ2 vector codecs.

## What this establishes
An independently checkable finite reference semantics with strict admission, negation, conjunction, queries, inconsistency handling and concrete counterexamples. At most four names means each axis needs at most 256 rank assignments; all finite real-coordinate weak orders have rank representatives.

This is designed symbolic interpretation, not learned reasoning, universal natural-language understanding or an efficiency result. The old English/vector adapters remain experimental and do not acquire this strict admission automatically. No scalability beyond the declared bounds is claimed. Next work is an explicit bridge between typed assertions and vector messages, followed by goals/actions and independently implemented agents.

Protocol: [AVL_LANGUAGE_CORE_PROTOCOL.md](AVL_LANGUAGE_CORE_PROTOCOL.md).
