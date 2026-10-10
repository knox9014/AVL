# AVL-L1 bounded language core design — 2026-10-09

## Scope and acceptance
Implement a standalone, deterministic spatial assertion language before connecting new semantics to learned vectors. Names are declared explicitly (2–4 distinct ASCII identifiers). Programs contain 0–32 assertions and exactly one query. Relations: left, right, above, below; explicit NOT is Boolean complement, never inverse direction. Coordinates are independent real axes; equal positions are permitted.

Strict source grammar: `assert [not ]NAME RELATION NAME;` repeated, followed by `ask [not ]NAME RELATION NAME;`. Keywords are lowercase, one space separates tokens; outer whitespace is permitted. No implicit English/OOD recovery. Unknown names, extra text, malformed punctuation, self-relations and excess bounds raise ValueError (unsupported input), never get a semantic guess.

Typed AST: exact version/names/assertions/query keys, exact subject/relation/object/negated atom keys; version AVL-L1. Canonical UTF-8 JSON round trip, reject duplicate/extra keys and invalid types. This is a new symbolic contract, not a change to AVL1 vector packets.

## Meaning
left(a,b): x(a)<x(b); right(a,b): x(a)>x(b); above(a,b): y(a)>y(b); below(a,b): y(a)<y(b). Negation complements the comparison, including equality. All assertions form a conjunction. Return inconsistent if no assignment satisfies the program, otherwise entailed/refuted/undetermined according to all satisfying assignments. Inconsistent programs do not entail arbitrary facts. Return concrete satisfying witnesses for each attainable query truth value.

For at most four names, enumerate each axis separately over integer ranks 0..n-1. Every finite real-coordinate weak ordering has such a rank representation, so this is complete for these comparisons. This is an exact bounded reference interpreter, not a scalable general theorem prover.

## Verification before publication
Write tests before implementation; observe missing-module failure. Test strict admission, NOT vs inverse, transitivity, unknown orthogonal relations, equality, inconsistent cycles, JSON round trips and duplicate rejection. Independently compare results against a Cartesian coordinate oracle on exhaustive two-name assertion pairs and queries. Run stdlib tests locally; disclose that full torch regression needs remote CI. Record counts and examples without interpreting formal correctness as learned generalization.

## Next milestones
Connect validated positive assertions to existing vector adapters through an explicit new API; then add typed goals/actions and agent interoperability evaluation. Negation transport and interpretation will remain explicitly symbolic until a separately registered learned experiment establishes more.
