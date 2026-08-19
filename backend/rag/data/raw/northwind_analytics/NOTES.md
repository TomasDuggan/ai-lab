# Fictional HR Policy Corpus (RAG v2 domain)

## Why this domain
Chosen specifically to satisfy the criteria for a real RAG use case
(private data, not something an LLM could already know) without
fabricating "facts" that pretend to be real - HR policies for a
fictional company are a standard, low-risk way to get realistic
structure and content without simulating false real-world events (unlike
e.g. fake incident reports for a real industry, which was considered and
rejected for feeling closer to fabricating evidence).

## Company
Northwind Analytics (fictional), 150-300 employees, remote-friendly -
sized deliberately to make hybrid/remote/international edge cases
plausible, since a fully in-office company would produce far fewer
interesting policy interactions.

## Corpus structure
12 documents, one policy per file, same format as the project's NOTES.md
convention (headers, bullets, no decorative prose):

```markdown
# [Policy name]
## Overview
## Rules
## Edge cases / exceptions
```

Documents: PTO/Vacation, Remote Work, Expense Reimbursement, Sick Leave
& Medical, Payroll & Compensation, Parental Leave, Code of Conduct,
Onboarding Process, Performance Review, Termination & Resignation,
IT & Security, Travel.

## Deliberate failure-case design
Unlike the Steam corpus (real but messy marketing text), this corpus is
fictional but *structurally engineered* to reproduce and extend the
retrieval failure modes found in the Steam project, in a domain where
they also carry real business meaning:

- **Cross-references between documents, on purpose** - e.g. Remote Work
  references Onboarding, IT & Security, and Performance Review; PTO and
  Sick Leave share the same 90-day accrual-pause rule and are explicitly
  interchangeable in one edge case. Mirrors the Garry's Mod problem from
  the Steam project (a chunk naming another entity pulls its embedding
  toward that entity) - but here cross-references are the *correct*,
  expected behavior of the domain, not noise. A good test of whether
  retrieval can still distinguish "policy A mentions policy B" from
  "policy A's chunk gets miscategorized as being about B."
- **Tables** - included deliberately in 2 documents only (Expense
  Reimbursement's approval-limit table, Travel's flight-class table),
  not spread across all 12. `RecursiveCharacterTextSplitter` doesn't
  understand tabular structure and can split a table mid-row, separating
  a role from its associated limit - a known, reproducible chunking
  failure mode worth testing directly.
- **Nested/conditional exceptions** - phrases like "X unless A or B,
  except when C" appear in most documents' edge-case sections, in
  moderation (1-2 per document, not exhaustively) - mainly a generation-
  stage test (can the LLM reason through a nested condition correctly
  when answering), not a retrieval-stage one.
- **One deliberately dangling reference** - IT & Security references an
  "Access Control Policy" that was never written. Left in on purpose to
  test how the pipeline behaves when context references something
  outside the corpus - should surface as "the context doesn't cover
  this" rather than a fabricated answer, per the RAG generation
  instructions already established (see RAG.md).
- **Explicitly NOT included yet**: contradictory information across
  document versions (e.g. two conflicting drafts of the same policy).
  Requires document versioning/metadata the current pipeline has no
  mechanism for (no "which version wins" logic exists yet) - would
  produce unattributable retrieval failures right now. Deferred to a
  later iteration, once the base pipeline works, as a deliberate
  "conflict resolution" exercise rather than an accidental confound in
  the first pass.

## What this domain is expected to test, beyond what Steam already covered
- Whether legitimate cross-referencing (a real, desired feature of this
  domain) can be told apart from unwanted semantic contamination (an
  unwanted side effect seen in Steam) - Steam only had the latter.
- Table-splitting failure, not present in the Steam corpus at all
  (no tabular data there).
- Faithfulness under precise, numeric, verifiable answers ("how many PTO
  days after 3 years?") rather than the more subjective "is this a good
  recommendation?" judgment calls Steam required - correctness here has
  a single right answer, making evaluation less ambiguous than curating
  Steam's `expected_games` ever was.