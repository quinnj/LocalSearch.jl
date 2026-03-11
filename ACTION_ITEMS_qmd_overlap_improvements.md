# Action Items: qmd overlap improvements for LocalSearch

## Context
- Repo: LocalSearch
- Worktree: /Users/jacob.quinn/.julia/dev/LocalSearch
- Branch: main

## Items

### [x] ITEM-001 (P0) Correct FTS score normalization
- Description: `search_fts` currently converts SQLite FTS5 BM25 scores with an inverted mapping, so stronger lexical matches can receive lower `SearchResult.score` values than weaker matches. This distorts `min_score` filtering and any downstream ranking logic that consumes the normalized score.
- Desired outcome: Stronger BM25 matches should consistently map to higher normalized `score` values, with tests covering the monotonic behavior.
- Affected files: `src/search.jl`, `test/runtests.jl`
- Implementation notes:
  - Replace the current FTS score normalization with a monotonic transformation that preserves relative strength for SQLite BM25 values.
  - Add focused tests that exercise strong vs weak lexical matches and guard against regression.
- Verification:
  - `julia --project=/Users/jacob.quinn/.julia/dev/LocalSearch -e 'using Pkg; Pkg.test()'`
- Assumptions:
  - SQLite FTS5 BM25 values remain negative for matches and become stronger as magnitude increases.
- Risks:
  - Existing callers may be relying on the old numeric range informally, so test coverage should pin the intended ordering behavior.
- Completion criteria:
  - BM25 score normalization is monotonic in the expected direction.
  - Tests prove stronger lexical hits get higher normalized scores.
- Verification evidence:
  - `julia --project=/Users/jacob.quinn/.julia/dev/LocalSearch -e 'using Pkg; Pkg.test()'` passed on 2026-03-11 after adding a regression test that compares raw BM25 ordering against normalized search scores.

### [x] ITEM-002 (P1) Add query/document embedding formatting
- Description: LocalSearch currently embeds raw query text and raw chunk text, while qmd now formats queries and documents differently and includes document titles when embedding content. LocalSearch already has titles available, but does not use them in embeddings.
- Desired outcome: Queries and documents should be embedded with explicit formatting helpers, and document embeddings should include titles when present.
- Affected files: `src/llm.jl`, `src/store.jl`, `src/search.jl`, `test/runtests.jl`
- Implementation notes:
  - Add helpers analogous to qmd’s query/document embedding formatters in the `Embed` module.
  - Ensure `embed_content!` uses formatted document text with titles.
  - Ensure vector query search uses formatted query text before embedding.
  - Add tests that verify the formatted inputs are what custom embedding functions receive.
- Verification:
  - `julia --project=/Users/jacob.quinn/.julia/dev/LocalSearch -e 'using Pkg; Pkg.test()'`
- Assumptions:
  - Keeping the default embedding model while changing prompt formatting is acceptable and desirable.
- Risks:
  - Embedding behavior will shift for existing stores, so tests need to focus on correctness of formatting and search behavior rather than exact score equality.
- Completion criteria:
  - LocalSearch formats query and document text explicitly before embedding.
  - Document formatting includes titles when available.
  - Tests cover both query and document formatting paths.
- Verification evidence:
  - `julia --project=/Users/jacob.quinn/.julia/dev/LocalSearch -e 'using Pkg; Pkg.test()'` passed on 2026-03-11 after adding helper coverage and recording-embed tests for formatted query/document inputs.

### [x] ITEM-003 (P1) Track embedding model metadata in stored vectors
- Description: LocalSearch stores chunk positions and timestamps for embeddings, but it does not record which embedding model produced those vectors. That makes it impossible to detect stale vectors after model changes.
- Desired outcome: Stored vector metadata should include the embedding model identifier, and LocalSearch should re-embed content when persisted vectors were generated with a different model URI.
- Affected files: `src/store.jl`, `src/llm.jl`, `test/runtests.jl`
- Implementation notes:
  - Extend the chunk/vector metadata schema to record the model URI.
  - Teach the store to resolve the effective embedding model for default embeddings and optionally custom embedding providers.
  - Re-embed or replace persisted vectors when the stored model metadata does not match the current embedding model.
  - Add tests for initial model recording and model-change re-embedding behavior.
- Verification:
  - `julia --project=/Users/jacob.quinn/.julia/dev/LocalSearch -e 'using Pkg; Pkg.test()'`
- Assumptions:
  - For built-in embeddings, the model URI string is the right identity to persist.
  - For custom embedding functions, a best-effort opaque model label is sufficient if explicit identity cannot be inferred automatically.
- Risks:
  - Schema changes can break existing databases if migration logic is incomplete.
- Completion criteria:
  - Stored embedding metadata includes a model identifier.
  - Existing content is re-embedded when the active model changes.
  - Tests cover both persistence and stale-vector replacement.
- Verification evidence:
  - `julia --project=/Users/jacob.quinn/.julia/dev/LocalSearch -e 'using Pkg; Pkg.test()'` passed on 2026-03-11 after adding metadata persistence, model-change refresh, title-change refresh, and legacy-schema migration coverage.

### [ ] ITEM-004 (P1) Improve token chunking with structural breakpoints
- Description: LocalSearch chunking is purely token-window based and repeatedly tokenizes substrings during binary search. qmd now uses markdown-aware breakpoints and code-fence protection to produce better chunks for the same underlying retrieval behavior.
- Desired outcome: Chunking should prefer structural markdown boundaries, avoid splitting inside fenced code blocks when possible, and retain token-limit guarantees.
- Affected files: `src/store.jl`, `test/runtests.jl`
- Implementation notes:
  - Add breakpoint scanning and code-fence detection helpers.
  - Use structural boundaries to choose better chunk cut points before or alongside token-limit refinement.
  - Preserve current overlap semantics and token safety guarantees.
  - Expand tests to cover headings, paragraph boundaries, and fenced code behavior.
- Verification:
  - `julia --project=/Users/jacob.quinn/.julia/dev/LocalSearch -e 'using Pkg; Pkg.test()'`
- Assumptions:
  - LocalSearch primarily indexes markdown or markdown-like prose where structural boundaries improve retrieval quality.
- Risks:
  - Chunk boundaries will shift, so tests should check invariants and meaningful behavior instead of hard-coding every chunk text.
- Completion criteria:
  - Chunking prefers useful structural cut points.
  - Fenced code blocks are not split internally when avoidable.
  - Tests cover the new chunking behavior and token limits still hold.

### [ ] ITEM-005 (P1) Improve lexical query parsing
- Description: LocalSearch’s current FTS query builder strips most punctuation and only supports simple ANDed prefix terms. qmd has improved its lexical parser to preserve quoted phrases and support exclusions, which is directly relevant to LocalSearch’s existing BM25 feature set.
- Desired outcome: LocalSearch should support quoted phrases and negated terms in lexical search while keeping invalid or degenerate cases safe.
- Affected files: `src/search.jl`, `test/runtests.jl`
- Implementation notes:
  - Replace the current split-and-sanitize builder with a parser that recognizes phrases and negation.
  - Maintain safe sanitization before handing the query to FTS5.
  - Return no query for unsupported all-negative input rather than producing invalid FTS syntax.
  - Add tests for phrase search, negation, unmatched quotes, and all-negative input.
- Verification:
  - `julia --project=/Users/jacob.quinn/.julia/dev/LocalSearch -e 'using Pkg; Pkg.test()'`
- Assumptions:
  - Backwards-compatible simple term behavior should remain the default for plain unquoted input.
- Risks:
  - FTS syntax bugs can be subtle, so regression tests should cover the builder behavior directly and through search results.
- Completion criteria:
  - Phrase and negation syntax work correctly in LocalSearch lexical search.
  - Invalid lexical input is handled safely.
  - Existing simple-term behavior still passes tests.

### [ ] ITEM-006 (P2) Retune FTS column weights
- Description: LocalSearch currently weights BM25 columns as key=1, title=5, body=10, which likely under-emphasizes identifiers and over-emphasizes body matches relative to titles. qmd has moved toward stronger weighting for high-signal metadata fields.
- Desired outcome: BM25 weighting should better reward identifier/title matches without harming ordinary content search.
- Affected files: `src/search.jl`, `test/runtests.jl`
- Implementation notes:
  - Revisit the FTS column weighting scheme used in `bm25(...)`.
  - Add ranking tests that cover key matches, title matches, and body-only matches.
  - Keep the tuning simple and evidence-driven rather than adding a new tuning abstraction.
- Verification:
  - `julia --project=/Users/jacob.quinn/.julia/dev/LocalSearch -e 'using Pkg; Pkg.test()'`
- Assumptions:
  - LocalSearch users benefit when identifiers and titles act as stronger signals than arbitrary body mentions.
- Risks:
  - Weight changes can shift ranking in edge cases, so tests need to capture intended ordering.
- Completion criteria:
  - BM25 weighting better favors key/title signal.
  - Tests demonstrate the intended ranking order for representative inputs.
