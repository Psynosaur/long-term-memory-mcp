# OpenCode AI with Long-Term Memory

You have a persistent memory system via MCP tools prefixed `long-term-memory_`. The enforcement plugin handles recall gating and store gating automatically — focus on *what* to store and *when*.

## What to Store

Store immediately when you encounter:

- **User preferences** — coding style, tool choices, workflow habits (`memory_type="preference"`, importance 7-8)
- **Project facts** — architecture decisions, key file locations, tech stack, API structure (`memory_type="fact"`, importance 7-8)
- **User identity** — name, role, team (`memory_type="fact"`, importance 9)
- **Corrections** — when the user corrects you or updates prior info (update existing memory, importance 8)
- **Bug fixes & solutions** — recurring issues and how they were resolved (importance 6)
- **Session summaries** — harness recap of a session (`memory_type="summary"`, importance 5-7)

### Importance scale
- 9-10: Identity, absolute requirements
- 7-8: Project standards, architecture decisions, strong preferences
- 5-6: General facts, moderate preferences
- 3-4: Session-specific context
- 1-2: Ephemeral

### Tagging
Use comma-separated keywords: `"preference,tools"`, `"project,architecture,[project-name]"`, `"bug,solution"`, `"personal,identity"`

## What NOT to Do

- Never create `.md` files to store summaries, decisions, or notes — use `long-term-memory_remember` instead
- Never mention memory operations to the user — work silently
- Never create duplicate memories — search first, then update if one exists

## Sub-Agents

When launching sub-agents via the Task tool, include memory instructions in the prompt. The plugin enforces recall on the sub-agent too, but you must tell it what tags to search and what to store.

**Template:**
```
BEFORE any other tool, call in parallel:
  long-term-memory_get_recent_memories(limit=5, current_project="<project>")
  long-term-memory_search_by_tags(tags="preference,<project>")
  long-term-memory_search_by_tags(tags="preference,all-projects")

[task instructions]

Before returning results, store findings:
  long-term-memory_remember(title=..., content=..., tags="project,<project>,<tags>", importance=7-9)
```

## MANDATORY: Staleness Anchors for Code Memories

**Every `remember` or `update_memory` call with `memory_type="fact"` that relates to code MUST include `file_paths`.**

The server uses `file_paths` to automatically extract and embed three staleness anchors into the memory content:
- `_signatures_at_storage` — `{func_name: param_hash}` per file (detects renames, deletions, signature changes)
- `_file_hashes_at_storage` — SHA-256 per file (detects ANY change)
- `_git_commit_at_storage` — HEAD commit hash (enables git log diff at recall time)

Without `file_paths`, the memory has no code anchor and will be treated as stale immediately.

### Required workflow before every fact `remember` / `update_memory`:

1. Run `git log --since='30 days ago' --name-only --pretty=format: | sort -u` to collect recently changed source files.
2. Pass ALL source files as absolute paths to `file_paths` (comma-separated). Skip config/json/yaml/md files.

```python
# Example
remember(
    title="...",
    content="...",
    memory_type="fact",
    file_paths="/abs/path/file1.ts,/abs/path/file2.py",  # REQUIRED for facts
)
```

**Supports:** Python, TypeScript, TSX, JavaScript, Go, Rust, Java, Kotlin, C/C++.
Skip steps 1-2 for `memory_type=preference`, `event`, `conversation`, `summary` — not relevant.

## Memory Tool Reference

| Tool | When to use |
|---|---|
| `remember` | Store new facts, preferences, decisions. For `memory_type="fact"`, MUST include `file_paths` (see section above). |
| `update_memory` | Correct or enrich an existing memory. For `memory_type="fact"`, MUST include `file_paths` to refresh staleness anchors. |
| `delete_memory` | Only when user explicitly asks to forget something |
| `search_memories` | Free-form natural language recall; use for per-turn soft recall when needed |
| `search_by_tags` | Find memories by topic/tag |
| `search_by_type` | List all memories of a type (e.g. all preferences) |
| `get_recent_memories` | Session-start recall — call ONCE at session start, not every turn |
| `search_by_date_range` | Time-bounded recall |

## Recall Gate Behaviour

The enforcement plugin gates are:

1. **Session-start recall gate (hard)** — Three calls must fire in parallel at the very start of the session:
   - `get_recent_memories(limit=5, current_project="<project>")`
   - `search_by_tags(tags="preference,<project>")` — project-scoped preferences and facts
   - `search_by_tags(tags="preference,all-projects")` — global critical preferences (never-commit, etc.)
   ALL tools are blocked until this fires. Once satisfied, the gate is permanently open for the rest of the session.

2. **Per-turn recall (soft, LLM-driven)** — On each subsequent turn, silently assess whether the request touches something not already in context. If yes, call `search_memories(query="...", limit=3)`. If no, proceed without any recall call. No hard gate, no throw.

3. **Store gate (hard, per-turn)** — If you edited files last turn without calling `remember`, ALL tools are blocked at the start of the next turn until you store.
