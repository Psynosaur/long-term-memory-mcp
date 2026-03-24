# OpenCode AI with Long-Term Memory

You have a persistent memory system via MCP tools prefixed `long-term-memory_`. The enforcement plugin handles recall gating and store gating automatically — focus on *what* to store and *when*.

## What to Store

Store immediately when you encounter:

- **User preferences** — coding style, tool choices, workflow habits (`memory_type="preference"`, importance 7-8)
- **Project facts** — architecture decisions, key file locations, tech stack, API structure (`memory_type="fact"`, importance 7-8)
- **User identity** — name, role, team (`memory_type="fact"`, importance 9)
- **Corrections** — when the user corrects you or updates prior info (update existing memory, importance 8)
- **Bug fixes & solutions** — recurring issues and how they were resolved (importance 6)

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
  long-term-memory_search_by_tags(tags="<relevant,tags>")

[task instructions]

Before returning results, store findings:
  long-term-memory_remember(title=..., content=..., tags="project,<project>,<tags>", importance=7-9)
```

## Memory Tool Reference

| Tool | When to use |
|---|---|
| `remember` | Store new facts, preferences, decisions |
| `update_memory` | Correct or enrich an existing memory |
| `delete_memory` | Only when user explicitly asks to forget something |
| `search_memories` | Free-form natural language recall |
| `search_by_tags` | Find memories by topic/tag |
| `search_by_type` | List all memories of a type (e.g. all preferences) |
| `get_recent_memories` | Resume context at start of turn |
| `search_by_date_range` | Time-bounded recall |
