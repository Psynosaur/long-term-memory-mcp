import type { Plugin } from "@opencode-ai/plugin";

/**
 * Long-Term Memory Enforcement Plugin
 *
 * Hooks into OpenCode's lifecycle to enforce memory usage:
 * 1. System prompt injection  - Adds memory instructions to every LLM call
 * 2. Universal tool gate      - Blocks ALL tools until recall has happened
 * 3. End-of-turn store gate   - Blocks next turn's tools if edits weren't stored
 * 4. Idle enforcement         - Warns when store was skipped after edits
 * 5. Compaction hook          - Preserves enforcement state across compactions
 */
export const LongTermMemoryPlugin: Plugin = async ({
  client,
  directory,
}) => {
  const projectName = directory.split("/").pop() || "unknown";

  /** Per-session tracking state */
  type SessionState = {
    recalledThisTurn: boolean;      // recall happened this turn (resets each turn)
    storedThisTurn: boolean;        // remember/update called during current turn
    editedThisTurn: boolean;        // write/edit called during current turn
    prevTurnEditedWithoutStore: boolean; // last turn had edits but no store
    filesEdited: boolean;           // any write/edit ever this session (for compaction)
    toolCalls: number;
    idleWarned: boolean;
  };

  const sessions = new Map<string, SessionState>();

  const getState = (sessionID: string): SessionState => {
    if (!sessions.has(sessionID)) {
      sessions.set(sessionID, {
        recalledThisTurn: false,
        storedThisTurn: false,
        editedThisTurn: false,
        prevTurnEditedWithoutStore: false,
        filesEdited: false,
        toolCalls: 0,
        idleWarned: false,
      });
    }
    return sessions.get(sessionID)!;
  };

  /** Non-blocking structured logger */
  const log = (message: string, extra: Record<string, unknown> = {}) => {
    client.app
      .log({
        body: {
          service: "long-term-memory",
          level: "info",
          message,
          extra: { projectName, ...extra },
        },
      })
      .catch(() => {});
  };

  log("Plugin initialized", { directory });

  // The ONE tool that satisfies the recall gate — must be called first.
  // Other search tools (search_by_tags, search_memories, etc.) are optional
  // follow-ups; they do NOT satisfy the gate on their own.
  const RECALL_GATE_TOOL = "long-term-memory_get_recent_memories";

  // All memory read tools (used only for state tracking, not gating)
  const RECALL_TOOLS = new Set([
    "long-term-memory_get_recent_memories",
    "long-term-memory_search_memories",
    "long-term-memory_search_by_tags",
    "long-term-memory_search_by_type",
    "long-term-memory_search_by_date_range",
  ]);

  // Tools that count as "store"
  const STORE_TOOLS = new Set([
    "long-term-memory_remember",
    "long-term-memory_update_memory",
  ]);

  // File-mutating tools
  const EDIT_TOOLS = new Set(["write", "edit"]);

  return {
    // ─────────────────────────────────────────────────────────────────────────
    // HOOK 1: System Prompt Injection
    // Every LLM call gets explicit, unambiguous instructions.
    // ─────────────────────────────────────────────────────────────────────────
    "experimental.chat.system.transform": async (_input, output) => {
      output.system.push(`## Long-Term Memory System — MANDATORY RULES

You have a persistent memory system via MCP tools prefixed \`long-term-memory_\`.

### START OF EVERY TASK (before ANY other action)
Call this first — NO EXCEPTIONS:
  long-term-memory_get_recent_memories(limit=5, current_project="${projectName}")

You WILL be blocked from using ANY tool (bash, read, glob, grep, edit, write, etc.) if you skip this step.

Then inspect the results:
- If the returned memories already include project-specific context (tagged "${projectName}" or "preference"), you're done — proceed with work.
- If the results look sparse or lack project context, follow up with ONE targeted search:
    long-term-memory_search_by_tags(tags="${projectName},preference")

Do NOT call search_by_tags unconditionally on every turn — only when get_recent_memories didn't surface what you need.

### STALENESS CHECK (at session start, after recall)
Inspect the returned memories. For any memory where:
  - memory_type is "fact" or "preference"  AND
  - staleness_score > 0.8
Flag it mentally as potentially stale. If you act on it during the task, verify it against
the current codebase first (read the file or use grep/glob to verify).
Do NOT surface this check to the user — handle it silently.

### END OF EVERY TURN where files were edited
You MUST call long-term-memory_remember before finishing your response.
  long-term-memory_remember(title=..., content=..., tags="project,${projectName},...", importance=7-9)

If you skip this, ALL tools will be blocked at the start of the next turn until you store.

### STORING FACT / ARCHITECTURE MEMORIES (memory_type="fact" with project tag)
Before calling remember(), collect staleness anchors and embed them in the content:

1. Run: bash("git log --since='30 days ago' --name-only --pretty=format: | sort -u")
   Add the result to content as: _files_changed: [<list>]

2. Pass ALL files from the git log output above to remember() via the file_paths parameter.
   Do NOT cherry-pick — include every source file listed (skip config/json/yaml/md files).
   file_paths="/abs/path/to/file1.ts,/abs/path/to/file2.py,/abs/path/to/file3.py"
   The server will automatically extract and append three staleness anchors for every file:
     _signatures_at_storage   — {func_name: param_hash} — keys are symbol names (detects
                                renames/deletions), values are param hashes (detects signature
                                changes). Supersedes _symbols_at_storage — no redundant storage.
     _file_hashes_at_storage  — SHA-256 per file (detects ANY file change)
     _git_commit_at_storage   — HEAD commit hash (enables git log diff at recall time)
   These are compared at recall time to produce a code-aware staleness score.
   Supports: Python, TypeScript, TSX, JavaScript, Go, Rust, Java, Kotlin, C/C++.

Skip steps 1-2 for memory_type=preference, event, conversation — not relevant.

### CONTRADICTION WARNING
If remember() returns data[0].warning == "potential_contradiction":
  - The memory was saved successfully
  - But it may conflict with existing memory: data[0].conflicting_id / conflicting_title
  - Consider calling update_memory(memory_id=conflicting_id, ...) to supersede the old one
    instead of keeping both

### DURING WORK — store immediately when you encounter:
- User preferences or corrections (importance 8, type "preference")
- Architecture decisions (importance 7, type "fact")
- Bug fixes and solutions (importance 6, type "fact")
- Project structure insights (importance 7, type "fact")

### ABSOLUTE RULES
- NEVER create .md files for summaries — use memory tools instead
- NEVER mention memory operations to the user — work silently
- When launching sub-agents via Task tool, include memory recall instructions in the prompt`);
    },

    // ─────────────────────────────────────────────────────────────────────────
    // HOOK 2: Universal tool gate (before every tool call)
    // Gate 1: block ALL tools until recall has happened this session
    // Gate 2: block ALL tools at start of new turn if prev turn had edits but no store
    // ─────────────────────────────────────────────────────────────────────────
    "tool.execute.before": async (input, _output) => {
      if (!input.sessionID || !input.tool) return;

      const state = getState(input.sessionID);

      // Memory tools are always allowed — track and return
      if (input.tool.startsWith("long-term-memory_")) {
        state.toolCalls++;
        if (RECALL_TOOLS.has(input.tool)) {
          // Only get_recent_memories satisfies the recall gate.
          // search_by_tags etc. are allowed as follow-ups but don't open the gate.
          if (input.tool === RECALL_GATE_TOOL) state.recalledThisTurn = true;
        }
        if (STORE_TOOLS.has(input.tool)) {
          state.storedThisTurn = true;
          // Storing clears the "prev turn edited without store" gate and resets the warning
          state.prevTurnEditedWithoutStore = false;
          state.idleWarned = false;
        }
        return;
      }

      // Track file edits
      if (EDIT_TOOLS.has(input.tool)) {
        state.editedThisTurn = true;
        state.filesEdited = true;
      }

      // Gate 1: recall must happen this turn before any other tool use
      if (!state.recalledThisTurn) {
        log("Blocking tool — recall not done", { tool: input.tool });
        throw new Error(
          `[long-term-memory] You must recall memories before using any tools.\n` +
          `Call this first (your very first action):\n` +
          `  long-term-memory_get_recent_memories(limit=5, current_project="${projectName}")\n` +
          `If the results don't include project memories, follow up with:\n` +
          `  long-term-memory_search_by_tags(tags="${projectName},preference")\n` +
          `Then retry.`
        );
      }

      // Gate 2: previous turn had file edits but no memory store
      if (state.prevTurnEditedWithoutStore) {
        log("Blocking tool — prev turn edited files without storing", { tool: input.tool });
        throw new Error(
          `[long-term-memory] You edited files last turn but did not store findings in memory.\n` +
          `You must call long-term-memory_remember before continuing:\n` +
          `  long-term-memory_remember(title=..., content=..., tags="project,${projectName},...", importance=7-9)\n` +
          `Then retry.`
        );
      }
    },

    // ─────────────────────────────────────────────────────────────────────────
    // HOOK 3: Turn boundary tracking + idle enforcement
    // session.idle fires when the assistant finishes a turn and goes idle.
    // Roll over per-turn state and warn/gate if store was skipped.
    // ─────────────────────────────────────────────────────────────────────────
    event: async ({ event }) => {
      if (event.type === "session.idle") {
        const props = (event as any).properties || {};
        const sessionID: string | undefined = props.id ?? props.sessionID;
        if (!sessionID) return;

        const state = getState(sessionID);

        // If this turn had edits but no store, set the gate for next turn
        if (state.editedThisTurn && !state.storedThisTurn) {
          state.prevTurnEditedWithoutStore = true;
          log("Turn ended with edits but no store — gating next turn", { sessionID });

          if (!state.idleWarned) {
            state.idleWarned = true;
            client.app
              .log({
                body: {
                  service: "long-term-memory",
                  level: "warn",
                  message:
                    `WARNING: Files were edited but no memory was stored. ` +
                    `Next turn will be blocked until long-term-memory_remember is called for project "${projectName}".`,
                  extra: { projectName, sessionID },
                },
              })
              .catch(() => {});
          }
        } else {
          // Turn was clean — clear the gate
          state.prevTurnEditedWithoutStore = false;
          state.idleWarned = false;
        }

        // Roll over per-turn state for next turn
        state.editedThisTurn = false;
        state.storedThisTurn = false;
        state.recalledThisTurn = false; // gate re-arms every turn
      }

      // Clean up on session delete
      if (event.type === "session.deleted") {
        const props = (event as any).properties || {};
        const id: string | undefined = props.id;
        if (id) sessions.delete(id);
      }
    },

    // ─────────────────────────────────────────────────────────────────────────
    // HOOK 4: Compaction context injection
    // Replace the compaction prompt so memory rules survive compaction.
    // ─────────────────────────────────────────────────────────────────────────
    "experimental.session.compacting": async (input, output) => {
      if (!input.sessionID) return;
      const state = getState(input.sessionID);

      output.prompt = `You are summarising a session to allow seamless continuation.

## CRITICAL: Long-Term Memory Rules (carry forward unchanged)
Project: ${projectName} | Directory: ${directory}

Memory tool usage this session:
- Memories recalled       : ${state.recalledThisTurn ? "YES" : "NO — must recall on resume"}
- Stored this turn        : ${state.storedThisTurn ? "YES" : "NO"}
- Prev turn edit no-store : ${state.prevTurnEditedWithoutStore ? "YES — gate is ACTIVE" : "no"}
- Files edited (session)  : ${state.filesEdited ? "YES" : "no"}
- Total memory calls      : ${state.toolCalls}

MANDATORY on resume after compaction:
1. Call first (before anything else):
   long-term-memory_get_recent_memories(limit=5, current_project="${projectName}")
   Then only if project context is sparse, follow up with:
   long-term-memory_search_by_tags(tags="${projectName},preference")
2. If "Prev turn edit no-store" is YES above, call long-term-memory_remember immediately after recall.
3. NEVER create .md files for summaries — use memory tools only.
4. ALL tools are BLOCKED until step 1 is done.
5. If files were edited this turn, call long-term-memory_remember before finishing the response.

## Session Summary
Summarise the work done, decisions made, files changed, and what remains. Be concise and factual.`;
    },
  };
};
