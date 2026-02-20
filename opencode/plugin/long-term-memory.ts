import type { Plugin } from "@opencode-ai/plugin";

/**
 * Long-Term Memory Enforcement Plugin
 *
 * Hooks into OpenCode's lifecycle to enforce memory usage:
 * 1. System prompt injection - Adds memory instructions to every LLM call
 * 2. Compaction hook - Preserves memory awareness across compactions
 */
export const LongTermMemoryPlugin: Plugin = async ({
  client,
  directory,
}) => {
  const projectName = directory.split("/").pop() || "unknown";

  // Track memory tool usage per session
  const sessionMemoryUsage = new Map<
    string,
    { recalled: boolean; stored: boolean; toolCalls: number }
  >();

  const getSessionState = (sessionID: string) => {
    if (!sessionMemoryUsage.has(sessionID)) {
      sessionMemoryUsage.set(sessionID, {
        recalled: false,
        stored: false,
        toolCalls: 0,
      });
    }
    return sessionMemoryUsage.get(sessionID)!;
  };

  /** Non-blocking logger */
  const log = (message: string, extra: Record<string, any> = {}) => {
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

  return {
    /**
     * HOOK 1: System Prompt Injection (PRIMARY ENFORCER)
     *
     * Injects memory instructions directly into the system prompt.
     * This is the most reliable enforcement - it's part of every LLM call.
     */
    "experimental.chat.system.transform": async (_input, output) => {
      output.system.push(`## Long-Term Memory System

You have access to a persistent long-term memory system via MCP tools prefixed with \`long-term-memory_\`.

### MANDATORY: Start of Every Task
Before doing ANY work, you MUST call these tools (in parallel if possible):
1. \`long-term-memory_get_recent_memories\` with limit=5, current_project="${projectName}"
2. \`long-term-memory_search_by_tags\` with tags="preference,project"

### MANDATORY: End of Every Task
After completing work, you MUST call:
- \`long-term-memory_remember\` to store what was accomplished, decisions made, and key findings
- Tag with: "project,${projectName}" plus relevant tags
- Set importance 7-9 for project facts, 5-6 for session context

### During Work
Store immediately when you encounter:
- User preferences or corrections
- Architecture decisions
- Solutions to problems
- Project structure insights

### Rules
- NEVER create .md files for summaries - use memory tools instead
- NEVER mention memory storage to the user - do it silently
- When launching sub-agents via Task tool, include memory instructions in the prompt`);
    },

    /**
     * HOOK 2: Compaction Context Injection
     *
     * Preserves memory awareness when session history is compacted.
     */
    "experimental.session.compacting": async (input, output) => {
      if (!input.sessionID) return;
      const state = getSessionState(input.sessionID);

      output.context.push(`## Long-Term Memory Context

This session has access to a persistent memory system (long-term-memory MCP tools).
Project: ${projectName} | Directory: ${directory}

Memory tool usage this session:
- Memories recalled: ${state.recalled ? "Yes" : "No"}
- Memories stored: ${state.stored ? "Yes" : "No"}
- Total memory tool calls: ${state.toolCalls}

CRITICAL for continuation:
1. After compaction, the agent MUST call long-term-memory_get_recent_memories(limit=5, current_project="${projectName}") to restore context
2. The agent MUST call long-term-memory_search_by_tags(tags="preference,project") to reload preferences
3. Any findings from this session that were not yet stored MUST be stored using long-term-memory_remember
4. Do NOT create markdown files for summaries - use memory tools
5. Include project tag "${projectName}" on all stored memories`);
    },

    /**
     * HOOK 3: Tool Execution Monitoring
     *
     * Tracks when memory tools are called for compaction context.
     */
    "tool.execute.before": async (input, _output) => {
      if (!input.sessionID || !input.tool) return;
      if (!input.tool.startsWith("long-term-memory_")) return;

      const state = getSessionState(input.sessionID);
      state.toolCalls++;

      if (
        input.tool === "long-term-memory_get_recent_memories" ||
        input.tool === "long-term-memory_search_memories" ||
        input.tool === "long-term-memory_search_by_tags" ||
        input.tool === "long-term-memory_search_by_type" ||
        input.tool === "long-term-memory_search_by_date_range"
      ) {
        state.recalled = true;
      }

      if (
        input.tool === "long-term-memory_remember" ||
        input.tool === "long-term-memory_update_memory"
      ) {
        state.stored = true;
      }
    },

    /**
     * HOOK 4: Event Handler
     *
     * Cleans up session state on deletion.
     */
    event: async ({ event }) => {
      if (event.type === "session.deleted") {
        const props = (event as any).properties || {};
        if (props.id) {
          sessionMemoryUsage.delete(props.id);
        }
      }
    },
  };
};
