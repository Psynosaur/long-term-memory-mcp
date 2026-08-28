import { useState, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { callTool } from '@/api/client'

// ── Tool schema: what args each tool takes ────────────────────────────────────

interface ArgDef {
  name: string
  type: 'string' | 'number' | 'float' | 'boolean' | 'textarea'
  required?: boolean
  placeholder?: string
  default?: string
}

interface ToolDef {
  name: string
  description: string
  args: ArgDef[]
}

const TOOLS: ToolDef[] = [
  {
    name: 'remember',
    description: 'Store a new memory.',
    args: [
      { name: 'title', type: 'string', required: true, placeholder: 'Short title' },
      { name: 'content', type: 'textarea', required: true, placeholder: 'Full content…' },
      { name: 'tags', type: 'string', placeholder: 'comma,separated' },
      { name: 'importance', type: 'number', placeholder: '5', default: '5' },
      { name: 'memory_type', type: 'string', placeholder: 'conversation', default: 'conversation' },
      { name: 'shared_with', type: 'string', placeholder: '* or uuid1,uuid2' },
      { name: 'file_paths', type: 'string', placeholder: '/abs/path/a.py,/abs/path/b.py' },
    ],
  },
  {
    name: 'search_memories',
    description: 'Semantic search across all memories.',
    args: [
      { name: 'query', type: 'string', required: true, placeholder: 'What are my preferences?' },
      { name: 'limit', type: 'number', placeholder: '10', default: '10' },
    ],
  },
  {
    name: 'search_by_type',
    description: 'Retrieve memories by type.',
    args: [
      { name: 'memory_type', type: 'string', required: true, placeholder: 'preference' },
      { name: 'limit', type: 'number', placeholder: '20', default: '20' },
    ],
  },
  {
    name: 'search_by_tags',
    description: 'Find memories by tags.',
    args: [
      { name: 'tags', type: 'string', required: true, placeholder: 'project,long-term-memory-mcp' },
      { name: 'limit', type: 'number', placeholder: '20', default: '20' },
    ],
  },
  {
    name: 'get_recent_memories',
    description: 'Most recent memories, optionally filtered by project.',
    args: [
      { name: 'limit', type: 'number', placeholder: '20', default: '20' },
      { name: 'current_project', type: 'string', placeholder: 'long-term-memory-mcp' },
    ],
  },
  {
    name: 'update_memory',
    description: 'Partially update a memory by ID.',
    args: [
      { name: 'memory_id', type: 'string', required: true, placeholder: 'mem_...' },
      { name: 'title', type: 'string', placeholder: 'New title' },
      { name: 'content', type: 'textarea', placeholder: 'New content…' },
      { name: 'tags', type: 'string', placeholder: 'tag1,tag2' },
      { name: 'importance', type: 'number', placeholder: '7' },
      { name: 'memory_type', type: 'string', placeholder: 'fact' },
      { name: 'shared_with', type: 'string', placeholder: '* or uuid' },
    ],
  },
  {
    name: 'delete_memory',
    description: 'Permanently delete a memory.',
    args: [
      { name: 'memory_id', type: 'string', required: true, placeholder: 'mem_...' },
    ],
  },
  {
    name: 'get_memory_stats',
    description: 'Memory system statistics.',
    args: [],
  },
  {
    name: 'create_backup',
    description: 'Trigger an on-demand backup.',
    args: [],
  },
  {
    name: 'search_by_date_range',
    description: 'Find memories in a date range.',
    args: [
      { name: 'date_from', type: 'string', required: true, placeholder: '2025-01-01' },
      { name: 'date_to', type: 'string', placeholder: '2025-12-31' },
      { name: 'limit', type: 'number', placeholder: '50', default: '50' },
    ],
  },
  {
    name: 'rebuild_vectors',
    description: 'Rebuild the vector index from SQLite.',
    args: [],
  },
  {
    name: 'kg_add',
    description: 'Add a fact triple to the knowledge graph.',
    args: [
      { name: 'subject', type: 'string', required: true, placeholder: 'Alice' },
      { name: 'predicate', type: 'string', required: true, placeholder: 'works_at' },
      { name: 'obj', type: 'string', required: true, placeholder: 'Acme Corp' },
      { name: 'valid_from', type: 'string', placeholder: '2023-01-01' },
      { name: 'valid_to', type: 'string', placeholder: '2025-01-01' },
      { name: 'confidence', type: 'float', placeholder: '1.0', default: '1.0' },
      { name: 'source_memory_id', type: 'string', placeholder: 'mem_...' },
    ],
  },
  {
    name: 'kg_invalidate',
    description: 'Expire an open triple.',
    args: [
      { name: 'subject', type: 'string', required: true, placeholder: 'Alice' },
      { name: 'predicate', type: 'string', required: true, placeholder: 'works_at' },
      { name: 'obj', type: 'string', required: true, placeholder: 'Acme Corp' },
      { name: 'ended', type: 'string', placeholder: 'today (leave blank)' },
    ],
  },
  {
    name: 'kg_query',
    description: 'Query facts about an entity, with optional time-travel.',
    args: [
      { name: 'entity', type: 'string', required: true, placeholder: 'Alice' },
      { name: 'as_of', type: 'string', placeholder: '2024-01-01 (optional)' },
      { name: 'direction', type: 'string', placeholder: 'both', default: 'both' },
    ],
  },
  {
    name: 'kg_timeline',
    description: 'Chronological fact history, optionally filtered to one entity.',
    args: [
      { name: 'entity', type: 'string', placeholder: 'Alice (leave blank for global)' },
      { name: 'limit', type: 'number', placeholder: '100', default: '100' },
    ],
  },
  {
    name: 'kg_stats',
    description: 'Knowledge graph entity and triple counts.',
    args: [],
  },
  {
    name: 'get_memory_system_info',
    description: 'Full system status: memories, KG, backends, storage.',
    args: [],
  },
]

// ── Styles ────────────────────────────────────────────────────────────────────

const inp: React.CSSProperties = {
  width: '100%', padding: '5px 8px', fontSize: 13, boxSizing: 'border-box',
  background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text)',
}
const lbl: React.CSSProperties = {
  fontSize: 11, color: 'var(--text-muted)', marginBottom: 2, display: 'block',
}

// ── Main ToolsTab component ───────────────────────────────────────────────────

export function ToolsTab() {
  const [selectedTool, setSelectedTool] = useState<ToolDef>(TOOLS[1]) // search_memories default
  const [args, setArgs] = useState<Record<string, string>>({})
  const [result, setResult] = useState<unknown>(null)
  const [error, setError] = useState<string | null>(null)

  // Reset args when tool changes
  useEffect(() => {
    const defaults: Record<string, string> = {}
    for (const a of selectedTool.args) {
      if (a.default !== undefined) defaults[a.name] = a.default
    }
    setArgs(defaults)
    setResult(null)
    setError(null)
  }, [selectedTool.name])

  const mut = useMutation({
    mutationFn: callTool,
    onSuccess: (data) => {
      setResult(data)
      setError(null)
    },
    onError: (e: Error) => {
      setError(e.message)
      setResult(null)
    },
  })

  function setArg(name: string, value: string) {
    setArgs(prev => ({ ...prev, [name]: value }))
  }

  function buildArgs(): Record<string, unknown> {
    const out: Record<string, unknown> = {}
    for (const def of selectedTool.args) {
      const raw = args[def.name]
      if (raw === undefined || raw === '') continue
      if (def.type === 'number') out[def.name] = parseInt(raw, 10)
      else if (def.type === 'float') out[def.name] = parseFloat(raw)
      else if (def.type === 'boolean') out[def.name] = raw === 'true'
      else out[def.name] = raw
    }
    return out
  }

  function submit(e: React.FormEvent) {
    e.preventDefault()
    // Required check
    for (const def of selectedTool.args) {
      if (def.required && !args[def.name]?.trim()) {
        setError(`"${def.name}" is required.`)
        return
      }
    }
    setError(null)
    mut.mutate({ tool: selectedTool.name, args: buildArgs() })
  }

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden', gap: 0 }}>
      {/* Left: tool list */}
      <aside style={{
        width: 220, flexShrink: 0, overflow: 'auto',
        borderRight: '1px solid var(--border)', padding: 8,
        display: 'flex', flexDirection: 'column', gap: 2,
      }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', padding: '4px 8px', textTransform: 'uppercase', letterSpacing: 1 }}>
          Tools
        </div>
        {TOOLS.map(t => (
          <button
            key={t.name}
            onClick={() => setSelectedTool(t)}
            style={{
              textAlign: 'left', padding: '6px 10px', borderRadius: 4,
              border: 'none', cursor: 'pointer', fontSize: 12, fontFamily: 'monospace',
              background: selectedTool.name === t.name ? 'var(--accent)' : 'transparent',
              color: selectedTool.name === t.name ? '#fff' : 'var(--text)',
              fontWeight: selectedTool.name === t.name ? 700 : 400,
            }}
          >
            {t.name}
          </button>
        ))}
      </aside>

      {/* Right: form + result */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: 16, gap: 12 }}>
        {/* Tool header */}
        <div>
          <div style={{ fontSize: 16, fontWeight: 700, fontFamily: 'monospace', color: 'var(--accent)' }}>
            {selectedTool.name}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
            {selectedTool.description}
          </div>
        </div>

        {/* Args form */}
        <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {selectedTool.args.length === 0 && (
            <div style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>No arguments required.</div>
          )}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 10 }}>
            {selectedTool.args.map(def => (
              <div key={def.name}>
                <label style={lbl}>
                  {def.name}
                  {def.required && <span style={{ color: 'var(--danger)', marginLeft: 3 }}>*</span>}
                  <span style={{ marginLeft: 6, color: 'var(--text-muted)', fontStyle: 'italic' }}>
                    ({def.type})
                  </span>
                </label>
                {def.type === 'textarea' ? (
                  <textarea
                    style={{ ...inp, height: 80, resize: 'vertical' }}
                    value={args[def.name] ?? ''}
                    onChange={e => setArg(def.name, e.target.value)}
                    placeholder={def.placeholder}
                  />
                ) : (
                  <input
                    style={inp}
                    type={def.type === 'number' || def.type === 'float' ? 'number' : 'text'}
                    step={def.type === 'float' ? '0.1' : undefined}
                    value={args[def.name] ?? ''}
                    onChange={e => setArg(def.name, e.target.value)}
                    placeholder={def.placeholder}
                  />
                )}
              </div>
            ))}
          </div>

          {error && (
            <div style={{ fontSize: 12, color: 'var(--danger)', padding: '4px 8px', background: 'var(--surface2)', borderRadius: 4 }}>
              {error}
            </div>
          )}

          <div>
            <button
              type="submit"
              disabled={mut.isPending}
              style={{ background: 'var(--accent)', color: '#fff', border: 'none', borderRadius: 4, padding: '6px 20px', cursor: 'pointer', fontWeight: 600 }}
            >
              {mut.isPending ? 'Running…' : 'Run'}
            </button>
          </div>
        </form>

        {/* Result */}
        {result !== null && (
          <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>
              Result
            </div>
            <pre style={{
              background: 'var(--surface2)', padding: 12, borderRadius: 6,
              fontSize: 12, overflow: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-all',
              border: '1px solid var(--border)', color: 'var(--text)', margin: 0,
            }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}
