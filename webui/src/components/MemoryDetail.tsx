import { useEffect, useRef, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { createMemory, updateMemory, deleteMemory } from '@/api/client'
import { useMemoryStore } from '@/store/memoryStore'
import type { Memory } from '@/api/types'

const TYPES = ['conversation', 'fact', 'preference', 'event', 'task', 'ephemeral']

interface MemoryDetailProps {
  onPeersClick: () => void
}

export function MemoryDetail({ onPeersClick }: MemoryDetailProps) {
  const { draft, patchDraft, setDraft, selectedId, setSelectedId, isNewMemory, setIsNewMemory, currentQueryKey } =
    useMemoryStore()
  const qc = useQueryClient()
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const saveTimeout = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    setError(null)
    setSaved(false)
  }, [selectedId, isNewMemory])

  const createMut = useMutation({
    mutationFn: createMemory,
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: currentQueryKey })
      qc.invalidateQueries({ queryKey: ['stats'] })
      const mem = res.data?.[0] as Memory | undefined
      if (mem) {
        setSelectedId(mem.id)
        setDraft({ ...mem })
        setIsNewMemory(false)
      }
      flashSaved()
    },
    onError: (e: Error) => setError(e.message),
  })

  const updateMut = useMutation({
    mutationFn: ({ id, body }: { id: string; body: Parameters<typeof updateMemory>[1] }) =>
      updateMemory(id, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: currentQueryKey })
      flashSaved()
    },
    onError: (e: Error) => setError(e.message),
  })

  const deleteMut = useMutation({
    mutationFn: deleteMemory,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: currentQueryKey })
      qc.invalidateQueries({ queryKey: ['stats'] })
      setSelectedId(null)
      setDraft(null)
      setIsNewMemory(false)
    },
    onError: (e: Error) => setError(e.message),
  })

  function flashSaved() {
    setSaved(true)
    if (saveTimeout.current) clearTimeout(saveTimeout.current)
    saveTimeout.current = setTimeout(() => setSaved(false), 2000)
  }

  function handleSave() {
    if (!draft) return
    if (!draft.title?.trim()) { setError('Title is required'); return }
    if (!draft.content?.trim()) { setError('Content is required'); return }
    setError(null)

    const tagList = typeof draft.tags === 'string'
      ? (draft.tags as string).split(',').map((t) => t.trim()).filter(Boolean)
      : draft.tags ?? []

    const sharedList = typeof draft.shared_with === 'string'
      ? (draft.shared_with as string).split(',').map((s) => s.trim()).filter(Boolean)
      : draft.shared_with ?? []

    if (isNewMemory) {
      createMut.mutate({
        title: draft.title!,
        content: draft.content!,
        memory_type: draft.memory_type ?? 'conversation',
        importance: draft.importance ?? 5,
        tags: tagList,
        shared_with: sharedList,
      })
    } else if (selectedId) {
      updateMut.mutate({
        id: selectedId,
        body: {
          title: draft.title,
          content: draft.content,
          memory_type: draft.memory_type,
          importance: draft.importance,
          tags: tagList,
          shared_with: sharedList,
        },
      })
    }
  }

  function handleDelete() {
    if (!selectedId || !draft?.title) return
    if (!window.confirm(`Delete "${draft.title}"? This cannot be undone.`)) return
    deleteMut.mutate(selectedId)
  }

  function handleNew() {
    setSelectedId(null)
    setIsNewMemory(true)
    setDraft({
      title: '',
      content: '',
      memory_type: 'conversation',
      importance: 5,
      tags: [],
      shared_with: [],
    })
    setError(null)
  }

  const isLoading = createMut.isPending || updateMut.isPending || deleteMut.isPending
  const tagsStr = Array.isArray(draft?.tags) ? draft!.tags.join(', ') : (draft?.tags ?? '')
  const sharedStr = Array.isArray(draft?.shared_with) ? draft!.shared_with.join(', ') : (draft?.shared_with ?? '')

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 6,
      overflow: 'hidden',
    }}>
      {/* Toolbar */}
      <div style={{
        display: 'flex', gap: 6, padding: '6px 8px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--surface2)',
        flexShrink: 0,
        flexWrap: 'wrap',
        alignItems: 'center',
      }}>
        <button onClick={handleNew} disabled={isLoading}>New Memory</button>
        <button className="btn-accent" onClick={handleSave} disabled={isLoading || !draft}>
          {isLoading ? 'Saving…' : saved ? 'Saved ✓' : 'Save Changes'}
        </button>
        <button className="btn-danger" onClick={handleDelete} disabled={!selectedId || isLoading}>
          Delete
        </button>
        {error && <span style={{ color: 'var(--danger)', fontSize: 11, marginLeft: 2 }}>{error}</span>}
      </div>

      {!draft ? (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
          Select a memory or click "New Memory"
        </div>
      ) : (
        // Outer flex column — fills all remaining height after toolbar
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>

          {/* ── Fixed-height fields section ──────────────────────────────── */}
          <div style={{ padding: '10px 10px 6px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
              Memory Details
            </div>

            {/* Title */}
            <Field label="Title">
              <input
                style={{ width: '100%', boxSizing: 'border-box' }}
                value={draft.title ?? ''}
                onChange={(e) => patchDraft({ title: e.target.value })}
              />
            </Field>

            {/* Type + Importance */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 8, marginTop: 6 }}>
              <Field label="Type">
                <select
                  style={{ width: '100%' }}
                  value={draft.memory_type ?? 'conversation'}
                  onChange={(e) => patchDraft({ memory_type: e.target.value })}
                >
                  {TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </Field>
              <Field label="Importance">
                <input
                  type="number"
                  min={1}
                  max={10}
                  style={{ width: 56 }}
                  value={draft.importance ?? 5}
                  onChange={(e) => patchDraft({ importance: Number(e.target.value) })}
                />
              </Field>
            </div>

            {/* Tags */}
            <div style={{ marginTop: 6 }}>
              <Field label="Tags (comma-separated)">
                <input
                  style={{ width: '100%', boxSizing: 'border-box' }}
                  value={tagsStr}
                  onChange={(e) => patchDraft({ tags: e.target.value.split(',').map((t) => t.trim()).filter(Boolean) })}
                />
              </Field>
            </div>

            {/* Share with */}
            <div style={{ marginTop: 6 }}>
              <Field label="Share with">
                <div style={{ display: 'flex', gap: 6 }}>
                  <input
                    style={{ flex: 1, minWidth: 0 }}
                    placeholder="UUIDs or * for everyone. Leave empty = private."
                    value={sharedStr}
                    onChange={(e) => patchDraft({ shared_with: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })}
                  />
                  <button onClick={onPeersClick} style={{ whiteSpace: 'nowrap', flexShrink: 0 }}>Peers…</button>
                </div>
              </Field>
            </div>
          </div>

          {/* ── Content — flex-grows to fill remaining space ─────────────── */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '6px 10px 6px', minHeight: 0 }}>
            <label style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, flexShrink: 0 }}>Content</label>
            <textarea
              style={{
                flex: 1,
                width: '100%',
                boxSizing: 'border-box',
                resize: 'none',         // no manual resize — it fills the column
                fontFamily: 'inherit',
                fontSize: 13,
                lineHeight: 1.65,
                minHeight: 0,           // allow shrinking below default
              }}
              value={draft.content ?? ''}
              onChange={(e) => patchDraft({ content: e.target.value })}
            />
          </div>

          {/* ── Metadata (read-only, scrollable) ─────────────────────────── */}
          {!isNewMemory && draft.id && (
            <div style={{
              flexShrink: 0,
              borderTop: '1px solid var(--border)',
              padding: '8px 10px',
              background: 'var(--bg)',
              maxHeight: 180,
              overflow: 'auto',
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>
                Metadata
              </div>
              <MetaRow label="ID" value={draft.id} mono />
              {draft.token_count !== undefined && <MetaRow label="Token Count" value={String(draft.token_count)} />}
              {draft.timestamp && <MetaRow label="Created" value={draft.timestamp} />}
              {draft.updated_at && <MetaRow label="Updated" value={draft.updated_at} />}
              {draft.last_accessed && <MetaRow label="Last Accessed" value={draft.last_accessed} />}
              {draft.staleness_score !== undefined && (
                <MetaRow label="Staleness" value={draft.staleness_score.toFixed(3)} warn={draft.staleness_warning} />
              )}
              {draft.metadata && Object.keys(draft.metadata).length > 0 && (
                <>
                  <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 6, marginBottom: 2 }}>Custom Metadata:</div>
                  {Object.entries(draft.metadata).map(([k, v]) => (
                    <MetaRow key={k} label={`  ${k}`} value={String(v)} />
                  ))}
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label style={{ display: 'block', fontSize: 11, color: 'var(--text-muted)', marginBottom: 3 }}>{label}</label>
      {children}
    </div>
  )
}

function MetaRow({ label, value, warn, mono }: { label: string; value: string; warn?: boolean; mono?: boolean }) {
  return (
    <div style={{ display: 'flex', gap: 8, marginBottom: 2, fontSize: 12 }}>
      <span style={{ color: 'var(--text-muted)', minWidth: 80, flexShrink: 0 }}>{label}:</span>
      <span style={{
        wordBreak: 'break-all',
        color: warn ? 'var(--warning)' : 'var(--text)',
        fontFamily: mono ? 'monospace' : 'inherit',
        fontSize: mono ? 11 : 12,
      }}>
        {value}
      </span>
    </div>
  )
}
