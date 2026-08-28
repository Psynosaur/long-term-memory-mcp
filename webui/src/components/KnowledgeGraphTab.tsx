import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getKgStats,
  getKgTimeline,
  queryKgEntity,
  addKgTriple,
  invalidateKgTriple,
} from '@/api/client'
import type { KgTriple } from '@/api/types'

// ── helpers ───────────────────────────────────────────────────────────────────

function Badge({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <span style={{
      display: 'inline-block', padding: '1px 7px', borderRadius: 10,
      fontSize: 11, fontWeight: 600, background: color, color: '#fff',
    }}>
      {children}
    </span>
  )
}

function TripleRow({ triple, onInvalidate }: { triple: KgTriple; onInvalidate: (t: KgTriple) => void }) {
  return (
    <tr style={{ borderBottom: '1px solid var(--border)' }}>
      <td style={{ padding: '5px 8px', fontWeight: 600 }}>{triple.subject}</td>
      <td style={{ padding: '5px 8px', color: 'var(--accent)', fontFamily: 'monospace', fontSize: 12 }}>
        {triple.predicate}
      </td>
      <td style={{ padding: '5px 8px' }}>{triple.object}</td>
      <td style={{ padding: '5px 8px', fontSize: 11, color: 'var(--text-muted)' }}>
        {triple.valid_from ?? '—'}
      </td>
      <td style={{ padding: '5px 8px', fontSize: 11, color: 'var(--text-muted)' }}>
        {triple.valid_to ?? '—'}
      </td>
      <td style={{ padding: '5px 8px' }}>
        {triple.current
          ? <Badge color="var(--success)">current</Badge>
          : <Badge color="var(--text-muted)">expired</Badge>}
      </td>
      <td style={{ padding: '5px 8px' }}>
        {triple.current && (
          <button
            onClick={() => onInvalidate(triple)}
            style={{ fontSize: 11, padding: '2px 8px', background: 'var(--danger)', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer' }}
          >
            Expire
          </button>
        )}
      </td>
    </tr>
  )
}

// ── Add Triple form ───────────────────────────────────────────────────────────

function AddTripleForm({ onDone }: { onDone: () => void }) {
  const [subject, setSubject] = useState('')
  const [predicate, setPredicate] = useState('')
  const [obj, setObj] = useState('')
  const [validFrom, setValidFrom] = useState('')
  const [validTo, setValidTo] = useState('')
  const [confidence, setConfidence] = useState('1.0')
  const [sourceId, setSourceId] = useState('')
  const [err, setErr] = useState<string | null>(null)

  const qc = useQueryClient()
  const mut = useMutation({
    mutationFn: addKgTriple,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['kg'] })
      onDone()
    },
    onError: (e: Error) => setErr(e.message),
  })

  function submit(e: React.FormEvent) {
    e.preventDefault()
    setErr(null)
    if (!subject.trim() || !predicate.trim() || !obj.trim()) {
      setErr('Subject, predicate, and object are required.')
      return
    }
    mut.mutate({
      subject: subject.trim(),
      predicate: predicate.trim(),
      obj: obj.trim(),
      valid_from: validFrom || undefined,
      valid_to: validTo || undefined,
      confidence: parseFloat(confidence) || 1.0,
      source_memory_id: sourceId || undefined,
    })
  }

  const inp: React.CSSProperties = {
    width: '100%', padding: '4px 8px', fontSize: 13, boxSizing: 'border-box',
    background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text)',
  }
  const lbl: React.CSSProperties = { fontSize: 11, color: 'var(--text-muted)', marginBottom: 2, display: 'block' }

  return (
    <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
        <div><label style={lbl}>Subject *</label><input style={inp} value={subject} onChange={e => setSubject(e.target.value)} placeholder="Alice" /></div>
        <div><label style={lbl}>Predicate *</label><input style={inp} value={predicate} onChange={e => setPredicate(e.target.value)} placeholder="works_at" /></div>
        <div><label style={lbl}>Object *</label><input style={inp} value={obj} onChange={e => setObj(e.target.value)} placeholder="Acme Corp" /></div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 8 }}>
        <div><label style={lbl}>Valid From</label><input style={inp} type="date" value={validFrom} onChange={e => setValidFrom(e.target.value)} /></div>
        <div><label style={lbl}>Valid To</label><input style={inp} type="date" value={validTo} onChange={e => setValidTo(e.target.value)} /></div>
        <div><label style={lbl}>Confidence (0–1)</label><input style={inp} type="number" min="0" max="1" step="0.1" value={confidence} onChange={e => setConfidence(e.target.value)} /></div>
        <div><label style={lbl}>Source Memory ID</label><input style={inp} value={sourceId} onChange={e => setSourceId(e.target.value)} placeholder="mem_..." /></div>
      </div>
      {err && <div style={{ fontSize: 12, color: 'var(--danger)' }}>{err}</div>}
      <div style={{ display: 'flex', gap: 8 }}>
        <button type="submit" disabled={mut.isPending} style={{ background: 'var(--accent)', color: '#fff', border: 'none', borderRadius: 4, padding: '5px 16px', cursor: 'pointer' }}>
          {mut.isPending ? 'Adding…' : 'Add Triple'}
        </button>
        <button type="button" onClick={onDone} style={{ padding: '5px 12px', borderRadius: 4 }}>Cancel</button>
      </div>
    </form>
  )
}

// ── Main KG Tab ───────────────────────────────────────────────────────────────

export function KnowledgeGraphTab() {
  const [mode, setMode] = useState<'timeline' | 'entity'>('timeline')
  const [entityInput, setEntityInput] = useState('')
  const [entityQuery, setEntityQuery] = useState('')
  const [asOf, setAsOf] = useState('')
  const [direction, setDirection] = useState<'outgoing' | 'incoming' | 'both'>('both')
  const [showAdd, setShowAdd] = useState(false)
  const [invalidateTarget, setInvalidateTarget] = useState<KgTriple | null>(null)
  const [statusMsg, setStatusMsg] = useState<string | null>(null)

  const qc = useQueryClient()

  const statsQ = useQuery({
    queryKey: ['kg', 'stats'],
    queryFn: () => getKgStats(),
  })

  const timelineQ = useQuery({
    queryKey: ['kg', 'timeline', mode],
    queryFn: () => getKgTimeline(undefined, 200),
    enabled: mode === 'timeline',
  })

  const entityQ = useQuery({
    queryKey: ['kg', 'entity', entityQuery, asOf, direction],
    queryFn: () => queryKgEntity(entityQuery, asOf || undefined, direction),
    enabled: mode === 'entity' && entityQuery.length > 0,
  })

  const invalidateMut = useMutation({
    mutationFn: invalidateKgTriple,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['kg'] })
      setStatusMsg('Triple expired.')
      setInvalidateTarget(null)
      setTimeout(() => setStatusMsg(null), 3000)
    },
    onError: (e: Error) => setStatusMsg(`Error: ${e.message}`),
  })

  const triples: KgTriple[] = mode === 'timeline'
    ? (timelineQ.data?.data ?? [])
    : (entityQ.data?.data ?? [])

  const stats = statsQ.data?.data?.[0]

  const inp: React.CSSProperties = {
    padding: '4px 8px', fontSize: 13,
    background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text)',
  }
  const tabBtn = (active: boolean): React.CSSProperties => ({
    padding: '5px 16px', borderRadius: 4, cursor: 'pointer', fontWeight: active ? 700 : 400,
    background: active ? 'var(--accent)' : 'var(--surface2)',
    color: active ? '#fff' : 'var(--text)',
    border: 'none',
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', padding: 16, gap: 12 }}>
      {/* Stats bar */}
      {stats && (
        <div style={{ display: 'flex', gap: 20, padding: '8px 12px', background: 'var(--surface2)', borderRadius: 6, flexShrink: 0 }}>
          <span><strong>{stats.entities}</strong> <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>entities</span></span>
          <span><strong>{stats.triples}</strong> <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>triples</span></span>
          <span><strong>{stats.current_facts}</strong> <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>current</span></span>
          <span><strong>{stats.expired_facts}</strong> <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>expired</span></span>
          {stats.relationship_types.length > 0 && (
            <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>
              predicates: {stats.relationship_types.join(', ')}
            </span>
          )}
        </div>
      )}

      {/* Controls */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', flexShrink: 0 }}>
        <button style={tabBtn(mode === 'timeline')} onClick={() => setMode('timeline')}>Timeline</button>
        <button style={tabBtn(mode === 'entity')} onClick={() => setMode('entity')}>Entity Query</button>

        {mode === 'entity' && (
          <>
            <input
              style={{ ...inp, width: 200 }}
              placeholder="Entity name…"
              value={entityInput}
              onChange={e => setEntityInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') setEntityQuery(entityInput.trim()) }}
            />
            <input style={{ ...inp, width: 130 }} type="date" value={asOf} onChange={e => setAsOf(e.target.value)} title="As-of date (time travel)" placeholder="as-of date" />
            <select style={inp} value={direction} onChange={e => setDirection(e.target.value as 'outgoing' | 'incoming' | 'both')}>
              <option value="both">both</option>
              <option value="outgoing">outgoing</option>
              <option value="incoming">incoming</option>
            </select>
            <button onClick={() => setEntityQuery(entityInput.trim())}>Query</button>
          </>
        )}

        <div style={{ flex: 1 }} />
        <button
          onClick={() => setShowAdd(v => !v)}
          style={{ background: 'var(--success)', color: '#fff', border: 'none', borderRadius: 4, padding: '5px 14px', cursor: 'pointer' }}
        >
          + Add Triple
        </button>
      </div>

      {/* Add triple form */}
      {showAdd && (
        <div style={{ padding: 12, background: 'var(--surface2)', borderRadius: 6, flexShrink: 0 }}>
          <AddTripleForm onDone={() => { setShowAdd(false); qc.invalidateQueries({ queryKey: ['kg'] }) }} />
        </div>
      )}

      {/* Status message */}
      {statusMsg && (
        <div style={{ fontSize: 12, padding: '4px 10px', borderRadius: 4, background: 'var(--surface2)', color: statusMsg.startsWith('Error') ? 'var(--danger)' : 'var(--success)' }}>
          {statusMsg}
        </div>
      )}

      {/* Invalidate confirm */}
      {invalidateTarget && (
        <div style={{ padding: 12, background: 'var(--surface2)', borderRadius: 6, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 13 }}>
            Expire <strong>{invalidateTarget.subject}</strong> → <em>{invalidateTarget.predicate}</em> → <strong>{invalidateTarget.object}</strong>?
          </span>
          <button
            onClick={() => invalidateMut.mutate({ subject: invalidateTarget.subject, predicate: invalidateTarget.predicate, obj: invalidateTarget.object })}
            disabled={invalidateMut.isPending}
            style={{ background: 'var(--danger)', color: '#fff', border: 'none', borderRadius: 4, padding: '4px 12px', cursor: 'pointer' }}
          >
            {invalidateMut.isPending ? 'Expiring…' : 'Confirm Expire'}
          </button>
          <button onClick={() => setInvalidateTarget(null)}>Cancel</button>
        </div>
      )}

      {/* Triple table */}
      <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
        {(mode === 'timeline' ? timelineQ.isLoading : entityQ.isLoading) && (
          <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)' }}>Loading…</div>
        )}
        {mode === 'entity' && !entityQuery && (
          <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)' }}>Enter an entity name and press Query.</div>
        )}
        {triples.length > 0 && (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: 'var(--surface2)', position: 'sticky', top: 0 }}>
                <th style={{ padding: '6px 8px', textAlign: 'left', fontWeight: 600 }}>Subject</th>
                <th style={{ padding: '6px 8px', textAlign: 'left', fontWeight: 600 }}>Predicate</th>
                <th style={{ padding: '6px 8px', textAlign: 'left', fontWeight: 600 }}>Object</th>
                <th style={{ padding: '6px 8px', textAlign: 'left', fontWeight: 600 }}>Valid From</th>
                <th style={{ padding: '6px 8px', textAlign: 'left', fontWeight: 600 }}>Valid To</th>
                <th style={{ padding: '6px 8px', textAlign: 'left', fontWeight: 600 }}>Status</th>
                <th style={{ padding: '6px 8px' }} />
              </tr>
            </thead>
            <tbody>
              {triples.map((t, i) => (
                <TripleRow key={i} triple={t} onInvalidate={setInvalidateTarget} />
              ))}
            </tbody>
          </table>
        )}
        {triples.length === 0 && !((mode === 'timeline' ? timelineQ.isLoading : entityQ.isLoading)) && entityQuery && (
          <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)' }}>No facts found.</div>
        )}
        {triples.length === 0 && mode === 'timeline' && !timelineQ.isLoading && (
          <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)' }}>Knowledge graph is empty. Add some triples above.</div>
        )}
      </div>
    </div>
  )
}
