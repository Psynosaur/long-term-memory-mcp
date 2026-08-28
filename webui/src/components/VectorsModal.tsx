import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { getVector, getVectorStats, rebuildVectors } from '@/api/client'
import { useMemoryStore } from '@/store/memoryStore'
import { X } from 'lucide-react'

interface VectorsModalProps {
  onClose: () => void
}

export function VectorsModal({ onClose }: VectorsModalProps) {
  const { selectedId } = useMemoryStore()
  const [queryId, setQueryId] = useState(selectedId ?? '')

  const statsQ = useQuery({ queryKey: ['vectorStats'], queryFn: getVectorStats })
  const vecQ = useQuery({
    queryKey: ['vector', queryId],
    queryFn: () => getVector(queryId),
    enabled: queryId.length > 0,
  })

  const rebuildMut = useMutation({
    mutationFn: rebuildVectors,
  })

  const stats = statsQ.data
  const vec = vecQ.data

  return (
    <Modal title="Vector Visualization" onClose={onClose} width={700}>
      {/* Backend stats */}
      <Section title="Vector Backend Stats">
        {statsQ.isLoading ? <Muted>Loading…</Muted> : stats ? (
          <div style={{ fontSize: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
            <KV k="Backend" v={stats.backend} />
            <KV k="Count" v={String(stats.count)} />
            <KV k="Model" v={stats.embedding_model} />
            <KV k="Configured Dims" v={String(stats.configured_dims)} />
            <KV k="Actual Dims" v={stats.actual_dims !== null ? String(stats.actual_dims) : '—'} />
            {stats.dims_warning && (
              <div style={{ gridColumn: '1/-1', color: 'var(--warning)', marginTop: 4 }}>
                ⚠ {stats.dims_warning}
              </div>
            )}
          </div>
        ) : null}
        <button
          style={{ marginTop: 8 }}
          onClick={() => rebuildMut.mutate()}
          disabled={rebuildMut.isPending}
        >
          {rebuildMut.isPending ? 'Rebuilding…' : 'Rebuild Vector Index'}
        </button>
        {rebuildMut.isSuccess && (
          <span style={{ color: 'var(--success)', fontSize: 12, marginLeft: 8 }}>
            Done — {rebuildMut.data?.data?.[0]?.count ?? '?'} vectors indexed
          </span>
        )}
        {rebuildMut.isError && (
          <span style={{ color: 'var(--danger)', fontSize: 12, marginLeft: 8 }}>
            {(rebuildMut.error as Error).message}
          </span>
        )}
      </Section>

      {/* Query by ID */}
      <Section title="Query Vector by Memory ID">
        <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
          <input
            style={{ flex: 1 }}
            placeholder="Memory ID…"
            value={queryId}
            onChange={(e) => setQueryId(e.target.value)}
          />
          <button className="btn-accent" onClick={() => setQueryId(queryId.trim())}>
            Query
          </button>
          <button onClick={() => setQueryId('')}>Clear</button>
        </div>

        {vecQ.isLoading && <Muted>Fetching…</Muted>}
        {vecQ.isError && <div style={{ color: 'var(--danger)', fontSize: 12 }}>{(vecQ.error as Error).message}</div>}
        {vec && (
          <div style={{ fontFamily: 'monospace', fontSize: 11, background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 4, padding: 10, maxHeight: 280, overflow: 'auto' }}>
            {vec.document_preview && (
              <div style={{ marginBottom: 6 }}>
                <span style={{ color: 'var(--text-muted)' }}>Document preview: </span>
                {vec.document_preview}
              </div>
            )}
            {vec.stats && (
              <>
                <KV k="Dimensions" v={String(vec.stats.dimensions)} />
                <KV k="Min" v={vec.stats.min.toFixed(6)} />
                <KV k="Max" v={vec.stats.max.toFixed(6)} />
                <KV k="Mean" v={vec.stats.mean.toFixed(6)} />
                <KV k="L2 Norm" v={vec.stats.l2_norm.toFixed(6)} />
              </>
            )}
            {vec.first_20 && vec.first_20.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <div style={{ color: 'var(--text-muted)', marginBottom: 3 }}>First 20 dimensions:</div>
                {vec.first_20.map((v, i) => (
                  <div key={i} style={{ display: 'flex', gap: 8 }}>
                    <span style={{ color: 'var(--text-muted)', minWidth: 30, textAlign: 'right' }}>[{i}]</span>
                    <span>{v.toFixed(8)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Section>
    </Modal>
  )
}

// ── Shared primitives ──────────────────────────────────────────────────────────

export function Modal({
  title,
  onClose,
  children,
  width = 600,
}: {
  title: string
  onClose: () => void
  children: React.ReactNode
  width?: number
}) {
  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 100,
        background: 'rgba(0,0,0,0.7)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 8,
        width, maxWidth: '95vw',
        maxHeight: '90vh',
        display: 'flex', flexDirection: 'column',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', borderBottom: '1px solid var(--border)', background: 'var(--surface2)' }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent)' }}>{title}</span>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', padding: 2, cursor: 'pointer', color: 'var(--text-muted)' }}
          >
            <X size={16} />
          </button>
        </div>
        <div style={{ overflow: 'auto', padding: 14, flex: 1 }}>
          {children}
        </div>
      </div>
    </div>
  )
}

export function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>{title}</div>
      {children}
    </div>
  )
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: 'flex', gap: 8, marginBottom: 2 }}>
      <span style={{ color: 'var(--text-muted)', minWidth: 130, flexShrink: 0 }}>{k}:</span>
      <span style={{ wordBreak: 'break-all' }}>{v}</span>
    </div>
  )
}

function Muted({ children }: { children: React.ReactNode }) {
  return <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>{children}</div>
}
