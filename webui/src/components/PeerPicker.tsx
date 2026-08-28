import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getPeers } from '@/api/client'
import type { Peer } from '@/api/types'
import { Modal } from './VectorsModal'

interface PeerPickerProps {
  currentValue: string
  onApply: (value: string) => void
  onClose: () => void
}

export function PeerPicker({ currentValue, onApply, onClose }: PeerPickerProps) {
  const { data, isLoading } = useQuery({
    queryKey: ['peers'],
    queryFn: getPeers,
  })

  const peers: Peer[] = data?.peers ?? []
  const [everyone, setEveryone] = useState(currentValue === '*')
  const [selected, setSelected] = useState<Set<string>>(
    () => new Set(currentValue.split(',').map((s) => s.trim()).filter((s) => s && s !== '*')),
  )

  function togglePeer(uuid: string) {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(uuid)) next.delete(uuid)
      else next.add(uuid)
      return next
    })
  }

  function handleApply() {
    if (everyone) {
      onApply('*')
    } else {
      onApply(Array.from(selected).join(','))
    }
    onClose()
  }

  return (
    <Modal title="Share With Peers" onClose={onClose} width={420}>
      {isLoading ? (
        <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>Discovering peers…</div>
      ) : peers.length === 0 ? (
        <div style={{ color: 'var(--text-muted)', fontSize: 13, marginBottom: 12 }}>
          No peers discovered. You can enter UUIDs manually in the "Share with" field.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 16 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 13 }}>
            <input type="checkbox" checked={everyone} onChange={(e) => setEveryone(e.target.checked)} />
            <span style={{ color: 'var(--accent)', fontWeight: 600 }}>Everyone (broadcast)</span>
          </label>
          <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}>
            {peers.map((p) => (
              <label key={p.node_uuid} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, cursor: 'pointer', fontSize: 13 }}>
                <input
                  type="checkbox"
                  disabled={everyone}
                  checked={selected.has(p.node_uuid)}
                  onChange={() => togglePeer(p.node_uuid)}
                />
                <span>{p.username}</span>
                <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>{p.node_uuid.slice(0, 8)}…</span>
              </label>
            ))}
          </div>
        </div>
      )}
      <div style={{ display: 'flex', gap: 8 }}>
        <button className="btn-accent" onClick={handleApply}>Apply</button>
        <button onClick={onClose}>Cancel</button>
      </div>
    </Modal>
  )
}
