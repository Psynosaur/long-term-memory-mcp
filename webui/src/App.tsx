import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { listMemories, createBackup, downloadExport } from '@/api/client'
import { useMemoryStore } from '@/store/memoryStore'
import { memoryQueryKey } from '@/api/queryKeys'
import { SearchPanel } from '@/components/SearchPanel'
import { MemoryList } from '@/components/MemoryList'
import { MemoryDetail } from '@/components/MemoryDetail'
import { StatsPanel } from '@/components/StatsPanel'
import { VectorsModal } from '@/components/VectorsModal'
import { MigrateModal } from '@/components/MigrateModal'
import { CompareModal } from '@/components/CompareModal'
import { PeerPicker } from '@/components/PeerPicker'
import type { Memory } from '@/api/types'

type ModalType = 'vectors' | 'migrate' | 'compare' | 'peers' | null

export default function App() {
  const {
    searchText, filterType, filterMinImportance,
    filterTags, sortOrder, searchMode,
    draft, patchDraft,
    page, setPage, pageSize,
    setCurrentQueryKey,
  } = useMemoryStore()

  const [modal, setModal] = useState<ModalType>(null)
  const [backupMsg, setBackupMsg] = useState<string | null>(null)
  const qc = useQueryClient()

  const queryKey = memoryQueryKey(
    searchText, filterType, filterMinImportance,
    filterTags, sortOrder, searchMode,
    page, pageSize,
  )

  // Keep the store in sync so MemoryDetail can invalidate the right key
  useEffect(() => { setCurrentQueryKey(queryKey) }, [queryKey.join(',')])  // eslint-disable-line

  const memoriesQ = useQuery({
    queryKey,
    queryFn: () => listMemories({
      q: searchText || undefined,
      type: filterType || undefined,
      min_importance: filterMinImportance > 1 ? filterMinImportance : undefined,
      tags: filterTags || undefined,
      sort: sortOrder,
      search_type: searchMode,
      limit: pageSize,
      offset: page * pageSize,
    }),
  })

  const memories: Memory[] = memoriesQ.data?.data ?? []
  const total: number = memoriesQ.data?.total ?? 0

  // Manual refresh: reset to page 0 and invalidate only the current key.
  // Other cached pages (different filters / page numbers) stay intact.
  function refresh() {
    setPage(0)
    qc.invalidateQueries({ queryKey: ['memories', searchText, filterType, filterMinImportance, filterTags, sortOrder, searchMode, 0, pageSize] })
    qc.invalidateQueries({ queryKey: ['stats'] })
  }

  const backupMut = useMutation({
    mutationFn: createBackup,
    onSuccess: (res) => {
      const path = res.data?.[0]?.backup_path ?? 'unknown'
      setBackupMsg(`Backup saved to: ${path}`)
      setTimeout(() => setBackupMsg(null), 5000)
    },
    onError: (e: Error) => setBackupMsg(`Backup failed: ${e.message}`),
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      {/* Header */}
      <header style={{
        background: 'var(--surface)',
        borderBottom: '1px solid var(--border)',
        padding: '10px 16px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: 'var(--accent)' }}>
            Memory Manager
          </h1>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 1 }}>
            View and manage AI companion memories
          </div>
        </div>

        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <button onClick={refresh}>Refresh</button>
          <button onClick={() => backupMut.mutate()} disabled={backupMut.isPending}>
            {backupMut.isPending ? 'Backing up…' : 'Backup'}
          </button>
          <button onClick={downloadExport}>Export</button>
          <button onClick={() => setModal('vectors')}>Vectors</button>
          <button onClick={() => setModal('migrate')}>Migrate</button>
          <button onClick={() => setModal('compare')}>Compare</button>
        </div>
      </header>

      {/* Backup message banner */}
      {backupMsg && (
        <div style={{
          background: 'var(--surface2)', borderBottom: '1px solid var(--border)',
          padding: '6px 16px', fontSize: 12,
          color: backupMsg.startsWith('Backup failed') ? 'var(--danger)' : 'var(--success)',
        }}>
          {backupMsg}
        </div>
      )}

      {/* Main layout */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        {/* Left sidebar: search + stats */}
        <aside style={{
          width: 260, flexShrink: 0,
          display: 'flex', flexDirection: 'column', gap: 8,
          padding: 8, borderRight: '1px solid var(--border)',
          overflow: 'auto',
        }}>
          <SearchPanel onSearch={refresh} />
          <StatsPanel />
        </aside>

        {/* Center: sortable memory list */}
        <div style={{ flex: 3, display: 'flex', flexDirection: 'column', padding: 8, overflow: 'hidden', minWidth: 0 }}>
          <MemoryList
            memories={memories}
            isLoading={memoriesQ.isLoading}
            total={total}
          />
        </div>

        {/* Right: detail / editor — flex:2 gives ~40% of the center+right area */}
        <div style={{ flex: 2, minWidth: 380, flexShrink: 0, padding: 8, overflow: 'hidden', display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--border)' }}>
          <MemoryDetail onPeersClick={() => setModal('peers')} />
        </div>
      </div>

      {/* Modals */}
      {modal === 'vectors' && <VectorsModal onClose={() => setModal(null)} />}
      {modal === 'migrate' && <MigrateModal onClose={() => setModal(null)} />}
      {modal === 'compare' && <CompareModal onClose={() => setModal(null)} />}
      {modal === 'peers' && draft && (
        <PeerPicker
          currentValue={Array.isArray(draft.shared_with) ? draft.shared_with.join(',') : (draft.shared_with ?? '')}
          onApply={(val) => patchDraft({ shared_with: val.split(',').map((s) => s.trim()).filter(Boolean) })}
          onClose={() => setModal(null)}
        />
      )}
    </div>
  )
}
