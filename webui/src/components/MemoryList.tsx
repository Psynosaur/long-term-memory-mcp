import { useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
} from '@tanstack/react-table'
import { ArrowUp, ArrowDown, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'
import type { Memory } from '@/api/types'
import { useMemoryStore, PAGE_SIZES } from '@/store/memoryStore'
import { getMemory } from '@/api/client'

const col = createColumnHelper<Memory>()

function fmt(ts: string) {
  if (!ts) return ''
  try {
    return new Date(ts).toLocaleString(undefined, {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit',
    })
  } catch {
    return ts
  }
}

interface MemoryListProps {
  memories: Memory[]
  isLoading: boolean
  total: number
}

export function MemoryList({ memories, isLoading, total }: MemoryListProps) {
  const { selectedId, setSelectedId, setDraft, setIsNewMemory, page, setPage, pageSize, setPageSize } = useMemoryStore()
  const qc = useQueryClient()
  const [sorting, setSorting] = useState<SortingState>([])

  const totalPages = Math.max(1, Math.ceil(total / pageSize))

  const columns = useMemo(
    () => [
      col.accessor('title', {
        header: 'Title',
        size: 320,
        cell: (info) => (
          <span style={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {info.getValue()}
          </span>
        ),
      }),
      col.accessor('memory_type', { header: 'Type', size: 100 }),
      col.accessor('importance', {
        header: 'Imp',
        size: 50,
        cell: (info) => <span style={{ display: 'block', textAlign: 'center' }}>{info.getValue()}</span>,
      }),
      col.accessor('shared_with', {
        header: 'Shared',
        size: 60,
        cell: (info) => (
          <span style={{ display: 'block', textAlign: 'center', color: 'var(--success)' }}>
            {(info.getValue()?.length ?? 0) > 0 ? '✓' : ''}
          </span>
        ),
      }),
      col.accessor('timestamp', {
        header: 'Date',
        size: 130,
        cell: (info) => (
          <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>{fmt(info.getValue())}</span>
        ),
      }),
      col.accessor('last_accessed', {
        header: 'Accessed',
        size: 130,
        cell: (info) => (
          <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>{fmt(info.getValue() ?? '')}</span>
        ),
      }),
      col.accessor('reinforcement_accum', {
        header: 'Reinf.',
        size: 58,
        cell: (info) => {
          const v = info.getValue()
          return (
            <span style={{ display: 'block', textAlign: 'right', color: v && v > 0 ? 'var(--success)' : 'var(--text-muted)', fontSize: 11 }}>
              {v !== undefined && v !== null ? v.toFixed(2) : '—'}
            </span>
          )
        },
      }),
    ],
    [],
  )

  const table = useReactTable({
    data: memories,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  // Fetch the full row (with updated_at, last_accessed) when a memory is clicked.
  // The list endpoint goes through MemoryRecord which drops those fields;
  // the single-fetch endpoint returns the raw DB row with all columns.
  async function selectMemory(mem: Memory) {
    setSelectedId(mem.id)
    setIsNewMemory(false)
    setDraft({ ...mem })  // optimistic: show list data immediately
    try {
      const full = await qc.fetchQuery({
        queryKey: ['memory', mem.id],
        queryFn: () => getMemory(mem.id),
        staleTime: 2 * 60 * 1000,
      })
      setDraft({ ...full })
    } catch {
      // leave optimistic draft as-is
    }
  }

  const from = total === 0 ? 0 : page * pageSize + 1
  const to = Math.min((page + 1) * pageSize, total)

  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 6,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      flex: 1,
      minHeight: 0,
    }}>
      {/* Header row */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '6px 10px', borderBottom: '1px solid var(--border)',
        flexShrink: 0,
      }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: 1 }}>
          Memories
          {total > 0 && (
            <span style={{ color: 'var(--text-muted)', fontWeight: 400, marginLeft: 6 }}>
              {from}–{to} of {total.toLocaleString()}
            </span>
          )}
        </span>

        {/* Page controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {/* Page size */}
          <select
            value={pageSize}
            onChange={(e) => { setPageSize(Number(e.target.value)); setPage(0) }}
            style={{ fontSize: 11, padding: '2px 4px', height: 24 }}
            title="Rows per page"
          >
            {PAGE_SIZES.map((s) => <option key={s} value={s}>{s} / page</option>)}
          </select>

          <PageBtn onClick={() => setPage(0)} disabled={page === 0} title="First page">
            <ChevronsLeft size={13} />
          </PageBtn>
          <PageBtn onClick={() => setPage(page - 1)} disabled={page === 0} title="Previous page">
            <ChevronLeft size={13} />
          </PageBtn>

          <span style={{ fontSize: 11, color: 'var(--text-muted)', minWidth: 60, textAlign: 'center' }}>
            {page + 1} / {totalPages}
          </span>

          <PageBtn onClick={() => setPage(page + 1)} disabled={page >= totalPages - 1} title="Next page">
            <ChevronRight size={13} />
          </PageBtn>
          <PageBtn onClick={() => setPage(totalPages - 1)} disabled={page >= totalPages - 1} title="Last page">
            <ChevronsRight size={13} />
          </PageBtn>
        </div>
      </div>

      {/* Table */}
      <div style={{ overflow: 'auto', flex: 1 }}>
        {isLoading ? (
          <div style={{ padding: 20, color: 'var(--text-muted)', textAlign: 'center' }}>Loading…</div>
        ) : memories.length === 0 ? (
          <div style={{ padding: 20, color: 'var(--text-muted)', textAlign: 'center' }}>No memories found</div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
            <thead>
              {table.getHeaderGroups().map((hg) => (
                <tr key={hg.id} style={{ background: 'var(--surface2)', position: 'sticky', top: 0, zIndex: 1 }}>
                  {hg.headers.map((header) => (
                    <th
                      key={header.id}
                      onClick={header.column.getToggleSortingHandler()}
                      style={{
                        width: header.getSize(),
                        padding: '5px 8px',
                        textAlign: 'left',
                        fontSize: 11,
                        fontWeight: 600,
                        color: 'var(--text-muted)',
                        borderBottom: '1px solid var(--border)',
                        cursor: 'pointer',
                        userSelect: 'none',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                      }}
                    >
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {header.column.getIsSorted() === 'asc' && <ArrowUp size={11} />}
                        {header.column.getIsSorted() === 'desc' && <ArrowDown size={11} />}
                      </span>
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row) => {
                const mem = row.original
                const isSelected = mem.id === selectedId
                return (
                  <tr
                    key={row.id}
                    onClick={() => selectMemory(mem)}
                    style={{
                      cursor: 'pointer',
                      background: isSelected ? 'var(--surface2)' : 'transparent',
                      borderBottom: '1px solid var(--border)',
                    }}
                    onMouseEnter={(e) => {
                      if (!isSelected) (e.currentTarget as HTMLElement).style.background = 'rgba(79,195,247,0.07)'
                    }}
                    onMouseLeave={(e) => {
                      if (!isSelected) (e.currentTarget as HTMLElement).style.background = 'transparent'
                    }}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <td
                        key={cell.id}
                        style={{
                          padding: '4px 8px',
                          overflow: 'hidden',
                          maxWidth: cell.column.getSize(),
                        }}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function PageBtn({ onClick, disabled, children, title }: {
  onClick: () => void
  disabled: boolean
  children: React.ReactNode
  title?: string
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        padding: '2px 5px', height: 24, display: 'inline-flex',
        alignItems: 'center', justifyContent: 'center', minWidth: 24,
      }}
    >
      {children}
    </button>
  )
}
