import { create } from 'zustand'
import type { Memory } from '@/api/types'

export const PAGE_SIZES = [25, 50, 100, 200]
export const DEFAULT_PAGE_SIZE = 50

interface MemoryStore {
  selectedId: string | null
  setSelectedId: (id: string | null) => void

  // Draft state for the detail editor
  draft: Partial<Memory> | null
  setDraft: (draft: Partial<Memory> | null) => void
  patchDraft: (fields: Partial<Memory>) => void

  isNewMemory: boolean
  setIsNewMemory: (v: boolean) => void

  // Search filters
  searchText: string
  setSearchText: (v: string) => void
  filterType: string
  setFilterType: (v: string) => void
  filterMinImportance: number
  setFilterMinImportance: (v: number) => void
  filterTags: string
  setFilterTags: (v: string) => void
  sortOrder: string
  setSortOrder: (v: string) => void
  searchMode: 'structured' | 'semantic'
  setSearchMode: (v: 'structured' | 'semantic') => void

  // Paging
  page: number
  setPage: (v: number) => void
  pageSize: number
  setPageSize: (v: number) => void

  // Current query key — set by App so mutations in MemoryDetail can invalidate
  // only the active page instead of the entire memories cache.
  currentQueryKey: readonly unknown[]
  setCurrentQueryKey: (k: readonly unknown[]) => void
}

export const useMemoryStore = create<MemoryStore>((set) => ({
  selectedId: null,
  setSelectedId: (id) => set({ selectedId: id }),

  draft: null,
  setDraft: (draft) => set({ draft }),
  patchDraft: (fields) =>
    set((state) => ({ draft: state.draft ? { ...state.draft, ...fields } : fields })),

  isNewMemory: false,
  setIsNewMemory: (v) => set({ isNewMemory: v }),

  searchText: '',
  setSearchText: (v) => set({ searchText: v }),
  filterType: '',
  setFilterType: (v) => set({ filterType: v }),
  filterMinImportance: 1,
  setFilterMinImportance: (v) => set({ filterMinImportance: v }),
  filterTags: '',
  setFilterTags: (v) => set({ filterTags: v }),
  sortOrder: 'importance DESC, timestamp DESC',
  setSortOrder: (v) => set({ sortOrder: v }),
  searchMode: 'structured',
  setSearchMode: (v) => set({ searchMode: v }),

  page: 0,
  setPage: (v) => set({ page: v }),
  pageSize: DEFAULT_PAGE_SIZE,
  setPageSize: (v) => set({ pageSize: v, page: 0 }),

  currentQueryKey: ['memories'] as readonly unknown[],
  setCurrentQueryKey: (k) => set({ currentQueryKey: k }),
}))
