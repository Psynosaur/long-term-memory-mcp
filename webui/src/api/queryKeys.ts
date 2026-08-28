// Shared query key factory so mutations in MemoryDetail can invalidate
// the exact current page without blowing away the whole memories cache.
export function memoryQueryKey(
  searchText: string,
  filterType: string,
  filterMinImportance: number,
  filterTags: string,
  sortOrder: string,
  searchMode: string,
  page: number,
  pageSize: number,
) {
  return [
    'memories',
    searchText, filterType, filterMinImportance,
    filterTags, sortOrder, searchMode,
    page, pageSize,
  ] as const
}
