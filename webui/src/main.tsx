import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import App from './App.tsx'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Memories only change when created/updated/deleted — treat cached data
      // as fresh for 5 minutes. Mutations call invalidateQueries with the exact
      // key to bust the cache immediately when a change is made.
      staleTime: 5 * 60 * 1000,
      // Don't refetch just because the user switched browser tabs
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)
