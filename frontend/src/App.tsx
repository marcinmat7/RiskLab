import { useEffect, useState } from 'react'

type HealthResponse = {
  status: string
  service: string
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        return response.json() as Promise<HealthResponse>
      })
      .then(setHealth)
      .catch((err: Error) => setError(err.message))
  }, [])

  return (
    <main className="app-shell">
      <section className="hero-card">
        <p className="eyebrow">RiskLab v0.1</p>
        <h1>Credit risk validation, built as a product.</h1>
        <p className="subtitle">
          The first vertical slice connects the React frontend to the FastAPI backend.
        </p>

        <div className="status-card">
          <span className="status-label">API status</span>
          {health && <strong className="status-ok">Connected · {health.service}</strong>}
          {!health && !error && <strong>Checking…</strong>}
          {error && <strong className="status-error">Disconnected · {error}</strong>}
        </div>
      </section>
    </main>
  )
}

export default App
