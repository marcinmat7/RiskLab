import { ChangeEvent, useEffect, useState } from 'react'

type HealthResponse = {
  status: string
  service: string
}

type DatasetPreview = {
  filename: string
  row_count: number
  column_count: number
  columns: string[]
  preview: Record<string, string | null>[]
  delimiter: string
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        return response.json() as Promise<HealthResponse>
      })
      .then(setHealth)
      .catch((err: Error) => setHealthError(err.message))
  }, [])

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setUploadError(null)
    setPreview(null)
    setIsUploading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://localhost:8000/datasets/preview', {
        method: 'POST',
        body: formData,
      })

      const body = await response.json()
      if (!response.ok) {
        throw new Error(body.detail ?? `HTTP ${response.status}`)
      }

      setPreview(body as DatasetPreview)
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed.')
    } finally {
      setIsUploading(false)
      event.target.value = ''
    }
  }

  return (
    <main className="app-shell">
      <section className="hero-card">
        <div className="top-row">
          <div>
            <p className="eyebrow">RiskLab v0.1</p>
            <h1>Upload a validation dataset.</h1>
            <p className="subtitle">
              Start with a CSV. RiskLab will inspect its structure and show a preview before any column mapping or model validation.
            </p>
          </div>

          <div className="api-pill">
            <span className={`status-dot ${health ? 'online' : healthError ? 'offline' : ''}`} />
            {health && <span>API connected</span>}
            {!health && !healthError && <span>Checking API…</span>}
            {healthError && <span>API disconnected</span>}
          </div>
        </div>

        <section className="upload-panel">
          <div>
            <h2>Dataset</h2>
            <p>CSV only for MVP v0.1. UTF-8 encoding is supported.</p>
          </div>

          <label className={`upload-button ${isUploading ? 'disabled' : ''}`}>
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={handleFileChange}
              disabled={isUploading}
            />
            {isUploading ? 'Reading file…' : 'Choose CSV'}
          </label>
        </section>

        {uploadError && <div className="message error-message">{uploadError}</div>}

        {preview && (
          <section className="preview-section">
            <div className="preview-header">
              <div>
                <p className="eyebrow">Dataset preview</p>
                <h2>{preview.filename}</h2>
              </div>

              <div className="dataset-stats">
                <div>
                  <span>Rows</span>
                  <strong>{preview.row_count.toLocaleString()}</strong>
                </div>
                <div>
                  <span>Columns</span>
                  <strong>{preview.column_count}</strong>
                </div>
              </div>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {preview.columns.map((column) => (
                      <th key={column}>{column}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.preview.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      {preview.columns.map((column) => (
                        <td key={`${rowIndex}-${column}`}>{row[column] || '—'}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="preview-note">
              Showing the first {preview.preview.length} rows. Next step: map target, prediction, date, ID and segment columns.
            </p>
          </section>
        )}
      </section>
    </main>
  )
}

export default App
