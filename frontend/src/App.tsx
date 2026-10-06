import { ChangeEvent, useEffect, useState } from 'react'

type HealthResponse = {
  status: string
  service: string
}

type PreviewRow = {
  row_number: number
  values: Record<string, string | null>
}

type DatasetPreview = {
  filename: string
  file_size_bytes: number | null
  row_count: number
  column_count: number
  columns: string[]
  first_preview: PreviewRow[]
  random_preview: PreviewRow[]
  delimiter: string
  warning: string | null
  max_rows: number
}

type UploadStatus = 'idle' | 'selected' | 'uploading' | 'ready' | 'error'
type Section = 'Overview' | 'Data' | 'Validation' | 'Discrimination' | 'Calibration' | 'Stability' | 'Segments' | 'Findings' | 'Reports'

const navItems: Section[] = ['Overview', 'Data', 'Validation', 'Discrimination', 'Calibration', 'Stability', 'Segments', 'Findings', 'Reports']

const metrics = [
  { label: 'AUC', value: '0.784', delta: '+0.012', tone: 'positive' },
  { label: 'Gini', value: '0.568', delta: '+0.015', tone: 'positive' },
  { label: 'KS', value: '0.421', delta: '+0.008', tone: 'positive' },
  { label: 'Brier score', value: '0.039', delta: '-0.003', tone: 'negative' },
  { label: 'Observations', value: '125,430', delta: '12 features', tone: 'neutral' },
  { label: 'Default rate', value: '3.2%', delta: '4,015 defaults', tone: 'neutral' },
]

const formatFileSize = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function PreviewTable({ columns, rows }: { columns: string[]; rows: PreviewRow[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Row</th>
            {columns.map((column) => <th key={column}>{column}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.row_number}>
              <td className="row-number">{row.row_number.toLocaleString()}</td>
              {columns.map((column) => <td key={`${row.row_number}-${column}`}>{row.values[column] || '—'}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function EmptyAnalysisState({ section, onOpenData }: { section: Section; onOpenData: () => void }) {
  return (
    <section className="page-content empty-page">
      <div className="empty-state-card">
        <div className="empty-state-icon" aria-hidden="true">
          <span className="empty-bar bar-one" />
          <span className="empty-bar bar-two" />
          <span className="empty-bar bar-three" />
          <span className="empty-spark">✦</span>
        </div>
        <p className="eyebrow">{section}</p>
        <h1>No analysis available yet</h1>
        <p className="empty-state-copy">
          Upload a validation dataset and run the analysis to populate this section with model diagnostics, metrics and findings.
        </p>
        <button className="primary-button empty-state-action" onClick={onOpenData}>Go to data upload</button>
        <div className="empty-state-steps">
          <span><strong>1</strong> Upload data</span>
          <span className="step-line" />
          <span><strong>2</strong> Configure analysis</span>
          <span className="step-line" />
          <span><strong>3</strong> Review results</span>
        </div>
      </div>
    </section>
  )
}

function App() {
  const [activeSection, setActiveSection] = useState<Section>('Overview')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>('idle')

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.json() as Promise<HealthResponse>
      })
      .then(setHealth)
      .catch((err: Error) => setHealthError(err.message))
  }, [])

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return

    setSelectedFile(file)
    setPreview(null)
    setUploadError(null)
    setUploadStatus('selected')
  }

  const removeFile = () => {
    setSelectedFile(null)
    setPreview(null)
    setUploadError(null)
    setUploadStatus('idle')
  }

  const uploadFile = async () => {
    if (!selectedFile) return

    setUploadError(null)
    setPreview(null)
    setUploadStatus('uploading')

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch('http://localhost:8000/datasets/preview', {
        method: 'POST',
        body: formData,
      })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail ?? `HTTP ${response.status}`)
      setPreview(body as DatasetPreview)
      setUploadStatus('ready')
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed.')
      setUploadStatus('error')
    }
  }

  const renderDataPage = () => (
    <section className="page-content">
      <div className="page-heading compact-heading">
        <div><p className="eyebrow">Dataset</p><h1>Upload validation data</h1><p>Select a CSV, review the file details, then upload it for structural inspection.</p></div>
      </div>

      <section className="panel upload-panel upload-workflow">
        <div className="upload-copy">
          <h2>Validation dataset</h2>
          <p>CSV only for MVP v0.1. UTF-8 encoding is supported. Maximum 1,000,000 data rows.</p>
        </div>

        {!selectedFile ? (
          <label className="primary-button upload-button">
            <input type="file" accept=".csv,text/csv" onChange={handleFileChange} />
            Choose CSV
          </label>
        ) : (
          <div className="selected-file-card">
            <div className="file-details">
              <span className="file-icon">CSV</span>
              <div>
                <strong>{selectedFile.name}</strong>
                <span>{formatFileSize(selectedFile.size)}</span>
              </div>
            </div>

            <div className={`upload-state ${uploadStatus}`}>
              <span className="state-dot" />
              {uploadStatus === 'selected' && 'Ready to upload'}
              {uploadStatus === 'uploading' && 'Uploading…'}
              {uploadStatus === 'ready' && 'Ready'}
              {uploadStatus === 'error' && 'Upload failed'}
            </div>

            <div className="file-actions">
              <label className={`secondary-button compact-button ${uploadStatus === 'uploading' ? 'disabled' : ''}`}>
                <input type="file" accept=".csv,text/csv" onChange={handleFileChange} disabled={uploadStatus === 'uploading'} />
                Replace
              </label>
              <button className="secondary-button compact-button" onClick={removeFile} disabled={uploadStatus === 'uploading'}>Remove</button>
              <button className="primary-button compact-button" onClick={uploadFile} disabled={uploadStatus === 'uploading'}>
                {uploadStatus === 'uploading' ? 'Uploading…' : uploadStatus === 'ready' ? 'Upload again' : 'Upload'}
              </button>
            </div>
          </div>
        )}
      </section>

      {uploadError && <div className="message error-message">{uploadError}</div>}
      {preview?.warning && <div className="message warning-message"><strong>Warning:</strong> {preview.warning}</div>}

      {preview && (
        <section className="panel preview-section">
          <div className="preview-header">
            <div><p className="eyebrow">Dataset preview</p><h2>{preview.filename}</h2><p className="preview-meta">Delimiter: <code>{preview.delimiter === '\t' ? 'tab' : preview.delimiter}</code> · Limit: {preview.max_rows.toLocaleString()} rows</p></div>
            <div className="dataset-stats">
              <div><span>Rows</span><strong>{preview.row_count.toLocaleString()}</strong></div>
              <div><span>Columns</span><strong>{preview.column_count}</strong></div>
            </div>
          </div>

          <div className="preview-block">
            <div className="preview-block-heading">
              <div><h3>First 10 rows</h3><p>The first {preview.first_preview.length} data rows in the uploaded CSV.</p></div>
              <span className="sample-badge">Rows 1–{preview.first_preview.length}</span>
            </div>
            <PreviewTable columns={preview.columns} rows={preview.first_preview} />
          </div>

          {preview.random_preview.length > 0 && (
            <div className="preview-block">
              <div className="preview-block-heading">
                <div><h3>Random sample</h3><p>{preview.random_preview.length} randomly selected rows from rows 11 through {preview.row_count.toLocaleString()}.</p></div>
                <span className="sample-badge">Random sample</span>
              </div>
              <PreviewTable columns={preview.columns} rows={preview.random_preview} />
            </div>
          )}
        </section>
      )}
    </section>
  )

  const renderOverview = () => (
    <section className="page-content">
      <div className="model-header">
        <div><div className="title-row"><h1>Consumer PD v1</h1><span className="validated-badge">● Validated</span></div><p>Retail credit risk model — Probability of Default</p></div>
        <div className="header-actions"><button className="secondary-button">Version 1.0⌄</button><button className="secondary-button">Validation run&nbsp;&nbsp;<strong>2024-12-01</strong></button><button className="primary-button">▷ Run validation</button></div>
      </div>

      <div className="tabs"><button className="active">Summary</button><button>Key metrics</button><button>Charts</button><button>Data quality</button><button>Recent findings</button></div>
      <div className="metric-grid">{metrics.map((metric) => <article className="metric-card" key={metric.label}><span>{metric.label}</span><strong>{metric.value}</strong><small className={metric.tone}>{metric.delta}</small></article>)}</div>

      <div className="chart-grid">
        <article className="panel chart-card"><div className="panel-title"><h2>ROC Curve</h2><span>•••</span></div><svg viewBox="0 0 600 240" className="chart-svg"><line x1="50" y1="200" x2="560" y2="30" className="dash-line"/><path d="M50,200 C90,105 165,65 250,49 C345,31 450,27 560,25" className="main-line"/><line x1="50" y1="200" x2="560" y2="200" className="axis"/><line x1="50" y1="200" x2="50" y2="25" className="axis"/></svg><div className="chart-legend"><span className="legend-blue"/>Model (AUC = 0.784)</div></article>
        <article className="panel chart-card"><div className="panel-title"><h2>Calibration Curve</h2><span>•••</span></div><svg viewBox="0 0 600 240" className="chart-svg"><line x1="50" y1="200" x2="560" y2="25" className="dash-line"/><polyline points="50,198 115,179 185,163 255,139 330,122 405,110 480,86 560,52" className="main-line"/><line x1="50" y1="200" x2="560" y2="200" className="axis"/><line x1="50" y1="200" x2="50" y2="25" className="axis"/></svg><div className="chart-legend"><span className="legend-blue"/>Model calibration</div></article>
      </div>

      <div className="lower-grid">
        <article className="panel"><div className="panel-title"><h2>Prediction Distribution</h2><span>•••</span></div><div className="bars">{[22,46,68,82,92,87,74,61,48,39,33,26,21,17,13,10].map((h, i) => <span key={i} style={{height:`${h}%`}} className={i > 8 ? 'bar warm' : 'bar'} />)}</div><div className="axis-labels"><span>0.001</span><span>0.01</span><span>0.1</span><span>1.0</span></div></article>
        <article className="panel data-quality"><div className="panel-title"><h2>Data Quality</h2><span className="quality-pill">● No critical issues</span></div>{[['Missing values','0.2%','ok'],['Duplicate rows','0','ok'],['Constant columns','0','ok'],['Invalid dates','0','ok'],['Outliers detected','1.3%','warn']].map(([label,value,state]) => <div className="quality-row" key={label}><span className={state === 'ok' ? 'check' : 'warning'}>{state === 'ok' ? '✓' : '!'}</span><span>{label}</span><strong>{value}</strong></div>)}</article>
      </div>

      <article className="panel findings"><div className="panel-title"><h2>Recent Findings</h2><a href="#">View all findings →</a></div><div className="table-wrap flat"><table><thead><tr><th>Severity</th><th>Category</th><th>Message</th><th>Status</th></tr></thead><tbody><tr><td><span className="severity high">High</span></td><td>Stability</td><td>PSI for feature “Credit utilization” is 0.18 (&gt; 0.1)</td><td><span className="status open">Open</span></td></tr><tr><td><span className="severity medium">Medium</span></td><td>Data quality</td><td>5 columns have &gt; 5% missing values</td><td><span className="status open">Open</span></td></tr><tr><td><span className="severity medium">Medium</span></td><td>Calibration</td><td>Calibration curve shows underestimation in high PD range</td><td><span className="status progress">In progress</span></td></tr><tr><td><span className="severity low">Low</span></td><td>Discrimination</td><td>KS decreased by 0.03 compared to previous validation</td><td><span className="status open">Open</span></td></tr></tbody></table></div></article>
    </section>
  )

  return (
    <div className="app-frame">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">▥</span><span>RiskLab</span></div>
        <nav>
          {navItems.map((item) => (
            <button key={item} className={activeSection === item ? 'nav-item active' : 'nav-item'} onClick={() => setActiveSection(item)}>
              <span className="nav-icon">{item.slice(0, 1)}</span>{item}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer"><span className={`status-dot ${health ? 'online' : healthError ? 'offline' : ''}`} />{health ? 'API connected' : healthError ? 'API offline' : 'Checking API…'}</div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div className="breadcrumbs">Models <span>›</span> Consumer PD v1</div>
          <div className="user-chip">MM <span>Marcin</span></div>
        </header>

        {activeSection === 'Overview'
          ? renderOverview()
          : activeSection === 'Data'
            ? renderDataPage()
            : <EmptyAnalysisState section={activeSection} onOpenData={() => setActiveSection('Data')} />}
      </main>
    </div>
  )
}

export default App
