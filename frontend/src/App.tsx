import { ChangeEvent, useEffect, useMemo, useState } from 'react'

type HealthResponse = { status: string; service: string }

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
  column_types: Record<string, PhysicalType>
  first_preview: PreviewRow[]
  random_preview: PreviewRow[]
  delimiter: string
  warning: string | null
  max_rows: number
}

type UploadStatus = 'idle' | 'selected' | 'uploading' | 'ready' | 'error'
type Section = 'Overview' | 'Data' | 'Validation' | 'Discrimination' | 'Calibration' | 'Stability' | 'Segments' | 'Findings' | 'Reports'
type AnalysisSection = Exclude<Section, 'Overview' | 'Data'>
type AnalysisStatus = 'not-run' | 'running' | 'ready' | 'failed' | 'blocked' | 'unavailable'
type PredictionType = 'pd' | 'score'
type ScoreDirection = 'higher-risk' | 'lower-risk'
type PhysicalType = 'integer' | 'numeric' | 'boolean' | 'datetime' | 'string' | 'empty'
type SemanticType = 'continuous' | 'categorical' | 'ordinal' | 'identifier' | 'datetime' | 'boolean' | 'text' | 'ignore'

type ValidationConfig = {
  predictionColumn: string
  predictionType: PredictionType
  targetColumn: string
  positiveClass: string
  scoreDirection: ScoreDirection
  timeColumn: string
  populationColumns: string[]
  sensitiveColumns: string[]
  featureColumns: string[]
  semanticTypes: Record<string, SemanticType>
}

const navItems: Section[] = ['Overview', 'Data', 'Validation', 'Discrimination', 'Calibration', 'Stability', 'Segments', 'Findings', 'Reports']
const analysisSections: AnalysisSection[] = ['Validation', 'Discrimination', 'Calibration', 'Stability', 'Segments', 'Findings', 'Reports']

const metrics = [
  { label: 'AUC', value: '0.784', delta: '+0.012', tone: 'positive' },
  { label: 'Gini', value: '0.568', delta: '+0.015', tone: 'positive' },
  { label: 'KS', value: '0.421', delta: '+0.008', tone: 'positive' },
  { label: 'Brier score', value: '0.039', delta: '-0.003', tone: 'negative' },
  { label: 'Observations', value: '125,430', delta: '12 features', tone: 'neutral' },
  { label: 'Default rate', value: '3.2%', delta: '4,015 defaults', tone: 'neutral' },
]

const analysisCopy: Record<AnalysisSection, { title: string; description: string; action: string }> = {
  Validation: { title: 'Validation setup', description: 'Define the model output, target and optional analysis roles for the uploaded dataset.', action: 'Run validation' },
  Discrimination: { title: 'Discrimination', description: 'Evaluate ranking performance of the submitted model.', action: 'Run discrimination' },
  Calibration: { title: 'Calibration', description: 'Evaluate how predicted probabilities align with observed default rates.', action: 'Run calibration' },
  Stability: { title: 'Stability', description: 'Assess population and model stability over time or across samples.', action: 'Run stability analysis' },
  Segments: { title: 'Segments', description: 'Compare model behaviour across selected populations and business segments.', action: 'Run segment analysis' },
  Findings: { title: 'Findings', description: 'Generate structured findings from completed validation analyses.', action: 'Generate findings' },
  Reports: { title: 'Reports', description: 'Generate a validation report from the analyses currently available.', action: 'Generate report' },
}

const initialConfig: ValidationConfig = {
  predictionColumn: '',
  predictionType: 'pd',
  targetColumn: '',
  positiveClass: '',
  scoreDirection: 'lower-risk',
  timeColumn: '',
  populationColumns: [],
  sensitiveColumns: [],
  featureColumns: [],
  semanticTypes: {},
}

const initialStatuses = (): Record<AnalysisSection, AnalysisStatus> => ({
  Validation: 'not-run',
  Discrimination: 'blocked',
  Calibration: 'blocked',
  Stability: 'blocked',
  Segments: 'blocked',
  Findings: 'blocked',
  Reports: 'blocked',
})

const formatFileSize = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const semanticTypeOptions: { value: SemanticType; label: string }[] = [
  { value: 'continuous', label: 'Continuous' },
  { value: 'categorical', label: 'Categorical' },
  { value: 'ordinal', label: 'Ordinal' },
  { value: 'identifier', label: 'Identifier' },
  { value: 'datetime', label: 'Date / time' },
  { value: 'boolean', label: 'Boolean' },
  { value: 'text', label: 'Text' },
  { value: 'ignore', label: 'Ignore' },
]

const suggestSemanticType = (physicalType: PhysicalType): SemanticType => {
  if (physicalType === 'integer' || physicalType === 'numeric') return 'continuous'
  if (physicalType === 'boolean') return 'boolean'
  if (physicalType === 'datetime') return 'datetime'
  if (physicalType === 'empty') return 'ignore'
  return 'categorical'
}

const statusLabel = (status: AnalysisStatus) => ({
  'not-run': 'Not run',
  running: 'Running',
  ready: 'Results available',
  failed: 'Failed',
  blocked: 'Blocked',
  unavailable: 'Not available',
}[status])

function PreviewTable({ columns, rows }: { columns: string[]; rows: PreviewRow[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Row</th>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
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

function UsedBy({ modules }: { modules: string[] }) {
  return <div className="used-by"><span>Used by</span>{modules.map((module) => <span className="use-badge" key={module}>{module}</span>)}</div>
}

function MultiColumnPicker({
  columns,
  value,
  onChange,
  excluded = [],
}: {
  columns: string[]
  value: string[]
  onChange: (value: string[]) => void
  excluded?: string[]
}) {
  const available = columns.filter((column) => !excluded.includes(column))
  return (
    <div className="multi-picker">
      {available.map((column) => {
        const selected = value.includes(column)
        return (
          <button
            type="button"
            key={column}
            className={selected ? 'column-chip selected' : 'column-chip'}
            onClick={() => onChange(selected ? value.filter((item) => item !== column) : [...value, column])}
          >
            <span>{selected ? '✓' : '+'}</span>{column}
          </button>
        )
      })}
    </div>
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
  const [validationConfig, setValidationConfig] = useState<ValidationConfig>(initialConfig)
  const [analysisStatus, setAnalysisStatus] = useState<Record<AnalysisSection, AnalysisStatus>>(initialStatuses)
  const [lastRun, setLastRun] = useState<Partial<Record<AnalysisSection, string>>>({})

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.json() as Promise<HealthResponse>
      })
      .then(setHealth)
      .catch((err: Error) => setHealthError(err.message))
  }, [])

  const targetValues = useMemo(() => {
    if (!preview || !validationConfig.targetColumn) return []
    const seen = new Set<string>()
    ;[...preview.first_preview, ...preview.random_preview].forEach((row) => {
      const value = row.values[validationConfig.targetColumn]
      if (value !== null && value !== undefined && value !== '') seen.add(String(value))
    })
    return [...seen]
  }, [preview, validationConfig.targetColumn])

  useEffect(() => {
    if (targetValues.length && !targetValues.includes(validationConfig.positiveClass)) {
      setValidationConfig((current) => ({ ...current, positiveClass: targetValues[0] }))
    }
  }, [targetValues, validationConfig.positiveClass])

  const resetAnalysis = () => {
    setValidationConfig(initialConfig)
    setAnalysisStatus(initialStatuses())
    setLastRun({})
  }

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setSelectedFile(file)
    setPreview(null)
    setUploadError(null)
    setUploadStatus('selected')
    resetAnalysis()
  }

  const removeFile = () => {
    setSelectedFile(null)
    setPreview(null)
    setUploadError(null)
    setUploadStatus('idle')
    resetAnalysis()
  }

  const uploadFile = async () => {
    if (!selectedFile) return
    setUploadError(null)
    setPreview(null)
    setUploadStatus('uploading')
    resetAnalysis()

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch('http://localhost:8000/datasets/preview', { method: 'POST', body: formData })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail ?? `HTTP ${response.status}`)
      const dataset = body as DatasetPreview
      setPreview(dataset)
      setValidationConfig({
        ...initialConfig,
        semanticTypes: Object.fromEntries(
          dataset.columns.map((column) => [column, suggestSemanticType(dataset.column_types[column] ?? 'string')])
        ) as Record<string, SemanticType>,
      })
      setUploadStatus('ready')
      setAnalysisStatus(initialStatuses())
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed.')
      setUploadStatus('error')
    }
  }

  const requiredConfigReady = Boolean(
    preview &&
    validationConfig.predictionColumn &&
    validationConfig.targetColumn &&
    validationConfig.positiveClass &&
    validationConfig.predictionColumn !== validationConfig.targetColumn
  )

  const validationReady = analysisStatus.Validation === 'ready'

  const runAnalysis = (section: AnalysisSection) => {
    if (section === 'Validation' && !requiredConfigReady) return
    if (section !== 'Validation' && !validationReady) return
    if (section === 'Calibration' && validationConfig.predictionType === 'score') return

    setAnalysisStatus((current) => ({ ...current, [section]: 'running' }))
    window.setTimeout(() => {
      setAnalysisStatus((current) => {
        const next = { ...current, [section]: 'ready' as AnalysisStatus }
        if (section === 'Validation') {
          analysisSections.forEach((item) => {
            if (item !== 'Validation') next[item] = item === 'Calibration' && validationConfig.predictionType === 'score' ? 'unavailable' : 'not-run'
          })
        }
        return next
      })
      setLastRun((current) => ({ ...current, [section]: new Date().toLocaleString() }))
    }, 650)
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
          <label className="primary-button upload-button"><input type="file" accept=".csv,text/csv" onChange={handleFileChange} />Choose CSV</label>
        ) : (
          <div className="selected-file-card">
            <div className="file-details">
              <span className="file-icon">CSV</span>
              <div><strong>{selectedFile.name}</strong><span>{formatFileSize(selectedFile.size)}</span></div>
            </div>
            <div className={`upload-state ${uploadStatus}`}><span className="state-dot" />{uploadStatus === 'selected' && 'Ready to upload'}{uploadStatus === 'uploading' && 'Uploading…'}{uploadStatus === 'ready' && 'Ready'}{uploadStatus === 'error' && 'Upload failed'}</div>
            <div className="file-actions">
              <label className={`secondary-button compact-button ${uploadStatus === 'uploading' ? 'disabled' : ''}`}><input type="file" accept=".csv,text/csv" onChange={handleFileChange} disabled={uploadStatus === 'uploading'} />Replace</label>
              <button className="secondary-button compact-button" onClick={removeFile} disabled={uploadStatus === 'uploading'}>Remove</button>
              <button className="primary-button compact-button" onClick={uploadFile} disabled={uploadStatus === 'uploading'}>{uploadStatus === 'uploading' ? 'Uploading…' : uploadStatus === 'ready' ? 'Upload again' : 'Upload'}</button>
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
            <div className="dataset-stats"><div><span>Rows</span><strong>{preview.row_count.toLocaleString()}</strong></div><div><span>Columns</span><strong>{preview.column_count}</strong></div></div>
          </div>
          <div className="preview-block">
            <div className="preview-block-heading"><div><h3>First 10 rows</h3><p>The first {preview.first_preview.length} data rows in the uploaded CSV.</p></div><span className="sample-badge">Rows 1–{preview.first_preview.length}</span></div>
            <PreviewTable columns={preview.columns} rows={preview.first_preview} />
          </div>
          {preview.random_preview.length > 0 && (
            <div className="preview-block">
              <div className="preview-block-heading"><div><h3>Random sample</h3><p>{preview.random_preview.length} randomly selected rows from rows 11 through {preview.row_count.toLocaleString()}.</p></div><span className="sample-badge">Random sample</span></div>
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
      <article className="panel findings"><div className="panel-title"><h2>Recent Findings</h2><a href="#">View all findings →</a></div><div className="table-wrap flat"><table><thead><tr><th>Severity</th><th>Category</th><th>Message</th><th>Status</th></tr></thead><tbody><tr><td><span className="severity high">High</span></td><td>Stability</td><td>PSI for feature “Credit utilization” is 0.18 (&gt; 0.1)</td><td><span className="status open">Open</span></td></tr><tr><td><span className="severity medium">Medium</span></td><td>Data quality</td><td>5 columns have &gt; 5% missing values</td><td><span className="status open">Open</span></td></tr></tbody></table></div></article>
    </section>
  )

  const renderValidation = () => {
    if (!preview) {
      return <EmptyAnalysisState section="Validation" onOpenData={() => setActiveSection('Data')} />
    }

    return (
      <section className="page-content validation-page">
        <div className="analysis-page-header">
          <div><p className="eyebrow">Validation</p><h1>Validation setup</h1><p>Map the uploaded dataset to the roles RiskLab needs before any analysis is run.</p></div>
          <div className={`analysis-status-pill ${analysisStatus.Validation}`}><span />{statusLabel(analysisStatus.Validation)}</div>
        </div>

        <section className="panel setup-card">
          <div className="setup-section-heading"><div><span className="setup-number">1</span><div><h2>Required setup</h2><p>These fields are required before validation can run.</p></div></div></div>
          <div className="form-grid">
            <label className="form-field"><span>Model output column <b>*</b></span><UsedBy modules={['Validation', 'Discrimination', 'Calibration', 'Stability', 'Segments', 'Fairness']} /><select value={validationConfig.predictionColumn} onChange={(e) => setValidationConfig({...validationConfig, predictionColumn:e.target.value})}><option value="">Select column…</option>{preview.columns.map((c) => <option key={c} value={c}>{c}</option>)}</select></label>
            <label className="form-field"><span>Output type <b>*</b></span><UsedBy modules={['Discrimination', 'Calibration']} /><select value={validationConfig.predictionType} onChange={(e) => setValidationConfig({...validationConfig, predictionType:e.target.value as PredictionType})}><option value="pd">Probability of Default (PD)</option><option value="score">Score</option></select></label>
            <label className="form-field"><span>Default indicator <b>*</b></span><UsedBy modules={['Validation', 'Discrimination', 'Calibration', 'Segments', 'Fairness']} /><select value={validationConfig.targetColumn} onChange={(e) => setValidationConfig({...validationConfig, targetColumn:e.target.value, positiveClass:''})}><option value="">Select column…</option>{preview.columns.map((c) => <option key={c} value={c}>{c}</option>)}</select></label>
            <label className="form-field"><span>Default class <b>*</b></span><UsedBy modules={['Validation', 'Discrimination', 'Calibration', 'Fairness']} /><select value={validationConfig.positiveClass} onChange={(e) => setValidationConfig({...validationConfig, positiveClass:e.target.value})} disabled={!validationConfig.targetColumn}><option value="">Select value…</option>{targetValues.map((value) => <option key={value} value={value}>{value}</option>)}</select><small>Values are detected from the current preview sample.</small></label>
          </div>

          {validationConfig.predictionType === 'score' && (
            <div className="score-direction">
              <span>Score direction</span>
              <label><input type="radio" checked={validationConfig.scoreDirection === 'lower-risk'} onChange={() => setValidationConfig({...validationConfig, scoreDirection:'lower-risk'})} /> Higher score = lower risk</label>
              <label><input type="radio" checked={validationConfig.scoreDirection === 'higher-risk'} onChange={() => setValidationConfig({...validationConfig, scoreDirection:'higher-risk'})} /> Higher score = higher risk</label>
            </div>
          )}

          {validationConfig.predictionColumn && validationConfig.predictionColumn === validationConfig.targetColumn && <div className="inline-validation-error">Model output and default indicator must use different columns.</div>}
        </section>

        <section className="panel setup-card column-types-card">
          <div className="setup-section-heading">
            <div><span className="setup-number">2</span><div><h2>Column types</h2><p>RiskLab detects the physical type from the CSV. Review or change the semantic type used by downstream analyses.</p></div></div>
            <UsedBy modules={['EDA', 'Segments', 'Stability', 'Challenger models']} />
          </div>
          <div className="column-type-table">
            <div className="column-type-row column-type-head"><span>Column</span><span>Detected physical type</span><span>Semantic type</span></div>
            {preview.columns.map((column) => (
              <div className="column-type-row" key={column}>
                <strong>{column}</strong>
                <span className="physical-type-badge">{preview.column_types[column] ?? 'string'}</span>
                <select
                  value={validationConfig.semanticTypes[column] ?? suggestSemanticType(preview.column_types[column] ?? 'string')}
                  onChange={(e) => setValidationConfig({
                    ...validationConfig,
                    semanticTypes: { ...validationConfig.semanticTypes, [column]: e.target.value as SemanticType },
                  })}
                >
                  {semanticTypeOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                </select>
              </div>
            ))}
          </div>
          <div className="type-note">Detected types are suggestions only. RiskLab never infers business meaning from a column name; the semantic type remains under user control.</div>
        </section>

        <section className="panel setup-card optional-card">
          <div className="setup-section-heading"><div><span className="setup-number optional">3</span><div><h2>Optional analysis settings</h2><p>These roles unlock time, representativeness, fairness and challenger-model analyses later.</p></div></div><span className="optional-badge">Optional</span></div>

          <div className="optional-setting">
            <div className="setting-copy"><h3>Time analysis</h3><p>Select a date or time column for performance and stability analysis over time.</p><UsedBy modules={['EDA', 'Stability']} /></div>
            <label className="form-field compact-field"><span>Time column</span><select value={validationConfig.timeColumn} onChange={(e) => setValidationConfig({...validationConfig, timeColumn:e.target.value})}><option value="">Not configured</option>{preview.columns.map((c) => <option key={c} value={c}>{c}</option>)}</select></label>
          </div>

          <div className="optional-setting stacked">
            <div className="setting-copy"><h3>Sample / population columns</h3><p>Use columns such as train/test, development/validation, country or portfolio to compare sample representativeness.</p><UsedBy modules={['EDA', 'Stability', 'Segments']} /></div>
            <MultiColumnPicker columns={preview.columns} value={validationConfig.populationColumns} onChange={(value) => setValidationConfig({...validationConfig, populationColumns:value})} excluded={[validationConfig.predictionColumn, validationConfig.targetColumn]} />
          </div>

          <div className="optional-setting stacked">
            <div className="setting-copy"><h3>Sensitive attributes</h3><p>Mark columns such as sex or age for future fairness and bias analysis.</p><UsedBy modules={['Fairness']} /></div>
            <MultiColumnPicker columns={preview.columns} value={validationConfig.sensitiveColumns} onChange={(value) => setValidationConfig({...validationConfig, sensitiveColumns:value})} excluded={[validationConfig.predictionColumn, validationConfig.targetColumn]} />
          </div>

          <div className="optional-setting stacked">
            <div className="setting-copy"><h3>Model features</h3><p>Select variables used to build the submitted model. RiskLab will later use them for challenger models and explainability.</p><UsedBy modules={['Challenger models', 'SHAP']} /></div>
            <div className="picker-toolbar"><button className="text-button" onClick={() => setValidationConfig({...validationConfig, featureColumns:preview.columns.filter((c) => ![validationConfig.predictionColumn, validationConfig.targetColumn].includes(c))})}>Select all available</button><button className="text-button" onClick={() => setValidationConfig({...validationConfig, featureColumns:[]})}>Clear</button></div>
            <MultiColumnPicker columns={preview.columns} value={validationConfig.featureColumns} onChange={(value) => setValidationConfig({...validationConfig, featureColumns:value})} excluded={[validationConfig.predictionColumn, validationConfig.targetColumn]} />
          </div>
        </section>

        <div className="run-footer">
          <div><strong>{requiredConfigReady ? 'Configuration ready' : 'Complete the required fields'}</strong><span>{requiredConfigReady ? 'RiskLab can now run structural validation.' : 'Choose model output, target and default class to continue.'}</span></div>
          <button className="primary-button run-button" disabled={!requiredConfigReady || analysisStatus.Validation === 'running'} onClick={() => runAnalysis('Validation')}>{analysisStatus.Validation === 'running' ? 'Running…' : analysisStatus.Validation === 'ready' ? 'Run validation again' : 'Run validation'}</button>
        </div>
      </section>
    )
  }

  const renderAnalysisPage = (section: Exclude<AnalysisSection, 'Validation'>) => {
    if (!preview) return <EmptyAnalysisState section={section} onOpenData={() => setActiveSection('Data')} />

    const status = analysisStatus[section]
    const blocked = !validationReady
    const unavailable = section === 'Calibration' && validationConfig.predictionType === 'score'
    const copy = analysisCopy[section]

    return (
      <section className="page-content analysis-module-page">
        <div className="analysis-page-header">
          <div><p className="eyebrow">{section}</p><h1>{copy.title}</h1><p>{copy.description}</p></div>
          <div className={`analysis-status-pill ${unavailable ? 'unavailable' : blocked ? 'blocked' : status}`}><span />{unavailable ? 'Not available' : blocked ? 'Blocked' : statusLabel(status)}</div>
        </div>

        <section className="panel analysis-run-card">
          <div className="analysis-run-icon" aria-hidden="true"><span>↗</span></div>
          {unavailable ? (
            <>
              <h2>Calibration requires probabilities</h2>
              <p>The uploaded model output is configured as a score. Calibration analysis becomes available when the output type is Probability of Default.</p>
            </>
          ) : blocked ? (
            <>
              <h2>Complete validation first</h2>
              <p>Run the Validation step to confirm the dataset mapping before this analysis can be executed.</p>
              <button className="secondary-button" onClick={() => setActiveSection('Validation')}>Open validation setup</button>
            </>
          ) : status === 'ready' ? (
            <>
              <h2>Results are available</h2>
              <p>This PR only implements the run framework. The actual {section.toLowerCase()} calculations will be connected in a later slice.</p>
              {lastRun[section] && <span className="last-run">Last run: {lastRun[section]}</span>}
              <button className="primary-button" onClick={() => runAnalysis(section)}>Run again</button>
            </>
          ) : (
            <>
              <h2>{status === 'running' ? 'Analysis is running…' : 'Ready to run'}</h2>
              <p>{status === 'running' ? 'RiskLab is preparing this analysis.' : 'The dataset is validated and this module is ready to execute independently.'}</p>
              <button className="primary-button" disabled={status === 'running'} onClick={() => runAnalysis(section)}>{status === 'running' ? 'Running…' : copy.action}</button>
            </>
          )}
        </section>
      </section>
    )
  }

  return (
    <div className="app-frame">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">▥</span><span>RiskLab</span></div>
        <nav>
          {navItems.map((item) => {
            const status = analysisSections.includes(item as AnalysisSection) ? analysisStatus[item as AnalysisSection] : null
            return (
              <button key={item} className={activeSection === item ? 'nav-item active' : 'nav-item'} onClick={() => setActiveSection(item)}>
                <span className="nav-icon">{item.slice(0, 1)}</span>
                <span className="nav-label">{item}</span>
                {status && <span className={`nav-analysis-status ${status}`} title={statusLabel(status)} />}
                {item === 'Data' && preview && <span className="nav-analysis-status ready" title="Dataset loaded" />}
              </button>
            )
          })}
        </nav>
        <div className="sidebar-footer"><span className={`status-dot ${health ? 'online' : healthError ? 'offline' : ''}`} />{health ? 'API connected' : healthError ? 'API offline' : 'Checking API…'}</div>
      </aside>

      <main className="workspace">
        <header className="topbar"><div className="breadcrumbs">Models <span>›</span> Consumer PD v1</div><div className="user-chip">MM <span>Marcin</span></div></header>
        {activeSection === 'Overview' && renderOverview()}
        {activeSection === 'Data' && renderDataPage()}
        {activeSection === 'Validation' && renderValidation()}
        {activeSection !== 'Overview' && activeSection !== 'Data' && activeSection !== 'Validation' && renderAnalysisPage(activeSection)}
      </main>
    </div>
  )
}

function EmptyAnalysisState({ section, onOpenData }: { section: Section; onOpenData: () => void }) {
  return (
    <section className="page-content empty-page">
      <div className="empty-state-card">
        <div className="empty-state-icon" aria-hidden="true"><span className="empty-bar bar-one" /><span className="empty-bar bar-two" /><span className="empty-bar bar-three" /><span className="empty-spark">✦</span></div>
        <p className="eyebrow">{section}</p>
        <h1>No dataset loaded</h1>
        <p className="empty-state-copy">Upload a validation dataset first. Once the dataset is available, configure Validation and run each analysis independently.</p>
        <button className="primary-button empty-state-action" onClick={onOpenData}>Go to data upload</button>
        <div className="empty-state-steps"><span><strong>1</strong> Upload data</span><span className="step-line" /><span><strong>2</strong> Configure validation</span><span className="step-line" /><span><strong>3</strong> Run analyses</span></div>
      </div>
    </section>
  )
}

export default App
