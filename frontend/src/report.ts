export type ReportFinding = {
  area: string
  description: string
  metrics: string[]
  severity: 'yellow' | 'red'
  criteria: string
}

export type ReportInput = {
  generatedAt: string
  dataset: {
    filename: string
    rows: number
    columns: number
  }
  validation: {
    predictionColumn: string
    predictionType: 'pd' | 'score'
    targetColumn: string
    positiveClass: string
    timeColumn: string
    timeCutoffDate: string
    populationColumns: string[]
    sensitiveColumns: string[]
    featureColumns: string[]
  }
  policy: {
    id: string
    name: string
    version: string
    schemaVersion: number
    minimumObservations: number
    minimumDefaults: number
    minimumNonDefaults: number
  }
  discrimination: null | {
    observations: number
    defaults: number
    defaultRate: number
    auc: number | null
    gini: number | null
    ks: number | null
    badCapture10: number | null
    roc: { fpr: number; tpr: number }[]
  }
  calibration: null | {
    brier: number | null
    oe: number | null
    meanPd: number
    defaultRate: number
    spiegelhalterP: number | null
    bins: { meanPd: number; observedRate: number }[]
  }
  stability: {
    variable: string
    psi: number
  }[]
  dataQuality: null | {
    missingRate: number
    duplicateRows: number
    duplicateRate: number
  }
  findings: ReportFinding[]
}

const esc = (value: string | number | null | undefined) => String(value ?? '—')
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;')
  .replaceAll("'", '&#039;')

const pct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`

const num = (value: number | null | undefined, digits = 3) =>
  value === null || value === undefined ? '—' : value.toFixed(digits)

const lineChartSvg = (
  title: string,
  series: { name: string; points: [number, number][]; className: string }[],
  xLabel: string,
  yLabel: string,
  maxX = 1,
  maxY = 1,
) => {
  const width = 760
  const height = 340
  const padLeft = 58
  const padRight = 24
  const padTop = 34
  const padBottom = 52
  const innerW = width - padLeft - padRight
  const innerH = height - padTop - padBottom
  const sx = (x: number) => padLeft + (Math.max(0, Math.min(maxX, x)) / maxX) * innerW
  const sy = (y: number) => padTop + innerH - (Math.max(0, Math.min(maxY, y)) / maxY) * innerH
  const grid = [0, .25, .5, .75, 1].map((v) => {
    const x = sx(v * maxX)
    const y = sy(v * maxY)
    return `<line x1="${x}" y1="${padTop}" x2="${x}" y2="${padTop + innerH}" class="grid"/><line x1="${padLeft}" y1="${y}" x2="${padLeft + innerW}" y2="${y}" class="grid"/><text x="${x}" y="${height - 24}" class="axis-text" text-anchor="middle">${(v * maxX).toFixed(2)}</text><text x="${padLeft - 10}" y="${y + 4}" class="axis-text" text-anchor="end">${(v * maxY).toFixed(2)}</text>`
  }).join('')

  const paths = series.map((item) => {
    const path = item.points.map(([x, y], index) => `${index ? 'L' : 'M'} ${sx(x).toFixed(1)} ${sy(y).toFixed(1)}`).join(' ')
    return `<path d="${path}" class="${item.className}"/>`
  }).join('')

  const legend = series.map((item) => `<span><i class="${item.className}"></i>${esc(item.name)}</span>`).join('')

  return `
    <figure class="report-chart">
      <figcaption><strong>${esc(title)}</strong><div class="legend">${legend}</div></figcaption>
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title)}">
        ${grid}
        <line x1="${padLeft}" y1="${padTop + innerH}" x2="${padLeft + innerW}" y2="${padTop + innerH}" class="axis"/>
        <line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${padTop + innerH}" class="axis"/>
        ${paths}
        <text x="${padLeft + innerW / 2}" y="${height - 4}" class="axis-title" text-anchor="middle">${esc(xLabel)}</text>
        <text x="14" y="${padTop + innerH / 2}" class="axis-title" text-anchor="middle" transform="rotate(-90 14 ${padTop + innerH / 2})">${esc(yLabel)}</text>
      </svg>
    </figure>`
}

const stabilityBars = (rows: { variable: string; psi: number }[]) => {
  const top = [...rows].sort((a, b) => b.psi - a.psi).slice(0, 12)
  const max = Math.max(...top.map((row) => row.psi), .01)
  return `
    <figure class="report-chart">
      <figcaption><strong>Top PSI values</strong><span>Highest available stability shifts</span></figcaption>
      <div class="bar-list">
        ${top.map((row) => `<div class="bar-row"><span>${esc(row.variable)}</span><div><i style="width:${Math.min(100, row.psi / max * 100).toFixed(1)}%"></i></div><strong>${row.psi.toFixed(3)}</strong></div>`).join('')}
      </div>
    </figure>`
}

export const generateSelfContainedReport = (input: ReportInput) => {
  const redCount = input.findings.filter((finding) => finding.severity === 'red').length
  const yellowCount = input.findings.filter((finding) => finding.severity === 'yellow').length

  const roc = input.discrimination?.roc?.length
    ? lineChartSvg(
        'ROC curve',
        [
          { name: 'Model', className: 'series-blue', points: input.discrimination.roc.map((point) => [point.fpr, point.tpr] as [number, number]) },
          { name: 'Random', className: 'series-muted', points: [[0, 0], [1, 1]] },
        ],
        'False positive rate',
        'True positive rate',
      )
    : ''

  const calibration = input.calibration?.bins?.length
    ? lineChartSvg(
        'Calibration curve',
        [
          { name: 'Observed', className: 'series-green', points: input.calibration.bins.map((row) => [row.meanPd, row.observedRate] as [number, number]) },
          { name: 'Ideal', className: 'series-muted', points: [[0, 0], [1, 1]] },
        ],
        'Mean predicted PD',
        'Observed default rate',
      )
    : ''

  const findingsRows = input.findings.length
    ? input.findings.map((finding) => `
      <tr>
        <td><span class="area">${esc(finding.area)}</span><strong>${esc(finding.description)}</strong></td>
        <td>${finding.metrics.map((metric) => `<span class="metric-chip">${esc(metric)}</span>`).join('')}</td>
        <td><span class="severity ${finding.severity}">${finding.severity.toUpperCase()}</span></td>
        <td>${esc(finding.criteria)}</td>
      </tr>`).join('')
    : '<tr><td colspan="4" class="empty">No findings triggered by the active policy.</td></tr>'

  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>RiskLab validation report</title>
<style>
  :root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#dfe8f5;background:#08111e}
  *{box-sizing:border-box}
  body{margin:0;background:#08111e;color:#dfe8f5}
  main{max-width:1220px;margin:0 auto;padding:36px}
  h1,h2,h3,p{margin-top:0}
  h1{font-size:34px;margin-bottom:8px}
  h2{font-size:20px;margin-bottom:12px}
  .muted{color:#71809a}
  .eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#7891b5;font-weight:800}
  .header{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:26px}
  .stamp{text-align:right;color:#7d8da5;font-size:12px}
  .card{border:1px solid #203149;border-radius:14px;background:#0c1523;padding:18px;margin-top:14px}
  .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
  .metric{padding:14px;border:1px solid #23344d;border-radius:10px;background:#0a1220}
  .metric span,.metric strong{display:block}.metric span{font-size:10px;color:#70819a;text-transform:uppercase}.metric strong{font-size:20px;margin-top:6px}
  .severity{display:inline-flex;padding:5px 9px;border-radius:999px;font-size:10px;font-weight:800}
  .severity.red{color:#ff8b8b;border:1px solid rgba(255,107,107,.35);background:rgba(255,107,107,.10)}
  .severity.yellow{color:#f2bc66;border:1px solid rgba(245,170,67,.35);background:rgba(245,170,67,.10)}
  table{width:100%;border-collapse:collapse;font-size:11px}
  th,td{padding:10px 11px;border-bottom:1px solid #1d2b40;text-align:left;vertical-align:top}
  th{color:#8293aa;text-transform:uppercase;font-size:9px;letter-spacing:.05em}
  td strong{display:block}.area{display:block;color:#6e829f;font-size:9px;text-transform:uppercase;margin-bottom:4px}
  .metric-chip{display:inline-flex;padding:3px 7px;border:1px solid #2a3b54;border-radius:999px;margin:2px;color:#a7b4c7;background:#0b1422}
  .report-chart{margin:0;padding:12px;border:1px solid #22324a;border-radius:12px;background:#091321}
  .report-chart figcaption{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:8px;color:#cfd9e8}
  .legend{display:flex;gap:12px;color:#788ba5;font-size:10px}.legend span{display:flex;align-items:center;gap:5px}
  .legend i{width:18px;height:2px;display:inline-block}.legend .series-blue,.series-blue{stroke:#4f8cff;background:#4f8cff}.legend .series-green,.series-green{stroke:#69e7ad;background:#69e7ad}.legend .series-muted,.series-muted{stroke:#67778f;background:#67778f}
  svg{width:100%;height:auto}.axis{stroke:#42516a;stroke-width:1}.grid{stroke:#1a293d;stroke-width:1}.axis-text{fill:#677991;font-size:9px}.axis-title{fill:#71839c;font-size:10px}
  .series-blue,.series-green,.series-muted{fill:none;stroke-width:2}.series-muted{stroke-dasharray:5 5;stroke-width:1.4}
  .chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .bar-list{display:grid;gap:8px}.bar-row{display:grid;grid-template-columns:180px 1fr 60px;gap:10px;align-items:center;font-size:10px}.bar-row>span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#90a0b6}.bar-row>div{height:9px;border-radius:999px;background:#142237;overflow:hidden}.bar-row i{display:block;height:100%;background:#4f8cff}.bar-row strong{text-align:right}
  .policy{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.policy div{padding:10px;border:1px solid #223149;border-radius:9px}.policy span,.policy strong{display:block}.policy span{font-size:9px;color:#6d7d95;text-transform:uppercase}.policy strong{margin-top:4px;font-size:12px}
  .empty{color:#71809a;text-align:center;padding:24px}
  footer{margin-top:24px;color:#61738c;font-size:10px;text-align:center}
  @media(max-width:900px){main{padding:20px}.grid,.policy,.chart-grid{grid-template-columns:1fr 1fr}.header{flex-direction:column}.stamp{text-align:left}}
  @media(max-width:560px){.grid,.policy,.chart-grid{grid-template-columns:1fr}.bar-row{grid-template-columns:120px 1fr 50px}}
  @media print{body{background:#fff;color:#111}main{max-width:none;padding:0}.card,.report-chart,.metric,.policy div{break-inside:avoid;background:#fff;border-color:#ccc}.muted,.eyebrow,.stamp,.area,th{color:#555}.grid,.policy,.chart-grid{gap:8px}.series-blue{stroke:#1f5fbf}.series-green{stroke:#2b8c62}.series-muted{stroke:#777}}
</style>
</head>
<body>
<main>
  <section class="header">
    <div>
      <p class="eyebrow">RiskLab validation report</p>
      <h1>Model validation summary</h1>
      <p class="muted">${esc(input.dataset.filename)} · ${input.dataset.rows.toLocaleString()} observations · ${input.dataset.columns.toLocaleString()} columns</p>
    </div>
    <div class="stamp">Generated ${esc(input.generatedAt)}<br/>Self-contained HTML</div>
  </section>

  <section class="grid">
    <div class="metric"><span>Red findings</span><strong>${redCount}</strong></div>
    <div class="metric"><span>Yellow findings</span><strong>${yellowCount}</strong></div>
    <div class="metric"><span>AUC</span><strong>${esc(input.discrimination?.auc === null || input.discrimination?.auc === undefined ? '—' : input.discrimination.auc.toFixed(3))}</strong></div>
    <div class="metric"><span>Default rate</span><strong>${pct(input.discrimination?.defaultRate)}</strong></div>
  </section>

  <section class="card">
    <p class="eyebrow">Validation configuration</p>
    <div class="policy">
      <div><span>Prediction</span><strong>${esc(input.validation.predictionColumn)} (${esc(input.validation.predictionType)})</strong></div>
      <div><span>Target</span><strong>${esc(input.validation.targetColumn)} = ${esc(input.validation.positiveClass)}</strong></div>
      <div><span>Time</span><strong>${esc(input.validation.timeColumn || 'Not configured')}</strong></div>
      <div><span>Cut-off</span><strong>${esc(input.validation.timeCutoffDate || 'Not configured')}</strong></div>
    </div>
  </section>

  <section class="card">
    <p class="eyebrow">Finding policy used</p>
    <h2>${esc(input.policy.name)} v${esc(input.policy.version)}</h2>
    <p class="muted"><code>${esc(input.policy.id)}</code> · schema v${input.policy.schemaVersion}</p>
    <div class="policy">
      <div><span>Minimum observations</span><strong>${input.policy.minimumObservations}</strong></div>
      <div><span>Minimum defaults</span><strong>${input.policy.minimumDefaults}</strong></div>
      <div><span>Minimum non-defaults</span><strong>${input.policy.minimumNonDefaults}</strong></div>
      <div><span>Severity scale</span><strong>Yellow / Red</strong></div>
    </div>
  </section>

  <section class="card">
    <p class="eyebrow">Executive metrics</p>
    <div class="grid">
      <div class="metric"><span>Observations</span><strong>${input.discrimination?.observations?.toLocaleString() ?? '—'}</strong></div>
      <div class="metric"><span>Defaults</span><strong>${input.discrimination?.defaults?.toLocaleString() ?? '—'}</strong></div>
      <div class="metric"><span>Gini</span><strong>${num(input.discrimination?.gini)}</strong></div>
      <div class="metric"><span>KS</span><strong>${num(input.discrimination?.ks)}</strong></div>
      <div class="metric"><span>Brier</span><strong>${num(input.calibration?.brier)}</strong></div>
      <div class="metric"><span>O/E</span><strong>${num(input.calibration?.oe)}</strong></div>
      <div class="metric"><span>Mean PD</span><strong>${pct(input.calibration?.meanPd)}</strong></div>
      <div class="metric"><span>Missing cells</span><strong>${pct(input.dataQuality?.missingRate)}</strong></div>
    </div>
  </section>

  <section class="card">
    <p class="eyebrow">Charts</p>
    <div class="chart-grid">${roc}${calibration}</div>
    ${input.stability.length ? stabilityBars(input.stability) : ''}
  </section>

  <section class="card">
    <p class="eyebrow">Findings</p>
    <h2>Policy-driven findings</h2>
    <table>
      <thead><tr><th>Finding</th><th>Key metrics</th><th>Severity</th><th>Policy criteria</th></tr></thead>
      <tbody>${findingsRows}</tbody>
    </table>
  </section>

  <section class="card">
    <p class="eyebrow">Scope metadata</p>
    <table>
      <tbody>
        <tr><th>Population columns</th><td>${esc(input.validation.populationColumns.join(', ') || 'None')}</td></tr>
        <tr><th>Sensitive attributes</th><td>${esc(input.validation.sensitiveColumns.join(', ') || 'None')}</td></tr>
        <tr><th>Model features</th><td>${esc(input.validation.featureColumns.join(', ') || 'None')}</td></tr>
        <tr><th>Duplicate rows</th><td>${input.dataQuality?.duplicateRows?.toLocaleString() ?? '—'} (${pct(input.dataQuality?.duplicateRate)})</td></tr>
        <tr><th>Spiegelhalter p-value</th><td>${num(input.calibration?.spiegelhalterP, 4)}</td></tr>
        <tr><th>Bad capture @10%</th><td>${pct(input.discrimination?.badCapture10)}</td></tr>
      </tbody>
    </table>
  </section>

  <footer>Generated by RiskLab. This report is self-contained and does not load external assets.</footer>
</main>
</body>
</html>`

  return html
}
