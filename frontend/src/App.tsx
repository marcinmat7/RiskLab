import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react'
import * as echarts from 'echarts'

type HealthResponse = { status: string; service: string }

type PreviewRow = {
  row_number: number
  values: Record<string, string | null>
}

type DatasetPreview = {
  dataset_id: string
  filename: string
  file_format: string
  file_size_bytes: number | null
  row_count: number
  column_count: number
  columns: string[]
  column_types: Record<string, PhysicalType>
  first_preview: PreviewRow[]
  random_preview: PreviewRow[]
  delimiter: string | null
  warning: string | null
  max_rows: number
}

type UploadStatus = 'idle' | 'selected' | 'uploading' | 'ready' | 'error'
type Section = 'Overview' | 'Data' | 'Validation' | 'EDA' | 'Discrimination' | 'Calibration' | 'Stability' | 'Segments' | 'Findings' | 'Reports' | 'Methodology'
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
  timeCutoffDate: string
  timeCutoffAsSegment: boolean
  timeCutoffBeforeLabel: string
  timeCutoffAfterLabel: string
  populationColumns: string[]
  sensitiveColumns: string[]
  featureColumns: string[]
  semanticTypes: Record<string, SemanticType>
}

type ValidationConfigFile = {
  schemaVersion: 1
  kind: 'risklab-validation-config'
  validation: ValidationConfig
}

type ValidationConfigMessage = {
  tone: 'success' | 'warning' | 'error'
  title: string
  details: string[]
}

type ValidationParsingWarning = {
  column: string
  detected_physical_type: PhysicalType
  requested_semantic_type: SemanticType
  effective_semantic_type: SemanticType
  non_missing_count: number
  invalid_count: number
  invalid_examples: string[]
  status: 'warning'
  message: string
}

type ValidationRunResult = {
  status: 'ok' | 'warning'
  effective_semantic_types: Record<string, SemanticType>
  column_results: {
    column: string
    detected_physical_type: PhysicalType
    requested_semantic_type: SemanticType
    effective_semantic_type: SemanticType
    non_missing_count: number
    invalid_count: number
    invalid_examples: string[]
    status: 'ok' | 'warning'
  }[]
  warnings: ValidationParsingWarning[]
}

type EdaTab = 'Overview' | 'Data quality' | 'Distributions' | 'Relationships' | 'Population comparison' | 'Time analysis' | 'Missingness'
type TimeGranularity = 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'yearly'

type EdaProfile = {
  column: string
  semantic_type: SemanticType
  missing_count: number
  missing_rate: number
  sample_missing_count: number
  unique_count: number | null
  unique_count_capped: boolean
  numeric_summary?: {
    count: number
    mean: number
    std: number
    min: number
    p25: number | null
    median: number | null
    p75: number | null
    max: number
    histogram: { label: string; count: number }[]
    cdf: { x: number; cdf: number }[]
  }
  categories?: { value: string; count: number; share: number }[]
  univariate_auc: number | null
  auc_direction: 'higher-is-riskier' | 'lower-is-riskier' | 'category-logit' | null
  category_logits: {
    value: string
    count: number
    defaults: number
    default_rate: number
    logit: number
  }[]
  uniqueness_rate?: number | null
  text_summary?: { mean_length: number; median_length: number; max_length: number }
}

type EdaResult = {
  sample_size: number
  sample_limit: number
  overview: {
    rows: number
    columns: number
    missing_cells: number
    missing_rate: number
    duplicate_rows: number
    duplicate_rate: number
  }
  profiles: EdaProfile[]
  relationships: { left: string; right: string; kind: string; score: number }[]
  relationship_columns: string[]
  relationship_columns_total: number
  relationship_columns_limit: number
  population_comparison: {
    column: string
    groups: { value: string; sample_rows: number; share: number; missing_rate: number }[]
    references: {
      reference: string
      comparisons: {
        comparison: string
        variables: {
          column: string
          semantic_type: SemanticType
          psi: number
          distribution: { label: string; reference_share: number; comparison_share: number }[]
        }[]
      }[]
    }[]
  }[]
  time_analysis: null | {
    time_column: string
    numeric_columns: string[]
    granularities: Record<TimeGranularity, {
      bucket: string
      observations: number
      missing_rate: number
      variables: Record<string, {
        min: number | null
        max: number | null
        mean: number | null
        median: number | null
        missing_rate: number
      }>
    }[]>
  }
  missingness_diagnostics: {
    column: string
    missing_count: number
    missing_rate: number
    assessment: string
    explanation: string
    strongest_associations: { column: string; score: number; evidence: string }[]
    mnar_note: string
  }[]
}


type DiscriminationPoint = { fpr: number; tpr: number; threshold: number }
type DiscriminationMetrics = {
  observations: number
  defaults: number
  default_rate: number
  auc: number | null
  gini: number | null
  ks: number | null
  bad_capture_10: number | null
  roc: DiscriminationPoint[]
  cap: { population: number; bad_capture: number }[]
}
type DiscriminationSegment = {
  key: string
  name: string
  columns: string[]
  groups: {
    value: string
    observations: number
    defaults: number
    default_rate: number
    auc: number | null
    gini: number | null
    ks: number | null
    bad_capture_10: number | null
    roc: DiscriminationPoint[]
    cap: { population: number; bad_capture: number }[]
  }[]
}
type DiscriminationTimeBucket = {
  bucket: string
  observations: number
  defaults: number
  default_rate: number
  auc: number | null
  gini: number | null
  ks: number | null
  bad_capture_10: number | null
  roc: DiscriminationPoint[]
  cap: { population: number; bad_capture: number }[]
}
type DiscriminationResult = {
  overall: DiscriminationMetrics
  segment_dimensions: {
    column: string
    semantic_type: SemanticType
    mode: 'quantiles' | 'categories'
    bins: string[]
    edges?: number[]
    display_name?: string
    cutoff_date?: string
  }[]
  segment_performance: DiscriminationSegment[]
  time_performance: null | {
    time_column: string
    granularities: Record<TimeGranularity, {
      overall: DiscriminationTimeBucket[]
      segments: {
        key: string
        name: string
        columns: string[]
        groups: { value: string; buckets: DiscriminationTimeBucket[] }[]
      }[]
    }>
  }
  excluded: { missing_prediction: number; missing_target: number }
  direction: string
  time_cutoff?: null | {
    date: string
    as_segment: boolean
    before_label: string
    after_label: string
  }
}

type CalibrationSummary = {
  observations: number
  defaults: number
  default_rate: number
  mean_pd: number
  expected_defaults: number
  oe_ratio: number | null
  brier_score: number | null
  spiegelhalter_z: number | null
  spiegelhalter_p_value: number | null
}

type CalibrationResult = {
  overall: CalibrationSummary & {
    calibration_bins: {
      bin: number
      observations: number
      defaults: number
      mean_pd: number
      observed_default_rate: number
      ci_lower: number | null
      ci_upper: number | null
    }[]
  }
  segment_dimensions: {
    column: string
    display_name: string
    semantic_type: SemanticType
    mode: 'quantiles' | 'categories'
    bins: string[]
  }[]
  segment_performance: {
    key: string
    name: string
    columns: string[]
    groups: ({ value: string } & CalibrationSummary)[]
  }[]
  time_performance: null | {
    time_column: string
    granularities: Record<TimeGranularity, {
      overall: ({ bucket: string } & CalibrationSummary)[]
      segments: {
        key: string
        name: string
        groups: { value: string; buckets: ({ bucket: string } & CalibrationSummary)[] }[]
      }[]
    }>
  }
  excluded: {
    missing_prediction: number
    missing_target: number
    invalid_pd: number
  }
}

type StabilityVariable = {
  column: string
  semantic_type: SemanticType
  psi: number
  reference_mean: number | null
  comparison_mean: number | null
  reference_missing_rate: number
  comparison_missing_rate: number
  missing_delta: number
  distribution: {
    label: string
    reference_share: number
    comparison_share: number
  }[]
}

type StabilityResult = {
  sources: {
    key: string
    name: string
    groups: { value: string; observations: number }[]
  }[]
  comparisons: {
    source_key: string
    source_name: string
    reference: string
    comparison: string
    reference_observations: number
    comparison_observations: number
    variables: StabilityVariable[]
  }[]
  variables: {
    column: string
    semantic_type: SemanticType
  }[]
  time_analysis: null | {
    time_column: string
    granularities: Record<TimeGranularity, {
      bucket: string
      observations: number
      references: {
        source_key: string
        reference: string
        variables: { column: string; psi: number }[]
      }[]
    }[]>
  }
  thresholds: {
    moderate: number
    high: number
    note: string
  }
}

const navItems: Section[] = ['Overview', 'Data', 'Validation', 'EDA', 'Discrimination', 'Calibration', 'Stability', 'Segments', 'Findings', 'Reports', 'Methodology']
const analysisSections: AnalysisSection[] = ['Validation', 'EDA', 'Discrimination', 'Calibration', 'Stability', 'Segments', 'Findings', 'Reports']

type MethodologyEntry = {
  id: string
  title: string
  aliases: string[]
  group: 'Discrimination' | 'Calibration' | 'Stability' | 'EDA'
  summary: string
  formula?: string
  interpretation: string[]
  implementation: string[]
  limitations: string[]
  related: string[]
}

const methodologyEntries: MethodologyEntry[] = [
  {
    id: 'roc-auc',
    title: 'ROC AUC',
    aliases: ['auc', 'area under roc', 'receiver operating characteristic'],
    group: 'Discrimination',
    summary: 'Measures how well a score ranks randomly selected defaulted observations above randomly selected non-defaulted observations.',
    formula: 'AUC = P(score_default > score_non-default) + 0.5 · P(tie)',
    interpretation: ['0.5 means no ranking power.', '1.0 means perfect ranking.', 'AUC describes ranking, not probability calibration.'],
    implementation: ['RiskLab normalizes model direction so higher internal risk score means higher risk.', 'ROC points are evaluated at score boundaries.'],
    limitations: ['AUC can look strong even when probability calibration is poor.', 'It can hide weak performance in a specific operating region or subgroup.'],
    related: ['gini', 'ks', 'cap'],
  },
  {
    id: 'gini',
    title: 'Gini coefficient',
    aliases: ['gini'],
    group: 'Discrimination',
    summary: 'A linear transformation of ROC AUC commonly used in credit risk.',
    formula: 'Gini = 2 · AUC − 1',
    interpretation: ['0 corresponds to AUC = 0.5.', '1 corresponds to perfect ranking.'],
    implementation: ['RiskLab derives Gini directly from the calculated ROC AUC.'],
    limitations: ['It contains the same ranking information as AUC and should not be interpreted as a separate independent performance test.'],
    related: ['roc-auc', 'ks'],
  },
  {
    id: 'ks',
    title: 'Kolmogorov–Smirnov statistic (KS)',
    aliases: ['ks', 'kolmogorov smirnov'],
    group: 'Discrimination',
    summary: 'Measures the largest separation between cumulative default and non-default score distributions.',
    formula: 'KS = max |TPR − FPR|',
    interpretation: ['Higher values indicate stronger separation.', 'The statistic is tied to the point of maximum separation, unlike AUC which summarizes the full ranking curve.'],
    implementation: ['RiskLab computes KS from the same score-ordered ROC traversal used for discrimination.'],
    limitations: ['A single maximum-separation point can hide behaviour elsewhere in the score range.'],
    related: ['roc-auc', 'gini'],
  },
  {
    id: 'cap',
    title: 'CAP / cumulative gains curve',
    aliases: ['cap', 'cumulative accuracy profile', 'gains'],
    group: 'Discrimination',
    summary: 'Shows how quickly defaults are captured when observations are ordered from highest to lowest predicted risk.',
    interpretation: ['A stronger model captures a larger share of defaults in a smaller share of the population.', 'The diagonal-like random benchmark represents no ranking power.'],
    implementation: ['RiskLab orders observations by normalized risk score and reports cumulative population share versus cumulative captured defaults.'],
    limitations: ['The shape depends on the portfolio default rate, so direct visual comparison across very different populations needs care.'],
    related: ['roc-auc', 'bad-capture'],
  },
  {
    id: 'bad-capture',
    title: 'Bad capture @ 10%',
    aliases: ['bad capture', 'capture at 10', 'top 10'],
    group: 'Discrimination',
    summary: 'Share of all defaults captured within the riskiest 10% of observations.',
    formula: 'Bad capture @ 10% = defaults in top-risk 10% / all defaults',
    interpretation: ['Higher values indicate that defaults are concentrated near the risky end of the ranking.'],
    implementation: ['RiskLab uses ceil(10% · N) observations after sorting by normalized risk score.'],
    limitations: ['It focuses on one operating point and is sensitive to sample size and the portfolio default rate.'],
    related: ['cap', 'roc-auc'],
  },
  {
    id: 'brier',
    title: 'Brier score',
    aliases: ['brier'],
    group: 'Calibration',
    summary: 'Mean squared error between predicted probabilities and binary outcomes.',
    formula: 'Brier = mean((p − y)²)',
    interpretation: ['Lower is better.', 'A value of 0 means perfect probability predictions on the observed sample.'],
    implementation: ['RiskLab calculates the score on usable PD/target observations.'],
    limitations: ['Brier score combines calibration and discrimination effects; it is not a pure calibration test.'],
    related: ['spiegelhalter', 'oe-ratio', 'calibration-curve'],
  },
  {
    id: 'spiegelhalter',
    title: 'Spiegelhalter Z-test',
    aliases: ['spiegelhalter', 'spiegelhalter test', 'z test calibration'],
    group: 'Calibration',
    summary: 'Tests whether submitted predicted probabilities are statistically consistent with observed binary outcomes.',
    formula: 'Z = Σ[(y − p)(1 − 2p)] / √Σ[(1 − 2p)²p(1 − p)]',
    interpretation: ['p ≥ 5%: no significant evidence of miscalibration.', '1% ≤ p < 5%: evidence of miscalibration.', 'p < 1%: strong evidence of miscalibration.'],
    implementation: ['RiskLab uses the observation-level two-sided test.', 'The displayed p-value is computed from the standard normal distribution.'],
    limitations: ['A non-significant result does not prove good calibration.', 'Statistical power depends on sample size and the distribution of submitted probabilities.'],
    related: ['brier', 'oe-ratio', 'calibration-curve'],
  },
  {
    id: 'oe-ratio',
    title: 'Observed / Expected ratio (O/E)',
    aliases: ['oe', 'o/e', 'observed expected'],
    group: 'Calibration',
    summary: 'Compares the observed number of defaults with the number expected from submitted PDs.',
    formula: 'O/E = Σy / Σp',
    interpretation: ['O/E = 1 indicates agreement in average risk level.', 'O/E > 1 means observed defaults exceed expected defaults.', 'O/E < 1 means expected defaults exceed observed defaults.'],
    implementation: ['Expected defaults are the sum of individual PD values.'],
    limitations: ['O/E measures average calibration and can miss offsetting calibration errors across the PD range.'],
    related: ['brier', 'spiegelhalter', 'calibration-curve'],
  },
  {
    id: 'calibration-curve',
    title: 'Calibration curve',
    aliases: ['calibration plot', 'reliability curve'],
    group: 'Calibration',
    summary: 'Compares average predicted PD with observed default rate across groups of observations.',
    interpretation: ['Points near the 45° line indicate agreement between predicted and observed rates.', 'Points above the line indicate underprediction of risk; points below indicate overprediction.'],
    implementation: ['RiskLab currently uses 10 equal-frequency PD bins for the overall curve.', 'Observed-rate confidence intervals are computed with Wilson intervals in the backend.'],
    limitations: ['The visual result depends on binning, and sparse bins can be noisy.'],
    related: ['wilson', 'brier', 'spiegelhalter'],
  },
  {
    id: 'wilson',
    title: 'Wilson confidence interval',
    aliases: ['wilson', 'confidence interval default rate'],
    group: 'Calibration',
    summary: 'Binomial proportion confidence interval used for observed default rates.',
    interpretation: ['Wider intervals indicate greater uncertainty, typically because a bin contains fewer observations or defaults.'],
    implementation: ['RiskLab computes 95% Wilson intervals for observed default rates in calibration bins.'],
    limitations: ['The interval describes uncertainty in the observed rate, not uncertainty in model parameters or predicted PDs.'],
    related: ['calibration-curve'],
  },
  {
    id: 'psi',
    title: 'Population Stability Index (PSI)',
    aliases: ['psi', 'population stability index', 'stability index'],
    group: 'Stability',
    summary: 'Measures distribution shift between a reference population and a comparison population.',
    formula: 'PSI = Σ (comparison_share − reference_share) · ln(comparison_share / reference_share)',
    interpretation: ['RiskLab default heuristic: PSI < 0.10 low shift.', '0.10–0.25 moderate shift.', '> 0.25 high shift.'],
    implementation: ['Numeric bin edges are defined from reference-population quantiles and reused for comparison.', 'Categorical variables use categories plus Other where needed.', 'Missing is treated as a separate bucket.', 'Small epsilon smoothing avoids division by zero.'],
    limitations: ['The 0.10/0.25 thresholds are heuristics, not universal regulatory rules.', 'PSI depends on binning and sample size and does not explain the cause of drift.'],
    related: ['missing-rate'],
  },
  {
    id: 'pearson',
    title: 'Pearson correlation',
    aliases: ['pearson', 'pearson r', 'correlation'],
    group: 'EDA',
    summary: 'Measures the strength and direction of a linear relationship between two numeric variables.',
    formula: 'r = cov(X,Y) / (σX · σY)',
    interpretation: ['Values range from −1 to 1.', 'The sign gives direction; absolute magnitude gives linear association strength.'],
    implementation: ['RiskLab computes Pearson correlation on paired non-missing numeric values from the EDA sample.'],
    limitations: ['Correlation does not imply causation.', 'Pearson correlation can miss strong nonlinear relationships and is sensitive to outliers.'],
    related: ['cramers-v', 'eta'],
  },
  {
    id: 'cramers-v',
    title: "Cramér's V",
    aliases: ['cramers v', 'cramér', 'categorical association'],
    group: 'EDA',
    summary: 'Measures association strength between two categorical variables using the chi-squared contingency-table statistic.',
    formula: 'V = √[(χ² / n) / min(r − 1, c − 1)]',
    interpretation: ['Values range from 0 to 1.', '0 indicates no detected association; larger values indicate stronger association.'],
    implementation: ['RiskLab calculates V for categorical/boolean pairs when cardinality is manageable.'],
    limitations: ['It has no direction or sign.', 'Large samples can make small relationships statistically detectable even when practical association is weak.'],
    related: ['pearson', 'eta'],
  },
  {
    id: 'eta',
    title: 'Correlation ratio η',
    aliases: ['eta', 'correlation ratio', 'η'],
    group: 'EDA',
    summary: 'Measures how strongly a categorical grouping explains variation in a numeric variable.',
    formula: 'η = √(between-group sum of squares / total sum of squares)',
    interpretation: ['Values range from 0 to 1.', 'Higher values mean numeric values differ more strongly between categories.'],
    implementation: ['RiskLab uses η for mixed numeric–categorical relationships in the EDA association heatmap.'],
    limitations: ['η gives strength but not direction.', 'Association does not imply causation.'],
    related: ['pearson', 'cramers-v'],
  },
  {
    id: 'missing-rate',
    title: 'Missing rate',
    aliases: ['missing', 'missingness', 'null rate'],
    group: 'EDA',
    summary: 'Share of observations for which a variable is missing according to RiskLab missing-value rules.',
    formula: 'Missing rate = missing observations / all observations',
    interpretation: ['Higher missingness can reduce usable sample size and may indicate data-quality or population-change issues.'],
    implementation: ['RiskLab treats empty values and common markers such as NA, N/A, null, none and NaN as missing in EDA.'],
    limitations: ['A missing rate alone does not identify the missingness mechanism (MCAR, MAR or MNAR).'],
    related: ['psi'],
  },
  {
    id: 'univariate-auc',
    title: 'Univariate AUC',
    aliases: ['univariate auc', 'single variable auc', 'category logit auc'],
    group: 'EDA',
    summary: 'Descriptive measure of how well one variable by itself ranks the configured default target.',
    interpretation: ['For numeric variables RiskLab direction-normalizes the displayed value to at least 0.5.', 'For categorical variables observations are scored by smoothed category default-rate logits.'],
    implementation: ['Calculated on the EDA reservoir sample.', 'Categorical smoothing uses (defaults + 0.5) / (count + 1) before the logit transform.'],
    limitations: ['It is in-sample and descriptive, not model-validation evidence.', 'Categorical AUC can be optimistic for high-cardinality variables.'],
    related: ['roc-auc'],
  },
]

const methodologyGroups: MethodologyEntry['group'][] = ['Discrimination', 'Calibration', 'Stability', 'EDA']

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
  EDA: { title: 'Exploratory Data Analysis', description: 'Profile data quality, distributions, relationships, population differences and time behaviour.', action: 'Run EDA' },
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
  timeCutoffDate: '',
  timeCutoffAsSegment: false,
  timeCutoffBeforeLabel: 'Pre cut-off',
  timeCutoffAfterLabel: 'Post cut-off',
  populationColumns: [],
  sensitiveColumns: [],
  featureColumns: [],
  semanticTypes: {},
}

const initialStatuses = (): Record<AnalysisSection, AnalysisStatus> => ({
  Validation: 'not-run',
  EDA: 'blocked',
  Discrimination: 'blocked',
  Calibration: 'blocked',
  Stability: 'blocked',
  Segments: 'blocked',
  Findings: 'blocked',
  Reports: 'blocked',
})

const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`
const formatMetric = (value: number | null | undefined) => value === null || value === undefined ? '—' : Math.abs(value) >= 1000 ? value.toLocaleString(undefined, { maximumFractionDigits: 1 }) : value.toLocaleString(undefined, { maximumFractionDigits: 3 })

const timeBucketForDate = (date: string, granularity: TimeGranularity) => {
  if (!date) return ''
  const [yearText, monthText, dayText] = date.split('-')
  const year = Number(yearText)
  const month = Number(monthText)
  const day = Number(dayText)
  if (!year || !month || !day) return ''
  if (granularity === 'daily') return date
  if (granularity === 'monthly') return `${yearText}-${monthText}`
  if (granularity === 'quarterly') return `${year}-Q${Math.floor((month - 1) / 3) + 1}`
  if (granularity === 'yearly') return String(year)
  const value = new Date(Date.UTC(year, month - 1, day))
  const weekday = value.getUTCDay() || 7
  value.setUTCDate(value.getUTCDate() + 4 - weekday)
  const isoYear = value.getUTCFullYear()
  const yearStart = new Date(Date.UTC(isoYear, 0, 1))
  const week = Math.ceil((((value.getTime() - yearStart.getTime()) / 86400000) + 1) / 7)
  return `${isoYear}-W${String(week).padStart(2, '0')}`
}

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



function EChart({ option, height = 360, group }: { option: echarts.EChartsOption; height?: number; group?: string }) {
  const ref = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (activeSection !== 'Methodology' || !methodologyTarget) return
    const timer = window.setTimeout(() => {
      document.getElementById(`methodology-${methodologyTarget}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 0)
    return () => window.clearTimeout(timer)
  }, [activeSection, methodologyTarget])

  const openMethodology = (metric: string) => {
    setMethodologyTarget(metric)
    setMethodologySearch('')
    setActiveSection('Methodology')
  }

  const metricHelp = (metric: string, label: string) => (
    <button
      type="button"
      className="metric-help-button"
      title={`Open methodology for ${label}`}
      aria-label={`Open methodology for ${label}`}
      onClick={(event) => {
        event.stopPropagation()
        openMethodology(metric)
      }}
    >?</button>
  )

  useEffect(() => {
    if (!ref.current) return
    const chart = echarts.init(ref.current)
    if (group) {
      chart.group = group
      echarts.connect(group)
    }
    chart.setOption(option)
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(ref.current)
    return () => {
      observer.disconnect()
      chart.dispose()
    }
  }, [option, group])

  return <div ref={ref} className="echart" style={{ height }} />
}

function MiniBarChart({ data }: { data: { label: string; value: number }[] }) {
  const max = Math.max(...data.map((item) => item.value), 1)
  return (
    <div className="eda-bar-chart">
      {data.map((item) => (
        <div className="eda-bar-row" key={item.label}>
          <span title={item.label}>{item.label}</span>
          <div><i style={{ width: `${(item.value / max) * 100}%` }} /></div>
          <strong>{item.value.toLocaleString()}</strong>
        </div>
      ))}
    </div>
  )
}

function MiniLineChart({ data, valueSuffix = '' }: { data: { label: string; value: number }[]; valueSuffix?: string }) {
  if (!data.length) return <div className="eda-no-data">No data available.</div>
  const width = 760
  const height = 220
  const pad = 28
  const values = data.map((item) => item.value)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const points = data.map((item, index) => {
    const x = pad + (index * (width - pad * 2)) / Math.max(data.length - 1, 1)
    const y = height - pad - ((item.value - min) / range) * (height - pad * 2)
    return `${x},${y}`
  }).join(' ')

  return (
    <div className="eda-line-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} className="eda-line-chart" role="img">
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} className="axis" />
        <polyline points={points} className="main-line" />
        {data.map((item, index) => {
          const [x, y] = points.split(' ')[index].split(',')
          return <circle key={item.label} cx={x} cy={y} r="3" className="eda-point"><title>{item.label}: {item.value.toFixed(2)}{valueSuffix}</title></circle>
        })}
      </svg>
      <div className="eda-line-labels"><span>{data[0]?.label}</span><span>{data[data.length - 1]?.label}</span></div>
    </div>
  )
}

function App() {
  const [activeSection, setActiveSection] = useState<Section>('Overview')
  const [methodologySearch, setMethodologySearch] = useState('')
  const [methodologyTarget, setMethodologyTarget] = useState('')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>('idle')
  const [validationConfig, setValidationConfig] = useState<ValidationConfig>(initialConfig)
  const [analysisStatus, setAnalysisStatus] = useState<Record<AnalysisSection, AnalysisStatus>>(initialStatuses)
  const [lastRun, setLastRun] = useState<Partial<Record<AnalysisSection, string>>>({})
  const [edaResult, setEdaResult] = useState<EdaResult | null>(null)
  const [edaError, setEdaError] = useState<string | null>(null)
  const [edaTab, setEdaTab] = useState<EdaTab>('Overview')
  const [distributionFilter, setDistributionFilter] = useState<'all' | 'numeric' | 'categorical' | 'other'>('all')
  const [distributionSearch, setDistributionSearch] = useState('')
  const [distributionSort, setDistributionSort] = useState<'dataset' | 'missing' | 'auc' | 'name'>('dataset')
  const [distributionDetailColumn, setDistributionDetailColumn] = useState('')
  const [edaTimeVariable, setEdaTimeVariable] = useState<string>('')
  const [edaTimeMissingColumn, setEdaTimeMissingColumn] = useState<string>('')
  const [timeGranularity, setTimeGranularity] = useState<TimeGranularity>('monthly')
  const [qualitySort, setQualitySort] = useState<'column' | 'semantic' | 'missing' | 'unique' | 'quality'>('missing')
  const [qualitySortDirection, setQualitySortDirection] = useState<'asc' | 'desc'>('desc')
  const [relationshipSearch, setRelationshipSearch] = useState('')
  const [relationshipMode, setRelationshipMode] = useState<'numeric' | 'mixed'>('numeric')
  const [relationshipFocus, setRelationshipFocus] = useState('')
  const [populationColumn, setPopulationColumn] = useState('')
  const [populationReference, setPopulationReference] = useState('')
  const [populationComparison, setPopulationComparison] = useState('')
  const [populationVariable, setPopulationVariable] = useState('')
  const [timeWorkspace, setTimeWorkspace] = useState<string[]>([])
  const [discriminationResult, setDiscriminationResult] = useState<DiscriminationResult | null>(null)
  const [discriminationError, setDiscriminationError] = useState<string | null>(null)
  const [discriminationGranularity, setDiscriminationGranularity] = useState<TimeGranularity>('monthly')
  const [discriminationSegmentKey, setDiscriminationSegmentKey] = useState('')
  const [discriminationTimeSegmentKey, setDiscriminationTimeSegmentKey] = useState('overall')
  const [calibrationResult, setCalibrationResult] = useState<CalibrationResult | null>(null)
  const [calibrationError, setCalibrationError] = useState<string | null>(null)
  const [calibrationSegmentKey, setCalibrationSegmentKey] = useState('')
  const [calibrationGranularity, setCalibrationGranularity] = useState<TimeGranularity>('monthly')
  const [calibrationTimeSegmentKey, setCalibrationTimeSegmentKey] = useState('overall')
  const [stabilityResult, setStabilityResult] = useState<StabilityResult | null>(null)
  const [stabilityError, setStabilityError] = useState<string | null>(null)
  const [stabilitySourceKey, setStabilitySourceKey] = useState('')
  const [stabilityReference, setStabilityReference] = useState('')
  const [stabilityComparison, setStabilityComparison] = useState('')
  const [stabilityVariable, setStabilityVariable] = useState('')
  const [stabilityGranularity, setStabilityGranularity] = useState<TimeGranularity>('monthly')
  const [validationConfigMessage, setValidationConfigMessage] = useState<ValidationConfigMessage | null>(null)
  const [validationParsingWarnings, setValidationParsingWarnings] = useState<ValidationParsingWarning[]>([])
  const [validationRunError, setValidationRunError] = useState<string | null>(null)

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
    if (targetValues.length && !validationConfig.positiveClass) {
      setValidationConfig((current) => ({ ...current, positiveClass: targetValues[0] }))
    }
  }, [targetValues, validationConfig.positiveClass])

  const resetAnalysis = () => {
    setValidationConfig(initialConfig)
    setAnalysisStatus(initialStatuses())
    setLastRun({})
    setEdaResult(null)
    setEdaError(null)
    setEdaTab('Overview')
    setDistributionFilter('all')
    setDistributionSearch('')
    setDistributionSort('dataset')
    setDistributionDetailColumn('')
    setEdaTimeVariable('')
    setEdaTimeMissingColumn('')
    setQualitySort('missing')
    setQualitySortDirection('desc')
    setRelationshipSearch('')
    setRelationshipMode('numeric')
    setRelationshipFocus('')
    setPopulationColumn('')
    setPopulationReference('')
    setPopulationComparison('')
    setPopulationVariable('')
    setTimeWorkspace([])
    setDiscriminationResult(null)
    setDiscriminationError(null)
    setDiscriminationGranularity('monthly')
    setDiscriminationSegmentKey('')
    setDiscriminationTimeSegmentKey('overall')
    setCalibrationResult(null)
    setCalibrationError(null)
    setCalibrationSegmentKey('')
    setCalibrationGranularity('monthly')
    setCalibrationTimeSegmentKey('overall')
    setStabilityResult(null)
    setStabilityError(null)
    setStabilitySourceKey('')
    setStabilityReference('')
    setStabilityComparison('')
    setStabilityVariable('')
    setStabilityGranularity('monthly')
    setValidationConfigMessage(null)
    setValidationParsingWarnings([])
    setValidationRunError(null)
  }

  const invalidateAnalysisResults = () => {
    setAnalysisStatus(initialStatuses())
    setLastRun({})
    setEdaResult(null)
    setEdaError(null)
    setDiscriminationResult(null)
    setDiscriminationError(null)
    setCalibrationResult(null)
    setCalibrationError(null)
    setCalibrationSegmentKey('')
    setCalibrationTimeSegmentKey('overall')
    setStabilityResult(null)
    setStabilityError(null)
    setStabilitySourceKey('')
    setStabilityReference('')
    setStabilityComparison('')
    setStabilityVariable('')
    setEdaTab('Overview')
    setDistributionDetailColumn('')
    setPopulationColumn('')
    setPopulationReference('')
    setPopulationComparison('')
    setPopulationVariable('')
    setTimeWorkspace([])
    setDiscriminationSegmentKey('')
    setDiscriminationTimeSegmentKey('overall')
    setValidationParsingWarnings([])
    setValidationRunError(null)
  }

  const exportValidationConfig = () => {
    if (!preview) return
    const payload: ValidationConfigFile = {
      schemaVersion: 1,
      kind: 'risklab-validation-config',
      validation: validationConfig,
    }
    const blob = new Blob([JSON.stringify(payload, null, 2) + '\n'], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    const baseName = (preview.filename || 'dataset').replace(/\.csv$/i, '').replace(/[^a-zA-Z0-9._-]+/g, '_')
    anchor.href = url
    anchor.download = `${baseName}_risklab_validation.json`
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)
    setValidationConfigMessage({ tone: 'success', title: 'Validation configuration exported', details: ['The JSON contains the complete current Validation setup.'] })
  }

  const importValidationConfig = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file || !preview) return

    try {
      const parsed = JSON.parse(await file.text()) as Partial<ValidationConfigFile>
      if (parsed.schemaVersion !== 1 || parsed.kind !== 'risklab-validation-config' || !parsed.validation || typeof parsed.validation !== 'object') {
        throw new Error('Unsupported RiskLab validation configuration file.')
      }

      const raw = parsed.validation as Partial<ValidationConfig>
      const warnings: string[] = []
      const columns = new Set(preview.columns)
      const semanticValues = new Set<SemanticType>(semanticTypeOptions.map((option) => option.value))

      const existingColumn = (value: unknown, role: string) => {
        if (typeof value !== 'string' || !value) return ''
        if (columns.has(value)) return value
        warnings.push(`${role}: column "${value}" was not found in the current dataset.`)
        return ''
      }

      const existingColumns = (value: unknown, role: string) => {
        if (!Array.isArray(value)) return []
        const result: string[] = []
        value.forEach((item) => {
          if (typeof item !== 'string') return
          if (columns.has(item)) result.push(item)
          else warnings.push(`${role}: column "${item}" was not found and was skipped.`)
        })
        return [...new Set(result)]
      }

      const predictionColumn = existingColumn(raw.predictionColumn, 'Model output')
      const targetColumn = existingColumn(raw.targetColumn, 'Default indicator')
      const timeColumn = existingColumn(raw.timeColumn, 'Time')
      const importedSemantic = raw.semanticTypes && typeof raw.semanticTypes === 'object' ? raw.semanticTypes : {}
      const semanticTypes = Object.fromEntries(preview.columns.map((column) => {
        const imported = importedSemantic[column]
        if (imported && semanticValues.has(imported)) return [column, imported]
        return [column, validationConfig.semanticTypes[column] ?? suggestSemanticType(preview.column_types[column] ?? 'string')]
      })) as Record<string, SemanticType>

      Object.keys(importedSemantic).forEach((column) => {
        if (!columns.has(column)) warnings.push(`Semantic type: column "${column}" was not found and was skipped.`)
      })

      const predictionType: PredictionType = raw.predictionType === 'score' ? 'score' : 'pd'
      const scoreDirection: ScoreDirection = raw.scoreDirection === 'higher-risk' ? 'higher-risk' : 'lower-risk'
      const timeCutoffDate = timeColumn && typeof raw.timeCutoffDate === 'string' ? raw.timeCutoffDate : ''
      const timeCutoffAsSegment = Boolean(timeColumn && timeCutoffDate && raw.timeCutoffAsSegment)

      const imported: ValidationConfig = {
        predictionColumn,
        predictionType,
        targetColumn,
        positiveClass: typeof raw.positiveClass === 'string' ? raw.positiveClass : '',
        scoreDirection,
        timeColumn,
        timeCutoffDate,
        timeCutoffAsSegment,
        timeCutoffBeforeLabel: typeof raw.timeCutoffBeforeLabel === 'string' && raw.timeCutoffBeforeLabel.trim() ? raw.timeCutoffBeforeLabel : 'Pre cut-off',
        timeCutoffAfterLabel: typeof raw.timeCutoffAfterLabel === 'string' && raw.timeCutoffAfterLabel.trim() ? raw.timeCutoffAfterLabel : 'Post cut-off',
        populationColumns: existingColumns(raw.populationColumns, 'Population'),
        sensitiveColumns: existingColumns(raw.sensitiveColumns, 'Sensitive attribute'),
        featureColumns: existingColumns(raw.featureColumns, 'Model feature'),
        semanticTypes,
      }

      if (predictionColumn && targetColumn && predictionColumn === targetColumn) {
        warnings.push('Model output and default indicator point to the same column; review the required setup.')
      }

      setValidationConfig(imported)
      invalidateAnalysisResults()
      setValidationConfigMessage({
        tone: warnings.length ? 'warning' : 'success',
        title: warnings.length ? 'Configuration imported with warnings' : 'Validation configuration imported',
        details: warnings.length ? warnings : ['All referenced columns were matched to the current dataset.'],
      })
    } catch (err) {
      setValidationConfigMessage({
        tone: 'error',
        title: 'Could not import configuration',
        details: [err instanceof Error ? err.message : 'The selected file is not a valid RiskLab validation configuration.'],
      })
    }
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
      const response = await fetch('http://localhost:8000/datasets/upload', { method: 'POST', body: formData })
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


  const runValidation = async () => {
    if (!preview || !requiredConfigReady) return

    setAnalysisStatus((current) => ({ ...current, Validation: 'running' }))
    setValidationRunError(null)
    setValidationParsingWarnings([])

    const formData = new FormData()
    formData.append('dataset_id', preview.dataset_id)
    formData.append('config', JSON.stringify(validationConfig))

    try {
      const response = await fetch('http://localhost:8000/validation/run', { method: 'POST', body: formData })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail ?? `HTTP ${response.status}`)
      const result = body as ValidationRunResult

      setValidationConfig((current) => ({
        ...current,
        semanticTypes: {
          ...current.semanticTypes,
          ...result.effective_semantic_types,
        },
      }))
      setValidationParsingWarnings(result.warnings)

      setAnalysisStatus((current) => {
        const next = { ...current, Validation: 'ready' as AnalysisStatus }
        analysisSections.forEach((item) => {
          if (item !== 'Validation') {
            next[item] = item === 'Calibration' && validationConfig.predictionType === 'score'
              ? 'unavailable'
              : 'not-run'
          }
        })
        return next
      })
      setLastRun((current) => ({ ...current, Validation: new Date().toLocaleString() }))
    } catch (err) {
      setValidationRunError(err instanceof Error ? err.message : 'Validation failed.')
      setAnalysisStatus((current) => ({ ...current, Validation: 'failed' }))
    }
  }


  const runEda = async () => {
    if (!preview || !validationReady) return

    setAnalysisStatus((current) => ({ ...current, EDA: 'running' }))
    setEdaError(null)

    const formData = new FormData()
    formData.append('dataset_id', preview.dataset_id)
    formData.append('config', JSON.stringify(validationConfig))
    formData.append('time_granularity', timeGranularity)

    try {
      const response = await fetch('http://localhost:8000/eda/run', { method: 'POST', body: formData })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail ?? `HTTP ${response.status}`)
      const result = body as EdaResult
      setEdaResult(result)
      setEdaTimeVariable((current) => current || result.profiles.find((profile) => ['continuous', 'ordinal'].includes(profile.semantic_type))?.column || '')
      setEdaTimeMissingColumn((current) => current || result.profiles.find((profile) => profile.missing_count > 0)?.column || result.profiles[0]?.column || '')
      const firstPopulation = result.population_comparison[0]
      if (firstPopulation) {
        setPopulationColumn((current) => current || firstPopulation.column)
        const firstReference = firstPopulation.references[0]
        if (firstReference) {
          setPopulationReference((current) => current || firstReference.reference)
          const firstComparison = firstReference.comparisons[0]
          if (firstComparison) {
            setPopulationComparison((current) => current || firstComparison.comparison)
            setPopulationVariable((current) => current || firstComparison.variables[0]?.column || '')
          }
        }
      }
      setTimeWorkspace((current) => current.length ? current : (result.time_analysis?.numeric_columns.slice(0, 2) ?? []))
      setAnalysisStatus((current) => ({ ...current, EDA: 'ready' }))
      setLastRun((current) => ({ ...current, EDA: new Date().toLocaleString() }))
    } catch (err) {
      setEdaError(err instanceof Error ? err.message : 'EDA failed.')
      setAnalysisStatus((current) => ({ ...current, EDA: 'failed' }))
    }
  }


  const runDiscrimination = async () => {
    if (!preview || !validationReady) return
    setAnalysisStatus((current) => ({ ...current, Discrimination: 'running' }))
    setDiscriminationError(null)

    const formData = new FormData()
    formData.append('dataset_id', preview.dataset_id)
    formData.append('config', JSON.stringify(validationConfig))
    formData.append('time_granularity', discriminationGranularity)

    try {
      const response = await fetch('http://localhost:8000/discrimination/run', { method: 'POST', body: formData })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail ?? `HTTP ${response.status}`)
      const result = body as DiscriminationResult
      setDiscriminationResult(result)
      setDiscriminationSegmentKey((current) => current || result.segment_performance[0]?.key || '')
      setDiscriminationTimeSegmentKey('overall')
      setAnalysisStatus((current) => ({ ...current, Discrimination: 'ready' }))
      setLastRun((current) => ({ ...current, Discrimination: new Date().toLocaleString() }))
    } catch (err) {
      setDiscriminationError(err instanceof Error ? err.message : 'Discrimination analysis failed.')
      setAnalysisStatus((current) => ({ ...current, Discrimination: 'failed' }))
    }
  }

  const runCalibration = async () => {
    if (!preview || !validationReady || validationConfig.predictionType !== 'pd') return
    setAnalysisStatus((current) => ({ ...current, Calibration: 'running' }))
    setCalibrationError(null)

    const formData = new FormData()
    formData.append('dataset_id', preview.dataset_id)
    formData.append('config', JSON.stringify(validationConfig))

    try {
      const response = await fetch('http://localhost:8000/calibration/run', { method: 'POST', body: formData })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail ?? `HTTP ${response.status}`)
      const result = body as CalibrationResult
      setCalibrationResult(result)
      setCalibrationSegmentKey((current) => current || result.segment_performance[0]?.key || '')
      setCalibrationTimeSegmentKey('overall')
      setAnalysisStatus((current) => ({ ...current, Calibration: 'ready' }))
      setLastRun((current) => ({ ...current, Calibration: new Date().toLocaleString() }))
    } catch (err) {
      setCalibrationError(err instanceof Error ? err.message : 'Calibration analysis failed.')
      setAnalysisStatus((current) => ({ ...current, Calibration: 'failed' }))
    }
  }

  const runStability = async () => {
    if (!preview || !validationReady) return
    setAnalysisStatus((current) => ({ ...current, Stability: 'running' }))
    setStabilityError(null)

    const formData = new FormData()
    formData.append('dataset_id', preview.dataset_id)
    formData.append('config', JSON.stringify(validationConfig))

    try {
      const response = await fetch('http://localhost:8000/stability/run', { method: 'POST', body: formData })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail ?? `HTTP ${response.status}`)
      const result = body as StabilityResult
      setStabilityResult(result)

      const firstSource = result.sources[0]
      const firstReference = firstSource?.groups[0]?.value ?? ''
      const firstComparison = firstSource?.groups.find((group) => group.value !== firstReference)?.value ?? ''
      setStabilitySourceKey(firstSource?.key ?? '')
      setStabilityReference(firstReference)
      setStabilityComparison(firstComparison)
      setStabilityVariable(result.comparisons.find((item) =>
        item.source_key === firstSource?.key &&
        item.reference === firstReference &&
        item.comparison === firstComparison
      )?.variables[0]?.column ?? result.variables[0]?.column ?? '')

      setAnalysisStatus((current) => ({ ...current, Stability: 'ready' }))
      setLastRun((current) => ({ ...current, Stability: new Date().toLocaleString() }))
    } catch (err) {
      setStabilityError(err instanceof Error ? err.message : 'Stability analysis failed.')
      setAnalysisStatus((current) => ({ ...current, Stability: 'failed' }))
    }
  }

  const renderMethodology = () => {
    const query = methodologySearch.trim().toLowerCase()
    const filtered = methodologyEntries.filter((entry) => {
      if (!query) return true
      return [entry.title, entry.group, entry.summary, ...entry.aliases, ...entry.related]
        .join(' ')
        .toLowerCase()
        .includes(query)
    })

    const visibleGroups = methodologyGroups
      .map((group) => ({ group, entries: filtered.filter((entry) => entry.group === group) }))
      .filter((item) => item.entries.length > 0)

    return (
      <section className="page-content methodology-page">
        <div className="analysis-page-header">
          <div>
            <p className="eyebrow">Reference</p>
            <h1>Methodology</h1>
            <p>Definitions, interpretation, implementation details and limitations for metrics used across RiskLab.</p>
          </div>
        </div>

        <section className="panel methodology-search-card">
          <div>
            <h2>Metric reference</h2>
            <p>Search by metric name, abbreviation or concept. Help icons across RiskLab link directly to these entries.</p>
          </div>
          <input
            className="methodology-search"
            value={methodologySearch}
            onChange={(event) => setMethodologySearch(event.target.value)}
            placeholder="Search AUC, PSI, Spiegelhalter, correlation…"
          />
        </section>

        <div className="methodology-layout">
          <aside className="panel methodology-index">
            <strong>Contents</strong>
            {methodologyGroups.map((group) => (
              <div key={group}>
                <span>{group}</span>
                {methodologyEntries.filter((entry) => entry.group === group).map((entry) => (
                  <button key={entry.id} onClick={() => {
                    setMethodologySearch('')
                    setMethodologyTarget(entry.id)
                    window.setTimeout(() => document.getElementById(`methodology-${entry.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 0)
                  }}>{entry.title}</button>
                ))}
              </div>
            ))}
          </aside>

          <div className="methodology-content">
            {visibleGroups.map(({ group, entries }) => (
              <section key={group} className="methodology-group">
                <div className="methodology-group-heading"><p className="eyebrow">{group}</p><h2>{group} metrics</h2></div>
                {entries.map((entry) => (
                  <article id={`methodology-${entry.id}`} className="panel methodology-entry" key={entry.id}>
                    <div className="methodology-entry-head">
                      <div><h3>{entry.title}</h3><p>{entry.summary}</p></div>
                      <span className="methodology-anchor">#{entry.id}</span>
                    </div>

                    {entry.formula && (
                      <div className="methodology-block">
                        <h4>Formula</h4>
                        <code>{entry.formula}</code>
                      </div>
                    )}

                    <div className="methodology-columns">
                      <div className="methodology-block">
                        <h4>Interpretation</h4>
                        <ul>{entry.interpretation.map((item) => <li key={item}>{item}</li>)}</ul>
                      </div>
                      <div className="methodology-block">
                        <h4>RiskLab implementation</h4>
                        <ul>{entry.implementation.map((item) => <li key={item}>{item}</li>)}</ul>
                      </div>
                    </div>

                    <div className="methodology-block methodology-limitations">
                      <h4>Important limitations</h4>
                      <ul>{entry.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
                    </div>

                    {entry.related.length > 0 && (
                      <div className="methodology-related">
                        <span>Related</span>
                        {entry.related.map((related) => {
                          const target = methodologyEntries.find((item) => item.id === related)
                          return target ? <button key={related} onClick={() => openMethodology(related)}>{target.title}</button> : null
                        })}
                      </div>
                    )}
                  </article>
                ))}
              </section>
            ))}
            {visibleGroups.length === 0 && <div className="panel eda-no-data">No methodology entries match “{methodologySearch}”.</div>}
          </div>
        </div>
      </section>
    )
  }

  const renderDataPage = () => (
    <section className="page-content">
      <div className="page-heading compact-heading">
        <div><p className="eyebrow">Dataset</p><h1>Upload validation data</h1><p>Select a dataset file, review the file details, then upload it for structural inspection.</p></div>
      </div>

      <section className="panel upload-panel upload-workflow">
        <div className="upload-copy">
          <h2>Validation dataset</h2>
          <p>CSV, TSV, Parquet and Feather. CSV automatically detects comma, semicolon, tab or pipe separators. Maximum 1,000,000 data rows.</p>
        </div>

        {!selectedFile ? (
          <label className="primary-button upload-button"><input type="file" accept=".csv,.tsv,.parquet,.feather,text/csv,text/tab-separated-values,application/vnd.apache.parquet,application/octet-stream" onChange={handleFileChange} />Choose file</label>
        ) : (
          <div className="selected-file-card">
            <div className="file-details">
              <span className="file-icon">{selectedFile.name.split('.').pop()?.toUpperCase() ?? 'DATA'}</span>
              <div><strong>{selectedFile.name}</strong><span>{formatFileSize(selectedFile.size)}</span></div>
            </div>
            <div className={`upload-state ${uploadStatus}`}><span className="state-dot" />{uploadStatus === 'selected' && 'Ready to upload'}{uploadStatus === 'uploading' && 'Uploading…'}{uploadStatus === 'ready' && 'Ready'}{uploadStatus === 'error' && 'Upload failed'}</div>
            <div className="file-actions">
              <label className={`secondary-button compact-button ${uploadStatus === 'uploading' ? 'disabled' : ''}`}><input type="file" accept=".csv,.tsv,.parquet,.feather,text/csv,text/tab-separated-values,application/vnd.apache.parquet,application/octet-stream" onChange={handleFileChange} disabled={uploadStatus === 'uploading'} />Replace</label>
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
            <div><p className="eyebrow">Dataset preview</p><h2>{preview.filename}</h2><p className="preview-meta">Format: <code>{preview.file_format}</code>{preview.delimiter ? <> · Delimiter: <code>{preview.delimiter === '\t' ? 'tab' : preview.delimiter}</code></> : null} · Limit: {preview.max_rows.toLocaleString()} rows</p></div>
            <div className="dataset-stats"><div><span>Rows</span><strong>{preview.row_count.toLocaleString()}</strong></div><div><span>Columns</span><strong>{preview.column_count}</strong></div></div>
          </div>
          <div className="preview-block">
            <div className="preview-block-heading"><div><h3>First 10 rows</h3><p>The first {preview.first_preview.length} data rows in the uploaded dataset.</p></div><span className="sample-badge">Rows 1–{preview.first_preview.length}</span></div>
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
          <div className="validation-header-actions">
            <label className="secondary-button config-file-button"><input type="file" accept=".json,application/json" onChange={importValidationConfig} />Import config</label>
            <button className="secondary-button" onClick={exportValidationConfig}>Export config</button>
            <div className={`analysis-status-pill ${analysisStatus.Validation}`}><span />{statusLabel(analysisStatus.Validation)}</div>
          </div>
        </div>

        {validationRunError && <div className="message error-message">{validationRunError}</div>}

        {validationConfigMessage && (
          <div className={`validation-config-message ${validationConfigMessage.tone}`}>
            <div><strong>{validationConfigMessage.title}</strong><button type="button" onClick={() => setValidationConfigMessage(null)} aria-label="Dismiss">×</button></div>
            {validationConfigMessage.details.length > 0 && <ul>{validationConfigMessage.details.map((detail, index) => <li key={`${detail}-${index}`}>{detail}</li>)}</ul>}
          </div>
        )}

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
            <div><span className="setup-number">2</span><div><h2>Column types</h2><p>RiskLab detects the physical type from the uploaded dataset. Review or change the semantic type used by downstream analyses.</p></div></div>
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
          <div className="type-note">Detected types are suggestions only. RiskLab never infers business meaning from a column name. On Run validation, incompatible semantic types fall back to the detected physical type and are reported as warnings.</div>
          {validationParsingWarnings.length > 0 && (
            <div className="validation-parsing-warnings">
              <div className="validation-parsing-warnings-head">
                <strong>Semantic type parsing warnings</strong>
                <span>{validationParsingWarnings.length} column{validationParsingWarnings.length === 1 ? '' : 's'} adjusted</span>
              </div>
              {validationParsingWarnings.map((warning) => (
                <div className="validation-parsing-warning" key={warning.column}>
                  <div>
                    <strong>{warning.column}</strong>
                    <span>{warning.requested_semantic_type} → {warning.effective_semantic_type}</span>
                  </div>
                  <p>{warning.message}</p>
                  {warning.invalid_examples.length > 0 && <small>Examples: {warning.invalid_examples.map((value) => `"${value}"`).join(', ')}</small>}
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="panel setup-card optional-card">
          <div className="setup-section-heading"><div><span className="setup-number optional">3</span><div><h2>Optional analysis settings</h2><p>These roles unlock time, representativeness, fairness and challenger-model analyses later.</p></div></div><span className="optional-badge">Optional</span></div>

          <div className="optional-setting time-analysis-setting">
            <div className="setting-copy"><h3>Time analysis</h3><p>Select a date or time column for analyses over time. An optional cut-off can be shown consistently across time charts and exposed as a reusable analysis segment.</p><UsedBy modules={['EDA', 'Discrimination', 'Calibration', 'Stability', 'Segments']} /></div>
            <div className="time-config-stack">
              <label className="form-field compact-field"><span>Time column</span><select value={validationConfig.timeColumn} onChange={(e) => setValidationConfig({...validationConfig, timeColumn:e.target.value, ...(e.target.value ? {} : { timeCutoffDate:'', timeCutoffAsSegment:false })})}><option value="">Not configured</option>{preview.columns.map((c) => <option key={c} value={c}>{c}</option>)}</select></label>
              {validationConfig.timeColumn && (
                <div className="time-cutoff-config">
                  <label className="form-field"><span>Time cut-off</span><input type="date" value={validationConfig.timeCutoffDate} onChange={(e) => setValidationConfig({...validationConfig, timeCutoffDate:e.target.value, timeCutoffAsSegment:e.target.value ? validationConfig.timeCutoffAsSegment : false})} /><small>Shown as a vertical dashed marker on time-series charts.</small></label>
                  <label className="cutoff-segment-toggle"><input type="checkbox" checked={validationConfig.timeCutoffAsSegment} disabled={!validationConfig.timeCutoffDate} onChange={(e) => setValidationConfig({...validationConfig, timeCutoffAsSegment:e.target.checked})} /><span><strong>Treat cut-off as an analysis segment</strong><small>Create reusable pre/post groups for subgroup analyses.</small></span></label>
                  {validationConfig.timeCutoffDate && validationConfig.timeCutoffAsSegment && (
                    <div className="cutoff-label-grid">
                      <label className="form-field"><span>Before label</span><input value={validationConfig.timeCutoffBeforeLabel} onChange={(e) => setValidationConfig({...validationConfig, timeCutoffBeforeLabel:e.target.value})} /></label>
                      <label className="form-field"><span>After label</span><input value={validationConfig.timeCutoffAfterLabel} onChange={(e) => setValidationConfig({...validationConfig, timeCutoffAfterLabel:e.target.value})} /></label>
                    </div>
                  )}
                </div>
              )}
            </div>
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
          <button className="primary-button run-button" disabled={!requiredConfigReady || analysisStatus.Validation === 'running'} onClick={runValidation}>{analysisStatus.Validation === 'running' ? 'Running…' : analysisStatus.Validation === 'ready' ? 'Run validation again' : 'Run validation'}</button>
        </div>
      </section>
    )
  }


  const renderEda = () => {
    if (!preview) return <EmptyAnalysisState section="EDA" onOpenData={() => setActiveSection('Data')} />

    if (!validationReady) {
      return (
        <section className="page-content analysis-module-page">
          <div className="analysis-page-header">
            <div><p className="eyebrow">EDA</p><h1>Exploratory Data Analysis</h1><p>Profile data structure, distributions and quality, with optional target-aware univariate diagnostics.</p></div>
            <div className="analysis-status-pill blocked"><span />Blocked</div>
          </div>
          <section className="panel analysis-run-card">
            <div className="analysis-run-icon" aria-hidden="true"><span>⌁</span></div>
            <h2>Complete validation first</h2>
            <p>Confirm the dataset mapping and semantic column types before running EDA.</p>
            <button className="secondary-button" onClick={() => setActiveSection('Validation')}>Open validation setup</button>
          </section>
        </section>
      )
    }

    const status = analysisStatus.EDA
    const tabs: EdaTab[] = ['Overview', 'Data quality', 'Distributions', 'Relationships', 'Population comparison', 'Time analysis', 'Missingness']
    const populationEnabled = validationConfig.populationColumns.length > 0
    const timeEnabled = Boolean(validationConfig.timeColumn)

    if (!edaResult) {
      return (
        <section className="page-content eda-page">
          <div className="analysis-page-header">
            <div><p className="eyebrow">EDA</p><h1>Exploratory Data Analysis</h1><p>Explore data quality and distributions. Univariate AUC is shown as descriptive target-aware context, not as model validation evidence.</p></div>
            <div className="eda-header-actions">
              <div className={`analysis-status-pill ${status}`}><span />{statusLabel(status)}</div>
              <button className="primary-button" disabled={status === 'running'} onClick={runEda}>{status === 'running' ? 'Running…' : 'Run EDA'}</button>
            </div>
          </div>
          {edaError && <div className="message error-message">{edaError}</div>}
          <section className="panel analysis-run-card">
            <div className="analysis-run-icon" aria-hidden="true"><span>⌁</span></div>
            <h2>{status === 'running' ? 'EDA is running…' : 'Ready to profile the dataset'}</h2>
            <p>RiskLab will calculate data quality, distributions, relationships and missingness diagnostics. Optional population and time views use the fields configured in Validation.</p>
            <button className="primary-button" disabled={status === 'running'} onClick={runEda}>{status === 'running' ? 'Running…' : 'Run EDA'}</button>
          </section>
        </section>
      )
    }

    const sortedProfiles = [...edaResult.profiles].sort((a, b) => {
      const direction = qualitySortDirection === 'asc' ? 1 : -1
      const qualityRank = (profile: EdaProfile) => profile.missing_rate > .2 ? 2 : profile.missing_rate > .05 ? 1 : 0
      let comparison = 0
      if (qualitySort === 'column') comparison = a.column.localeCompare(b.column)
      if (qualitySort === 'semantic') comparison = a.semantic_type.localeCompare(b.semantic_type)
      if (qualitySort === 'missing') comparison = a.missing_rate - b.missing_rate
      if (qualitySort === 'unique') comparison = (a.unique_count ?? Number.MAX_SAFE_INTEGER) - (b.unique_count ?? Number.MAX_SAFE_INTEGER)
      if (qualitySort === 'quality') comparison = qualityRank(a) - qualityRank(b)
      return comparison * direction
    })

    const changeQualitySort = (column: typeof qualitySort) => {
      if (qualitySort === column) setQualitySortDirection((current) => current === 'asc' ? 'desc' : 'asc')
      else {
        setQualitySort(column)
        setQualitySortDirection(column === 'column' || column === 'semantic' ? 'asc' : 'desc')
      }
    }

    const histogramOption = (profile: EdaProfile, logarithmic: boolean): echarts.EChartsOption => {
      const base = profile.numeric_summary
        ? profile.numeric_summary.histogram.map((item) => ({ label: item.label, count: item.count, missing: false }))
        : (profile.categories ?? []).map((item) => ({ label: item.value, count: item.count, missing: false }))
      const rows = [...base, { label: 'Missing', count: profile.sample_missing_count, missing: true }]
      return {
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        grid: { left: 54, right: 18, top: 28, bottom: profile.categories ? 86 : 62 },
        xAxis: {
          type: 'category',
          data: rows.map((item) => item.label),
          axisLabel: { color: '#71809a', rotate: profile.categories ? 35 : 20, interval: 0, hideOverlap: true },
          axisLine: { lineStyle: { color: '#263650' } },
        },
        yAxis: {
          type: logarithmic ? 'log' : 'value',
          min: logarithmic ? 1 : 0,
          name: logarithmic ? 'Count (log)' : 'Count',
          nameTextStyle: { color: '#71809a' },
          axisLabel: { color: '#71809a' },
          splitLine: { lineStyle: { color: '#1b293d' } },
        },
        series: [{
          type: 'bar',
          data: rows.map((item) => ({
            value: logarithmic && item.count === 0 ? null : item.count,
            itemStyle: { color: item.missing ? '#7b879a' : '#4f8cff' },
          })),
          barMaxWidth: 38,
        }],
      }
    }

    const cdfOption = (profile: EdaProfile): echarts.EChartsOption => ({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const point = Array.isArray(params) ? params[0] as { data?: [number, number] } : undefined
          return point?.data ? 'Value: ' + formatMetric(point.data[0]) + '<br/>CDF: ' + (point.data[1] * 100).toFixed(1) + '%' : ''
        },
      },
      grid: { left: 58, right: 20, top: 28, bottom: 48 },
      xAxis: { type: 'value', name: profile.column, nameTextStyle: { color: '#71809a' }, axisLabel: { color: '#71809a' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      yAxis: { type: 'value', min: 0, max: 1, name: 'CDF', nameTextStyle: { color: '#71809a' }, axisLabel: { color: '#71809a', formatter: (value: string | number) => Math.round(Number(value) * 100) + '%' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      series: [{
        type: 'line',
        showSymbol: false,
        smooth: false,
        data: profile.numeric_summary?.cdf.map((point) => [point.x, point.cdf]) ?? [],
        lineStyle: { width: 2, color: '#6ea8ff' },
      }],
    })

    const distributionProfiles = edaResult.profiles
      .filter((profile) => profile.semantic_type !== 'ignore')
      .filter((profile) => {
        if (distributionFilter === 'numeric') return ['continuous', 'ordinal'].includes(profile.semantic_type)
        if (distributionFilter === 'categorical') return ['categorical', 'boolean'].includes(profile.semantic_type)
        if (distributionFilter === 'other') return ['identifier', 'datetime', 'text'].includes(profile.semantic_type)
        return true
      })
      .filter((profile) => profile.column.toLowerCase().includes(distributionSearch.trim().toLowerCase()))

    const distributionOrder = new Map(edaResult.profiles.map((profile, index) => [profile.column, index]))
    distributionProfiles.sort((a, b) => {
      if (distributionSort === 'missing') return b.missing_rate - a.missing_rate
      if (distributionSort === 'auc') return (b.univariate_auc ?? -1) - (a.univariate_auc ?? -1)
      if (distributionSort === 'name') return a.column.localeCompare(b.column)
      return (distributionOrder.get(a.column) ?? 0) - (distributionOrder.get(b.column) ?? 0)
    })

    const distributionDetailProfile = edaResult.profiles.find((profile) => profile.column === distributionDetailColumn)

    const miniBars = (profile: EdaProfile) => {
      const values = profile.numeric_summary
        ? profile.numeric_summary.histogram.map((item) => item.count)
        : (profile.categories ?? []).slice(0, 8).map((item) => item.count)
      const max = Math.max(1, ...values)
      return (
        <div className="distribution-mini-bars" aria-hidden="true">
          {values.map((value, index) => <span key={index} style={{ height: `${Math.max(5, value / max * 100)}%` }} />)}
        </div>
      )
    }

    const relationshipLookup = new Map(
      edaResult.relationships.flatMap((item) => [
        [item.left + '|||' + item.right, item] as const,
        [item.right + '|||' + item.left, item] as const,
      ])
    )
    const numericRelationshipColumns = edaResult.relationship_columns.filter((column) => ['continuous', 'ordinal'].includes(validationConfig.semanticTypes[column] ?? 'categorical'))
    const relationshipBaseColumns = relationshipMode === 'numeric' ? numericRelationshipColumns : edaResult.relationship_columns
    const searchedRelationshipColumns = relationshipBaseColumns.filter((column) => column.toLowerCase().includes(relationshipSearch.trim().toLowerCase()))
    const rankedRelationshipColumns = [...searchedRelationshipColumns].sort((a, b) => {
      const maxFor = (column: string) => Math.max(0, ...edaResult.relationships.filter((item) => item.left === column || item.right === column).map((item) => Math.abs(item.score)))
      return maxFor(b) - maxFor(a)
    })
    const focusedRelationshipColumns = relationshipFocus && relationshipBaseColumns.includes(relationshipFocus)
      ? [relationshipFocus, ...rankedRelationshipColumns.filter((column) => column !== relationshipFocus).slice(0, 29)]
      : rankedRelationshipColumns.slice(0, 30)

    const heatmapOption = (mode: 'numeric' | 'mixed'): echarts.EChartsOption => {
      const columns = focusedRelationshipColumns
      const data: [number, number, number][] = []
      columns.forEach((left, y) => {
        columns.forEach((right, x) => {
          if (left === right) {
            data.push([x, y, 1])
            return
          }
          const item = relationshipLookup.get(left + '|||' + right)
          if (!item) return
          if (mode === 'numeric' && item.kind !== 'Pearson correlation') return
          const score = mode === 'numeric' ? item.score : (item.kind === 'Pearson correlation' ? Math.abs(item.score) : item.score)
          data.push([x, y, score])
        })
      })

      return {
        backgroundColor: 'transparent',
        tooltip: {
          formatter: (params: unknown) => {
            const point = params as { data?: [number, number, number] }
            if (!point.data) return ''
            return columns[point.data[1]] + ' ↔ ' + columns[point.data[0]] + '<br/><strong>' + point.data[2].toFixed(3) + '</strong>'
          },
        },
        grid: { left: 140, right: 38, top: 34, bottom: 120 },
        xAxis: { type: 'category', data: columns, axisLabel: { color: '#71809a', rotate: 45, interval: 0 }, splitArea: { show: true } },
        yAxis: { type: 'category', data: columns, axisLabel: { color: '#71809a', interval: 0 }, splitArea: { show: true } },
        visualMap: {
          min: mode === 'numeric' ? -1 : 0,
          max: 1,
          calculable: true,
          orient: 'horizontal',
          left: 'center',
          bottom: 8,
          textStyle: { color: '#8b9aaf' },
          inRange: mode === 'numeric'
            ? { color: ['#ef6b73', '#17243a', '#4f8cff'] }
            : { color: ['#101827', '#275eaf', '#78adff'] },
        },
        series: [{ type: 'heatmap', data, emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,.45)' } } }],
      }
    }

    const activePopulation = edaResult.population_comparison.find((item) => item.column === populationColumn) ?? edaResult.population_comparison[0]
    const activeReference = activePopulation?.references.find((item) => item.reference === populationReference) ?? activePopulation?.references[0]
    const activeComparison = activeReference?.comparisons.find((item) => item.comparison === populationComparison) ?? activeReference?.comparisons[0]
    const activePopulationVariable = activeComparison?.variables.find((item) => item.column === populationVariable) ?? activeComparison?.variables[0]
    const psiRanking = activeReference ? Array.from(new Set(activeReference.comparisons.flatMap((comparison) => comparison.variables.map((variable) => variable.column)))).map((column) => {
      const values = Object.fromEntries(activeReference.comparisons.map((comparison) => [
        comparison.comparison,
        comparison.variables.find((variable) => variable.column === column)?.psi ?? 0,
      ]))
      const first = activeReference.comparisons.flatMap((comparison) => comparison.variables).find((variable) => variable.column === column)
      return {
        column,
        semantic_type: first?.semantic_type ?? 'categorical' as SemanticType,
        values,
        maxPsi: Math.max(0, ...Object.values(values)),
      }
    }).sort((a, b) => b.maxPsi - a.maxPsi) : []

    const populationDistributionOption = (): echarts.EChartsOption => {
      const variable = activePopulationVariable
      const rows = variable?.distribution ?? []
      return {
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        legend: { top: 0, textStyle: { color: '#8b9aaf' } },
        grid: { left: 58, right: 20, top: 42, bottom: 90 },
        xAxis: { type: 'category', data: rows.map((row) => row.label), axisLabel: { color: '#71809a', rotate: 30, interval: 0, hideOverlap: true } },
        yAxis: { type: 'value', name: 'Share', max: 1, axisLabel: { color: '#71809a', formatter: (value: string | number) => Math.round(Number(value) * 100) + '%' }, splitLine: { lineStyle: { color: '#1b293d' } } },
        series: [
          { name: activeReference?.reference ?? 'Reference', type: 'bar', data: rows.map((row) => row.reference_share), itemStyle: { color: '#4f8cff' }, barMaxWidth: 34 },
          { name: activeComparison?.comparison ?? 'Comparison', type: 'bar', data: rows.map((row) => row.comparison_share), itemStyle: { color: '#8b99ad' }, barMaxWidth: 34 },
        ],
      }
    }

    const currentTimeBuckets = edaResult.time_analysis?.granularities[timeGranularity] ?? []
    const edaCutoffBucket = timeBucketForDate(validationConfig.timeCutoffDate, timeGranularity)
    const edaCutoffMarkLine = validationConfig.timeCutoffDate && edaCutoffBucket ? {
      silent: true,
      symbol: 'none',
      label: { formatter: `Cut-off · ${validationConfig.timeCutoffDate}`, color: '#9aa9bd', position: 'insideEndTop' as const },
      lineStyle: { type: 'dashed' as const, width: 1, color: '#9aa9bd' },
      data: [{ xAxis: edaCutoffBucket }],
    } : undefined
    const timeMissingOption = (column: string): echarts.EChartsOption => ({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      toolbox: { right: 8, feature: { dataZoom: { yAxisIndex: 'none' }, restore: {} } },
      grid: { left: 58, right: 18, top: 28, bottom: 66 },
      xAxis: { type: 'category', data: currentTimeBuckets.map((bucket) => bucket.bucket), boundaryGap: false, axisLabel: { color: '#71809a' } },
      yAxis: { type: 'value', min: 0, max: 1, name: 'Missing', axisLabel: { color: '#71809a', formatter: (value: string | number) => Math.round(Number(value) * 100) + '%' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      dataZoom: [{ type: 'inside', xAxisIndex: 0 }, { type: 'slider', xAxisIndex: 0, bottom: 12, height: 18 }],
      series: [{
        name: 'Missing %',
        type: 'line',
        showSymbol: false,
        data: currentTimeBuckets.map((bucket) => bucket.variables[column]?.missing_rate ?? null),
        lineStyle: { width: 2, color: '#8b99ad' },
        areaStyle: { opacity: .08, color: '#8b99ad' },
        markLine: edaCutoffMarkLine,
      }],
    })

    const timeStatsOption = (column: string): echarts.EChartsOption => ({
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      toolbox: { right: 8, feature: { dataZoom: { yAxisIndex: 'none' }, restore: {} } },
      legend: { top: 0, textStyle: { color: '#8b9aaf' } },
      grid: { left: 62, right: 18, top: 42, bottom: 66 },
      xAxis: { type: 'category', data: currentTimeBuckets.map((bucket) => bucket.bucket), boundaryGap: false, axisLabel: { color: '#71809a' } },
      yAxis: { type: 'value', name: column, axisLabel: { color: '#71809a' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      dataZoom: [{ type: 'inside', xAxisIndex: 0 }, { type: 'slider', xAxisIndex: 0, bottom: 12, height: 18 }],
      series: [
        { name: 'Min', type: 'line', showSymbol: false, data: currentTimeBuckets.map((bucket) => bucket.variables[column]?.min ?? null), lineStyle: { width: 1, type: 'dashed', color: '#65758d' } },
        { name: 'Max', type: 'line', showSymbol: false, data: currentTimeBuckets.map((bucket) => bucket.variables[column]?.max ?? null), lineStyle: { width: 1, type: 'dashed', color: '#91a0b7' } },
        { name: 'Mean', type: 'line', showSymbol: false, data: currentTimeBuckets.map((bucket) => bucket.variables[column]?.mean ?? null), lineStyle: { width: 2, color: '#4f8cff' }, markLine: edaCutoffMarkLine },
        { name: 'Median', type: 'line', showSymbol: false, data: currentTimeBuckets.map((bucket) => bucket.variables[column]?.median ?? null), lineStyle: { width: 2, color: '#69e7ad' } },
      ],
    })

    const addTimeVariable = (column: string) => {
      setTimeWorkspace((current) => current.includes(column) ? current : [...current, column])
    }

    const renderLocked = (kind: 'population' | 'time') => (
      <section className="panel eda-locked-card">
        <div className="analysis-run-icon" aria-hidden="true"><span>🔒</span></div>
        <h2>{kind === 'population' ? 'Population comparison needs configuration' : 'Time analysis needs configuration'}</h2>
        <p>{kind === 'population'
          ? 'Select at least one Sample / population column in Validation to compare groups.'
          : 'Select a Time column in Validation to analyse volume, missingness and variables over time.'}</p>
        <button className="secondary-button" onClick={() => setActiveSection('Validation')}>Open validation setup</button>
      </section>
    )

    return (
      <section className="page-content eda-page">
        <div className="analysis-page-header">
          <div><p className="eyebrow">EDA</p><h1>Exploratory Data Analysis</h1><p>Generic profiling only — no credit-risk-specific interpretation is applied here.</p></div>
          <div className="eda-header-actions">
            <div className={`analysis-status-pill ${status}`}><span />{statusLabel(status)}</div>
            <button className="primary-button" disabled={status === 'running'} onClick={runEda}>{status === 'running' ? 'Running…' : edaResult ? 'Run EDA again' : 'Run EDA'}</button>
          </div>
        </div>

        {edaError && <div className="message error-message">{edaError}</div>}

        <>
            <div className="eda-tabs">
              {tabs.map((tab) => {
                const locked = (tab === 'Population comparison' && !populationEnabled) || (tab === 'Time analysis' && !timeEnabled)
                return <button key={tab} className={edaTab === tab ? 'active' : ''} onClick={() => setEdaTab(tab)}>{tab}{locked ? '  🔒' : ''}</button>
              })}
            </div>
            <div className="eda-sample-note">Rows and exact missingness are calculated on the full dataset. Distribution, relationship, population and time diagnostics use a reservoir sample of up to {edaResult.sample_limit.toLocaleString()} rows ({edaResult.sample_size.toLocaleString()} used).</div>

            {edaTab === 'Overview' && (
              <>
                <div className="eda-kpi-grid">
                  <article className="metric-card"><span>Rows</span><strong>{edaResult.overview.rows.toLocaleString()}</strong><small className="neutral">Full dataset</small></article>
                  <article className="metric-card"><span>Columns</span><strong>{edaResult.overview.columns}</strong><small className="neutral">Configured schema</small></article>
                  <article className="metric-card"><span>Missing cells</span><strong>{formatPercent(edaResult.overview.missing_rate)}</strong><small className={edaResult.overview.missing_rate > .05 ? 'negative' : 'neutral'}>{edaResult.overview.missing_cells.toLocaleString()} cells</small></article>
                  <article className="metric-card"><span>Duplicate rows</span><strong>{edaResult.overview.duplicate_rows.toLocaleString()}</strong><small className={edaResult.overview.duplicate_rows ? 'negative' : 'positive'}>{formatPercent(edaResult.overview.duplicate_rate)}</small></article>
                </div>
                <section className="panel eda-section">
                  <div className="panel-title"><h2>Columns by semantic type</h2></div>
                  <MiniBarChart data={Object.entries(edaResult.profiles.reduce<Record<string, number>>((acc, profile) => {
                    acc[profile.semantic_type] = (acc[profile.semantic_type] || 0) + 1
                    return acc
                  }, {})).map(([label, value]) => ({ label, value }))} />
                </section>
              </>
            )}

            {edaTab === 'Data quality' && (
              <section className="panel eda-section">
                <div className="panel-title"><div><h2>Column quality</h2><p className="eda-muted">Click a column header to sort. Click it again to reverse the order.</p></div><span>{edaResult.profiles.length} columns</span></div>
                <div className="table-wrap">
                  <table className="sortable-table">
                    <thead><tr>
                      <th><button onClick={() => changeQualitySort('column')}>Column <span>{qualitySort === 'column' ? (qualitySortDirection === 'asc' ? '↑' : '↓') : '↕'}</span></button></th>
                      <th><button onClick={() => changeQualitySort('semantic')}>Semantic type <span>{qualitySort === 'semantic' ? (qualitySortDirection === 'asc' ? '↑' : '↓') : '↕'}</span></button></th>
                      <th><button onClick={() => changeQualitySort('missing')}>Missing <span>{qualitySort === 'missing' ? (qualitySortDirection === 'asc' ? '↑' : '↓') : '↕'}</span></button></th>
                      <th><button onClick={() => changeQualitySort('unique')}>Unique <span>{qualitySort === 'unique' ? (qualitySortDirection === 'asc' ? '↑' : '↓') : '↕'}</span></button></th>
                      <th><button onClick={() => changeQualitySort('quality')}>Quality signal <span>{qualitySort === 'quality' ? (qualitySortDirection === 'asc' ? '↑' : '↓') : '↕'}</span></button></th>
                    </tr></thead>
                    <tbody>{sortedProfiles.map((profile) => (
                      <tr key={profile.column}>
                        <td>{profile.column}</td><td>{profile.semantic_type}</td><td>{formatPercent(profile.missing_rate)}</td>
                        <td>{profile.unique_count_capped ? '>10,000' : (profile.unique_count ?? '—')}</td>
                        <td><span className={profile.missing_rate > .2 ? 'eda-signal bad' : profile.missing_rate > .05 ? 'eda-signal warn' : 'eda-signal good'}>{profile.missing_rate > .2 ? 'Review' : profile.missing_rate > .05 ? 'Watch' : 'OK'}</span></td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
              </section>
            )}

            {edaTab === 'Distributions' && (
              <section className="panel eda-section distribution-gallery-section">
                <div className="eda-control-row distribution-gallery-header">
                  <div>
                    <h2>Distribution gallery</h2>
                    <p>Scan all variables at once. Univariate AUC uses the configured default target and the EDA sample; click any card for a detailed view.</p>
                  </div>
                  <div className="distribution-gallery-controls">
                    <div className="segmented-control">
                      <button className={distributionFilter === 'all' ? 'active' : ''} onClick={() => setDistributionFilter('all')}>All</button>
                      <button className={distributionFilter === 'numeric' ? 'active' : ''} onClick={() => setDistributionFilter('numeric')}>Numeric</button>
                      <button className={distributionFilter === 'categorical' ? 'active' : ''} onClick={() => setDistributionFilter('categorical')}>Categorical</button>
                      <button className={distributionFilter === 'other' ? 'active' : ''} onClick={() => setDistributionFilter('other')}>Other</button>
                    </div>
                    <input className="relationship-search" value={distributionSearch} onChange={(e) => setDistributionSearch(e.target.value)} placeholder="Search variables…" />
                    <label className="form-field compact-field"><span>Sort by</span><select value={distributionSort} onChange={(e) => setDistributionSort(e.target.value as 'dataset' | 'missing' | 'auc' | 'name')}><option value="dataset">Dataset order</option><option value="missing">Missing rate</option><option value="auc">Univariate AUC</option><option value="name">Name</option></select></label>
                  </div>
                </div>

                <div className="distribution-card-grid">
                  {distributionProfiles.map((profile) => (
                    <button
                      type="button"
                      className="distribution-card"
                      key={profile.column}
                      onClick={() => setDistributionDetailColumn(profile.column)}
                    >
                      <div className="distribution-card-head">
                        <div><strong>{profile.column}</strong><span>{profile.semantic_type}</span></div>
                        {profile.univariate_auc !== null && <span className="distribution-auc">AUC {profile.univariate_auc.toFixed(3)}</span>}
                      </div>

                      {(profile.numeric_summary || profile.categories) ? miniBars(profile) : (
                        <div className="distribution-card-placeholder">{profile.semantic_type === 'identifier' ? 'Identifier' : profile.semantic_type === 'text' ? 'Text' : 'No histogram'}</div>
                      )}

                      {profile.numeric_summary && (
                        <div className="distribution-card-stats">
                          <span><b>Mean</b>{formatMetric(profile.numeric_summary.mean)}</span>
                          <span><b>Median</b>{formatMetric(profile.numeric_summary.median)}</span>
                          <span><b>Range</b>{formatMetric(profile.numeric_summary.min)}–{formatMetric(profile.numeric_summary.max)}</span>
                          <span><b>Missing</b>{formatPercent(profile.missing_rate)}</span>
                        </div>
                      )}

                      {profile.categories && (
                        <div className="distribution-card-stats">
                          <span><b>Unique</b>{profile.unique_count_capped ? '>10k' : (profile.unique_count ?? '—')}</span>
                          <span><b>Top share</b>{profile.categories[0] ? formatPercent(profile.categories[0].share) : '—'}</span>
                          <span><b>Missing</b>{formatPercent(profile.missing_rate)}</span>
                          <span><b>Logit AUC</b>{profile.univariate_auc === null ? '—' : profile.univariate_auc.toFixed(3)}</span>
                        </div>
                      )}

                      {profile.semantic_type === 'identifier' && <div className="distribution-card-stats"><span><b>Unique</b>{profile.unique_count_capped ? '>10k' : (profile.unique_count ?? '—')}</span><span><b>Missing</b>{formatPercent(profile.missing_rate)}</span></div>}
                      {profile.text_summary && <div className="distribution-card-stats"><span><b>Mean length</b>{formatMetric(profile.text_summary.mean_length)}</span><span><b>Max length</b>{profile.text_summary.max_length}</span><span><b>Missing</b>{formatPercent(profile.missing_rate)}</span></div>}
                    </button>
                  ))}
                </div>

                {distributionProfiles.length === 0 && <div className="eda-no-data">No variables match the current distribution filters.</div>}

                {distributionDetailProfile && (
                  <div className="distribution-modal-backdrop" role="presentation" onClick={() => setDistributionDetailColumn('')}>
                    <div className="distribution-modal" role="dialog" aria-modal="true" aria-label={`Distribution details for ${distributionDetailProfile.column}`} onClick={(event) => event.stopPropagation()}>
                      <div className="distribution-modal-head">
                        <div><p className="eyebrow">Distribution detail</p><h2>{distributionDetailProfile.column}</h2><p>{distributionDetailProfile.semantic_type} · sample n={edaResult.sample_size.toLocaleString()}</p></div>
                        <button className="secondary-button compact-button" onClick={() => setDistributionDetailColumn('')}>Close</button>
                      </div>

                      <div className="eda-inline-stats">
                        <div><span>Type</span><strong>{distributionDetailProfile.semantic_type}</strong></div>
                        <div><span>Missing</span><strong>{formatPercent(distributionDetailProfile.missing_rate)}</strong></div>
                        <div><span>Unique</span><strong>{distributionDetailProfile.unique_count_capped ? '>10k' : (distributionDetailProfile.unique_count ?? '—')}</strong></div>
                        <div><span className="metric-label-with-help">Univariate AUC {metricHelp('univariate-auc', 'Univariate AUC')}</span><strong>{distributionDetailProfile.univariate_auc === null ? '—' : distributionDetailProfile.univariate_auc.toFixed(3)}</strong></div>
                      </div>

                      {distributionDetailProfile.numeric_summary && (
                        <>
                          <div className="eda-inline-stats wide">
                            <div><span>Mean</span><strong>{formatMetric(distributionDetailProfile.numeric_summary.mean)}</strong></div>
                            <div><span>Median</span><strong>{formatMetric(distributionDetailProfile.numeric_summary.median)}</strong></div>
                            <div><span>Std</span><strong>{formatMetric(distributionDetailProfile.numeric_summary.std)}</strong></div>
                            <div><span>P25</span><strong>{formatMetric(distributionDetailProfile.numeric_summary.p25)}</strong></div>
                            <div><span>P75</span><strong>{formatMetric(distributionDetailProfile.numeric_summary.p75)}</strong></div>
                            <div><span>Min / max</span><strong>{formatMetric(distributionDetailProfile.numeric_summary.min)} / {formatMetric(distributionDetailProfile.numeric_summary.max)}</strong></div>
                          </div>
                          <div className="eda-info-card">AUC direction: <strong>{distributionDetailProfile.auc_direction === 'lower-is-riskier' ? 'lower values are riskier' : 'higher values are riskier'}</strong>. The displayed AUC is direction-normalized to be at least 0.5 and is calculated on the EDA sample.</div>
                          <div className="eda-chart-grid distribution-grid">
                            <article className="eda-chart-panel"><h3>Histogram · linear count scale</h3><EChart option={histogramOption(distributionDetailProfile, false)} /></article>
                            <article className="eda-chart-panel"><h3>Histogram · logarithmic count scale</h3><EChart option={histogramOption(distributionDetailProfile, true)} /></article>
                          </div>
                          <article className="eda-chart-panel cdf-panel"><div><h3>Empirical cumulative distribution</h3><p className="eda-muted">CDF is calculated from the EDA reservoir sample.</p></div><EChart option={cdfOption(distributionDetailProfile)} height={390} /></article>
                        </>
                      )}

                      {distributionDetailProfile.categories && (
                        <>
                          <div className="eda-chart-grid distribution-grid">
                            <article className="eda-chart-panel"><h3>Category counts · linear scale</h3><EChart option={histogramOption(distributionDetailProfile, false)} /></article>
                            <article className="eda-chart-panel"><h3>Category counts · logarithmic scale</h3><EChart option={histogramOption(distributionDetailProfile, true)} /></article>
                          </div>
                          <div className="category-logit-section">
                            <div><h3>Category logit diagnostics</h3><p className="eda-muted">Each category is scored with its smoothed observed default-rate logit. The variable-level AUC measures how well those category logits rank the configured target. This is an in-sample descriptive diagnostic and can be optimistic for high-cardinality variables.</p></div>
                            <div className="table-wrap">
                              <table>
                                <thead><tr><th>Category</th><th>Count</th><th>Defaults</th><th>Default rate</th><th>Smoothed logit</th></tr></thead>
                                <tbody>{distributionDetailProfile.category_logits.map((row) => <tr key={row.value}><td><strong>{row.value}</strong></td><td>{row.count.toLocaleString()}</td><td>{row.defaults.toLocaleString()}</td><td>{formatPercent(row.default_rate)}</td><td>{row.logit.toFixed(3)}</td></tr>)}</tbody>
                              </table>
                            </div>
                          </div>
                        </>
                      )}

                      {distributionDetailProfile.semantic_type === 'identifier' && <div className="eda-info-card">Uniqueness rate: <strong>{distributionDetailProfile.uniqueness_rate === null || distributionDetailProfile.uniqueness_rate === undefined ? 'Unavailable for very high cardinality' : formatPercent(distributionDetailProfile.uniqueness_rate)}</strong></div>}
                      {distributionDetailProfile.text_summary && <div className="eda-info-card">Text length — mean <strong>{formatMetric(distributionDetailProfile.text_summary.mean_length)}</strong>, median <strong>{formatMetric(distributionDetailProfile.text_summary.median_length)}</strong>, max <strong>{distributionDetailProfile.text_summary.max_length}</strong>.</div>}
                    </div>
                  </div>
                )}
              </section>
            )}

            {edaTab === 'Relationships' && (
              <section className="panel eda-section relationships-section">
                <div className="eda-control-row">
                  <div><h2>Relationship heatmap</h2><p>Numeric view shows signed Pearson correlation {metricHelp('pearson', 'Pearson correlation')}. Mixed view uses |Pearson|, Cramér's V {metricHelp('cramers-v', "Cramér's V")} and correlation ratio η {metricHelp('eta', 'Correlation ratio eta')}.</p></div>
                  <div className="relationship-controls">
                    <div className="segmented-control">
                      <button className={relationshipMode === 'numeric' ? 'active' : ''} onClick={() => { setRelationshipMode('numeric'); setRelationshipFocus('') }}>Numeric correlations</button>
                      <button className={relationshipMode === 'mixed' ? 'active' : ''} onClick={() => { setRelationshipMode('mixed'); setRelationshipFocus('') }}>All-variable associations</button>
                    </div>
                    <input className="relationship-search" value={relationshipSearch} onChange={(e) => setRelationshipSearch(e.target.value)} placeholder="Search variables…" />
                    <label className="form-field compact-field"><span>Focus variable</span><select value={relationshipFocus} onChange={(e) => setRelationshipFocus(e.target.value)}><option value="">Top associated variables</option>{relationshipBaseColumns.map((column) => <option key={column} value={column}>{column}</option>)}</select></label>
                  </div>
                </div>
                {edaResult.relationship_columns_total > edaResult.relationship_columns_limit && <div className="message warning-message">This dataset has {edaResult.relationship_columns_total} eligible variables. The backend currently computes pairwise relationships for the first {edaResult.relationship_columns_limit}; the heatmap then shows up to 30 at once for readability.</div>}
                {focusedRelationshipColumns.length ? (
                  <>
                    <div className="heatmap-meta">Showing {focusedRelationshipColumns.length} of {relationshipBaseColumns.length} variables. Search or choose a focus variable to navigate large matrices.</div>
                    <div className="heatmap-scroll"><EChart option={heatmapOption(relationshipMode)} height={Math.max(520, focusedRelationshipColumns.length * 27 + 190)} /></div>
                    <div className="table-wrap relationship-table">
                      <table><thead><tr><th>Variable A</th><th>Variable B</th><th>Metric</th><th>Score</th></tr></thead><tbody>{edaResult.relationships.filter((item) => relationshipMode === 'mixed' || item.kind === 'Pearson correlation').slice(0, 20).map((item) => <tr key={`${item.left}-${item.right}`}><td>{item.left}</td><td>{item.right}</td><td>{item.kind}</td><td>{formatMetric(item.score)}</td></tr>)}</tbody></table>
                    </div>
                  </>
                ) : <div className="eda-no-data">No variables match the current relationship view.</div>}
              </section>
            )}

            {edaTab === 'Population comparison' && (!populationEnabled ? renderLocked('population') : (
              <section className="panel eda-section population-analysis">
                <div className="eda-control-row">
                  <div><h2 className="heading-with-help">Population comparison {metricHelp('psi', 'Population Stability Index')}</h2><p>PSI is calculated for every eligible variable using the selected reference population. Numeric bins are defined on the reference population; Missing is a separate bin.</p></div>
                  <div className="population-controls">
                    <label className="form-field compact-field"><span>Population column</span><select value={activePopulation?.column ?? ''} onChange={(e) => { const next = edaResult.population_comparison.find((item) => item.column === e.target.value); setPopulationColumn(e.target.value); setPopulationReference(next?.references[0]?.reference ?? ''); setPopulationComparison(next?.references[0]?.comparisons[0]?.comparison ?? ''); setPopulationVariable(next?.references[0]?.comparisons[0]?.variables[0]?.column ?? '') }}>{edaResult.population_comparison.map((population) => <option key={population.column} value={population.column}>{population.column}</option>)}</select></label>
                    <label className="form-field compact-field"><span>Reference</span><select value={activeReference?.reference ?? ''} onChange={(e) => { const next = activePopulation?.references.find((item) => item.reference === e.target.value); setPopulationReference(e.target.value); setPopulationComparison(next?.comparisons[0]?.comparison ?? ''); setPopulationVariable(next?.comparisons[0]?.variables[0]?.column ?? '') }}>{activePopulation?.references.map((reference) => <option key={reference.reference} value={reference.reference}>{reference.reference}</option>)}</select></label>
                    <label className="form-field compact-field"><span>Compare with</span><select value={activeComparison?.comparison ?? ''} onChange={(e) => { const next = activeReference?.comparisons.find((item) => item.comparison === e.target.value); setPopulationComparison(e.target.value); setPopulationVariable(next?.variables[0]?.column ?? '') }}>{activeReference?.comparisons.map((comparison) => <option key={comparison.comparison} value={comparison.comparison}>{comparison.comparison}</option>)}</select></label>
                  </div>
                </div>

                {activePopulation && <div className="population-group-strip">{activePopulation.groups.map((group) => <div key={group.value}><span>{group.value}</span><strong>{group.sample_rows.toLocaleString()}</strong><small>{formatPercent(group.share)} · missing {formatPercent(group.missing_rate)}</small></div>)}</div>}

                {activeComparison && (
                  <div className="population-layout">
                    <div className="population-ranking">
                      <div className="panel-title"><div><h3>PSI ranking</h3><p className="eda-muted">Variables are ranked by their highest PSI against any comparison group. Click a variable to inspect the currently selected comparison.</p></div><span>Reference: {activeReference?.reference}</span></div>
                      <div className="table-wrap">
                        <table className="clickable-table psi-table">
                          <thead><tr><th>Variable</th><th>Type</th>{activeReference?.comparisons.map((comparison) => <th key={comparison.comparison}>PSI · {comparison.comparison}</th>)}<th>Max PSI</th><th>Signal</th></tr></thead>
                          <tbody>{psiRanking.map((variable) => (
                            <tr key={variable.column} className={activePopulationVariable?.column === variable.column ? 'selected-row' : ''} onClick={() => setPopulationVariable(variable.column)}>
                              <td>{variable.column}</td><td>{variable.semantic_type}</td>
                              {activeReference?.comparisons.map((comparison) => <td key={comparison.comparison}>{(variable.values[comparison.comparison] ?? 0).toFixed(4)}</td>)}
                              <td><strong>{variable.maxPsi.toFixed(4)}</strong></td>
                              <td><span className={variable.maxPsi >= .25 ? 'eda-signal bad' : variable.maxPsi >= .1 ? 'eda-signal warn' : 'eda-signal good'}>{variable.maxPsi >= .25 ? 'High' : variable.maxPsi >= .1 ? 'Moderate' : 'Low'}</span></td>
                            </tr>
                          ))}</tbody>
                        </table>
                      </div>
                    </div>
                    <div className="population-drilldown">
                      {activePopulationVariable ? (
                        <>
                          <div className="population-drilldown-head"><div><h3>{activePopulationVariable.column}</h3><p>{activeReference?.reference} vs {activeComparison.comparison}</p></div><strong>PSI {activePopulationVariable.psi.toFixed(4)}</strong></div>
                          <EChart option={populationDistributionOption()} height={430} />
                        </>
                      ) : <div className="eda-no-data">Select a variable to inspect its distributions.</div>}
                    </div>
                  </div>
                )}
              </section>
            ))}

            {edaTab === 'Time analysis' && (!timeEnabled ? renderLocked('time') : (
              <section className="panel eda-section time-workspace-section">
                <div className="eda-control-row">
                  <div><h2>Time analysis workspace</h2><p>Change aggregation instantly, add variables to the workspace, then drag or use the add button. Zoom one chart and all time charts stay synchronized.</p></div>
                  <label className="form-field compact-field"><span>Aggregation</span><select value={timeGranularity} onChange={(e) => setTimeGranularity(e.target.value as TimeGranularity)}><option value="daily">Daily</option><option value="weekly">Weekly</option><option value="monthly">Monthly</option><option value="quarterly">Quarterly</option><option value="yearly">Yearly</option></select><small>Updates immediately — no EDA rerun required.</small></label>
                </div>

                {edaResult.time_analysis && currentTimeBuckets.length ? (
                  <div className="time-workspace-layout">
                    <aside className="time-variable-palette">
                      <div><h3>Available variables</h3><p>Drag into the workspace or click +.</p></div>
                      <div className="time-variable-list">{edaResult.time_analysis.numeric_columns.map((column) => (
                        <button
                          key={column}
                          draggable
                          className={timeWorkspace.includes(column) ? 'time-variable-chip selected' : 'time-variable-chip'}
                          onDragStart={(event) => event.dataTransfer.setData('text/plain', column)}
                          onClick={() => addTimeVariable(column)}
                        ><span>⋮⋮</span>{column}<strong>+</strong></button>
                      ))}</div>
                    </aside>

                    <div
                      className="time-drop-zone"
                      onDragOver={(event) => event.preventDefault()}
                      onDrop={(event) => { event.preventDefault(); const column = event.dataTransfer.getData('text/plain'); if (column) addTimeVariable(column) }}
                    >
                      {timeWorkspace.length === 0 ? (
                        <div className="time-empty-drop"><strong>Drop numeric variables here</strong><span>Each variable creates a missingness chart and a min / max / mean / median chart.</span></div>
                      ) : timeWorkspace.map((column) => (
                        <article className="time-variable-panel" key={column}>
                          <div className="time-variable-panel-head"><div><h3>{column}</h3><span>{timeGranularity} aggregation</span></div><button onClick={() => setTimeWorkspace((current) => current.filter((item) => item !== column))}>Remove</button></div>
                          <div className="time-chart-pair">
                            <div className="eda-chart-panel"><h4>Missing values</h4><EChart option={timeMissingOption(column)} height={330} group="risklab-time-workspace" /></div>
                            <div className="eda-chart-panel"><h4>Min / max / mean / median</h4><EChart option={timeStatsOption(column)} height={330} group="risklab-time-workspace" /></div>
                          </div>
                        </article>
                      ))}
                    </div>
                  </div>
                ) : <div className="eda-no-data">No parseable time values were found for {validationConfig.timeColumn}.</div>}
              </section>
            ))}

            {edaTab === 'Missingness' && (
              <section className="panel eda-section">
                <div className="panel-title"><div><h2>Missingness mechanism diagnostics</h2><p className="eda-muted">RiskLab provides evidence, not a definitive MCAR / MAR / MNAR classification. MNAR cannot be established from observed data alone.</p></div></div>
                {edaResult.missingness_diagnostics.length ? (
                  <div className="missingness-grid">{edaResult.missingness_diagnostics.map((item) => (
                    <article className="missingness-card" key={item.column}>
                      <div className="missingness-card-head"><div><h3>{item.column}</h3><span>{formatPercent(item.missing_rate)} missing · {item.missing_count.toLocaleString()} rows</span></div><span className={item.assessment === 'MAR plausible' ? 'missingness-assessment warn' : item.assessment.startsWith('MCAR') ? 'missingness-assessment good' : 'missingness-assessment neutral'}>{item.assessment}</span></div>
                      <p>{item.explanation}</p>
                      {item.strongest_associations.length > 0 && <div className="association-list"><strong>Strongest observed associations</strong>{item.strongest_associations.map((association) => <span key={association.column}>{association.column}: {association.evidence}</span>)}</div>}
                      <div className="mnar-note">{item.mnar_note}</div>
                    </article>
                  ))}</div>
                ) : <div className="eda-no-data">No missing values were detected.</div>}
              </section>
            )}

            <div className="eda-footer-meta">Last run: {lastRun.EDA ?? '—'}</div>
        </>
      </section>
    )
  }


  const renderDiscrimination = () => {
    if (!preview) return <EmptyAnalysisState section="Discrimination" onOpenData={() => setActiveSection('Data')} />

    const status = analysisStatus.Discrimination
    if (!validationReady) {
      return (
        <section className="page-content analysis-module-page">
          <div className="analysis-page-header">
            <div><p className="eyebrow">Discrimination</p><h1>Discrimination</h1><p>Evaluate how well the submitted model ranks defaults above non-defaults.</p></div>
            <div className="analysis-status-pill blocked"><span />Blocked</div>
          </div>
          <section className="panel analysis-run-card">
            <div className="analysis-run-icon" aria-hidden="true"><span>↗</span></div>
            <h2>Complete validation first</h2>
            <p>RiskLab needs the prediction, target, positive class and score direction before discrimination can be calculated.</p>
            <button className="secondary-button" onClick={() => setActiveSection('Validation')}>Open validation setup</button>
          </section>
        </section>
      )
    }

    if (!discriminationResult) {
      return (
        <section className="page-content analysis-module-page">
          <div className="analysis-page-header">
            <div><p className="eyebrow">Discrimination</p><h1>Discrimination</h1><p>ROC AUC, Gini, KS, CAP / gains, subgroup ROC curves and performance over time.</p></div>
            <div className={`analysis-status-pill ${status}`}><span />{statusLabel(status)}</div>
          </div>
          {discriminationError && <div className="message error-message">{discriminationError}</div>}
          <section className="panel analysis-run-card">
            <div className="analysis-run-icon" aria-hidden="true"><span>↗</span></div>
            <h2>{status === 'running' ? 'Discrimination analysis is running…' : 'Ready to evaluate ranking performance'}</h2>
            <p>Continuous segment variables are automatically split into three quantile groups. Configured population and sensitive variables can also be analysed as pairwise intersections.</p>
            <button className="primary-button" disabled={status === 'running'} onClick={runDiscrimination}>{status === 'running' ? 'Running…' : 'Run discrimination'}</button>
          </section>
        </section>
      )
    }

    const result = discriminationResult
    const rocOption: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      legend: { top: 0, textStyle: { color: '#8b9aaf' } },
      grid: { left: 58, right: 20, top: 42, bottom: 48 },
      xAxis: { type: 'value', min: 0, max: 1, name: 'False positive rate', axisLabel: { color: '#71809a' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      yAxis: { type: 'value', min: 0, max: 1, name: 'True positive rate', axisLabel: { color: '#71809a' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      series: [
        { name: 'ROC', type: 'line', showSymbol: false, data: result.overall.roc.map((point) => [point.fpr, point.tpr]), lineStyle: { width: 2, color: '#4f8cff' } },
        { name: 'Random', type: 'line', showSymbol: false, data: [[0,0],[1,1]], lineStyle: { width: 1, type: 'dashed', color: '#65758d' } },
      ],
    }

    const ksOption: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      legend: { top: 0, textStyle: { color: '#8b9aaf' } },
      grid: { left: 58, right: 20, top: 42, bottom: 48 },
      xAxis: { type: 'category', data: result.overall.roc.map((_, index) => index), axisLabel: { show: false } },
      yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: '#71809a' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      series: [
        { name: 'Bad cumulative', type: 'line', showSymbol: false, data: result.overall.roc.map((point) => point.tpr), lineStyle: { color: '#4f8cff', width: 2 } },
        { name: 'Good cumulative', type: 'line', showSymbol: false, data: result.overall.roc.map((point) => point.fpr), lineStyle: { color: '#8b99ad', width: 2 } },
      ],
    }

    const capOption: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      legend: { top: 0, textStyle: { color: '#8b9aaf' } },
      grid: { left: 58, right: 20, top: 42, bottom: 48 },
      xAxis: { type: 'value', min: 0, max: 1, name: 'Population share', axisLabel: { color: '#71809a', formatter: (value: string | number) => Math.round(Number(value) * 100) + '%' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      yAxis: { type: 'value', min: 0, max: 1, name: 'Bad capture', axisLabel: { color: '#71809a', formatter: (value: string | number) => Math.round(Number(value) * 100) + '%' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      series: [
        { name: 'Model', type: 'line', showSymbol: false, data: result.overall.cap.map((point) => [point.population, point.bad_capture]), lineStyle: { color: '#69e7ad', width: 2 } },
        { name: 'Random', type: 'line', showSymbol: false, data: [[0,0],[1,1]], lineStyle: { color: '#65758d', type: 'dashed', width: 1 } },
      ],
    }

    const selectedSegment = result.segment_performance.find((segment) => segment.key === discriminationSegmentKey) ?? result.segment_performance[0]
    const segmentRocOption = selectedSegment ? {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      legend: { type: 'scroll', top: 0, textStyle: { color: '#8b9aaf' } },
      grid: { left: 58, right: 20, top: 54, bottom: 48 },
      xAxis: { type: 'value', min: 0, max: 1, name: 'False positive rate', axisLabel: { color: '#71809a' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      yAxis: { type: 'value', min: 0, max: 1, name: 'True positive rate', axisLabel: { color: '#71809a' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      series: [
        ...selectedSegment.groups.filter((group) => group.roc.length).map((group) => ({
          name: `${group.value} · AUC ${group.auc?.toFixed(3) ?? '—'}`,
          type: 'line' as const,
          showSymbol: false,
          data: group.roc.map((point) => [point.fpr, point.tpr]),
          lineStyle: { width: 2 },
        })),
        { name: 'Random', type: 'line' as const, showSymbol: false, data: [[0,0],[1,1]], lineStyle: { width: 1, type: 'dashed' as const, color: '#65758d' } },
      ],
    } satisfies echarts.EChartsOption : null

    const timeData = result.time_performance?.granularities[discriminationGranularity]
    const selectedTimeSegment = discriminationTimeSegmentKey === 'overall'
      ? null
      : timeData?.segments.find((segment) => segment.key === discriminationTimeSegmentKey)
    const timeCategories = timeData?.overall.map((bucket) => bucket.bucket) ?? []
    const discriminationCutoffBucket = timeBucketForDate(validationConfig.timeCutoffDate, discriminationGranularity)
    const discriminationCutoffMarkLine = validationConfig.timeCutoffDate && discriminationCutoffBucket ? {
      silent: true,
      symbol: 'none',
      label: { formatter: `Cut-off · ${validationConfig.timeCutoffDate}`, color: '#9aa9bd', position: 'insideEndTop' as const },
      lineStyle: { type: 'dashed' as const, width: 1, color: '#9aa9bd' },
      data: [{ xAxis: discriminationCutoffBucket }],
    } : undefined

    const timeOption: echarts.EChartsOption | null = timeData ? {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      legend: { type: 'scroll', top: 0, textStyle: { color: '#8b9aaf' } },
      grid: { left: 58, right: 20, top: 54, bottom: 62 },
      xAxis: { type: 'category', data: timeCategories, boundaryGap: false, axisLabel: { color: '#71809a' } },
      yAxis: { type: 'value', min: 0, max: 1, name: 'ROC AUC', axisLabel: { color: '#71809a' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 12, height: 18 }],
      series: discriminationTimeSegmentKey === 'overall'
        ? [{
            name: 'Overall AUC',
            type: 'line',
            showSymbol: true,
            connectNulls: false,
            data: timeData.overall.map((bucket) => bucket.auc),
            lineStyle: { width: 2, color: '#4f8cff' },
            markLine: discriminationCutoffMarkLine,
          }]
        : (selectedTimeSegment?.groups.map((group, index) => {
            const lookup = new Map(group.buckets.map((bucket) => [bucket.bucket, bucket.auc]))
            return {
              name: group.value,
              type: 'line' as const,
              showSymbol: true,
              connectNulls: false,
              data: timeCategories.map((bucket) => lookup.get(bucket) ?? null),
              lineStyle: { width: 2 },
              markLine: index === 0 ? discriminationCutoffMarkLine : undefined,
            }
          }) ?? []),
    } : null

    return (
      <section className="page-content discrimination-page">
        <div className="analysis-page-header">
          <div><p className="eyebrow">Discrimination</p><h1>Discrimination</h1><p>Ranking performance for {validationConfig.predictionColumn} against {validationConfig.targetColumn}.</p></div>
          <div className="eda-header-actions"><div className={`analysis-status-pill ${status}`}><span />{statusLabel(status)}</div><button className="primary-button" disabled={status === 'running'} onClick={runDiscrimination}>{status === 'running' ? 'Running…' : 'Run again'}</button></div>
        </div>
        {discriminationError && <div className="message error-message">{discriminationError}</div>}

        <div className="discrimination-kpis">
          <article className="metric-card"><span className="metric-label-with-help">ROC AUC {metricHelp('roc-auc', 'ROC AUC')}</span><strong>{formatMetric(result.overall.auc)}</strong><small className="neutral">{result.overall.observations.toLocaleString()} observations</small></article>
          <article className="metric-card"><span className="metric-label-with-help">Gini {metricHelp('gini', 'Gini')}</span><strong>{formatMetric(result.overall.gini)}</strong><small className="neutral">2 × AUC − 1</small></article>
          <article className="metric-card"><span className="metric-label-with-help">KS {metricHelp('ks', 'KS')}</span><strong>{formatMetric(result.overall.ks)}</strong><small className="neutral">Max cumulative separation</small></article>
          <article className="metric-card"><span className="metric-label-with-help">Bad capture @ 10% {metricHelp('bad-capture', 'Bad capture at 10%')}</span><strong>{result.overall.bad_capture_10 === null ? '—' : formatPercent(result.overall.bad_capture_10)}</strong><small className="neutral">{result.overall.defaults.toLocaleString()} defaults</small></article>
          <article className="metric-card"><span>Default rate</span><strong>{formatPercent(result.overall.default_rate)}</strong><small className="neutral">Positive class: {validationConfig.positiveClass}</small></article>
        </div>

        <div className="discrimination-chart-grid">
          <article className="panel discrimination-chart"><div className="panel-title"><h2>ROC curve</h2></div><EChart option={rocOption} height={380} /></article>
          <article className="panel discrimination-chart"><div className="panel-title"><h2>KS cumulative curves</h2></div><EChart option={ksOption} height={380} /></article>
        </div>

        <article className="panel discrimination-chart"><div className="panel-title"><div><h2 className="heading-with-help">CAP / cumulative gains {metricHelp('cap', 'CAP / cumulative gains')}</h2><p className="eda-muted">Shows how quickly the riskiest observations capture observed defaults.</p></div></div><EChart option={capOption} height={390} /></article>

        {result.segment_performance.length > 0 && (
          <section className="panel discrimination-section segment-performance-section">
            <div className="eda-control-row">
              <div><h2>ROC by subgroup</h2><p>Compare discriminatory power across configured populations, sensitive attributes and pairwise intersections.</p></div>
              <label className="form-field compact-field"><span>Segment view</span><select value={selectedSegment?.key ?? ''} onChange={(e) => setDiscriminationSegmentKey(e.target.value)}>{result.segment_performance.map((segment) => <option key={segment.key} value={segment.key}>{segment.name}</option>)}</select></label>
            </div>
            {selectedSegment && (
              <>
                <div className="segment-chip-row">
                  {selectedSegment.groups.map((group) => <span className="segment-auc-chip" key={group.value}><strong>{group.value}</strong><small>AUC {formatMetric(group.auc)} · n={group.observations.toLocaleString()}</small></span>)}
                </div>
                {segmentRocOption && <EChart option={segmentRocOption} height={470} />}
              </>
            )}
            <div className="segment-method-note">
              {result.segment_dimensions.map((dimension) => <span key={dimension.column}><strong>{dimension.display_name ?? dimension.column}</strong>: {dimension.mode === 'quantiles' ? '3 quantile groups' : `${dimension.bins.length} categorical groups`}</span>)}
            </div>
          </section>
        )}

        {result.time_performance && timeData && (
          <section className="panel discrimination-section">
            <div className="eda-control-row">
              <div><h2>ROC AUC over time</h2><p>Aggregation and subgroup selection update immediately from the completed discrimination run.</p></div>
              <div className="discrimination-time-controls">
                <label className="form-field compact-field"><span>Aggregation</span><select value={discriminationGranularity} onChange={(e) => setDiscriminationGranularity(e.target.value as TimeGranularity)}><option value="daily">Daily</option><option value="weekly">Weekly</option><option value="monthly">Monthly</option><option value="quarterly">Quarterly</option><option value="yearly">Yearly</option></select><small>No rerun required.</small></label>
                <label className="form-field compact-field"><span>Population</span><select value={discriminationTimeSegmentKey} onChange={(e) => setDiscriminationTimeSegmentKey(e.target.value)}><option value="overall">Overall</option>{timeData.segments.map((segment) => <option key={segment.key} value={segment.key}>{segment.name}</option>)}</select></label>
              </div>
            </div>
            {timeOption && <EChart option={timeOption} height={440} />}
          </section>
        )}

        {(result.excluded.missing_prediction > 0 || result.excluded.missing_target > 0) && <div className="eda-sample-note">Excluded rows: {result.excluded.missing_prediction.toLocaleString()} missing prediction, {result.excluded.missing_target.toLocaleString()} missing target.</div>}
        <div className="eda-footer-meta">Last run: {lastRun.Discrimination ?? '—'}</div>
      </section>
    )
  }

  const renderCalibration = () => {
    if (!preview) return <EmptyAnalysisState section="Calibration" onOpenData={() => setActiveSection('Data')} />

    const status = analysisStatus.Calibration
    if (!validationReady) {
      return (
        <section className="page-content analysis-module-page">
          <div className="analysis-page-header">
            <div><p className="eyebrow">Calibration</p><h1>Calibration</h1><p>Evaluate whether predicted probabilities align with observed default rates.</p></div>
            <div className="analysis-status-pill blocked"><span />Blocked</div>
          </div>
          <section className="panel analysis-run-card">
            <div className="analysis-run-icon" aria-hidden="true"><span>⌁</span></div>
            <h2>Complete validation first</h2>
            <p>Calibration requires a validated prediction and target mapping.</p>
            <button className="secondary-button" onClick={() => setActiveSection('Validation')}>Open validation setup</button>
          </section>
        </section>
      )
    }

    if (validationConfig.predictionType !== 'pd') {
      return (
        <section className="page-content analysis-module-page">
          <div className="analysis-page-header">
            <div><p className="eyebrow">Calibration</p><h1>Calibration</h1><p>Evaluate whether predicted probabilities align with observed default rates.</p></div>
            <div className="analysis-status-pill unavailable"><span />Not available</div>
          </div>
          <section className="panel analysis-run-card">
            <div className="analysis-run-icon" aria-hidden="true"><span>⌁</span></div>
            <h2>Calibration requires probabilities</h2>
            <p>The model output is configured as a score. Change Output type to Probability of Default (PD) to run calibration analysis.</p>
          </section>
        </section>
      )
    }

    if (!calibrationResult) {
      return (
        <section className="page-content analysis-module-page">
          <div className="analysis-page-header">
            <div><p className="eyebrow">Calibration</p><h1>Calibration</h1><p>Spiegelhalter test, observed vs predicted default rates and calibration over time.</p></div>
            <div className={`analysis-status-pill ${status}`}><span />{statusLabel(status)}</div>
          </div>
          {calibrationError && <div className="message error-message">{calibrationError}</div>}
          <section className="panel analysis-run-card">
            <div className="analysis-run-icon" aria-hidden="true"><span>⌁</span></div>
            <h2>{status === 'running' ? 'Calibration analysis is running…' : 'Ready to evaluate calibration'}</h2>
            <p>The analysis uses observation-level PDs and the configured default indicator. Subgroup definitions are shared with Discrimination.</p>
            <button className="primary-button" disabled={status === 'running'} onClick={runCalibration}>{status === 'running' ? 'Running…' : 'Run calibration'}</button>
          </section>
        </section>
      )
    }

    const result = calibrationResult
    const selectedSegment = result.segment_performance.find((segment) => segment.key === calibrationSegmentKey) ?? result.segment_performance[0]
    const summaryRows = [
      { value: 'All', ...result.overall },
      ...(selectedSegment?.groups ?? []),
    ]

    const spiegelhalterBadge = (pValue: number | null) => {
      if (pValue === null) return <span className="calibration-test-badge neutral">Unavailable</span>
      if (pValue < 0.01) return <span className="calibration-test-badge bad">{pValue.toFixed(4)} · Strong evidence</span>
      if (pValue < 0.05) return <span className="calibration-test-badge warn">{pValue.toFixed(4)} · Evidence</span>
      return <span className="calibration-test-badge neutral">{pValue.toFixed(4)} · No significant evidence</span>
    }

    const calibrationBins = result.overall.calibration_bins
    const maxCalibrationValue = Math.max(
      0.05,
      ...calibrationBins.flatMap((row) => [row.mean_pd, row.observed_default_rate, row.ci_upper ?? 0]),
    )
    const calibrationAxisMax = Math.min(1, Math.ceil(maxCalibrationValue * 20) / 20)

    const calibrationOption: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      legend: { top: 0, textStyle: { color: '#8b9aaf' } },
      grid: { left: 64, right: 24, top: 48, bottom: 56 },
      xAxis: {
        type: 'value',
        min: 0,
        max: calibrationAxisMax,
        name: 'Mean predicted PD',
        axisLabel: { color: '#71809a', formatter: (value: string | number) => (Number(value) * 100).toFixed(0) + '%' },
        splitLine: { lineStyle: { color: '#1b293d' } },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: calibrationAxisMax,
        name: 'Observed default rate',
        axisLabel: { color: '#71809a', formatter: (value: string | number) => (Number(value) * 100).toFixed(0) + '%' },
        splitLine: { lineStyle: { color: '#1b293d' } },
      },
      series: [
        {
          name: 'Observed',
          type: 'line',
          showSymbol: true,
          symbolSize: 8,
          data: calibrationBins.map((row) => [row.mean_pd, row.observed_default_rate]),
          lineStyle: { width: 2, color: '#4f8cff' },
        },
        {
          name: 'Ideal calibration',
          type: 'line',
          showSymbol: false,
          data: [[0,0],[calibrationAxisMax, calibrationAxisMax]],
          lineStyle: { width: 1, type: 'dashed', color: '#65758d' },
        },
      ],
    }

    const timeData = result.time_performance?.granularities[calibrationGranularity]
    const selectedTimeSegment = calibrationTimeSegmentKey === 'overall'
      ? null
      : timeData?.segments.find((segment) => segment.key === calibrationTimeSegmentKey)
    const timeCategories = timeData?.overall.map((bucket) => bucket.bucket) ?? []
    const cutoffBucket = timeBucketForDate(validationConfig.timeCutoffDate, calibrationGranularity)
    const cutoffMarkLine = validationConfig.timeCutoffDate && cutoffBucket ? {
      silent: true,
      symbol: 'none',
      label: { formatter: `Cut-off · ${validationConfig.timeCutoffDate}`, color: '#9aa9bd', position: 'insideEndTop' as const },
      lineStyle: { type: 'dashed' as const, width: 1, color: '#9aa9bd' },
      data: [{ xAxis: cutoffBucket }],
    } : undefined

    const calibrationTimeOption: echarts.EChartsOption | null = timeData ? {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      legend: { type: 'scroll', top: 0, textStyle: { color: '#8b9aaf' } },
      grid: { left: 62, right: 20, top: 54, bottom: 64 },
      xAxis: { type: 'category', data: timeCategories, boundaryGap: false, axisLabel: { color: '#71809a' } },
      yAxis: { type: 'value', min: 0, name: 'Rate', axisLabel: { color: '#71809a', formatter: (value: string | number) => (Number(value) * 100).toFixed(1) + '%' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 12, height: 18 }],
      series: calibrationTimeSegmentKey === 'overall'
        ? [
            {
              name: 'Mean PD',
              type: 'line',
              showSymbol: true,
              data: timeData.overall.map((bucket) => bucket.mean_pd),
              lineStyle: { width: 2, color: '#4f8cff' },
              markLine: cutoffMarkLine,
            },
            {
              name: 'Observed default rate',
              type: 'line',
              showSymbol: true,
              data: timeData.overall.map((bucket) => bucket.default_rate),
              lineStyle: { width: 2, color: '#69e7ad' },
            },
          ]
        : (selectedTimeSegment?.groups.flatMap((group, groupIndex) => {
            const lookup = new Map(group.buckets.map((bucket) => [bucket.bucket, bucket]))
            return [
              {
                name: `${group.value} · Mean PD`,
                type: 'line' as const,
                showSymbol: false,
                data: timeCategories.map((bucket) => lookup.get(bucket)?.mean_pd ?? null),
                lineStyle: { width: 1, type: 'dashed' as const },
                markLine: groupIndex === 0 ? cutoffMarkLine : undefined,
              },
              {
                name: `${group.value} · Observed`,
                type: 'line' as const,
                showSymbol: true,
                data: timeCategories.map((bucket) => lookup.get(bucket)?.default_rate ?? null),
                lineStyle: { width: 2 },
              },
            ]
          }) ?? []),
    } : null

    return (
      <section className="page-content calibration-page">
        <div className="analysis-page-header">
          <div><p className="eyebrow">Calibration</p><h1>Calibration</h1><p>Compare predicted PD with observed defaults overall, across segments and over time.</p></div>
          <div className="eda-header-actions"><div className={`analysis-status-pill ${status}`}><span />{statusLabel(status)}</div><button className="primary-button" disabled={status === 'running'} onClick={runCalibration}>{status === 'running' ? 'Running…' : 'Run again'}</button></div>
        </div>
        {calibrationError && <div className="message error-message">{calibrationError}</div>}

        <section className="panel calibration-summary-section">
          <div className="eda-control-row">
            <div>
              <h2>Calibration summary</h2>
              <p>Spiegelhalter tests the null hypothesis that the submitted probabilities are calibrated to the observed binary outcomes.</p>
            </div>
            {result.segment_performance.length > 0 && (
              <label className="form-field compact-field"><span>Segment view</span><select value={selectedSegment?.key ?? ''} onChange={(e) => setCalibrationSegmentKey(e.target.value)}>{result.segment_performance.map((segment) => <option key={segment.key} value={segment.key}>{segment.name}</option>)}</select></label>
            )}
          </div>
          <div className="table-wrap">
            <table className="calibration-summary-table">
              <thead><tr><th>Segment</th><th>Observations</th><th>Defaults</th><th>Default rate</th><th>Mean PD</th><th><span className="metric-label-with-help">Spiegelhalter p-value {metricHelp('spiegelhalter', 'Spiegelhalter p-value')}</span></th></tr></thead>
              <tbody>{summaryRows.map((row) => (
                <tr key={row.value}>
                  <td><strong>{row.value}</strong></td>
                  <td>{row.observations.toLocaleString()}</td>
                  <td>{row.defaults.toLocaleString()}</td>
                  <td>{formatPercent(row.default_rate)}</td>
                  <td>{formatPercent(row.mean_pd)}</td>
                  <td>{spiegelhalterBadge(row.spiegelhalter_p_value)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <div className="calibration-test-legend"><span><i className="neutral" />p ≥ 5% · no significant evidence of miscalibration</span><span><i className="warn" />1% ≤ p &lt; 5% · evidence of miscalibration</span><span><i className="bad" />p &lt; 1% · strong evidence of miscalibration</span></div>
        </section>

        <div className="calibration-kpis">
          <article className="metric-card"><span className="metric-label-with-help">Brier score {metricHelp('brier', 'Brier score')}</span><strong>{formatMetric(result.overall.brier_score)}</strong><small className="neutral">Mean squared probability error</small></article>
          <article className="metric-card"><span className="metric-label-with-help">O / E ratio {metricHelp('oe-ratio', 'Observed / Expected ratio')}</span><strong>{formatMetric(result.overall.oe_ratio)}</strong><small className="neutral">Observed / expected defaults</small></article>
          <article className="metric-card"><span>Expected defaults</span><strong>{formatMetric(result.overall.expected_defaults)}</strong><small className="neutral">Σ predicted PD</small></article>
          <article className="metric-card"><span className="metric-label-with-help">Spiegelhalter Z {metricHelp('spiegelhalter', 'Spiegelhalter Z-test')}</span><strong>{formatMetric(result.overall.spiegelhalter_z)}</strong><small className="neutral">Two-sided calibration test</small></article>
        </div>

        <section className="panel calibration-chart-section">
          <div className="panel-title"><div><h2 className="heading-with-help">Calibration curve {metricHelp('calibration-curve', 'Calibration curve')}</h2><p className="eda-muted">Ten equal-frequency PD bins. The dashed diagonal represents ideal calibration.</p></div></div>
          <EChart option={calibrationOption} height={440} />
          <div className="calibration-bin-strip">{calibrationBins.map((row) => <span key={row.bin}><strong>Bin {row.bin}</strong><small>PD {formatPercent(row.mean_pd)} · DR {formatPercent(row.observed_default_rate)} · n={row.observations.toLocaleString()}</small></span>)}</div>
        </section>

        {result.time_performance && timeData && (
          <section className="panel calibration-chart-section">
            <div className="eda-control-row">
              <div><h2>Calibration over time</h2><p>Compare mean predicted PD with observed default rate. Aggregation and subgroup selection update without rerunning the analysis.</p></div>
              <div className="discrimination-time-controls">
                <label className="form-field compact-field"><span>Aggregation</span><select value={calibrationGranularity} onChange={(e) => setCalibrationGranularity(e.target.value as TimeGranularity)}><option value="daily">Daily</option><option value="weekly">Weekly</option><option value="monthly">Monthly</option><option value="quarterly">Quarterly</option><option value="yearly">Yearly</option></select><small>No rerun required.</small></label>
                <label className="form-field compact-field"><span>Population</span><select value={calibrationTimeSegmentKey} onChange={(e) => setCalibrationTimeSegmentKey(e.target.value)}><option value="overall">Overall</option>{timeData.segments.map((segment) => <option key={segment.key} value={segment.key}>{segment.name}</option>)}</select></label>
              </div>
            </div>
            {calibrationTimeOption && <EChart option={calibrationTimeOption} height={460} />}
          </section>
        )}

        {(result.excluded.missing_prediction > 0 || result.excluded.missing_target > 0 || result.excluded.invalid_pd > 0) && (
          <div className="eda-sample-note">Excluded rows: {result.excluded.missing_prediction.toLocaleString()} missing prediction, {result.excluded.missing_target.toLocaleString()} missing target, {result.excluded.invalid_pd.toLocaleString()} PD outside [0, 1].</div>
        )}
        <div className="eda-footer-meta">Last run: {lastRun.Calibration ?? '—'}</div>
      </section>
    )
  }

  const renderStability = () => {
    if (!preview) return <EmptyAnalysisState section="Stability" onOpenData={() => setActiveSection('Data')} />

    const status = analysisStatus.Stability
    if (!validationReady) {
      return (
        <section className="page-content analysis-module-page">
          <div className="analysis-page-header">
            <div><p className="eyebrow">Stability</p><h1>Stability</h1><p>Assess population and model-output drift between reference and comparison populations.</p></div>
            <div className="analysis-status-pill blocked"><span />Blocked</div>
          </div>
          <section className="panel analysis-run-card">
            <div className="analysis-run-icon" aria-hidden="true"><span>≈</span></div>
            <h2>Complete validation first</h2>
            <p>Stability uses the configured population columns, features, model output and optional time cut-off.</p>
            <button className="secondary-button" onClick={() => setActiveSection('Validation')}>Open validation setup</button>
          </section>
        </section>
      )
    }

    if (!stabilityResult) {
      return (
        <section className="page-content analysis-module-page">
          <div className="analysis-page-header">
            <div><p className="eyebrow">Stability</p><h1>Stability</h1><p>PSI, distribution shift and volume stability across populations and time.</p></div>
            <div className={`analysis-status-pill ${status}`}><span />{statusLabel(status)}</div>
          </div>
          {stabilityError && <div className="message error-message">{stabilityError}</div>}
          <section className="panel analysis-run-card">
            <div className="analysis-run-icon" aria-hidden="true"><span>≈</span></div>
            <h2>{status === 'running' ? 'Stability analysis is running…' : 'Ready to assess stability'}</h2>
            <p>RiskLab compares the model output and configured features between population groups. The optional time cut-off can also act as a reference/comparison definition.</p>
            <button className="primary-button" disabled={status === 'running'} onClick={runStability}>{status === 'running' ? 'Running…' : 'Run stability analysis'}</button>
          </section>
        </section>
      )
    }

    const result = stabilityResult
    const selectedSource = result.sources.find((source) => source.key === stabilitySourceKey) ?? result.sources[0]

    if (!selectedSource) {
      return (
        <section className="page-content stability-page">
          <div className="analysis-page-header">
            <div><p className="eyebrow">Stability</p><h1>Stability</h1><p>PSI, distribution shift and volume stability across populations and time.</p></div>
            <div className="eda-header-actions"><div className={`analysis-status-pill ${status}`}><span />{statusLabel(status)}</div><button className="primary-button" onClick={runStability}>Run again</button></div>
          </div>
          <section className="panel analysis-run-card">
            <div className="analysis-run-icon" aria-hidden="true"><span>≈</span></div>
            <h2>Configure a comparison population</h2>
            <p>Select a Sample / population column with at least two groups in Validation, or enable “Treat cut-off as an analysis segment”.</p>
            <button className="secondary-button" onClick={() => setActiveSection('Validation')}>Open validation setup</button>
          </section>
        </section>
      )
    }

    const sourceGroups = selectedSource.groups
    const resolvedReference = sourceGroups.some((group) => group.value === stabilityReference) ? stabilityReference : sourceGroups[0]?.value ?? ''
    const resolvedComparison = sourceGroups.some((group) => group.value === stabilityComparison && group.value !== resolvedReference)
      ? stabilityComparison
      : sourceGroups.find((group) => group.value !== resolvedReference)?.value ?? ''

    const activeComparison = result.comparisons.find((item) =>
      item.source_key === selectedSource.key &&
      item.reference === resolvedReference &&
      item.comparison === resolvedComparison
    )

    const activeVariable = activeComparison?.variables.find((variable) => variable.column === stabilityVariable)
      ?? activeComparison?.variables[0]

    const psiTone = (psi: number) => psi > result.thresholds.high ? 'bad' : psi >= result.thresholds.moderate ? 'warn' : 'neutral'
    const psiStatus = (psi: number) => psi > result.thresholds.high ? 'High shift' : psi >= result.thresholds.moderate ? 'Moderate shift' : 'Low shift'

    const distributionOption: echarts.EChartsOption | null = activeVariable ? {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { top: 0, textStyle: { color: '#8b9aaf' } },
      grid: { left: 58, right: 20, top: 44, bottom: 92 },
      xAxis: { type: 'category', data: activeVariable.distribution.map((row) => row.label), axisLabel: { color: '#71809a', rotate: 30, interval: 0, hideOverlap: true } },
      yAxis: { type: 'value', min: 0, name: 'Share', axisLabel: { color: '#71809a', formatter: (value: string | number) => Math.round(Number(value) * 100) + '%' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      series: [
        { name: resolvedReference, type: 'bar', data: activeVariable.distribution.map((row) => row.reference_share), itemStyle: { color: '#4f8cff' }, barMaxWidth: 34 },
        { name: resolvedComparison, type: 'bar', data: activeVariable.distribution.map((row) => row.comparison_share), itemStyle: { color: '#8b99ad' }, barMaxWidth: 34 },
      ],
    } : null

    const timeBuckets = result.time_analysis?.granularities[stabilityGranularity] ?? []
    const psiTimeValues = timeBuckets.map((bucket) => {
      const reference = bucket.references.find((item) => item.source_key === selectedSource.key && item.reference === resolvedReference)
      return reference?.variables.find((item) => item.column === activeVariable?.column)?.psi ?? null
    })
    const cutoffBucket = timeBucketForDate(validationConfig.timeCutoffDate, stabilityGranularity)
    const cutoffMarkLine = validationConfig.timeCutoffDate && cutoffBucket ? {
      silent: true,
      symbol: 'none',
      label: { formatter: `Cut-off · ${validationConfig.timeCutoffDate}`, color: '#9aa9bd', position: 'insideEndTop' as const },
      lineStyle: { type: 'dashed' as const, width: 1, color: '#9aa9bd' },
      data: [{ xAxis: cutoffBucket }],
    } : undefined

    const psiTimeOption: echarts.EChartsOption | null = result.time_analysis && activeVariable ? {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      grid: { left: 58, right: 22, top: 34, bottom: 64 },
      xAxis: { type: 'category', data: timeBuckets.map((bucket) => bucket.bucket), boundaryGap: false, axisLabel: { color: '#71809a' } },
      yAxis: { type: 'value', min: 0, name: 'PSI', axisLabel: { color: '#71809a' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 12, height: 18 }],
      series: [{
        name: activeVariable.column,
        type: 'line',
        showSymbol: true,
        data: psiTimeValues,
        lineStyle: { width: 2, color: '#4f8cff' },
        markLine: {
          silent: true,
          symbol: 'none',
          label: { color: '#91a0b7' },
          lineStyle: { type: 'dashed', width: 1 },
          data: [
            { yAxis: result.thresholds.moderate, label: { formatter: '0.10 · moderate' }, lineStyle: { color: '#d99a3e' } },
            { yAxis: result.thresholds.high, label: { formatter: '0.25 · high' }, lineStyle: { color: '#e56f6f' } },
            ...(cutoffMarkLine ? cutoffMarkLine.data : []),
          ],
        },
      }],
    } : null

    const volumeOption: echarts.EChartsOption | null = result.time_analysis ? {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      grid: { left: 64, right: 22, top: 34, bottom: 64 },
      xAxis: { type: 'category', data: timeBuckets.map((bucket) => bucket.bucket), boundaryGap: false, axisLabel: { color: '#71809a' } },
      yAxis: { type: 'value', min: 0, name: 'Observations', axisLabel: { color: '#71809a' }, splitLine: { lineStyle: { color: '#1b293d' } } },
      dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 12, height: 18 }],
      series: [{
        name: 'Observations',
        type: 'line',
        showSymbol: true,
        areaStyle: { opacity: .08, color: '#8b99ad' },
        data: timeBuckets.map((bucket) => bucket.observations),
        lineStyle: { width: 2, color: '#8b99ad' },
        markLine: cutoffMarkLine,
      }],
    } : null

    const changeSource = (key: string) => {
      const source = result.sources.find((item) => item.key === key)
      const reference = source?.groups[0]?.value ?? ''
      const comparison = source?.groups.find((group) => group.value !== reference)?.value ?? ''
      const firstVariable = result.comparisons.find((item) => item.source_key === key && item.reference === reference && item.comparison === comparison)?.variables[0]?.column ?? result.variables[0]?.column ?? ''
      setStabilitySourceKey(key)
      setStabilityReference(reference)
      setStabilityComparison(comparison)
      setStabilityVariable(firstVariable)
    }

    const changeReference = (reference: string) => {
      const comparison = sourceGroups.find((group) => group.value !== reference)?.value ?? ''
      const firstVariable = result.comparisons.find((item) => item.source_key === selectedSource.key && item.reference === reference && item.comparison === comparison)?.variables[0]?.column ?? result.variables[0]?.column ?? ''
      setStabilityReference(reference)
      setStabilityComparison(comparison)
      setStabilityVariable(firstVariable)
    }

    return (
      <section className="page-content stability-page">
        <div className="analysis-page-header">
          <div><p className="eyebrow">Stability</p><h1>Stability</h1><p>Compare population distributions and model output between a fixed reference and comparison population.</p></div>
          <div className="eda-header-actions"><div className={`analysis-status-pill ${status}`}><span />{statusLabel(status)}</div><button className="primary-button" disabled={status === 'running'} onClick={runStability}>{status === 'running' ? 'Running…' : 'Run again'}</button></div>
        </div>
        {stabilityError && <div className="message error-message">{stabilityError}</div>}

        <section className="panel stability-controls">
          <div className="eda-control-row">
            <div><h2>Population comparison</h2><p>Select one reference population and one comparison population. All PSI values below use the same definition.</p></div>
            <div className="stability-selector-grid">
              <label className="form-field compact-field"><span>Population definition</span><select value={selectedSource.key} onChange={(e) => changeSource(e.target.value)}>{result.sources.map((source) => <option key={source.key} value={source.key}>{source.name}</option>)}</select></label>
              <label className="form-field compact-field"><span>Reference</span><select value={resolvedReference} onChange={(e) => changeReference(e.target.value)}>{sourceGroups.map((group) => <option key={group.value} value={group.value}>{group.value} · n={group.observations.toLocaleString()}</option>)}</select></label>
              <label className="form-field compact-field"><span>Comparison</span><select value={resolvedComparison} onChange={(e) => setStabilityComparison(e.target.value)}>{sourceGroups.filter((group) => group.value !== resolvedReference).map((group) => <option key={group.value} value={group.value}>{group.value} · n={group.observations.toLocaleString()}</option>)}</select></label>
            </div>
          </div>
        </section>

        {activeComparison && (
          <>
            <section className="panel stability-summary-section">
              <div className="panel-title"><div><h2 className="heading-with-help">PSI summary {metricHelp('psi', 'Population Stability Index')}</h2><p className="eda-muted">Default thresholds are heuristic: PSI &lt; 0.10 low, 0.10–0.25 moderate, &gt; 0.25 high.</p></div></div>
              <div className="table-wrap">
                <table className="stability-summary-table">
                  <thead><tr><th>Variable</th><th><span className="metric-label-with-help">PSI {metricHelp('psi', 'Population Stability Index')}</span></th><th>Reference mean</th><th>Comparison mean</th><th>Missing Δ</th><th>Status</th></tr></thead>
                  <tbody>{activeComparison.variables.map((variable) => (
                    <tr key={variable.column} className={activeVariable?.column === variable.column ? 'selected-row' : ''} onClick={() => setStabilityVariable(variable.column)}>
                      <td><strong>{variable.column}</strong><small>{variable.semantic_type}</small></td>
                      <td>{variable.psi.toFixed(3)}</td>
                      <td>{formatMetric(variable.reference_mean)}</td>
                      <td>{formatMetric(variable.comparison_mean)}</td>
                      <td>{variable.missing_delta >= 0 ? '+' : ''}{(variable.missing_delta * 100).toFixed(1)} pp</td>
                      <td><span className={`stability-badge ${psiTone(variable.psi)}`}>{psiStatus(variable.psi)}</span></td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
              <div className="eda-sample-note">{result.thresholds.note} Click a variable row to inspect its distribution.</div>
            </section>

            {activeVariable && distributionOption && (
              <section className="panel stability-drilldown">
                <div className="eda-control-row">
                  <div><h2>{activeVariable.column} distribution</h2><p>Reference-bin distribution is reused for the comparison population. Missing values are a dedicated bucket.</p></div>
                  <label className="form-field compact-field"><span>Variable</span><select value={activeVariable.column} onChange={(e) => setStabilityVariable(e.target.value)}>{activeComparison.variables.map((variable) => <option key={variable.column} value={variable.column}>{variable.column} · PSI {variable.psi.toFixed(3)}</option>)}</select></label>
                </div>
                <EChart option={distributionOption} height={430} />
              </section>
            )}

            {result.time_analysis && psiTimeOption && volumeOption && (
              <section className="panel stability-time-section">
                <div className="eda-control-row">
                  <div><h2>Stability over time</h2><p>Each time bucket is compared with the selected fixed reference population. Volume is shown alongside PSI for context.</p></div>
                  <label className="form-field compact-field"><span>Aggregation</span><select value={stabilityGranularity} onChange={(e) => setStabilityGranularity(e.target.value as TimeGranularity)}><option value="daily">Daily</option><option value="weekly">Weekly</option><option value="monthly">Monthly</option><option value="quarterly">Quarterly</option><option value="yearly">Yearly</option></select><small>No rerun required.</small></label>
                </div>
                <div className="stability-time-grid">
                  <article className="eda-chart-panel"><h4>PSI over time · {activeVariable?.column}</h4><EChart option={psiTimeOption} height={360} /></article>
                  <article className="eda-chart-panel"><h4>Population volume</h4><EChart option={volumeOption} height={360} /></article>
                </div>
              </section>
            )}
          </>
        )}

        <div className="eda-footer-meta">Last run: {lastRun.Stability ?? '—'}</div>
      </section>
    )
  }

  const renderAnalysisPage = (section: Exclude<AnalysisSection, 'Validation' | 'EDA' | 'Discrimination' | 'Calibration' | 'Stability'>) => {
    if (!preview) return <EmptyAnalysisState section={section} onOpenData={() => setActiveSection('Data')} />

    const status = analysisStatus[section]
    const blocked = !validationReady
    const copy = analysisCopy[section]

    return (
      <section className="page-content analysis-module-page">
        <div className="analysis-page-header">
          <div><p className="eyebrow">{section}</p><h1>{copy.title}</h1><p>{copy.description}</p></div>
          <div className={`analysis-status-pill ${blocked ? 'blocked' : status}`}><span />{blocked ? 'Blocked' : statusLabel(status)}</div>
        </div>

        <section className="panel analysis-run-card">
          <div className="analysis-run-icon" aria-hidden="true"><span>↗</span></div>
          {blocked ? (
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
        {activeSection === 'EDA' && renderEda()}
        {activeSection === 'Discrimination' && renderDiscrimination()}
        {activeSection === 'Calibration' && renderCalibration()}
        {activeSection === 'Stability' && renderStability()}
        {activeSection === 'Methodology' && renderMethodology()}
        {activeSection !== 'Overview' && activeSection !== 'Data' && activeSection !== 'Validation' && activeSection !== 'EDA' && activeSection !== 'Discrimination' && activeSection !== 'Calibration' && activeSection !== 'Stability' && activeSection !== 'Methodology' && renderAnalysisPage(activeSection)}
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
