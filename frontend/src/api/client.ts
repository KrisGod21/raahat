/**
 * API client. Spec 9.5: the frontend must render fully with the backend
 * switched off, so every call has a short timeout and a visible, useful
 * error state rather than a spinner that never resolves.
 */

const BASE = '/api/v1'
// Spec 9.5 asks for a 3-second timeout so a dead backend shows a useful error
// rather than an endless spinner. That is right for the small per-date calls,
// but the district geometry is ~1 MB of polygons parsed once at startup, and
// on a loaded machine it can take well over 3 seconds. Sharing one timeout
// aborted the geometry fetch and left a permanently blank map with no error --
// the failure looked like a rendering bug for a long time.
// Serverless cold starts and slower connections can exceed three seconds.
const TIMEOUT_MS = 12000
const SLOW_TIMEOUT_MS = 30000

export type Colour = 0 | 1 | 2 | 3
export const COLOUR_NAMES = ['GREEN', 'YELLOW', 'ORANGE', 'RED'] as const
export const COLOUR_HEX = ['#1B8A3F', '#F2C200', '#F07C00', '#D31F26'] as const
export const COLOUR_ACTION = [
  'No warning', 'Be updated', 'Be prepared', 'Take action',
] as const

export interface Health {
  status: string
  model_version: string
  data_from: string | null
  data_through: string | null
  mode: string
  attribution: string[]
}

export interface ForecastValue {
  district_id: number
  value: number | null
  colour_code: Colour
  regime_argmax: string | null
}

export interface DistrictDetail {
  district: { district_id: number; district_name?: string; state?: string; zone_code?: string }
  date: string
  lead: number
  raw: { multi_model_mean: number | null }
  corrected: { p10: number | null; median: number | null; p90: number | null }
  exceedance: { p64_5: number | null; p115_6: number | null; p204_5: number | null }
  regime: { probs: Record<string, number | null>; argmax: string | null }
  explanation: {
    drivers: { feature: string; label: string; contribution: number; direction: string }[]
    narrative: string
  }
  analogs: { date: string | null; observed_mm: number | null }[]
  analog_narrative: string
  warning: {
    suggested_colour: string
    colour_code: Colour
    alpha_used: number
    raw_colour: string
    guard_applied: boolean
  }
}

export interface RegimeTimeline {
  date: string
  timeline: { lead: number; probs: Record<string, number | null>; entropy: number | null }[]
  note: string
}

export class ApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message)
  }
}

async function get<T>(path: string, timeoutMs = TIMEOUT_MS): Promise<T> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const r = await fetch(`${BASE}${path}`, { signal: ctrl.signal })
    if (!r.ok) {
      let msg = `Request failed (${r.status})`
      try {
        const body = await r.json()
        if (body?.error?.message) msg = body.error.message
      } catch { /* keep the status-based message */ }
      throw new ApiError(msg, r.status)
    }
    return (await r.json()) as T
  } catch (e) {
    if (e instanceof ApiError) throw e
    if ((e as Error).name === 'AbortError') {
      throw new ApiError(`Backend did not respond within ${timeoutMs / 1000} seconds.`)
    }
    throw new ApiError('Backend not reachable. Start it with: uvicorn raahat.api.main:app')
  } finally {
    clearTimeout(timer)
  }
}

export const api = {
  health: () => get<Health>('/health'),
  dates: (lead: number) => get<{ lead: number; dates: string[] }>(`/dates?lead=${lead}`),
  districts: async () => {
    try {
      const response = await fetch('/districts.geojson')
      if (response.ok) return await response.json() as GeoJSON.FeatureCollection
    } catch { /* local development can fall back to the API */ }
    return get<GeoJSON.FeatureCollection>('/districts', SLOW_TIMEOUT_MS)
  },
  forecast: (date: string, lead: number, layer: string, alpha: number) =>
    get<{ values: ForecastValue[]; layer: string; alpha: number }>(
      `/forecast?date=${date}&lead=${lead}&layer=${layer}&alpha=${alpha}`),
  district: (id: number, date: string, lead: number, alpha: number) =>
    get<DistrictDetail>(`/district/${id}?date=${date}&lead=${lead}&alpha=${alpha}`),
  regime: (date: string) => get<RegimeTimeline>(`/regime?date=${date}&lead_max=5`),
  costLoss: (date: string, lead: number) =>
    get<{ curve: { alpha: number; warned: number; warn_rate: number }[]; note: string }>(
      `/cost-loss?date=${date}&lead=${lead}`),
  bulletin: (id: number, date: string, lead: number) =>
    get<{ text_en: string; colour: string; caveat: string }>(
      `/bulletin/${id}?date=${date}&lead=${lead}`),
}

/** Human labels for the regime probability columns. */
export const REGIME_LABEL: Record<string, string> = {
  p_ACTIVE: 'Active spell',
  p_BREAK: 'Break',
  p_LPS: 'Low-pressure system',
  p_WD: 'Western disturbance',
  p_EASTERLY_COASTAL: 'Easterly / coastal',
  p_WEAK: 'Unsettled',
}

/**
 * Fixed hue per weather situation, shared by the regime band, the district
 * table and the atlas so a colour means the same thing everywhere. All cool
 * (or, for Break, grey): none of them can be mistaken for a warning colour.
 * Break is deliberately the quietest -- it is a lull.
 */
export const REGIME_HEX: Record<string, string> = {
  LPS: '#3A4FC4',               // indigo  -- low-pressure system
  ACTIVE: '#0E8F8F',            // teal    -- active spell
  BREAK: '#8593A4',             // grey    -- a lull, deliberately quiet
  WEAK: '#1B87C9',              // cyan    -- unsettled
  WD: '#6B4EC7',                // violet  -- western disturbance
  EASTERLY_COASTAL: '#2B7A9E',  // steel   -- easterly / coastal
}

/**
 * The API returns `p_LPS` in probability maps but `LPS` in argmax fields, and
 * one regime has two spellings: contract.py stores WEAK_TRANSITION in the
 * column `p_WEAK` because spec 13.2's column list uses the short form. The
 * atlas and the district table therefore send WEAK_TRANSITION while the regime
 * band sends p_WEAK. Without this alias the atlas printed a bare
 * "WEAK_TRANSITION" in grey next to three properly labelled, coloured rows.
 */
const ALIAS: Record<string, string> = { WEAK_TRANSITION: 'WEAK' }
export const regimeKey = (s: string) => {
  const k = s.replace(/^p_/, '')
  return ALIAS[k] ?? k
}
export const regimeHex = (s?: string | null) =>
  s ? REGIME_HEX[regimeKey(s)] ?? '#8593A4' : '#8593A4'
export const regimeLabel = (s?: string | null) =>
  s ? REGIME_LABEL[`p_${regimeKey(s)}`] ?? regimeKey(s) : '—'
