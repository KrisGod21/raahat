/**
 * RAAHAT forecast desk (spec 9.3).
 *
 * Design constraints that are deliberate, not accidental:
 *  - The four IMD warning colours are the only saturated colours ON THE MAP
 *    and on the pills that restate a warning (spec 9.2). The chrome around
 *    them is a deep indigo that cannot be confused with any of the four.
 *  - One animated moment only: the choropleth cross-fades when the lead
 *    changes. Nothing else moves.
 *  - The replay banner is always visible. Spec 10.3 is emphatic that we must
 *    never imply live capability we do not have.
 *  - Errors state what to DO, not just that something failed (spec 9.5).
 *  - The control bar is hidden on the three tabs it cannot affect. Leaving a
 *    lead selector visible above a page it does nothing to is a small lie
 *    about how the interface works.
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import { DistrictMap } from './components/DistrictMap'
import { Card, DistrictPanel, RegimePanel } from './components/Panels'
import { AboutScreen, AtlasScreen, EvidenceScreen, TableScreen } from './components/Screens'
import {
  api, ApiError, COLOUR_ACTION, COLOUR_HEX,
  type DistrictDetail, type ForecastValue, type Health, type RegimeTimeline,
} from './api/client'

const LAYERS = [
  { id: 'colour', label: 'Warning' },
  { id: 'corrected', label: 'Corrected' },
  { id: 'raw', label: 'Raw' },
  { id: 'delta', label: 'Change' },
  { id: 'p64', label: 'P(>64.5)' },
] as const

const TABS = [
  { id: 'forecast', label: 'Forecast' },
  { id: 'table', label: 'District table' },
  { id: 'atlas', label: 'Regime Error Atlas' },
  { id: 'evidence', label: 'Evidence' },
  { id: 'about', label: 'About the data' },
] as const
type Tab = typeof TABS[number]['id']

/** Tabs that the date / lead / caution / layer controls actually drive. */
const CONTROLLED: ReadonlySet<string> = new Set(['forecast', 'table'])

/** A raindrop, so the wordmark reads as a mark rather than six capitals. */
function Mark() {
  return (
    <svg width="17" height="21" viewBox="0 0 17 21" aria-hidden="true">
      <defs>
        <linearGradient id="dropg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#8FD3F4" />
          <stop offset="100%" stopColor="#3358E8" />
        </linearGradient>
      </defs>
      <path d="M8.5 1.2C8.5 1.2 1.6 9.1 1.6 13.4a6.9 6.9 0 0 0 13.8 0C15.4 9.1 8.5 1.2 8.5 1.2Z"
            fill="url(#dropg)" />
      <path d="M5.6 13.6a2.9 2.9 0 0 0 2.9 2.9" fill="none"
            stroke="#FFFFFF" strokeOpacity=".75" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  )
}

export default function App() {
  const [tab, setTab] = useState<Tab>('forecast')
  const [names, setNames] = useState<Map<number, any>>(new Map())
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [geo, setGeo] = useState<GeoJSON.FeatureCollection | null>(null)
  const [dates, setDates] = useState<string[]>([])
  const [dateIdx, setDateIdx] = useState(0)
  const [lead, setLead] = useState(3)
  const [alpha, setAlpha] = useState(0.15)
  const [layer, setLayer] = useState<string>('colour')
  const [values, setValues] = useState<ForecastValue[]>([])
  const [selected, setSelected] = useState<number | null>(null)
  const [detail, setDetail] = useState<DistrictDetail | null>(null)
  const [missing, setMissing] = useState<string | null>(null)
  const [dateNote, setDateNote] = useState<string | null>(null)
  const [regime, setRegime] = useState<RegimeTimeline | null>(null)
  const [fading, setFading] = useState(false)
  const [playing, setPlaying] = useState(false)

  const date = dates[dateIdx]

  useEffect(() => {
    if (!playing || dates.length < 2) return
    const timer = window.setInterval(() => {
      setDateIdx((current) => {
        if (current >= dates.length - 1) {
          setPlaying(false)
          return current
        }
        return current + 1
      })
    }, 4200)
    return () => window.clearInterval(timer)
  }, [playing, dates.length])

  // boot: health, geometry, available dates
  useEffect(() => {
    (async () => {
      try {
        const [h, g, d] = await Promise.all([
          api.health(), api.districts(), api.dates(3),
        ])
        setHealth(h); setGeo(g); setDates(d.dates)
        const m = new Map<number, any>()
        for (const f of (g as any).features ?? []) {
          const pr = f.properties ?? {}
          m.set(Number(pr.district_id), {
            district_name: pr.DISTRICT ?? pr.district_name,
            state: pr.ST_NM ?? pr.state, zone_code: pr.zone_code,
          })
        }
        setNames(m)
        setDateIdx(Math.min(92, Math.max(0, d.dates.length - 1)))
      } catch (e) {
        setError(e instanceof ApiError ? e.message : String(e))
      }
    })()
  }, [])

  // forecast layer + regime for the current date/lead/alpha
  useEffect(() => {
    if (!date) return
    let cancelled = false
    setFading(true)
    ;(async () => {
      try {
        const [f, r] = await Promise.all([
          api.forecast(date, lead, layer, alpha),
          api.regime(date),
        ])
        if (cancelled) return
        setValues(f.values); setRegime(r); setError(null)
        setSelected((current) => current ??
          f.values.find((v) => v.colour_code === 3 && !/island/i.test(names.get(v.district_id)?.state ?? ''))?.district_id ??
          [...f.values].sort((a, b) => b.colour_code - a.colour_code)[0]?.district_id ?? null)
      } catch (e) {
        if (!cancelled) setError(e instanceof ApiError ? e.message : String(e))
      } finally {
        if (!cancelled) setTimeout(() => setFading(false), 60)
      }
    })()
    return () => { cancelled = true }
  }, [date, lead, layer, alpha])

  // district detail
  //
  // A failed lookup is NOT the same as nothing being selected. The map draws
  // all 641 districts and only the fetched ones have predictions, so clicking a
  // pale district used to redisplay "Select a district on the map" -- which
  // reads as a dead click rather than as an honest gap in the archive.
  useEffect(() => {
    if (!date || selected == null) { setDetail(null); setMissing(null); return }
    let cancelled = false
    api.district(selected, date, lead, alpha)
      .then((d) => { if (!cancelled) { setDetail(d); setMissing(null) } })
      .catch(() => {
        if (cancelled) return
        setDetail(null)
        setMissing(names.get(selected)?.district_name ?? `District ${selected}`)
      })
    return () => { cancelled = true }
  }, [selected, date, lead, alpha, names])

  const onSelect = useCallback((id: number) => setSelected(id), [])

  const counts = useMemo(() => {
    const c = [0, 0, 0, 0]
    for (const v of values) c[v.colour_code] = (c[v.colour_code] ?? 0) + 1
    return c
  }, [values])

  const step = (d: number) => {
    setPlaying(false)
    setDateNote(null)
    setDateIdx((i) => Math.min(Math.max(i + d, 0), Math.max(dates.length - 1, 0)))
  }

  /**
   * Jump to a typed or picked date.
   *
   * The archive is a list of specific days, not a continuous range -- a native
   * date input cannot grey out the gaps, and min/max only bound the ends. So a
   * date we do not hold snaps to the nearest one we do, and says so. Silently
   * showing a different day than the one asked for would be the one thing this
   * control must never do.
   */
  const pickDate = useCallback((iso: string) => {
    if (!iso || dates.length === 0) return
    setPlaying(false)
    const exact = dates.indexOf(iso)
    if (exact >= 0) { setDateIdx(exact); setDateNote(null); return }

    const want = Date.parse(iso)
    if (!isFinite(want)) return
    let best = 0
    let bestGap = Infinity
    dates.forEach((d, i) => {
      const gap = Math.abs(Date.parse(d) - want)
      if (gap < bestGap) { bestGap = gap; best = i }
    })
    setDateIdx(best)
    setDateNote(`${iso} is not in the archive — showing ${dates[best]}`)
  }, [dates])

  if (error && !values.length) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <div className="panel max-w-lg">
          <div className="panel-head"
               style={{ background: 'var(--amber-soft)', borderColor: 'var(--amber-line)' }}>
            <h1 className="text-[15px] font-semibold" style={{ color: 'var(--amber)' }}>
              Backend not reachable
            </h1>
          </div>
          <div className="p-4">
            <p className="mb-3 text-sm" style={{ color: 'var(--slate)' }}>{error}</p>
            <button className="btn mb-3 px-3 py-1.5 text-sm" onClick={() => window.location.reload()}>
              Try again
            </button>
            <pre className="whitespace-pre-wrap rounded-md p-3 text-xs"
                 style={{ background: 'var(--brand-950)', color: '#C9D6FF' }}>
              cd &lt;project root&gt;{'\n'}
              PYTHONPATH=src uvicorn raahat.api.main:app --port 8000
            </pre>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="app-shell flex h-full flex-col">
      {/* ------------------------------------------------------ app bar --- */}
      <header className="masthead shrink-0">
        <div className="masthead-main flex flex-wrap items-center gap-x-4 gap-y-2">
          <div className="brand-lockup flex items-center gap-3">
            <span className="brand-mark"><Mark /></span>
            <div>
              <span className="brand-name">RAAHAT<span className="brand-period">.</span></span>
              <span className="brand-subtitle">Rainfall intelligence for every district</span>
            </div>
          </div>

          {/* spec 10.3: never imply live capability we do not have */}
          <span className="archive-status">
            <span className="status-pulse" />
            Archive replay <span className="status-date">{date ?? 'Loading…'}</span>
            <span className="status-qualifier">Historical data</span>
          </span>

          <div className="warning-summary ml-auto flex items-center gap-2" aria-label="District warning counts">
            <span className="warning-summary-label">DISTRICT OUTLOOK</span>
            {counts.map((n, i) => (
              <span key={i} className="warning-count" title={`${n} districts: ${COLOUR_ACTION[i as 0 | 1 | 2 | 3]}`}
                    style={{
                      background: n > 0 ? COLOUR_HEX[i as 0 | 1 | 2 | 3] + '2E' : 'rgba(255,255,255,.05)',
                      border: '1px solid ' +
                        (n > 0 ? COLOUR_HEX[i as 0 | 1 | 2 | 3] + '80' : 'rgba(255,255,255,.10)'),
                      color: n > 0 ? '#FFFFFF' : 'var(--on-brand-dim)',
                    }}>
                <span className="h-2.5 w-2.5 rounded-[3px]"
                      style={{ background: COLOUR_HEX[i as 0 | 1 | 2 | 3] }} />
                <span className="num">{n}</span>
              </span>
            ))}
          </div>
        </div>

        <nav className="app-nav flex gap-0.5 overflow-x-auto" aria-label="Main navigation">
          {TABS.map((t) => {
            const on = t.id === tab
            return (
              <button key={t.id} onClick={() => setTab(t.id)}
                      className={'tab whitespace-nowrap rounded-t-md px-3 py-2 text-[13px]' + (on ? ' on' : '')}
                      style={{
                        color: on ? '#FFFFFF' : 'var(--on-brand-dim)',
                        fontWeight: on ? 600 : 450,
                        background: on ? 'rgba(255,255,255,.10)' : undefined,
                        boxShadow: on ? 'inset 0 -2.5px 0 var(--accent-lift)' : undefined,
                      }}>
                {t.label}
              </button>
            )
          })}
        </nav>
      </header>

      {/* ------------------------------------------------- control bar --- */}
      {CONTROLLED.has(tab) && (
        <div className="control-deck flex shrink-0 flex-wrap items-center gap-x-6 gap-y-2.5">
          <div className="control-group flex items-center gap-2">
            <span className="text-xs" style={{ color: 'var(--mist)' }}>Lead</span>
            <div className="seg flex">
              {[1, 2, 3, 4, 5].map((l) => (
                <button key={l} onClick={() => setLead(l)}
                        className={'px-2.5 py-1 text-xs' + (l === lead ? ' on' : '')}
                        style={{
                          background: l === lead ? 'var(--accent)' : 'transparent',
                          color: l === lead ? 'white' : 'var(--slate)',
                          fontWeight: l === lead ? 600 : 450,
                          borderLeft: l > 1 ? '1px solid var(--rule)' : undefined,
                        }}>
                  {l}
                </button>
              ))}
            </div>
            <span className="text-xs" style={{ color: 'var(--mist)' }}>
              day{lead > 1 ? 's' : ''} ahead
            </span>
          </div>

          <div className="control-group flex items-center gap-2.5">
            <span className="text-xs" style={{ color: 'var(--mist)' }}>Caution</span>
            <input type="range" min={0.05} max={0.6} step={0.05} value={alpha}
                   onChange={(e) => setAlpha(Number(e.target.value))} className="w-24" />
            <span className="num w-8 text-xs font-semibold" style={{ color: 'var(--accent-deep)' }}>
              {alpha.toFixed(2)}
            </span>
            <span className="chip" style={{ background: 'var(--accent-soft)', color: 'var(--accent-deep)' }}>
              {alpha <= 0.15 ? 'warn early' : alpha >= 0.4 ? 'warn late' : 'balanced'}
            </span>
          </div>

          <div className="control-group date-control flex min-w-[340px] flex-1 items-center gap-2">
            <span className="text-xs" style={{ color: 'var(--mist)' }}>Date</span>
            <input type="date" aria-label="forecast date"
                   className="field num px-2 py-1 text-xs"
                   value={date ?? ''}
                   min={dates[0] ?? undefined}
                   max={dates[dates.length - 1] ?? undefined}
                   onChange={(e) => pickDate(e.target.value)} />
            <button onClick={() => step(-1)} disabled={dateIdx <= 0}
                    className="seg px-2 py-0.5 text-xs disabled:opacity-40"
                    style={{ color: 'var(--slate)' }} aria-label="previous day">&#8249;</button>
            <button onClick={() => { if (dateIdx >= dates.length - 1) setDateIdx(0); setPlaying(!playing) }}
                    disabled={dates.length < 2} className="play-control" aria-label={playing ? 'Pause replay' : 'Play archive replay'}
                    title={playing ? 'Pause replay' : 'Play archive replay'}>{playing ? 'Ⅱ' : '▶'}</button>
            <input type="range" min={0} max={Math.max(dates.length - 1, 0)} value={dateIdx}
                   aria-label="scrub through the archive"
                   onChange={(e) => { setDateNote(null); setDateIdx(Number(e.target.value)) }}
                   className="min-w-[90px] flex-1" />
            <button onClick={() => step(1)} disabled={dateIdx >= dates.length - 1}
                    className="seg px-2 py-0.5 text-xs disabled:opacity-40"
                    style={{ color: 'var(--slate)' }} aria-label="next day">&#8250;</button>
            <span className="num shrink-0 text-[11px]" style={{ color: 'var(--mist)' }}>
              {dates.length ? `${dateIdx + 1} / ${dates.length}` : '—'}
            </span>
          </div>

          {dateNote && (
            <span className="chip"
                  style={{ background: 'var(--amber-soft)', color: 'var(--amber)' }}>
              {dateNote}
            </span>
          )}

          <div className="seg layer-control flex">
            {LAYERS.map((l, i) => (
              <button key={l.id} onClick={() => setLayer(l.id)}
                      className={'px-2.5 py-1 text-xs' + (l.id === layer ? ' on' : '')}
                      style={{
                        background: l.id === layer ? 'var(--accent-deep)' : 'transparent',
                        color: l.id === layer ? 'white' : 'var(--slate)',
                        fontWeight: l.id === layer ? 600 : 450,
                        borderLeft: i > 0 ? '1px solid var(--rule)' : undefined,
                      }}>
                {l.label}
              </button>
            ))}
          </div>
        </div>
      )}

      <main className="workspace flex min-h-0 flex-1">
        {tab === 'forecast' && (
          <>
            <div className="map-stage min-w-0 flex-1">
              <DistrictMap geojson={geo} values={values} selected={selected}
                           onSelect={onSelect} fading={fading} date={date} lead={lead} />
            </div>
            <aside className="insight-rail w-[25rem] shrink-0 overflow-y-auto">
              <div className="rail-intro">
                <span className="rail-kicker">FORECAST INTELLIGENCE / {date ?? '—'}</span>
                <h1>Read the rain<span>.</span><br />Act ahead.</h1>
                <p>District-level outlook, corrected for the weather situation.</p>
              </div>
              <Card title="Weather situation" tone="teal"
                    right={<span className="chip" style={{ background: 'var(--sunk)', color: 'var(--slate)' }}>
                      {lead} day{lead > 1 ? 's' : ''} ahead</span>}>
                <RegimePanel data={regime} lead={lead} />
              </Card>
              <div style={{ borderTop: '1px solid var(--rule)' }} />
              <Card title="District" tone="accent">
                <DistrictPanel d={detail} missing={missing} />
              </Card>
            </aside>
          </>
        )}
        {tab === 'table' && (
          <div className="min-h-0 flex-1">
            <TableScreen values={values} names={names} date={date ?? ''} lead={lead}
                         layer={layer}
                         layerLabel={LAYERS.find((l) => l.id === layer)?.label ?? 'Value'} />
          </div>
        )}
        {tab === 'atlas' && <div className="min-h-0 flex-1 overflow-auto"><AtlasScreen /></div>}
        {tab === 'evidence' && <div className="min-h-0 flex-1 overflow-auto"><EvidenceScreen /></div>}
        {tab === 'about' && <div className="min-h-0 flex-1 overflow-auto"><AboutScreen /></div>}
      </main>

      <footer className="app-footer flex shrink-0 flex-wrap items-center gap-x-2 px-4 py-1.5 text-[11px]"
              style={{ background: 'var(--brand-950)', color: 'var(--on-brand-dim)' }}>
        <span>
          {health?.attribution?.join(' · ') ??
            'Weather data by Open-Meteo.com (CC BY 4.0) · Observations: IMD Pune'}
        </span>
        {health?.model_version && (
          <span className="chip ml-auto"
                style={{ background: 'rgba(127,166,255,.14)', color: '#BFCEFF' }}>
            model {health.model_version}
          </span>
        )}
      </footer>
    </div>
  )
}
