/**
 * Secondary screens (spec 9.4): district table, evidence, atlas, about.
 *
 * Spec 9.4 is right that "the table is what a forecaster actually wants; the
 * map is what a judge wants" -- so both exist. The evidence screen deliberately
 * shows the rows where we did NOT improve, because spec 10.2 is correct that
 * pointing at your own weak cell buys more credibility than anything else.
 *
 * Colour here is structural, not decorative: each screen carries a tone so you
 * know which one you are on without reading the tab, warning state keeps the
 * four IMD colours, and every other value uses the one blue/brown ramp that
 * means "more rain / less rain" throughout the product.
 */
import { useEffect, useMemo, useState } from 'react'
import {
  COLOUR_ACTION, COLOUR_HEX, regimeHex, regimeLabel, type ForecastValue,
} from '../api/client'
import { Card, DOWN, UP } from './Panels'

const n2 = (v: unknown, d = 3) =>
  typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—'

/** Frozen reference views load from the CDN and fall back to the API locally. */
function useReferenceData(asset: string, endpoint: string, requiredArray: string) {
  const [data, setData] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    let cancelled = false
    setData(null)
    setError(null)
    const read = async (url: string) => {
      const controller = new AbortController()
      const timer = window.setTimeout(() => controller.abort(), 12000)
      try {
        const response = await fetch(url, { signal: controller.signal })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const body = await response.json()
        if (!Array.isArray(body?.[requiredArray]) || body[requiredArray].length === 0)
          throw new Error('Missing reference data')
        return body
      } finally { window.clearTimeout(timer) }
    }
    ;(async () => {
      try {
        let result
        try { result = await read(asset) }
        catch { result = await read(endpoint) }
        if (!cancelled) setData(result)
      } catch {
        if (!cancelled) setError('This reference view could not load. Check the connection and try again.')
      }
    })()
    return () => { cancelled = true }
  }, [asset, endpoint, requiredArray, attempt])
  return { data, error, retry: () => setAttempt((n) => n + 1) }
}

function ReferenceStatus({ error, retry }: { error: string | null; retry: () => void }) {
  if (!error) return <p className="p-6 text-sm" style={{ color: 'var(--mist)' }}>Loading reference data…</p>
  return (
    <div className="panel m-3 max-w-lg p-5">
      <h2 className="text-base font-semibold">Unable to load this view</h2>
      <p className="my-2 text-sm" style={{ color: 'var(--slate)' }}>{error}</p>
      <button className="btn px-3 py-1.5 text-sm" onClick={retry}>Try again</button>
    </div>
  )
}

/** Yellow is too light for white text; every other warning colour is not. */
const onWarning = (c: number) => (c === 1 ? '#3A2B00' : '#FFFFFF')

function WarningPill({ c }: { c: number }) {
  return (
    <span className="chip" style={{ background: COLOUR_HEX[c as 0 | 1 | 2 | 3], color: onWarning(c) }}>
      {COLOUR_ACTION[c as 0 | 1 | 2 | 3]}
    </span>
  )
}

function RegimeChip({ r }: { r: string | null }) {
  if (!r) return <span style={{ color: 'var(--mist)' }}>—</span>
  const hex = regimeHex(r)
  return (
    <span className="chip" style={{ background: hex + '1F', color: hex }}>
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: hex }} />
      {regimeLabel(r)}
    </span>
  )
}

/** A number worth looking at, with its label under it. */
function Stat({ label, value, tone = 'var(--accent-deep)', wash = 'var(--accent-soft)' }: {
  label: string; value: React.ReactNode; tone?: string; wash?: string
}) {
  return (
    <div className="rounded-lg px-3 py-2" style={{ background: wash }}>
      <div className="num text-[22px] font-semibold leading-tight" style={{ color: tone }}>{value}</div>
      <div className="text-[11px]" style={{ color: 'var(--slate)' }}>{label}</div>
    </div>
  )
}

/* ------------------------------------------------------ district table --- */

interface Row { district_id: number; name: string; state: string; zone: string
                value: number | null; colour: number; regime: string | null }

export function TableScreen({ values, names, date, lead, layer, layerLabel }: {
  values: ForecastValue[]
  names: Map<number, { district_name: string; state: string; zone_code: string }>
  date: string; lead: number; layer: string; layerLabel: string
}) {
  // On the Warning layer the API's `value` IS the colour code, so a numeric
  // column headed "Value" showed 2.0 / 1.0 / 0.0 beside a Warning column that
  // already said "Be prepared". Drop the column there and let the pill carry
  // it; on every other layer the number is the point of the table.
  const showValue = layer !== 'colour'
  const [sort, setSort] = useState<{ key: keyof Row; dir: 1 | -1 }>({ key: 'value', dir: -1 })
  const [filter, setFilter] = useState('')
  const [onlyWarned, setOnlyWarned] = useState(false)

  const rows: Row[] = useMemo(() => values.map((v) => {
    const m = names.get(v.district_id)
    return {
      district_id: v.district_id,
      name: m?.district_name ?? `District ${v.district_id}`,
      state: m?.state ?? '', zone: m?.zone_code ?? '',
      value: v.value, colour: v.colour_code, regime: v.regime_argmax,
    }
  }), [values, names])

  const counts = useMemo(() => {
    const c = [0, 0, 0, 0]
    for (const r of rows) c[r.colour] = (c[r.colour] ?? 0) + 1
    return c
  }, [rows])

  const shown = useMemo(() => {
    const q = filter.trim().toLowerCase()
    return rows
      .filter((r) => (!onlyWarned || r.colour > 0) &&
        (!q || r.name.toLowerCase().includes(q) || r.state.toLowerCase().includes(q)))
      .sort((a, b) => {
        const x = a[sort.key], y = b[sort.key]
        if (x == null) return 1
        if (y == null) return -1
        return (x > y ? 1 : x < y ? -1 : 0) * sort.dir
      })
  }, [rows, sort, filter, onlyWarned])

  const exportCsv = () => {
    const head = 'district_id,district,state,zone,value,warning,regime\n'
    const body = shown.map((r) =>
      `${r.district_id},"${r.name}","${r.state}",${r.zone},${r.value ?? ''},` +
      `${['GREEN', 'YELLOW', 'ORANGE', 'RED'][r.colour]},${r.regime ?? ''}`).join('\n')
    const url = URL.createObjectURL(new Blob([head + body], { type: 'text/csv' }))
    const a = document.createElement('a')
    a.href = url; a.download = `raahat_${date}_day${lead}.csv`; a.click()
    URL.revokeObjectURL(url)
  }

  // With the value column hidden, sorting by `value` and sorting by `colour`
  // are the same ordering -- so put the arrow where the reader can see it.
  const arrowKey: keyof Row = !showValue && sort.key === 'value' ? 'colour' : sort.key

  const th = (key: keyof Row, label: string, right = false) => (
    <th className={`cursor-pointer select-none px-2.5 py-2 ${right ? 'text-right' : 'text-left'}`}
        onClick={() => setSort((s) => ({ key, dir: s.key === key && s.dir === -1 ? 1 : -1 }))}
        style={{ color: arrowKey === key ? 'var(--accent-deep)' : 'var(--slate)' }}>
      {label}{arrowKey === key ? (sort.dir === -1 ? ' ▼' : ' ▲') : ''}
    </th>
  )

  return (
    <div className="flex h-full flex-col gap-3 p-3">
      <div className="grid shrink-0 grid-cols-2 gap-2 sm:grid-cols-5">
        <Stat label="districts shown" value={shown.length} />
        {[1, 2, 3].map((i) => (
          <Stat key={i} label={COLOUR_ACTION[i as 1 | 2 | 3].toLowerCase()} value={counts[i]}
                tone={COLOUR_HEX[i as 1 | 2 | 3]} wash={COLOUR_HEX[i as 1 | 2 | 3] + '14'} />
        ))}
        <Stat label="no warning" value={counts[0]} tone="var(--slate)" wash="var(--sunk)" />
      </div>

      <div className="flex shrink-0 flex-wrap items-center gap-3">
        <input value={filter} onChange={(e) => setFilter(e.target.value)}
               placeholder="Filter by district or state"
               className="field px-2.5 py-1.5 text-sm" style={{ width: 240 }} />
        <label className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--slate)' }}>
          <input type="checkbox" checked={onlyWarned}
                 onChange={(e) => setOnlyWarned(e.target.checked)} />
          Warned districts only
        </label>
        <button onClick={exportCsv} className="btn ml-auto px-3 py-1.5 text-sm">
          Export CSV
        </button>
      </div>

      <div className="panel min-h-0 flex-1 overflow-auto">
        <table className="grid-table w-full text-sm">
          <thead className="sticky top-0 z-10">
            <tr>
              {th('name', 'District')}{th('state', 'State')}{th('zone', 'Zone')}
              {showValue && th('value', layerLabel, true)}
              {th('regime', 'Situation')}{th('colour', 'Warning')}
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.district_id}>
                <td className="px-2.5 py-1.5 font-medium">{r.name}</td>
                <td className="px-2.5 py-1.5" style={{ color: 'var(--slate)' }}>{r.state}</td>
                <td className="px-2.5 py-1.5">
                  <span className="chip" style={{ background: 'var(--sunk)', color: 'var(--slate)' }}>
                    {r.zone || '—'}
                  </span>
                </td>
                {showValue && (
                  <td className="num px-2.5 py-1.5 text-right font-semibold"
                      style={{ color: r.value == null ? 'var(--mist)' : 'var(--accent-deep)' }}>
                    {r.value == null ? '—' : r.value.toFixed(1)}
                  </td>
                )}
                <td className="px-2.5 py-1.5"><RegimeChip r={r.regime} /></td>
                <td className="px-2.5 py-1.5"><WarningPill c={r.colour} /></td>
              </tr>
            ))}
            {shown.length === 0 && (
              <tr>
                <td colSpan={showValue ? 6 : 5} className="px-2.5 py-8 text-center text-sm"
                    style={{ color: 'var(--mist)' }}>
                  No districts match that filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/* ----------------------------------------------------------- evidence --- */

export function EvidenceScreen() {
  const { data, error, retry } = useReferenceData('/verification.json', '/api/v1/verification', 'scorecard')
  if (!data) return <ReferenceStatus error={error} retry={retry} />

  const lead3 = (data.scorecard ?? []).filter((r: any) => r.lead === 3)

  const headRow = (labels: [string, boolean][]) => (
    <tr>
      {labels.map(([l, right], i) => (
        <th key={i} className={`px-2 py-1.5 ${right ? 'text-right' : 'text-left'}`}>{l}</th>
      ))}
    </tr>
  )

  return (
    <div className="space-y-3 p-3">
      <Card panel tone="accent"
            title="Held-out scorecard — day 3, 168 heavy-rain events"
            note="Every configuration is scored on the same test block, opened once. The
                  highlighted row is the full system; the rows above it are what you get if
                  you take a piece of it away."
            right={<span className="chip" style={{ background: 'var(--card)', color: 'var(--slate)' }}>
              opened once · {data.manifest?.run_id ?? ''}</span>}>
        <table className="grid-table w-full text-sm">
          <thead>{headRow([['Configuration', false], ['RMSE', true], ['Corr', true],
                           ['Detection', true], ['False alarms', true], ['CSI', true]])}</thead>
          <tbody>
            {lead3.map((r: any) => {
              const best = String(r.config).startsWith('5')
              return (
                <tr key={r.config} style={{
                  background: best ? 'var(--accent-soft)' : undefined,
                  fontWeight: best ? 600 : 450,
                  color: best ? 'var(--accent-deep)' : undefined,
                  boxShadow: best ? 'inset 3px 0 0 var(--accent)' : undefined,
                }}>
                  <td className="px-2 py-1.5">{r.config}</td>
                  <td className="num px-2 py-1.5 text-right">{n2(r.rmse, 2)}</td>
                  <td className="num px-2 py-1.5 text-right">{n2(r.corr)}</td>
                  <td className="num px-2 py-1.5 text-right">{n2(r['POD@64.5'])}</td>
                  <td className="num px-2 py-1.5 text-right">{n2(r['FAR@64.5'])}</td>
                  <td className="num px-2 py-1.5 text-right">{n2(r['CSI@64.5'])}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </Card>

      <Card panel tone="teal" title="Per weather situation — where the regime step actually pays"
            note="A cell with too few events is reported as insufficient rather than given a
                  number that would not survive a second season.">
        <table className="grid-table w-full text-sm">
          <thead>{headRow([['Situation', false], ['Events', true], ['Raw', true],
                           ['Without regime', true], ['With regime', true]])}</thead>
          <tbody>
            {(data.per_regime ?? []).map((r: any) => (
              <tr key={r.regime}>
                <td className="px-2 py-1.5"><RegimeChip r={r.regime} /></td>
                <td className="num px-2 py-1.5 text-right" style={{ color: 'var(--slate)' }}>
                  {r.n_events}
                </td>
                {r.sufficient ? (
                  <>
                    <td className="num px-2 py-1.5 text-right">{n2(r.raw_csi)}</td>
                    <td className="num px-2 py-1.5 text-right">{n2(r['4 RAAHAT no-regime'])}</td>
                    <td className="num px-2 py-1.5 text-right font-semibold"
                        style={{ color: 'var(--accent-deep)' }}>{n2(r['5 RAAHAT full'])}</td>
                  </>
                ) : (
                  <td colSpan={3} className="px-2 py-1.5 text-right">
                    <span className="chip" style={{ background: 'var(--sunk)', color: 'var(--slate)' }}>
                      insufficient sample — not reported
                    </span>
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card panel tone="violet" title="Is 70% really 70%? — calibration on held-out data"
            note="If the probabilities are honest, the first two columns should be close.
                  The bar shows how far apart they actually are.">
        <table className="grid-table w-full text-sm">
          <thead>{headRow([['We said', false], ['Reality', true], ['', false], ['Days', true]])}</thead>
          <tbody>
            {(data.reliability ?? []).filter((r: any) => r.n > 0).map((r: any, i: number) => {
              const said = Number(r.forecast_prob) || 0
              const obs = Number(r.observed_freq) || 0
              return (
                <tr key={i}>
                  <td className="num px-2 py-1.5 font-medium">{n2(said)}</td>
                  <td className="num px-2 py-1.5 text-right font-semibold"
                      style={{ color: obs >= said ? UP : DOWN }}>{n2(obs)}</td>
                  <td className="px-2 py-1.5" style={{ width: '46%' }}>
                    <span className="relative block h-2 w-full rounded-full"
                          style={{ background: 'var(--sunk)' }}>
                      <span className="absolute h-2 rounded-full" style={{
                        left: `${Math.min(said, obs) * 100}%`,
                        width: `${Math.abs(obs - said) * 100}%`,
                        background: obs >= said ? 'var(--accent)' : 'var(--amber)',
                      }} />
                      <span className="absolute top-[-3px] h-3.5 w-[2px]"
                            style={{ left: `${said * 100}%`, background: 'var(--ink)' }} />
                    </span>
                  </td>
                  <td className="num px-2 py-1.5 text-right" style={{ color: 'var(--mist)' }}>{r.n}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <p className="mt-2 text-xs" style={{ color: 'var(--mist)' }}>
          The tick is what we forecast; the bar runs to what actually happened. Blue means
          it rained more often than we said, brown means less.
        </p>
      </Card>

      <Card panel tone="amber" title="What we are not claiming">
        <ul className="space-y-2 text-sm">
          {(data.caveats ?? []).map((c: string, i: number) => (
            <li key={i} className="flex gap-2.5">
              <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full"
                    style={{ background: 'var(--amber)' }} />
              <span>{c}</span>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  )
}

/* -------------------------------------------------------------- atlas --- */

// bias -> a cool/warm ramp that is deliberately NOT the warning palette.
// Blue = the raw forecast was too low, brown = too high. The same two colours
// carry the same meaning in the district panel's driver list.
const shade = (v: number) => {
  const t = Math.max(-1, Math.min(1, v / 12))
  return t < 0 ? `rgba(51, 88, 232, ${Math.abs(t) * 0.72 + 0.08})`
               : `rgba(169, 96, 15, ${t * 0.72 + 0.08})`
}

export function AtlasScreen() {
  const { data, error, retry } = useReferenceData('/atlas.json', '/api/v1/atlas', 'cells')
  if (!data) return <ReferenceStatus error={error} retry={retry} />

  const cells = (data.cells ?? []).filter((c: any) => c.lead_day === 3)
  const regimes = [...new Set(cells.map((c: any) => c.regime))] as string[]
  const zones = [...new Set(cells.map((c: any) => c.zone))].sort() as string[]
  const get = (r: string, z: string) => cells.find((c: any) => c.regime === r && c.zone === z)

  return (
    <div className="space-y-3 p-3">
      <Card panel tone="violet" title="Regime Error Atlas — how wrong the raw forecast is, day 3"
            note="Each cell is the raw forecast's average error in that weather situation and
                  region. This is a measurement, not a model — it is useful to a forecaster
                  who never runs our system."
            right={
              <span className="flex items-center gap-2 text-[11px]" style={{ color: 'var(--slate)' }}>
                under-forecast
                <span className="h-2.5 w-24 rounded-full" style={{
                  background: 'linear-gradient(90deg, rgba(51,88,232,.8) 0%,' +
                              ' rgba(255,255,255,.9) 50%, rgba(169,96,15,.8) 100%)',
                  border: '1px solid var(--rule)',
                }} />
                over-forecast
              </span>
            }>
        <div className="overflow-x-auto">
          <table className="text-sm" style={{ borderSpacing: '3px', borderCollapse: 'separate' }}>
            <thead>
              <tr>
                <th className="px-2.5 py-1.5 text-left text-xs font-semibold"
                    style={{ color: 'var(--slate)' }}>Situation</th>
                {zones.map((z) => (
                  <th key={z} className="px-2.5 py-1.5 text-center text-xs font-semibold"
                      style={{ color: 'var(--slate)' }}>{z}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {regimes.map((r) => (
                <tr key={r}>
                  <td className="py-1 pr-3"><RegimeChip r={r} /></td>
                  {zones.map((z) => {
                    const c = get(r, z)
                    if (!c) {
                      return <td key={z} className="px-2 py-1.5 text-center"
                                 style={{ color: 'var(--mist)' }}>—</td>
                    }
                    if (!c.sufficient) {
                      return (
                        <td key={z} className="rounded-md px-2 py-1.5 text-center text-xs"
                            style={{ color: 'var(--mist)', background: 'var(--sunk)' }}>
                          n={c.n_events}
                        </td>
                      )
                    }
                    return (
                      <td key={z} className="num rounded-md px-2 py-1.5 text-center"
                          style={{
                            background: shade(c.raw_bias),
                            color: c.raw_bias >= 0 ? DOWN : UP,
                            fontWeight: 600,
                          }}>
                        <div>{c.raw_bias >= 0 ? '+' : ''}{n2(c.raw_bias, 1)} mm</div>
                        <div className="text-[10px] font-normal" style={{ color: 'var(--slate)' }}>
                          n={c.n_events}
                        </div>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs" style={{ color: 'var(--mist)' }}>{data.note}</p>
      </Card>
    </div>
  )
}

/* -------------------------------------------------------------- about --- */

export function AboutScreen() {
  const { data, error, retry } = useReferenceData('/about.json', '/api/v1/about', 'sources')
  if (!data) return <ReferenceStatus error={error} retry={retry} />
  return (
    <div className="space-y-3 p-3">
      <Card panel tone="teal" title="Where the data comes from">
        <table className="grid-table w-full text-sm">
          <tbody>
            {data.sources.map((s: any, i: number) => (
              <tr key={i}>
                <td className="py-2.5 pr-3 align-top font-semibold"
                    style={{ width: '22%', color: 'var(--accent-deep)' }}>{s.what}</td>
                <td className="py-2.5 pr-3 align-top">{s.source}
                  <div className="text-xs" style={{ color: 'var(--mist)' }}>{s.detail}</div>
                </td>
                <td className="py-2.5 align-top" style={{ width: '18%' }}>
                  <span className="chip" style={{ background: 'var(--teal-soft)', color: '#0B6E6E' }}>
                    {s.licence}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card panel tone="accent" title="Sample size — the honest number">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {Object.entries(data.sample_size).map(([k, v]) => (
            <Stat key={k} label={k.replace(/_/g, ' ')} value={String(v)} />
          ))}
        </div>
      </Card>

      <Card panel tone="amber" title="What we deliberately do not use">
        <ul className="space-y-2.5 text-sm">
          {data.not_used.map((x: any, i: number) => (
            <li key={i} className="flex gap-2.5">
              <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full"
                    style={{ background: 'var(--amber)' }} />
              <span>
                <span className="font-semibold">{x.what}</span>
                <div style={{ color: 'var(--slate)' }}>{x.why}</div>
              </span>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  )
}
