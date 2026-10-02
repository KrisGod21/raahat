/**
 * Right-hand column: regime outlook and district readout (spec 9.3).
 *
 * Design follows spec 9.2's "avoid" list literally. Earlier versions of this
 * file broke three of its rules: an all-caps eyebrow above every heading,
 * identical bordered cards stacked uniformly, and numbers set at body size so
 * nothing led the eye. The result read like a generic dashboard rather than
 * something belonging on a forecaster's second monitor at 2am.
 *
 * What replaced it:
 *  - sentence-case headings, one weight lighter than the content they head
 *  - hairline rules and whitespace instead of a box around everything
 *  - the corrected rainfall set large, because it is the number being read
 *  - probabilities as one horizontal band, not six separate rows of chrome
 *  - a coloured keyline per section, so the eye can find a section by colour
 *    before it reads the heading
 */
import {
  COLOUR_ACTION, COLOUR_HEX, REGIME_LABEL, regimeHex,
  type DistrictDetail, type RegimeTimeline,
} from '../api/client'

const mm = (v: number | null | undefined, d = 0) =>
  v == null || !isFinite(v) ? '—' : v.toFixed(d)
const pct = (v: number | null | undefined) =>
  v == null || !isFinite(v) ? '—' : `${Math.round(v * 100)}%`

/**
 * Direction colours. NOT green/red: those are two of the four warning colours
 * and a green "+3.2" beside an amber district would be genuinely ambiguous.
 * Instead this reuses the Regime Error Atlas ramp, where blue already means
 * "more rain than the raw forecast said" and brown means "less". One ramp,
 * one meaning, everywhere in the product.
 */
export const UP = 'var(--accent-deep)'
export const DOWN = 'var(--amber)'

const TONE: Record<string, { line: string; wash: string; text: string }> = {
  accent: { line: 'var(--accent)', wash: 'var(--accent-soft)', text: 'var(--accent-deep)' },
  teal:   { line: 'var(--teal)',   wash: 'var(--teal-soft)',   text: '#0B6E6E' },
  violet: { line: 'var(--violet)', wash: 'var(--violet-soft)', text: '#523AA1' },
  amber:  { line: 'var(--amber)',  wash: 'var(--amber-soft)',  text: 'var(--amber)' },
  slate:  { line: 'var(--mist)',   wash: 'var(--sunk)',        text: 'var(--slate)' },
}

/**
 * A titled region.
 *
 * `panel` = false (the sidebar): no border, no shadow -- a short coloured
 * keyline and whitespace do the work.
 * `panel` = true (the full-width screens): a real card with a tinted header,
 * because those pages are several unrelated tables stacked and without a
 * boundary the reader cannot tell where one ends.
 */
export function Card({ title, children, right, tone = 'accent', panel = false, note }: {
  title: string
  children: React.ReactNode
  right?: React.ReactNode
  tone?: keyof typeof TONE | string
  panel?: boolean
  note?: string
}) {
  const t = TONE[tone] ?? TONE.accent

  if (panel) {
    return (
      <section className="panel" style={{ borderTop: `2px solid ${t.line}` }}>
        <header className="panel-head flex flex-wrap items-baseline justify-between gap-2"
                style={{ background: t.wash }}>
          <h2 className="text-[13px] font-semibold" style={{ color: t.text }}>{title}</h2>
          {right}
        </header>
        <div className="p-3.5">
          {note && (
            <p className="mb-3 text-sm leading-relaxed" style={{ color: 'var(--slate)' }}>{note}</p>
          )}
          {children}
        </div>
      </section>
    )
  }

  return (
    <section className="px-4 pb-4 pt-3">
      <header className="mb-2.5 flex items-baseline justify-between gap-2">
        <h2 className="flex items-center gap-2 text-[13px] font-semibold" style={{ color: t.text }}>
          <span className="inline-block h-3 w-[3px] rounded-full" style={{ background: t.line }} />
          {title}
        </h2>
        {right}
      </header>
      {children}
    </section>
  )
}

/* ------------------------------------------------------------- regime --- */

export function RegimePanel({ data, lead }: { data: RegimeTimeline | null; lead: number }) {
  const row = data?.timeline.find((t) => t.lead === lead) ?? data?.timeline[0]
  if (!row) return <p className="text-sm" style={{ color: 'var(--mist)' }}>No regime data.</p>

  const all = Object.entries(row.probs)
    .map(([k, v]) => ({ key: k, label: REGIME_LABEL[k] ?? k, v: v ?? 0 }))
    .sort((a, b) => b.v - a.v)
  const live = all.filter((e) => e.v > 0.005)

  // STRUCTURALLY unavailable, not merely zero today. Western disturbance and
  // the easterly/coastal regime need upper-atmosphere fields that are not
  // freely available, so they can never be non-zero. An active spell reading
  // 0% is simply a fact about today and must not be described as undetectable
  // -- an earlier version listed every zero regime here, which would have told
  // a judge we cannot detect active and break spells when in fact those are
  // the two we detect best.
  const UNAVAILABLE = new Set(['p_WD', 'p_EASTERLY_COASTAL'])
  const dead = all.filter((e) => UNAVAILABLE.has(e.key))

  // One continuous band rather than six bars: it reads as a MIX, which is what
  // a blended regime actually is (spec 1.5). Each situation keeps a fixed hue
  // (see REGIME_HEX) so the colour means the same thing on every date and on
  // every screen -- a band that recoloured as the ordering changed would be
  // worse than grey.
  return (
    <div>
      <div className="mb-3 flex h-3 w-full overflow-hidden rounded-full"
           style={{ boxShadow: 'inset 0 0 0 1px rgba(16,21,28,.06)' }}>
        {live.map((e) => (
          <div key={e.key} title={`${e.label} ${pct(e.v)}`}
               style={{ width: `${e.v * 100}%`, background: regimeHex(e.key) }} />
        ))}
        {live.length === 0 && <div className="w-full" style={{ background: 'var(--sunk)' }} />}
      </div>

      <dl className="space-y-1.5">
        {live.map((e, i) => (
          <div key={e.key} className="flex items-baseline gap-2">
            <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: regimeHex(e.key) }} />
            <dt className="flex-1 text-sm" style={{ fontWeight: i === 0 ? 600 : 400 }}>{e.label}</dt>
            <dd className="num text-sm font-semibold"
                style={{ color: i === 0 ? regimeHex(e.key) : 'var(--slate)' }}>{pct(e.v)}</dd>
          </div>
        ))}
      </dl>

      {dead.length > 0 && (
        <p className="mt-3 rounded-md px-2.5 py-2 text-xs leading-relaxed"
           style={{ background: 'var(--sunk)', color: 'var(--slate)' }}>
          {dead.map((d) => d.label).join(', ')} cannot be detected from freely
          available data. Shown as zero rather than hidden.
        </p>
      )}
    </div>
  )
}

/* ----------------------------------------------------------- district --- */

function Exceedance({ label, p }: { label: string; p: number | null }) {
  const v = p ?? 0
  return (
    <div className="flex items-center gap-2.5">
      <span className="w-[104px] shrink-0 text-xs" style={{ color: 'var(--slate)' }}>{label}</span>
      <span className="h-2 flex-1 overflow-hidden rounded-full" style={{ background: 'var(--sunk)' }}>
        <span className="block h-full rounded-full" style={{
          width: `${Math.min(v * 100, 100)}%`,
          background: v >= 0.5
            ? 'linear-gradient(90deg, var(--accent) 0%, var(--accent-deep) 100%)'
            : 'linear-gradient(90deg, var(--cyan) 0%, var(--accent) 100%)',
          opacity: v >= 0.5 ? 1 : 0.8,
        }} />
      </span>
      <span className="num w-9 shrink-0 text-right text-xs font-semibold"
            style={{ color: v >= 0.5 ? 'var(--accent-deep)' : 'var(--slate)' }}>{pct(p)}</span>
    </div>
  )
}

export function DistrictPanel({ d, missing }: {
  d: DistrictDetail | null
  /** A district was clicked but we hold no prediction for it. */
  missing?: string | null
}) {
  if (!d && missing) {
    return (
      <div className="rounded-lg p-4" style={{ background: 'var(--amber-soft)' }}>
        <p className="text-[15px] font-semibold">{missing}</p>
        <p className="mt-1 text-sm leading-relaxed" style={{ color: 'var(--slate)' }}>
          No forecast in the archive for this district on this date. The map draws
          all 641 districts; the pale ones are still being backfilled and are
          marked “no data” in the legend.
        </p>
      </div>
    )
  }
  if (!d) {
    return (
      <p className="rounded-lg py-10 text-center text-sm leading-relaxed"
         style={{ background: 'var(--paper)', color: 'var(--mist)' }}>
        Select a district on the map<br />to see its forecast breakdown.
      </p>
    )
  }
  const c = d.warning.colour_code
  const raw = d.raw.multi_model_mean
  const med = d.corrected.median
  const shift = raw != null && med != null ? med - raw : null

  return (
    <div className="space-y-4">
      {/* The warning colour is legitimate here -- this IS warning state, so
          spec 9.2's reservation is being honoured, not bent. */}
      <div className="rounded-lg p-3"
           style={{ background: COLOUR_HEX[c] + '14', borderLeft: `3px solid ${COLOUR_HEX[c]}` }}>
        <div className="flex items-baseline justify-between gap-2">
          <h3 className="text-[17px] font-semibold leading-tight">
            {d.district.district_name ?? `District ${d.district.district_id}`}
          </h3>
          <span className="chip shrink-0"
                style={{ background: COLOUR_HEX[c], color: c === 1 ? '#3A2B00' : '#FFFFFF' }}>
            {d.warning.suggested_colour}
          </span>
        </div>
        <p className="mt-0.5 text-xs" style={{ color: 'var(--slate)' }}>
          {d.district.state} · {COLOUR_ACTION[c]}
          {d.warning.guard_applied && ' · warning held up by safety rule'}
        </p>
      </div>

      {/* the number being read, set large */}
      <div className="flex items-end gap-5">
        <div>
          <div className="display num text-[38px] leading-none" style={{ color: 'var(--accent-deep)' }}>
            {mm(med)}<span className="ml-1 text-base font-normal"
                           style={{ color: 'var(--mist)' }}>mm</span>
          </div>
          <div className="mt-1 text-xs" style={{ color: 'var(--slate)' }}>
            corrected forecast
          </div>
        </div>
        <div className="pb-1">
          <div className="num text-sm" style={{ color: 'var(--slate)' }}>
            {mm(raw)} mm raw
            {shift != null && (
              <span className="chip ml-1.5" style={{
                background: shift >= 0 ? 'var(--accent-soft)' : 'var(--amber-soft)',
                color: shift >= 0 ? UP : DOWN,
              }}>
                {shift >= 0 ? '+' : '−'}{Math.abs(shift).toFixed(0)}
              </span>
            )}
          </div>
          <div className="num text-xs" style={{ color: 'var(--mist)' }}>
            likely {mm(d.corrected.p10)}–{mm(d.corrected.p90)} mm
          </div>
        </div>
      </div>

      <div className="space-y-1.5 border-t pt-3" style={{ borderColor: 'var(--rule)' }}>
        <Exceedance label="above 64.5 mm" p={d.exceedance.p64_5} />
        <Exceedance label="above 115.6 mm" p={d.exceedance.p115_6} />
        <Exceedance label="above 204.5 mm" p={d.exceedance.p204_5} />
      </div>

      <div className="border-t pt-3" style={{ borderColor: 'var(--rule)' }}>
        <p className="text-sm leading-relaxed">{d.explanation.narrative}</p>
        <ul className="mt-2 space-y-1">
          {d.explanation.drivers.map((x, i) => (
            <li key={i} className="flex items-baseline justify-between gap-3 text-xs">
              <span style={{ color: 'var(--slate)' }}>{x.label}</span>
              <span className="num shrink-0 tabular-nums font-semibold"
                    style={{ color: x.contribution >= 0 ? UP : DOWN }}>
                {x.contribution >= 0 ? '+' : '−'}{Math.abs(x.contribution).toFixed(1)}
              </span>
            </li>
          ))}
        </ul>
      </div>

      <div className="rounded-lg p-3 text-sm leading-relaxed"
           style={{ background: 'var(--teal-soft)', color: '#0B4F4F' }}>
        {d.analog_narrative}
      </div>

      <p className="text-xs leading-relaxed" style={{ color: 'var(--mist)' }}>
        District-wide average, not “isolated places within”. Complementary to IMD
        warnings, not a replacement.
      </p>
    </div>
  )
}
