/**
 * District choropleth (spec 9.3).
 *
 * MapLibre with no basemap tiles: the districts ARE the map. A satellite or
 * street basemap would add colour competing with the four warning colours,
 * which spec 9.2 reserves exclusively for warning state. It also means the map
 * needs no network at all, which matters for spec 10.3 (the demo runs with the
 * cable unplugged).
 *
 * The map instance lives in STATE, not a ref. React StrictMode mounts,
 * unmounts and remounts in development, so the map is created twice. With a
 * ref, the effect that adds the district layers does not re-run for the second
 * map -- its declared dependencies never changed -- and you get a permanently
 * blank map with no error. Putting the instance in state makes it a real
 * dependency, so every dependent effect re-runs against the live map.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { COLOUR_HEX, type ForecastValue } from '../api/client'

interface Props {
  geojson: GeoJSON.FeatureCollection | null
  values: ForecastValue[]
  selected: number | null
  onSelect: (id: number) => void
  fading: boolean
  date?: string
  lead: number
  layer: string
}

/** A district we hold no prediction for. Deliberately lighter than the sea. */
const NO_DATA = '#EFF3F7'

const RAIN = ['#c9e1ee', '#a0cde3', '#79b9da', '#418fb9', '#20678f', '#124969']
const PROB = ['#c9e1ee', '#a4d0e6', '#77b7d9', '#4e9dc6', '#155d8c']
const CHANGE = ['#a4612a', '#d3a377', '#f1e7dc', '#dcecf3', '#83bddc', '#1b729e']

const legendFor = (layer: string) => {
  if (layer === 'corrected' || layer === 'raw') return {
    title: `${layer === 'raw' ? 'Raw' : 'Corrected'} rainfall · mm`,
    items: ['<10', '10–25', '25–50', '50–100', '100–200', '≥200'].map((label, i) => ({ label, colour: RAIN[i] })),
  }
  if (layer === 'delta') return {
    title: 'Correction from raw · mm',
    items: ['<−20', '−20 to −5', 'near 0', '+5 to +20', '+20 to +50', '≥+50'].map((label, i) => ({ label, colour: CHANGE[i] })),
  }
  if (layer === 'p64') return {
    title: 'Chance of >64.5 mm',
    items: ['<10%', '10–25%', '25–50%', '50–75%', '≥75%'].map((label, i) => ({ label, colour: PROB[i] })),
  }
  return {
    title: 'Suggested warning',
    items: ['No warning', 'Be updated', 'Be prepared', 'Take action'].map((label, i) => ({ label, colour: COLOUR_HEX[i as 0 | 1 | 2 | 3] })),
  }
}

function colourFor(v: ForecastValue, layer: string) {
  if (layer === 'colour') return COLOUR_HEX[v.colour_code] ?? NO_DATA
  const x = v.value
  if (x == null || !Number.isFinite(x)) return NO_DATA
  if (layer === 'corrected' || layer === 'raw')
    return RAIN[x < 10 ? 0 : x < 25 ? 1 : x < 50 ? 2 : x < 100 ? 3 : x < 200 ? 4 : 5]
  if (layer === 'p64')
    return PROB[x < .1 ? 0 : x < .25 ? 1 : x < .5 ? 2 : x < .75 ? 3 : 4]
  return CHANGE[x < -20 ? 0 : x < -5 ? 1 : x < 5 ? 2 : x < 20 ? 3 : x < 50 ? 4 : 5]
}

export function DistrictMap({ geojson, values, selected, onSelect, fading, date, lead, layer }: Props) {
  const boxRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const paintedIds = useRef<Set<number>>(new Set())
  // A COUNTER, not a boolean. React StrictMode creates the map, tears it down
  // and creates another; effects that add layers must re-run for whichever map
  // is actually alive. Keying them off the map object in a ref does not work --
  // a ref change never re-runs an effect -- so readiness is state, and it
  // increments so that a second map with the same "true" value still fires.
  const [ready, setReady] = useState(0)
  //: true once the district source and layers exist on the live map
  const [layered, setLayered] = useState(false)
  //: the layers never attached within the retry window -- say so rather than
  //  showing an empty sea that reads as "no districts have any data"
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    const el = boxRef.current
    if (!el) return
    const m = new maplibregl.Map({
      container: el,
      // no `glyphs` key at all: MapLibre validates it as a string and rejects
      // an explicit undefined. We draw no labels, so it is not needed.
      //
      // The one background layer is a flat, pale blue -- not a basemap. Without
      // it the sea is the page background and the country looks like a shape
      // floating on a form. It is far too pale to compete with a warning
      // colour, which is what spec 9.2 is actually protecting against.
      //
      // It must stay clearly DARKER than NO_DATA below. A district we have not
      // fetched yet and the sea are different facts, and at the first attempt
      // the two colours were four points apart, so the un-fetched half of the
      // country simply looked like ocean.
      style: {
        version: 8, sources: {},
        layers: [{ id: 'sea', type: 'background', paint: { 'background-color': '#D8E6F0' } }],
      },
      center: [80, 22],
      zoom: 3.6,
      attributionControl: false,
      dragRotate: false,
    })
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right')
    mapRef.current = m
    const onLoad = () => setReady((n) => n + 1)
    m.on('load', onLoad)
    if (m.loaded()) onLoad()
    return () => {
      mapRef.current = null
      setLayered(false)
      m.remove()
    }
  }, [])

  // add the district polygons once both the map and the geometry exist
  useEffect(() => {
    const map = mapRef.current
    // Deliberately NOT gated on the load event. MapLibre's 'load' does not
    // always fire before the geometry arrives, and gating on it left the map
    // intermittently blank with nothing in the console. Instead: try to add
    // immediately, and if the style is not ready yet the call throws and the
    // styledata listener retries. `ready` stays in the dependency list so a
    // freshly created map re-runs this.
    if (!map || !geojson) return
    let done = false

    // Returns true only when there is nothing left to do. Returning true when
    // the style was merely NOT READY YET was a real bug: `add()` is also the
    // signal for whether to install the retry listeners, so "come back later"
    // was being read as "finished", no retry was ever scheduled, and the map
    // stayed blank permanently. It survived earlier testing only because the
    // geometry usually arrived after the style had loaded; at a different
    // window size the order flipped and the map never drew.
    const add = () => {
      if (done) return true
      if (map.getSource('districts')) { done = true; setLayered(true); return true }
      // getStyle() returns undefined (or throws) until the style is applied.
      let style: unknown = null
      try { style = map.getStyle() } catch { style = null }
      if (!style) return false          // not ready -- MUST retry
      try {
        map.addSource('districts', {
          type: 'geojson', data: geojson, promoteId: 'district_id',
        })
        map.addLayer({
          id: 'fill', type: 'fill', source: 'districts',
          paint: {
            'fill-color': ['coalesce', ['feature-state', 'colour'], NO_DATA],
            'fill-opacity': 0.92,
          },
        })
        map.addLayer({
          id: 'outline', type: 'line', source: 'districts',
          paint: { 'line-color': '#FFFFFF', 'line-width': 0.4 },
        })
        map.addLayer({
          id: 'selected', type: 'line', source: 'districts',
          paint: { 'line-color': '#14181D', 'line-width': 2 },
          filter: ['==', ['get', 'district_id'], -1],
        })
      } catch {
        return false          // style not ready yet; the listener retries
      }

      map.on('click', 'fill', (e) => {
        const f = e.features?.[0]
        if (f) onSelect(Number(f.properties?.district_id))
      })
      map.on('mouseenter', 'fill', () => { map.getCanvas().style.cursor = 'pointer' })
      map.on('mouseleave', 'fill', () => { map.getCanvas().style.cursor = '' })

      // Fit against the actual measured panel, including after responsive
      // layout changes. Fitting before resize can crop the southern districts.
      map.resize()
      try {
        const b = new maplibregl.LngLatBounds()
        for (const f of geojson.features) {
          const g: any = f.geometry
          const rings = g?.type === 'Polygon' ? g.coordinates : g?.coordinates?.flat()
          for (const ring of rings ?? []) for (const c of ring) b.extend(c as [number, number])
        }
        if (!b.isEmpty()) map.fitBounds(b, {
          padding: { top: 30, right: 28, bottom: 125, left: 28 },
          duration: 850,
        })
      } catch { /* bounds are cosmetic; never let them blank the map */ }

      // a map created before its container had size renders nothing until
      // told to re-measure
      done = true
      setLayered(true)
      return true
    }

    if (add()) { setFailed(false); return }

    // Declared before retry(), which clears both: a listener that fired before
    // the assignment would otherwise hit the temporal dead zone.
    let timer: number | undefined
    let giveUp: number | undefined

    const retry = () => {
      if (!add()) return
      setFailed(false)
      map.off('styledata', retry)
      window.clearTimeout(timer)
      window.clearTimeout(giveUp)
    }
    map.on('styledata', retry)
    map.once('load', retry)

    // Last resort: MapLibre can settle without emitting anything we listen
    // for -- and when the window is occluded it may never run a frame, so the
    // 'load' event never arrives at all. Poll rather than leave a blank map.
    //
    // A self-rescheduling timeout rather than a fixed interval, because this
    // never gives up (see below) and a permanent 4-times-a-second timer for a
    // tab nobody is looking at is not something to leave running. It starts
    // responsive and settles to once every two seconds.
    let delay = 250
    const tick = () => {
      retry()
      if (done) return
      delay = Math.min(delay * 1.5, 2000)
      timer = window.setTimeout(tick, delay)
    }
    timer = window.setTimeout(tick, delay)

    // A browser that is not painting -- a background tab, an occluded window --
    // throttles requestAnimationFrame to nothing, and MapLibre cannot finish
    // its first render without a frame, so the style never reports loaded.
    // That is temporary: the moment the tab is looked at, frames resume. So
    // retry on visibility as well, and keep polling rather than giving up.
    const onVisible = () => { if (!document.hidden) retry() }
    document.addEventListener('visibilitychange', onVisible)

    // Spec 9.5: say what to DO. A blank map with no message is the worst
    // outcome, because it looks like the country simply has no data. The
    // message is advisory only -- polling continues underneath it, and it
    // disappears by itself if the layers do arrive.
    giveUp = window.setTimeout(() => setFailed(true), 15000)

    return () => {
      map.off('styledata', retry)
      map.off('load', retry)
      document.removeEventListener('visibilitychange', onVisible)
      window.clearTimeout(timer)
      window.clearTimeout(giveUp)
    }
  }, [ready, geojson, onSelect])

  // repaint on new values
  useEffect(() => {
    const map = mapRef.current
    if (!map || !layered) return
    const nextIds = new Set(values.map((v) => v.district_id))
    for (const id of paintedIds.current) {
      if (!nextIds.has(id)) map.setFeatureState({ source: 'districts', id }, { colour: NO_DATA })
    }
    for (const v of values) {
      map.setFeatureState(
        { source: 'districts', id: v.district_id },
        { colour: colourFor(v, layer) },
      )
    }
    paintedIds.current = nextIds
  }, [ready, layered, values, layer])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !layered) return
    map.setFilter('selected', ['==', ['get', 'district_id'], selected ?? -1])
  }, [ready, layered, selected])

  return (
    <div className="map-canvas relative h-full w-full">
      <div ref={boxRef} className="crossfade h-full w-full"
           style={{ opacity: fading ? 0.55 : 1 }} />

      <div className="map-heading" aria-hidden="true">
        <span className="map-heading-kicker"><span className="map-heading-dot" /> INDIA / DISTRICT FORECAST</span>
        <strong>Monsoon outlook</strong>
        <span>{date ?? 'Loading archive'} <span className="map-heading-divider">/</span> Day {lead} forecast</span>
      </div>

      <div className="map-instruction">Select a district to explore its outlook <span>↗</span></div>

      {failed && (
        <div className="panel absolute left-1/2 top-1/2 max-w-sm -translate-x-1/2 -translate-y-1/2 p-4">
          <h3 className="mb-1 text-sm font-semibold" style={{ color: 'var(--amber)' }}>
            The map did not draw
          </h3>
          <p className="text-xs leading-relaxed" style={{ color: 'var(--slate)' }}>
            District boundaries loaded but the map layer never attached. Reload the
            page. The District table tab has the same forecast as a sortable list
            and does not depend on the map.
          </p>
        </div>
      )}
      <div className="panel map-legend absolute bottom-3 left-3 px-3 py-2 text-xs"
           style={{ background: 'rgba(255,255,255,.96)', backdropFilter: 'blur(6px)' }}>
        <div className="mb-1.5 text-[11px] font-semibold" style={{ color: 'var(--accent-deep)' }}>
          {legendFor(layer).title}
        </div>
        <div className="flex gap-2">
          {legendFor(layer).items.map(({ label, colour }) => (
            <span key={label} className="chip"
                  style={{
                    background: colour + '25',
                    color: 'var(--ink)',
                  }}>
              <span className="inline-block h-2.5 w-2.5 rounded-[3px]"
                    style={{ background: colour }} />
              {label}
            </span>
          ))}
          <span className="chip" style={{ background: 'var(--sunk)', color: 'var(--mist)' }}>
            <span className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: NO_DATA }} />
            no data
          </span>
        </div>
      </div>
    </div>
  )
}
