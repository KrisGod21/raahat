import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const publicDir = resolve(root, 'frontend/public')
const fromRoot = (path) => resolve(root, path)
const readJson = (path) => JSON.parse(readFileSync(fromRoot(path), 'utf8'))

function parseCsv(path) {
  const source = readFileSync(fromRoot(path), 'utf8').replace(/^\uFEFF/, '')
  const lines = []
  let row = []
  let field = ''
  let quoted = false
  for (let i = 0; i < source.length; i++) {
    const char = source[i]
    if (char === '"') {
      if (quoted && source[i + 1] === '"') { field += '"'; i++ }
      else quoted = !quoted
    } else if (char === ',' && !quoted) {
      row.push(field); field = ''
    } else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && source[i + 1] === '\n') i++
      row.push(field); field = ''
      if (row.some((cell) => cell !== '')) lines.push(row)
      row = []
    } else field += char
  }
  if (field !== '' || row.length) { row.push(field); lines.push(row) }
  const [headers, ...records] = lines
  const value = (text) => {
    if (text === '') return null
    if (text === 'True') return true
    if (text === 'False') return false
    if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(text)) return Number(text)
    return text
  }
  return records.map((record) => Object.fromEntries(headers.map((header, index) => [header, value(record[index] ?? '')])))
}

mkdirSync(publicDir, { recursive: true })
copyFileSync(fromRoot('data/static/districts_simplified.geojson'), resolve(publicDir, 'districts.geojson'))
copyFileSync(fromRoot('data/static/about.json'), resolve(publicDir, 'about.json'))

const final = 'results/final/'
const manifest = readJson(final + 'MANIFEST.json')
const meta = readJson(final + 'reference_meta.json')
const verification = {
  scorecard: parseCsv(final + 'scorecard_test.csv'),
  per_regime: parseCsv(final + 'per_regime_csi_test.csv'),
  reliability: parseCsv(final + 'reliability_test.csv'),
  manifest: Object.fromEntries(['run_id', 'created_utc', 'splits', 'alpha', 'frozen_before_test_opened'].map((key) => [key, manifest[key] ?? null])),
  caveats: meta.caveats,
}
const atlas = {
  cells: parseCsv(final + 'regime_error_atlas_test.csv'),
  min_events: meta.atlas_min_events,
  note: meta.atlas_note,
}
writeFileSync(resolve(publicDir, 'verification.json'), JSON.stringify(verification))
writeFileSync(resolve(publicDir, 'atlas.json'), JSON.stringify(atlas))
