import { copyFileSync, mkdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const publicDir = resolve(root, 'frontend/public')
mkdirSync(publicDir, { recursive: true })
copyFileSync(
  resolve(root, 'data/static/districts_simplified.geojson'),
  resolve(publicDir, 'districts.geojson'),
)
