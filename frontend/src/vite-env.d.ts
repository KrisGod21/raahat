/// <reference types="vite/client" />
// GeoJSON types are used in the API client and map component.
declare namespace GeoJSON {
  interface FeatureCollection { type: string; features: any[] }
}
