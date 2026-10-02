import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/tokens.css'

/**
 * React.StrictMode is deliberately NOT used.
 *
 * StrictMode double-invokes effects in development: it creates the map,
 * synchronously tears it down, and creates another. MapLibre allocates a WebGL
 * context per map, and that create/destroy/create cycle leaves the replacement
 * map in a state where addSource throws "Style is not done loading" and the
 * map never draws -- with no error surfaced to the user.
 *
 * Production builds never double-invoke, so this makes development behave like
 * the build we actually ship. The map component is still written to survive a
 * genuine remount (the ref is nulled and the layer flag reset on cleanup), so
 * switching tabs works correctly either way.
 */
ReactDOM.createRoot(document.getElementById('root')!).render(<App />)
