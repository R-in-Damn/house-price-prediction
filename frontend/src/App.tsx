import { FormEvent, useState } from 'react'

type Property = { bedrooms: number; bathrooms: number; area: number; zipcode: string; asking_price: number; monthly_rent?: number; annual_appreciation_rate?: number; images?: string[] }
type Valuation = { estimated_value: number; lower_bound: number; upper_bound: number; currency: string; method: string; confidence: string; price_gap_percent?: number; comparable_count: number; assumptions: string[] }
type Investment = { decision: string; score: number; rental_yield?: number; roi_percent?: number; opportunity_cost_percent?: number; valuation: Valuation; uncertainty: string; assumptions: string[] }
type Comparable = { property_id: string; bedrooms: number; bathrooms: number; area: number; zipcode: string; price: number; price_per_area: number }
type RAGResult = { answer: string; citations: { id: string; title: string; source: string }[]; grounded: boolean; llm_provider: string }

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
const money = (value?: number) => value == null ? 'Unavailable' : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value)

export default function App() {
  const [property, setProperty] = useState<Property>({ bedrooms: 3, bathrooms: 2, area: 1800, zipcode: '91901', asking_price: 500000, monthly_rent: 2800, annual_appreciation_rate: 0.04 })
  const [result, setResult] = useState<Investment | null>(null)
  const [comps, setComps] = useState<Comparable[]>([])
  const [imageData, setImageData] = useState<string[]>([])
  const [question, setQuestion] = useState('What documents should I verify before buying?')
  const [ragResult, setRagResult] = useState<RAGResult | null>(null)
  const [ragError, setRagError] = useState('')
  const [error, setError] = useState('')
  const update = (key: keyof Property, value: string) => setProperty((current) => ({ ...current, [key]: key === 'zipcode' ? value : Number(value) }))
  async function analyse(event: FormEvent) {
    event.preventDefault(); setError(''); setResult(null)
    try {
      const request = { ...property, images: imageData }
      const response = await fetch(`${API}/api/v1/investment-analysis`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) })
      if (!response.ok) throw new Error((await response.json()).detail ?? 'Request failed')
      const data: Investment = await response.json(); setResult(data)
      const compResponse = await fetch(`${API}/api/v1/comparables/search`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) })
      setComps(await compResponse.json())
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Unable to reach the valuation service') }
  }
  async function askAssistant(event: FormEvent) {
    event.preventDefault(); setRagError(''); setRagResult(null)
    try {
      const response = await fetch(`${API}/api/v1/rag/query`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }) })
      if (!response.ok) throw new Error((await response.json()).detail ?? 'RAG request failed')
      setRagResult(await response.json())
    } catch (caught) { setRagError(caught instanceof Error ? caught.message : 'Unable to reach the knowledge assistant') }
  }
  return <main>
    <header><div className="kicker">ESTATE SIGNAL / DATA-AWARE VALUATION</div><h1>See the property<br /><em>in context.</em></h1><p className="lede">A practical valuation workspace combining structured details, comparable homes, and investment math without pretending uncertainty away.</p></header>
    <section className="workspace">
      <form className="panel form-panel" onSubmit={analyse}><div className="panel-label">01 / PROPERTY SNAPSHOT</div><div className="fields">
        {([['bedrooms','Bedrooms'],['bathrooms','Bathrooms'],['area','Area'],['asking_price','Asking price'],['monthly_rent','Monthly rent'],['annual_appreciation_rate','Annual appreciation']] as const).map(([key,label]) => <label key={key}>{label}<input type="number" step={key === 'annual_appreciation_rate' ? '0.01' : '1'} value={property[key] ?? ''} onChange={(event) => update(key, event.target.value)} /></label>)}
        <label>ZIP / locality<input value={property.zipcode} onChange={(event) => update('zipcode', event.target.value)} /></label>
        <label className="upload">Property images<input type="file" accept="image/*" multiple onChange={(event) => { const files = Array.from(event.target.files ?? []).slice(0, 8); Promise.all(files.map((file) => new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = reject; reader.readAsDataURL(file) }))).then(setImageData) }} /><small>{imageData.length ? `${imageData.length} image${imageData.length === 1 ? '' : 's'} ready for visual analysis` : 'Optional: add room or exterior images'}</small></label>
      </div><button type="submit">Run intelligence <span>→</span></button>{error && <p className="error">{error}</p>}</form>
      <div className="results">{result ? <><div className="panel hero-result"><div className="panel-label">02 / VALUATION SIGNAL</div><div className="value">{money(result.valuation.estimated_value)}</div><div className="range">Range {money(result.valuation.lower_bound)} — {money(result.valuation.upper_bound)}</div><div className="signal-row"><span className={`decision ${result.decision.toLowerCase()}`}>{result.decision}</span><strong>{result.score}/100</strong><span>{result.valuation.confidence} confidence</span></div><p className="method">{result.valuation.method}. Asking price is {result.valuation.price_gap_percent == null ? 'not compared' : `${Math.abs(result.valuation.price_gap_percent).toFixed(1)}% ${result.valuation.price_gap_percent > 0 ? 'above' : 'below'} estimated fair value`}.</p></div><div className="metrics">{[['Rental yield', result.rental_yield == null ? 'Unavailable' : `${result.rental_yield}%`],['Projected ROI', result.roi_percent == null ? 'Unavailable' : `${result.roi_percent}%`],['Opportunity benchmark', `${result.opportunity_cost_percent}%`]].map(([label,value]) => <div className="metric" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div><div className="panel"><div className="panel-label">03 / NEARBY EVIDENCE</div><table><thead><tr><th>Home</th><th>Area</th><th>ZIP</th><th>Price</th></tr></thead><tbody>{comps.map((comp) => <tr key={comp.property_id}><td>#{comp.property_id}</td><td>{comp.area.toLocaleString()} sq ft</td><td>{comp.zipcode}</td><td>{money(comp.price)}</td></tr>)}</tbody></table></div><div className="panel notes"><div className="panel-label">ASSUMPTIONS & UNCERTAINTY</div>{result.assumptions.map((item) => <p key={item}>• {item}</p>)}<p><strong>{result.uncertainty}</strong></p></div></> : <div className="empty"><span>◎</span><h2>Your market readout will land here.</h2><p>Enter a property snapshot to compare fair value, evidence, and investment scenarios.</p></div>}</div>
    </section><section className="panel assistant"><div className="panel-label">04 / DOCUMENT ASSISTANT</div><h2>Ask the source-backed guide.</h2><form onSubmit={askAssistant}><input value={question} onChange={(event) => setQuestion(event.target.value)} /><button type="submit">Ask assistant <span>→</span></button></form>{ragResult && <div className="answer"><p>{ragResult.answer}</p><small>{ragResult.grounded ? 'Grounded in indexed documents' : 'No supporting source found'} · {ragResult.llm_provider}</small><div className="sources">{ragResult.citations.map((citation) => <span key={citation.id}>[{citation.id}] {citation.title} · {citation.source}</span>)}</div></div>}{ragError && <p className="error">{ragError}</p>}</section><footer>Dataset-aware analysis · Results are decision support, not financial advice.</footer>
  </main>
}
