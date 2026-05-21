import React, { useState, useEffect } from 'react';
import MapView from './components/MapView';
import MetricsPanel from './components/MetricsPanel';
import './App.css';

const API = 'http://localhost:8000/api';

function App() {
  const [data, setData] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [nResources, setNResources] = useState(8);
  const [nRequests, setNRequests] = useState(10);
  const [activeAlgo, setActiveAlgo] = useState('greedy');

  const fetchData = async () => {
    const res = await fetch(`${API}/sample-data?n_resources=${nResources}&n_requests=${nRequests}&seed=${Date.now() % 10000}`);
    const json = await res.json();
    setData(json);
    setResults(null);
  };

  const runAllocation = async () => {
    if (!data) return;
    setLoading(true);
    const res = await fetch(`${API}/allocate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ resources: data.resources, requests: data.requests }),
    });
    const json = await res.json();
    setResults(json);
    setLoading(false);
  };

  useEffect(() => { fetchData(); }, []);

  const currentResult = results ? results[activeAlgo] : null;

  return (
    <div className="app">
      <header className="header">
        <h1>🚚 Delivery Fleet Allocation Engine</h1>
        <p>Optimally assign trucks to delivery orders</p>
      </header>

      <div className="controls">
        <label>Trucks: <input type="number" min="1" max="20" value={nResources} onChange={e => setNResources(+e.target.value)} /></label>
        <label>Orders: <input type="number" min="1" max="30" value={nRequests} onChange={e => setNRequests(+e.target.value)} /></label>
        <button onClick={fetchData}>Generate Data</button>
        <button onClick={runAllocation} disabled={!data || loading}>
          {loading ? 'Running...' : 'Run Allocation'}
        </button>
      </div>

      {results && (
        <div className="algo-tabs">
          <button className={activeAlgo === 'greedy' ? 'active' : ''} onClick={() => setActiveAlgo('greedy')}>
            Greedy (Nearest-First)
          </button>
          <button className={activeAlgo === 'hungarian' ? 'active' : ''} onClick={() => setActiveAlgo('hungarian')}>
            Hungarian (Optimal Batch)
          </button>
          <button className={activeAlgo === 'ml' ? 'active' : ''} onClick={() => setActiveAlgo('ml')}>
            ML-Based (Learned)
          </button>
        </div>
      )}

      <div className="main-content">
        <div className="map-container">
          <MapView data={data} result={currentResult} />
        </div>
        <div className="side-panel">
          {results && <MetricsPanel greedy={results.greedy} hungarian={results.hungarian} ml={results.ml} />}
          {currentResult && (
            <div className="assignments-list">
              <h3>Assignments ({currentResult.assignments.length})</h3>
              {currentResult.assignments.map((a, i) => (
                <div key={i} className="assignment-card">
                  <strong>{a.resource_id} → {a.request_id}</strong>
                  <span className="distance">{a.distance_km} km</span>
                  <p className="explanation">{a.explanation}</p>
                </div>
              ))}
              {currentResult.unassigned_requests.length > 0 && (
                <div className="unassigned">
                  <h4>Unassigned Orders</h4>
                  {currentResult.unassigned_requests.map(id => <span key={id} className="badge">{id}</span>)}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
