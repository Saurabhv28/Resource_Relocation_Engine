import React from 'react';

function MetricsPanel({ greedy, hungarian }) {
  return (
    <div className="metrics-panel">
      <h3>Algorithm Comparison</h3>
      <div className="comparison-row">
        <div className="algo-col">
          <h4>Greedy (Nearest-First)</h4>
          <div className="metric-card"><div className="value">{greedy.total_distance_km} km</div><div className="label">Total Distance</div></div>
          <div className="metric-card"><div className="value">{greedy.avg_distance_km} km</div><div className="label">Avg Distance</div></div>
          <div className="metric-card green"><div className="value">{greedy.assignment_rate}%</div><div className="label">Fulfillment</div></div>
          <div className="metric-card"><div className="value">{greedy.computation_time_ms} ms</div><div className="label">Compute Time</div></div>
        </div>
        <div className="algo-col">
          <h4>Hungarian (Optimal)</h4>
          <div className="metric-card"><div className="value">{hungarian.total_distance_km} km</div><div className="label">Total Distance</div></div>
          <div className="metric-card"><div className="value">{hungarian.avg_distance_km} km</div><div className="label">Avg Distance</div></div>
          <div className="metric-card green"><div className="value">{hungarian.assignment_rate}%</div><div className="label">Fulfillment</div></div>
          <div className="metric-card"><div className="value">{hungarian.computation_time_ms} ms</div><div className="label">Compute Time</div></div>
        </div>
      </div>
      {hungarian.total_distance_km < greedy.total_distance_km && (
        <p style={{ marginTop: 12, fontSize: '0.8rem', color: '#276749', background: '#f0fff4', padding: 8, borderRadius: 4 }}>
          ✓ Hungarian saves <strong>{(greedy.total_distance_km - hungarian.total_distance_km).toFixed(1)} km</strong> ({((1 - hungarian.total_distance_km / greedy.total_distance_km) * 100).toFixed(1)}% reduction) over Greedy.
        </p>
      )}
    </div>
  );
}

export default MetricsPanel;
