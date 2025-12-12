import React from "react";

export default function CountBoard({ counts }) {
  return (
    <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 6 }}>
      <h3>Counts</h3>
      {Object.keys(counts || {}).length === 0 && <div>No data</div>}
      {Object.entries(counts || {}).map(([k,v]) => (
        <div key={k} style={{marginBottom:6}}>{k}: <strong>{v}</strong></div>
      ))}
    </div>
  );
}
