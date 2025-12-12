import React from "react";

export default function HistoryTable({ history }) {
  return (
    <div style={{ marginTop: 12 }}>
      <h3>History (recent)</h3>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ border: "1px solid #ddd", padding: 6 }}>Time</th>
            <th style={{ border: "1px solid #ddd", padding: 6 }}>Event</th>
          </tr>
        </thead>
        <tbody>
          {history.map((h, idx) => (
            <tr key={idx}>
              <td style={{ border: "1px solid #ddd", padding: 6 }}>{new Date(h.timestamp).toLocaleString()}</td>
              <td style={{ border: "1px solid #ddd", padding: 6 }}>{JSON.stringify(h, null, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
