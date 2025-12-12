import React, { useEffect, useState } from "react";
import LiveStream from "./components/LiveStream";
import CountBoard from "./components/CountBoard";
import HistoryTable from "./components/HistoryTable";
import Stats from "./components/Stats";
import { getHistory } from "./api";

export default function App() {
  const [counts, setCounts] = useState({});
  const [history, setHistory] = useState([]);

  useEffect(() => {
   const backendUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
const wsUrl = backendUrl.replace(/^http/, "ws") + "/ws";

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => console.log("WS connected");
    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.counts) setCounts(data.counts);
        if (data.events?.length > 0) {
          setHistory((h) => [...data.events, ...h].slice(0, 200));
        }
      } catch (e) {
        console.error(e);
      }
    };

    ws.onclose = () => console.log("WS closed");
    return () => ws.close();
  }, []);

  useEffect(() => {
    getHistory(50).then((h) => setHistory(h)).catch(() => {});
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h2>🚗 Vehicle Monitoring</h2>

      <LiveStream />

      <div style={{ display: "flex", gap: 20, marginTop: 12 }}>
        <div style={{ flex: 1 }}>
          <CountBoard counts={counts} />
        </div>
        <div style={{ flex: 2 }}>
          <Stats />
        </div>
      </div>

      <HistoryTable history={history} />
    </div>
  );
}
