import React, { useEffect, useState } from "react";
import { getDaily, getHourly } from "../api";
import { BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";

export default function Stats() {
  const [daily, setDaily] = useState([]);
  const [hourly, setHourly] = useState([]);

  useEffect(() => {
    getDaily().then(d => {
      setDaily(d.map(x => ({ day: `${x._id.day}`, count: x.count })));
    }).catch(()=>{});
    getHourly().then(d => {
      setHourly(d.map(x => ({ hour: `${x._id.hour}`, count: x.count })));
    }).catch(()=>{});
  }, []);

  return (
    <div style={{display:"flex", gap:20, marginTop:20}}>
      <div>
        <h4>Daily</h4>
        <BarChart width={400} height={250} data={daily}>
          <XAxis dataKey="day"/>
          <YAxis/>
          <Tooltip/>
          <Bar dataKey="count" />
        </BarChart>
      </div>
      <div>
        <h4>Hourly</h4>
        <BarChart width={400} height={250} data={hourly}>
          <XAxis dataKey="hour"/>
          <YAxis/>
          <Tooltip/>
          <Bar dataKey="count" />
        </BarChart>
      </div>
    </div>
  );
}
