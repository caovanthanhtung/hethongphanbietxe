import React from "react";

export default function LiveStream() {
  const url = import.meta.env.VITE_VIDEO_URL || "http://localhost:8000/video";

  return (
    <div style={{ width: "100%", maxWidth: 960 }}>
      <img src={url} alt="live" style={{ width: "100%", borderRadius: 8 }} />
    </div>
  );
}
