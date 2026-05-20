import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, CircleMarker } from 'react-leaflet';
import L from 'leaflet';

// Fix default marker icon issue with webpack
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const truckIcon = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});

function MapView({ data, result }) {
  const center = [12.9716, 77.5946]; // Bangalore

  const assignmentLines = result?.assignments?.map((a, i) => {
    const res = data.resources.find(r => r.id === a.resource_id);
    const req = data.requests.find(r => r.id === a.request_id);
    if (!res || !req) return null;
    return (
      <Polyline
        key={i}
        positions={[[res.location.lat, res.location.lng], [req.destination.lat, req.destination.lng]]}
        color="#3182ce"
        weight={2}
        opacity={0.7}
      />
    );
  });

  return (
    <MapContainer center={center} zoom={12} style={{ height: '100%', width: '100%' }}>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {data?.resources?.map(r => (
        <Marker key={r.id} position={[r.location.lat, r.location.lng]} icon={truckIcon}>
          <Popup><strong>{r.name}</strong><br/>{r.size} — {r.capacity_kg} kg</Popup>
        </Marker>
      ))}
      {data?.requests?.map(r => (
        <CircleMarker
          key={r.id}
          center={[r.destination.lat, r.destination.lng]}
          radius={6 + r.priority * 2}
          fillColor={r.priority === 3 ? '#e53e3e' : r.priority === 2 ? '#dd6b20' : '#38a169'}
          color="#fff"
          weight={2}
          fillOpacity={0.8}
        >
          <Popup><strong>{r.id}</strong><br/>Weight: {r.weight_kg} kg<br/>Priority: {r.priority}</Popup>
        </CircleMarker>
      ))}
      {assignmentLines}
    </MapContainer>
  );
}

export default MapView;
