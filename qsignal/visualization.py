"""
Visualization helpers for Q-Signal.
Returns Folium maps and Matplotlib figures; Streamlit renders them.
No st.* calls in this module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    import folium
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False

try:
    import altair as alt
    _ALTAIR_AVAILABLE = True
except ImportError:
    _ALTAIR_AVAILABLE = False


from qsignal.models import (
    TrafficState,
    PHASE_NS,
    PHASE_EW,
    PHASE_YELLOW,
    PHASE_ALL_RED,
)
from qsignal.network import (
    schematic_to_latlon,
    get_landmark_name,
    TIRUPPUR_CENTER_LAT,
    TIRUPPUR_CENTER_LON,
    get_road_waypoints,
    get_vehicle_latlon,
)

# ---------------------------------------------------------------------------
# Phase colours
# ---------------------------------------------------------------------------
PHASE_COLORS = {
    PHASE_NS: "#22c55e",       # green
    PHASE_EW: "#3b82f6",       # blue
    PHASE_YELLOW: "#f59e0b",   # amber
    PHASE_ALL_RED: "#ef4444",  # red
}
PHASE_LABELS = {
    PHASE_NS: "NS Green",
    PHASE_EW: "EW Green",
    PHASE_YELLOW: "Yellow",
    PHASE_ALL_RED: "All Red",
}


# ---------------------------------------------------------------------------
# Folium network map
# ---------------------------------------------------------------------------

def build_network_map(
    state: TrafficState,
    emergency_route: Optional[List[str]] = None,
    center_lat: float = TIRUPPUR_CENTER_LAT,
    center_lon: float = TIRUPPUR_CENTER_LON,
    zoom: int = 14,
    mapbox_token: Optional[str] = None,
    theme: str = "mild",
    **kwargs,
) -> Optional[object]:
    """
    Build a real-time Folium map of Tiruppur (Tamil Nadu, India) showing intersections,
    queue intensity, vehicle telemetry, and emergency green corridors.
    Supports OpenStreetMap, Satellite imagery, and CartoDB themes.
    Returns None if Folium is not available.
    """
    if not _FOLIUM_AVAILABLE:
        return None

    import os

    is_mild = (theme == "mild")

    # Convert schematic positions to Tiruppur lat/lon
    positions = {nid: inter.position for nid, inter in state.intersections.items()}
    latlon = schematic_to_latlon(positions, center_lat, center_lon, spacing_deg=0.007)

    token = (mapbox_token or os.getenv("MAPBOX_API_KEY") or os.getenv("RC_API_KEY") or "").strip()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # If Mapbox API key is provided, add premium Mapbox HD vector layers
    if token:
        mapbox_streets = f"https://api.mapbox.com/styles/v1/mapbox/streets-v12/tiles/256/{{z}}/{{x}}/{{y}}?access_token={token}"
        mapbox_sat = f"https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/tiles/256/{{z}}/{{x}}/{{y}}?access_token={token}"
        mapbox_dark = f"https://api.mapbox.com/styles/v1/mapbox/dark-v11/tiles/256/{{z}}/{{x}}/{{y}}?access_token={token}"

        folium.TileLayer(
            tiles=mapbox_streets,
            attr="© Mapbox © OpenStreetMap",
            name="💎 Mapbox HD Streets (API Key)",
            overlay=False,
            control=True,
        ).add_to(m)
        folium.TileLayer(
            tiles=mapbox_sat,
            attr="© Mapbox © OpenStreetMap",
            name="🛰️ Mapbox HD Satellite (API Key)",
            overlay=False,
            control=True,
        ).add_to(m)
        folium.TileLayer(
            tiles=mapbox_dark,
            attr="© Mapbox © OpenStreetMap",
            name="🌙 Mapbox Dark GIS (API Key)",
            overlay=False,
            control=True,
        ).add_to(m)

    # Add Esri Satellite Imagery Layer (Real Earth view of Tiruppur streets)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="🛰️ Real Satellite Imagery (Tiruppur)",
        overlay=False,
    ).add_to(m)

    # Add standard base layer options in layer control
    # Add standard base layer options in layer control using direct fastly/OSM/Esri CDN (zero API key warning)
    osm_tiles = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    osm_attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    carto_dark = "https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png"
    carto_light = "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"
    carto_attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; CARTO'

    if is_mild:
        folium.TileLayer(osm_tiles, attr=osm_attr, name="🗺️ OpenStreetMap Standard", overlay=False).add_to(m)
        folium.TileLayer(carto_light, attr=carto_attr, name="🏙️ CartoDB Positron (Light)", overlay=False).add_to(m)
        folium.TileLayer(carto_dark, attr=carto_attr, name="🌙 Dark Matter (Night)", overlay=False).add_to(m)
    else:
        folium.TileLayer(carto_dark, attr=carto_attr, name="🌙 CartoDB Dark Matter", overlay=False).add_to(m)
        folium.TileLayer(osm_tiles, attr=osm_attr, name="🗺️ OpenStreetMap Standard", overlay=False).add_to(m)
        folium.TileLayer(carto_light, attr=carto_attr, name="🏙️ CartoDB Positron (Light)", overlay=False).add_to(m)

    folium.LayerControl(position="topright").add_to(m)

    # Group approaches by intersection
    approach_by_node: Dict[str, Dict[str, float]] = {}
    for approach in state.approaches.values():
        approach_by_node.setdefault(approach.intersection_id, {})[approach.movement] = approach.queue

    # Draw edges (roads) along real street geometry
    node_ids = list(state.intersections.keys())
    drawn = set()
    road_color = "#94a3b8" if is_mild else "#475569"

    for nid in node_ids:
        idx = node_ids.index(nid)
        n_total = len(node_ids)
        import math
        cols = math.ceil(math.sqrt(n_total))
        # horizontal
        right_idx = idx + 1
        if right_idx < n_total and (right_idx % cols) != 0:
            rn = node_ids[right_idx]
            key = tuple(sorted([nid, rn]))
            if key not in drawn:
                pts = get_road_waypoints(nid, rn, latlon)
                folium.PolyLine(
                    pts, color=road_color, weight=5 if is_mild else 4, opacity=0.75
                ).add_to(m)
                drawn.add(key)
        # vertical
        down_idx = idx + cols
        if down_idx < n_total:
            dn = node_ids[down_idx]
            key = tuple(sorted([nid, dn]))
            if key not in drawn:
                pts = get_road_waypoints(nid, dn, latlon)
                folium.PolyLine(
                    pts, color=road_color, weight=5 if is_mild else 4, opacity=0.75
                ).add_to(m)
                drawn.add(key)

    # Draw emergency route along real street paths
    if emergency_route and len(emergency_route) >= 2:
        route_pts = []
        for i in range(len(emergency_route) - 1):
            seg = get_road_waypoints(emergency_route[i], emergency_route[i+1], latlon)
            if i > 0 and seg:
                route_pts.extend(seg[1:])
            else:
                route_pts.extend(seg)
        if route_pts:
            folium.PolyLine(
                route_pts,
                color="#ea580c" if is_mild else "#f97316",
                weight=7,
                opacity=0.92,
                dash_array="8, 6",
                tooltip="🚨 Emergency Priority Corridor",
            ).add_to(m)

    # Draw emergency vehicle icon if active
    if emergency_route and len(emergency_route) > 0:
        first_node = emergency_route[0]
        if first_node in latlon:
            amb_lat, amb_lon = latlon[first_node]
            folium.Marker(
                location=[amb_lat, amb_lon],
                icon=folium.DivIcon(
                    html='<div style="font-size:24px;filter:drop-shadow(0 0 6px rgba(234,88,12,0.8));line-height:1;">🚨</div>',
                    icon_size=(30, 30),
                    icon_anchor=(15, 15),
                ),
                tooltip="🚨 Priority Emergency Vehicle En Route",
            ).add_to(m)

    # Draw intersections with directional arrow badges
    for nid, inter in state.intersections.items():
        lat, lon = latlon.get(nid, (center_lat, center_lon))
        queues = approach_by_node.get(nid, {})
        total_q = sum(queues.values())

        # Determine directional indicator symbol and color
        if inter.phase == PHASE_NS:
            shape_symbol = "▲▼"
            symbol_color = "#16a34a" if is_mild else "#22c55e"
            phase_desc = "NS Green"
        elif inter.phase == PHASE_EW:
            shape_symbol = "◀▶"
            symbol_color = "#2563eb" if is_mild else "#38bdf8"
            phase_desc = "EW Green"
        elif inter.phase == PHASE_YELLOW or inter.is_in_clearance:
            shape_symbol = "●"
            symbol_color = "#d97706" if is_mild else "#f59e0b"
            phase_desc = "Yellow"
        else:
            shape_symbol = "■"
            symbol_color = "#dc2626" if is_mild else "#ef4444"
            phase_desc = "All-Red"

        # Emergency highlight
        is_reserved = inter.is_reserved and (nid in (emergency_route or []))
        
        if is_mild:
            border_style = "2px solid #ea580c; box-shadow: 0 2px 10px rgba(234,88,12,0.4);" if is_reserved else f"1.5px solid {symbol_color};"
            reserved_tag = '<div style="color:#ea580c;font-size:8px;font-weight:700;margin-top:1px;">🚨CORRIDOR</div>' if is_reserved else ''
            card_bg = "#ffffff"
            title_color = "#0f172a"
            subtext_color = "#64748b"
            card_shadow = "0 2px 8px rgba(0,0,0,0.12)"
        else:
            border_style = "2px solid #f97316; box-shadow: 0 0 14px rgba(249,115,22,0.8);" if is_reserved else f"1.5px solid {symbol_color};"
            reserved_tag = '<div style="color:#f97316;font-size:8px;font-weight:700;margin-top:1px;">🚨CORRIDOR</div>' if is_reserved else ''
            card_bg = "#0b132b"
            title_color = "#f1f5f9"
            subtext_color = "#94a3b8"
            card_shadow = "0 3px 8px rgba(0,0,0,0.6)"

        landmark_title = get_landmark_name(nid)
        popup_html = (
            f"<b>📍 {nid}: {landmark_title}</b><br>"
            f"<i>Tiruppur Metropolitan Network</i><br>"
            f"Phase: {shape_symbol} {phase_desc}<br>"
            f"NS Queue: {queues.get('NS',0):.1f} veh<br>"
            f"EW Queue: {queues.get('EW',0):.1f} veh<br>"
            f"Total Queue: {total_q:.1f} veh<br>"
            f"Green Time: {inter.elapsed_green}s<br>"
            f"GPS: {lat:.4f}° N, {lon:.4f}° E<br>"
            f"{'🚨 EMERGENCY GREEN CORRIDOR' if is_reserved else ''}"
        )

        short_landmark = landmark_title.split('/')[0].strip()
        marker_card_html = f"""
        <div style="background:{card_bg};border:{border_style}border-radius:8px;padding:3px 7px;text-align:center;min-width:60px;color:{title_color};font-family:Inter,sans-serif;box-shadow:{card_shadow};user-select:none;transform:translate(-50%,-50%);">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:4px;">
                <span style="font-weight:800;font-size:11px;color:{title_color};">{nid}</span>
                <span style="color:{symbol_color};font-size:11px;font-weight:900;">{shape_symbol}</span>
            </div>
            <div style="font-size:8px;font-weight:700;color:{symbol_color};white-space:nowrap;max-width:90px;overflow:hidden;text-overflow:ellipsis;">{short_landmark}</div>
            <div style="font-size:9px;color:{subtext_color};font-weight:600;">Q:{total_q:.1f}</div>
            {reserved_tag}
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=marker_card_html,
                icon_size=(62, 38),
                icon_anchor=(31, 19),
            ),
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"📍 {nid} ({landmark_title}) | {shape_symbol} {phase_desc} | Q={total_q:.1f}",
        ).add_to(m)

        # Draw realistic queued vehicle expression badges along the road
        if total_q >= 1.0:
            car_icons = ["🚗", "🚙", "🚕", "🚌", "🚚"]
            chosen_icon = car_icons[int(total_q) % len(car_icons)]
            q_bg = "rgba(255,255,255,0.95)" if is_mild else "rgba(15,23,42,0.9)"
            q_txt = "#b91c1c" if is_mild else "#fca5a5"
            folium.Marker(
                location=[lat + 0.0013, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:11px;background:{q_bg};border:1px solid #ef4444;border-radius:10px;padding:1px 5px;box-shadow:0 2px 6px rgba(0,0,0,0.15);white-space:nowrap;user-select:none;">🛑 {chosen_icon} <span style="font-size:9px;color:{q_txt};font-weight:700;">{int(total_q)} waiting</span></div>',
                    icon_size=(80, 20),
                    icon_anchor=(40, 10),
                ),
                tooltip=f"{nid} Queue: {total_q:.1f} vehicles idling at red signal",
            ).add_to(m)

    # Draw moving and stopped vehicles along GIS road coordinates following real street curves
    for v in state.vehicles.values():
        if v.completed or not v.from_node or not v.to_node:
            continue
        if v.from_node not in latlon or v.to_node not in latlon:
            continue

        p = max(0.05, min(0.95, v.progress))
        v_lat, v_lon = get_vehicle_latlon(v.from_node, v.to_node, p, latlon)

        if v.emergency:
            folium.Marker(
                location=[v_lat, v_lon],
                icon=folium.DivIcon(
                    html='<div style="font-size:24px;filter:drop-shadow(0 0 6px rgba(234,88,12,0.9));transform:translate(-12px,-12px);">🚑</div>',
                    icon_size=(28, 28),
                    icon_anchor=(14, 14),
                ),
                tooltip="🚨 Priority Emergency Vehicle en route (75 km/h)",
            ).add_to(m)
        elif v.is_stopped:
            stop_bg = "rgba(255,255,255,0.95)" if is_mild else "rgba(15,23,42,0.92)"
            stop_txt = "#b91c1c" if is_mild else "#fca5a5"
            folium.Marker(
                location=[v_lat, v_lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:12px;background:{stop_bg};border:1px solid #ef4444;border-radius:10px;padding:1px 5px;box-shadow:0 2px 6px rgba(0,0,0,0.15);white-space:nowrap;transform:translate(-50%,-50%);">🛑 {v.icon} <span style="font-size:8px;color:{stop_txt};font-weight:700;">Wait</span></div>',
                    icon_size=(48, 18),
                    icon_anchor=(24, 9),
                ),
                tooltip=f"{v.vehicle_id} | 🛑 Stopped at red signal",
            ).add_to(m)
        else:
            glow = "drop-shadow(0 1px 3px rgba(0,0,0,0.25))" if is_mild else "drop-shadow(0 0 3px rgba(34,197,94,0.8))"
            folium.Marker(
                location=[v_lat, v_lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:14px;filter:{glow};transform:translate(-50%,-50%);">{v.icon}</div>',
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                ),
                tooltip=f"{v.vehicle_id} | 🟢 Moving ({v.speed_kmh:.0f} km/h)",
            ).add_to(m)

    return m


# ---------------------------------------------------------------------------
# Real Live Interactive Leaflet Map (Flicker-Free with Camera State Retention)
# ---------------------------------------------------------------------------

def build_real_live_map_html(
    state: TrafficState,
    emergency_route: Optional[List[str]] = None,
    mapbox_token: Optional[str] = None,
    theme: str = "mild",
    height: int = 460,
) -> str:
    """
    Generate an authentic, ultra-smooth Real Live Interactive Street Map of Tiruppur.
    Features:
      - High-detail OpenStreetMap, Esri World Imagery (Real Satellite), and Dark GIS layers
      - Preserves user pan & zoom across simulation ticks via browser sessionStorage (NO resetting!)
      - Real arterial road waypoints with live traffic flow congestion heat colors (Green, Amber, Red)
      - Interactive signal heads with live countdown timers and directional phase indicators
      - Live animated moving vehicles and emergency vehicle green wave routing
      - Quick map controls: Center Tiruppur, Fit Network, Locate Emergency Vehicle, and Layer Switcher
    """
    import json
    import os
    import math

    is_mild = (theme == "mild")
    token = (mapbox_token or os.getenv("MAPBOX_API_KEY") or os.getenv("RC_API_KEY") or "").strip()

    # Lat/lon mapping of junctions
    positions = {nid: inter.position for nid, inter in state.intersections.items()}
    latlon = schematic_to_latlon(positions, TIRUPPUR_CENTER_LAT, TIRUPPUR_CENTER_LON, spacing_deg=0.007)

    # Approaches by node
    approach_by_node: Dict[str, Dict[str, float]] = {}
    for approach in state.approaches.values():
        approach_by_node.setdefault(approach.intersection_id, {})[approach.movement] = approach.queue

    # Prepare node data
    nodes_data = []
    for nid, inter in state.intersections.items():
        lat, lon = latlon.get(nid, (TIRUPPUR_CENTER_LAT, TIRUPPUR_CENTER_LON))
        queues = approach_by_node.get(nid, {})
        total_q = sum(queues.values())

        if inter.phase == PHASE_NS:
            shape_symbol = "▲▼"
            symbol_color = "#059669" if is_mild else "#10b981"  # Soft Sage / Forest Mint
            phase_desc = "NS Green"
        elif inter.phase == PHASE_EW:
            shape_symbol = "◀▶"
            symbol_color = "#4338ca" if is_mild else "#6366f1"  # Mild Royal Slate / Indigo
            phase_desc = "EW Green"
        elif inter.phase == PHASE_YELLOW or inter.is_in_clearance:
            shape_symbol = "●"
            symbol_color = "#d97706" if is_mild else "#f59e0b"  # Warm Honey Amber
            phase_desc = "Yellow Clearance"
        else:
            shape_symbol = "■"
            symbol_color = "#be123c" if is_mild else "#f43f5e"  # Soft Rose Terracotta
            phase_desc = "All-Red"

        is_reserved = bool(inter.is_reserved and (nid in (emergency_route or [])))
        landmark = get_landmark_name(nid)
        short_landmark = landmark.split("/")[0].strip()

        nodes_data.append({
            "id": nid,
            "lat": lat,
            "lon": lon,
            "name": landmark,
            "shortName": short_landmark,
            "phase": inter.phase,
            "phaseDesc": phase_desc,
            "symbol": shape_symbol,
            "color": symbol_color,
            "timer": inter.elapsed_green,
            "nsQueue": round(queues.get("NS", 0.0), 1),
            "ewQueue": round(queues.get("EW", 0.0), 1),
            "totalQueue": round(total_q, 1),
            "isReserved": is_reserved,
        })

    # Prepare road data following real street geometry
    node_ids = list(state.intersections.keys())
    n_total = len(node_ids)
    cols = math.ceil(math.sqrt(n_total)) if n_total > 0 else 1
    drawn_pairs = set()
    roads_data = []

    for nid in node_ids:
        idx = node_ids.index(nid)
        # horizontal (EW)
        right_idx = idx + 1
        if right_idx < n_total and (right_idx % cols) != 0:
            rn = node_ids[right_idx]
            pair = tuple(sorted([nid, rn]))
            if pair not in drawn_pairs:
                pts = get_road_waypoints(nid, rn, latlon)
                q_u = approach_by_node.get(nid, {}).get("EW", 0.0)
                q_v = approach_by_node.get(rn, {}).get("EW", 0.0)
                avg_q = (q_u + q_v) / 2.0

                if avg_q < 2.5:
                    congestion_color = "#059669" if is_mild else "#10b981"
                    glow_color = "rgba(5, 150, 105, 0.35)"
                    flow_status = "Free Flow (~45 km/h)"
                elif avg_q < 5.5:
                    congestion_color = "#d97706" if is_mild else "#f59e0b"
                    glow_color = "rgba(217, 119, 6, 0.35)"
                    flow_status = "Moderate Flow (~25 km/h)"
                else:
                    congestion_color = "#be123c" if is_mild else "#f43f5e"
                    glow_color = "rgba(190, 18, 60, 0.40)"
                    flow_status = "Heavy Bottleneck (~10 km/h)"

                roads_data.append({
                    "u": nid,
                    "v": rn,
                    "name": f"{get_landmark_name(nid).split('/')[0].strip()} ↔ {get_landmark_name(rn).split('/')[0].strip()}",
                    "pts": [[p[0], p[1]] for p in pts],
                    "queue": round(avg_q, 1),
                    "color": congestion_color,
                    "glowColor": glow_color,
                    "status": flow_status,
                })
                drawn_pairs.add(pair)

        # vertical (NS)
        down_idx = idx + cols
        if down_idx < n_total:
            dn = node_ids[down_idx]
            pair = tuple(sorted([nid, dn]))
            if pair not in drawn_pairs:
                pts = get_road_waypoints(nid, dn, latlon)
                q_u = approach_by_node.get(nid, {}).get("NS", 0.0)
                q_v = approach_by_node.get(dn, {}).get("NS", 0.0)
                avg_q = (q_u + q_v) / 2.0

                if avg_q < 2.5:
                    congestion_color = "#059669" if is_mild else "#10b981"
                    glow_color = "rgba(5, 150, 105, 0.35)"
                    flow_status = "Free Flow (~45 km/h)"
                elif avg_q < 5.5:
                    congestion_color = "#d97706" if is_mild else "#f59e0b"
                    glow_color = "rgba(217, 119, 6, 0.35)"
                    flow_status = "Moderate Flow (~25 km/h)"
                else:
                    congestion_color = "#be123c" if is_mild else "#f43f5e"
                    glow_color = "rgba(190, 18, 60, 0.40)"
                    flow_status = "Heavy Bottleneck (~10 km/h)"

                roads_data.append({
                    "u": nid,
                    "v": dn,
                    "name": f"{get_landmark_name(nid).split('/')[0].strip()} ↔ {get_landmark_name(dn).split('/')[0].strip()}",
                    "pts": [[p[0], p[1]] for p in pts],
                    "queue": round(avg_q, 1),
                    "color": congestion_color,
                    "glowColor": glow_color,
                    "status": flow_status,
                })
                drawn_pairs.add(pair)

    # Emergency corridor polyline
    corridor_pts = []
    if emergency_route and len(emergency_route) >= 2:
        for i in range(len(emergency_route) - 1):
            seg = get_road_waypoints(emergency_route[i], emergency_route[i+1], latlon)
            if i > 0 and seg:
                corridor_pts.extend(seg[1:])
            else:
                corridor_pts.extend(seg)

    # Vehicles data along real road curves
    vehicles_data = []
    amb_coords = None
    for v in state.vehicles.values():
        if v.completed or not v.from_node or not v.to_node:
            continue
        if v.from_node not in latlon or v.to_node not in latlon:
            continue

        p = max(0.04, min(0.96, v.progress))
        v_lat, v_lon = get_vehicle_latlon(v.from_node, v.to_node, p, latlon)

        if v.emergency:
            amb_coords = [v_lat, v_lon]

        vehicles_data.append({
            "id": v.vehicle_id,
            "lat": v_lat,
            "lon": v_lon,
            "icon": "🚑" if v.emergency else v.icon,
            "speed": round(v.speed_kmh, 1),
            "isStopped": v.is_stopped,
            "isEmergency": v.emergency,
        })

    # Summary numbers for HUD
    total_network_queue = sum(a.queue for a in state.approaches.values())
    active_emergency_count = sum(1 for v in state.vehicles.values() if v.emergency and not v.completed)

    # Serialize JSON
    nodes_json = json.dumps(nodes_data)
    roads_json = json.dumps(roads_data)
    corridor_json = json.dumps([[p[0], p[1]] for p in corridor_pts])
    vehicles_json = json.dumps(vehicles_data)
    amb_json = json.dumps(amb_coords)

    # Styles
    bg_hud = "rgba(255, 255, 255, 0.98)" if is_mild else "rgba(15, 23, 42, 0.95)"
    text_hud = "#0f172a" if is_mild else "#f8fafc"
    border_hud = "#cbd5e1" if is_mild else "#334155"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
    <script>
        if (typeof L === 'undefined') {{
            document.write('<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"><\\/script>');
        }}
    </script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{ 
            width: 100%; 
            height: 100%; 
            margin: 0; 
            padding: 0; 
            font-family: 'Inter', sans-serif; 
            overflow: hidden; 
            background: {'#f8fafc' if is_mild else '#0f172a'}; 
        }}
        #map {{ 
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            width: 100%; 
            height: 100%; 
            min-height: 460px;
        }}
        
        /* Floating Toolbar - High Contrast and Clear Typography */
        .map-hud {{
            position: absolute;
            top: 12px;
            left: 12px;
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 8px;
            background: {bg_hud};
            border: 1.5px solid {border_hud};
            border-radius: 8px;
            padding: 7px 14px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12);
            backdrop-filter: blur(8px);
            font-size: 13px;
            font-weight: 600;
            color: {text_hud};
        }}
        .hud-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #059669;
            box-shadow: 0 0 8px #059669;
            animation: pulse-dot 1.5s infinite;
        }}
        @keyframes pulse-dot {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.5; transform: scale(1.2); }}
        }}

        /* Control Action Buttons */
        .map-actions {{
            position: absolute;
            top: 12px;
            right: 12px;
            z-index: 1000;
            display: flex;
            gap: 6px;
        }}
        .map-btn {{
            background: {bg_hud};
            color: {text_hud};
            border: 1.5px solid {border_hud};
            border-radius: 6px;
            padding: 7px 13px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.10);
            transition: all 0.15s ease;
            backdrop-filter: blur(6px);
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }}
        .map-btn:hover {{
            background: {'#f1f5f9' if is_mild else '#1e293b'};
            border-color: #94a3b8;
            transform: translateY(-1px);
        }}
        .map-btn.active {{
            background: {'#0f172a' if is_mild else '#38bdf8'};
            color: {'#ffffff' if is_mild else '#0f172a'};
            border-color: {'#0f172a' if is_mild else '#38bdf8'};
        }}

        /* Bottom Legend */
        .map-legend {{
            position: absolute;
            bottom: 12px;
            left: 12px;
            z-index: 1000;
            background: {bg_hud};
            border: 1.5px solid {border_hud};
            border-radius: 8px;
            padding: 7px 14px;
            font-size: 12px;
            font-weight: 600;
            color: {text_hud};
            display: flex;
            align-items: center;
            gap: 14px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12);
            backdrop-filter: blur(8px);
        }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; font-weight: 600; font-size: 12px; color: {'#1e293b' if is_mild else '#f1f5f9'}; }}
        .color-bar {{ width: 16px; height: 6px; border-radius: 3px; }}

        /* Tooltip High-Contrast Typography */
        .leaflet-tooltip {{
            font-family: 'Inter', sans-serif !important;
            font-size: 12.5px !important;
            font-weight: 600 !important;
            color: #0f172a !important;
            background: rgba(255, 255, 255, 0.98) !important;
            border: 1.5px solid #94a3b8 !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.18) !important;
            padding: 6px 12px !important;
        }}

        /* Corridor Pulse */
        @keyframes dash-animation {{
            to {{ stroke-dashoffset: -24; }}
        }}
        .corridor-dash {{
            animation: dash-animation 0.8s linear infinite;
        }}

        /* Custom Marker Card - Clear, Bold, Legible */
        .junction-card {{
            background: {'#ffffff' if is_mild else '#0f172a'};
            border-radius: 10px;
            padding: 6px 10px;
            text-align: left;
            min-width: 114px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.20);
            user-select: none;
            cursor: pointer;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .junction-card:hover {{
            transform: scale(1.06) translateY(-2px);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.28);
        }}

        /* Ambulance Siren Beacon */
        @keyframes siren-pulse {{
            0% {{ transform: scale(1); filter: drop-shadow(0 0 4px #ea580c); }}
            50% {{ transform: scale(1.25); filter: drop-shadow(0 0 14px #f97316); }}
            100% {{ transform: scale(1); filter: drop-shadow(0 0 4px #ea580c); }}
        }}
        .amb-siren {{
            animation: siren-pulse 0.7s infinite;
        }}
    </style>
</head>
<body>
    <div id="map"></div>

    <!-- HUD Status -->
    <div class="map-hud">
        <div class="hud-dot"></div>
        <span style="font-weight:800;letter-spacing:0.2px;">LIVE TIRUPPUR STREET NETWORK</span>
        <span style="color:#94a3b8;">·</span>
        <span style="font-family:'JetBrains Mono',monospace;font-weight:700;">t = {state.time_s}s</span>
        <span style="color:#94a3b8;">·</span>
        <span>🚦 Total Queue: <b style="color:{'#0f172a' if is_mild else '#f8fafc'};">{total_network_queue:.1f} veh</b></span>
        {'<span style="background:#ea580c;color:#ffffff;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:800;margin-left:6px;letter-spacing:0.3px;">🚨 EMERGENCY VEHICLE ACTIVE</span>' if active_emergency_count > 0 else ''}
    </div>

    <!-- Map Layer & Camera Actions -->
    <div class="map-actions">
        <button id="btn-streets" class="map-btn" onclick="switchLayer('streets')">🗺️ Streets</button>
        <button id="btn-sat" class="map-btn" onclick="switchLayer('sat')">🛰️ Satellite</button>
        <button id="btn-dark" class="map-btn" onclick="switchLayer('dark')">🌙 Dark GIS</button>
        <button class="map-btn" onclick="centerTiruppur()" title="Reset to Tiruppur City Center">🎯 Center</button>
        <button class="map-btn" onclick="fitNetwork()" title="Fit all intersection nodes">🔍 Fit</button>
        <button id="btn-emergency" class="map-btn" onclick="locateEmergencyVehicle()" style="display:{'inline-flex' if amb_coords else 'none'};border-color:#ea580c;color:#ea580c;font-weight:800;" title="Pan to Emergency Vehicle">🚨 Locate EV</button>
    </div>

    <!-- Legend -->
    <div class="map-legend">
        <span style="font-weight:800;color:{text_hud};">Flow:</span>
        <div class="legend-item"><div class="color-bar" style="background:#059669;"></div> Free Flow (&lt;3)</div>
        <div class="legend-item"><div class="color-bar" style="background:#d97706;"></div> Slow (3-6)</div>
        <div class="legend-item"><div class="color-bar" style="background:#be123c;"></div> Congested (&gt;6)</div>
        <div class="legend-item"><div class="color-bar" style="background:#ea580c;"></div> 🚨 Green Wave</div>
    </div>

    <script>
        const nodesData = {nodes_json};
        const roadsData = {roads_json};
        const corridorPts = {corridor_json};
        const vehiclesData = {vehicles_json};
        const ambulanceCoords = {amb_json};
        const defaultCenter = [{TIRUPPUR_CENTER_LAT}, {TIRUPPUR_CENTER_LON}];
        const defaultZoom = 14;

        // Safe storage helpers that never crash in restricted/sandboxed iframes
        function safeGet(k, defVal) {{
            try {{
                return (typeof window !== 'undefined' && window.sessionStorage) ? (window.sessionStorage.getItem(k) || defVal) : defVal;
            }} catch (e) {{
                return defVal;
            }}
        }}

        function safeSet(k, val) {{
            try {{
                if (typeof window !== 'undefined' && window.sessionStorage) window.sessionStorage.setItem(k, val);
            }} catch (e) {{}}
        }}

        // 1. Camera Persistence: Retrieve user's previous zoom and pan coordinates safely
        let initialLat = parseFloat(safeGet('qsignal_center_lat', defaultCenter[0]));
        let initialLon = parseFloat(safeGet('qsignal_center_lon', defaultCenter[1]));
        let initialZoom = parseInt(safeGet('qsignal_zoom', defaultZoom));
        let savedLayer = safeGet('qsignal_layer', ('{theme}' === 'mild' ? 'streets' : 'dark'));

        // Sanity check coordinates
        if (isNaN(initialLat) || isNaN(initialLon) || Math.abs(initialLat - defaultCenter[0]) > 0.5) {{
            initialLat = defaultCenter[0];
            initialLon = defaultCenter[1];
            initialZoom = defaultZoom;
        }}

        // 2. Tile Layers
        const osmStreets = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }});

        const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            maxZoom: 19,
            attribution: 'Tiles &copy; Esri'
        }});

        const cartoDark = L.tileLayer('https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '&copy; OpenStreetMap &copy; CARTO'
        }});

        const map = L.map('map', {{
            center: [initialLat, initialLon],
            zoom: initialZoom,
            zoomControl: true
        }});

        // Force Leaflet to calculate container dimensions immediately
        setTimeout(function() {{
            try {{
                map.invalidateSize();
            }} catch(e) {{}}
        }}, 200);

        window.addEventListener('resize', function() {{
            try {{
                map.invalidateSize();
            }} catch(e) {{}}
        }});

        // Layer dictionary
        const layers = {{
            'streets': osmStreets,
            'sat': esriSatellite,
            'dark': cartoDark
        }};

        // Apply saved or default tile layer
        let currentLayerKey = layers[savedLayer] ? savedLayer : ('{theme}' === 'mild' ? 'streets' : 'dark');
        layers[currentLayerKey].addTo(map);
        updateButtonStyles(currentLayerKey);

        function switchLayer(key) {{
            if (!layers[key]) return;
            Object.values(layers).forEach(l => map.removeLayer(l));
            layers[key].addTo(map);
            currentLayerKey = key;
            safeSet('qsignal_layer', key);
            updateButtonStyles(key);
        }}

        function updateButtonStyles(activeKey) {{
            ['streets', 'sat', 'dark'].forEach(k => {{
                const btn = document.getElementById('btn-' + k);
                if (btn) {{
                    if (k === activeKey) {{
                        btn.classList.add('active');
                    }} else {{
                        btn.classList.remove('active');
                    }}
                }}
            }});
        }}

        // 3. Save Camera Position on User Pan / Zoom (Persists through Streamlit ticks)
        map.on('moveend', function() {{
            try {{
                const c = map.getCenter();
                safeSet('qsignal_center_lat', c.lat);
                safeSet('qsignal_center_lon', c.lng);
                safeSet('qsignal_zoom', map.getZoom());
            }} catch(e) {{}}
        }});

        // 4. Render Roads along Real Street Geometry with Congestion Heat
        roadsData.forEach(road => {{
            // Outer soft glow line representing traffic load
            L.polyline(road.pts, {{
                color: road.color,
                weight: 9,
                opacity: 0.38,
                lineCap: 'round',
                lineJoin: 'round',
                smoothFactor: 1.0
            }}).addTo(map);

            // Core street line
            const coreLine = L.polyline(road.pts, {{
                color: road.color,
                weight: 4.5,
                opacity: 0.95,
                lineCap: 'round',
                lineJoin: 'round',
                smoothFactor: 1.0
            }}).addTo(map);

            coreLine.bindTooltip(
                `<b>🛣️ ${{road.name}}</b><br>` +
                `Flow: <span style="font-weight:700;color:${{road.color}}">${{road.status}}</span><br>` +
                `Average Approach Queue: <b>${{road.queue}} veh</b>`,
                {{ sticky: true }}
            );
        }});

        // 5. Render Emergency Priority Green Wave Corridor
        if (corridorPts && corridorPts.length >= 2) {{
            // Outer emergency glow
            L.polyline(corridorPts, {{
                color: '#ea580c',
                weight: 12,
                opacity: 0.45,
                lineCap: 'round'
            }}).addTo(map);

            // Animated pulsating dash
            const corridorLine = L.polyline(corridorPts, {{
                color: '#f97316',
                weight: 6,
                opacity: 1.0,
                dashArray: '10, 8',
                className: 'corridor-dash'
            }}).addTo(map);

            corridorLine.bindTooltip(
                `🚨 <b>EMERGENCY PRIORITY GREEN WAVE CORRIDOR</b><br>` +
                `Signals pre-cleared for incoming emergency vehicle (75 km/h)`,
                {{ sticky: true }}
            );
        }}

        // 6. Render Real-time Intersection Signal Heads (Clear, Bold, High Visibility)
        const allMarkers = [];
        nodesData.forEach(node => {{
            const borderStyle = node.isReserved 
                ? '2.5px solid #ea580c; box-shadow: 0 0 16px rgba(234, 88, 12, 0.85);' 
                : `2px solid ${{node.color}};`;

            const reservedBadge = node.isReserved 
                ? `<div style="color:#ea580c;font-size:9.5px;font-weight:800;letter-spacing:0.5px;margin-top:3px;background:rgba(234,88,12,0.12);padding:1px 4px;border-radius:4px;">🚨CORRIDOR</div>`
                : '';

            const cardHtml = `
                <div class="junction-card" style="border:${{borderStyle}}">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">
                        <span style="font-weight:800;font-size:13px;color:{text_hud};">${{node.id}}</span>
                        <span style="color:${{node.color}};font-size:13px;font-weight:900;background:rgba(0,0,0,0.04);padding:1px 5px;border-radius:4px;">${{node.symbol}}</span>
                    </div>
                    <div style="font-size:11.5px;font-weight:700;color:{'#1e293b' if is_mild else '#f1f5f9'};white-space:nowrap;max-width:115px;overflow:hidden;text-overflow:ellipsis;margin-bottom:3px;" title="${{node.name}}">${{node.shortName}}</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;font-size:11px;padding-top:3px;border-top:1px solid {'#f1f5f9' if is_mild else '#334155'};">
                        <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:{'#475569' if is_mild else '#94a3b8'};">⏱️${{node.timer}}s</span>
                        <span style="font-weight:800;color:${{node.color}};">Q:${{node.totalQueue}}</span>
                    </div>
                    ${{reservedBadge}}
                </div>
            `;

            const icon = L.divIcon({{
                html: cardHtml,
                className: '',
                iconSize: [116, 60],
                iconAnchor: [58, 30]
            }});

            const popupHtml = `
                <div style="font-family:Inter,sans-serif;min-width:200px;">
                    <div style="font-weight:800;font-size:14px;margin-bottom:2px;color:#0f172a;">📍 ${{node.id}}: ${{node.name}}</div>
                    <div style="font-size:11px;color:#64748b;margin-bottom:8px;">Tiruppur Metropolitan Network</div>
                    <div style="background:{'#f8fafc' if is_mild else '#1e293b'};padding:8px;border-radius:6px;font-size:12px;line-height:1.6;border:1px solid #e2e8f0;">
                        <div>Phase: <b style="color:${{node.color}}">${{node.symbol}} ${{node.phaseDesc}}</b></div>
                        <div>Elapsed Green: <b>${{node.timer}} seconds</b></div>
                        <div>NS Queue: <b>${{node.nsQueue}} veh</b></div>
                        <div>EW Queue: <b>${{node.ewQueue}} veh</b></div>
                        <div>Total Junction Queue: <b>${{node.totalQueue}} veh</b></div>
                        ${{node.isReserved ? '<div style="color:#ea580c;font-weight:800;margin-top:4px;">🚨 EMERGENCY GREEN WAVE CLEARED</div>' : ''}}
                    </div>
                </div>
            `;

            const marker = L.marker([node.lat, node.lon], {{ icon: icon }})
                .addTo(map)
                .bindPopup(popupHtml, {{ maxWidth: 300 }})
                .bindTooltip(`📍 ${{node.id}}: ${{node.shortName}} | ${{node.symbol}} ${{node.phaseDesc}} | Q=${{node.totalQueue}}`);

            allMarkers.push([node.lat, node.lon]);
        }});

        // 7. Render Moving & Stopped Vehicles along the Street Geometry
        vehiclesData.forEach(v => {{
            let vehicleIconHtml = '';
            let iconSize = [28, 28];
            let iconAnchor = [14, 14];

            if (v.isEmergency) {{
                vehicleIconHtml = `<div class="amb-siren" style="font-size:26px;filter:drop-shadow(0 0 8px rgba(234,88,12,0.95));">🚑</div>`;
                iconSize = [32, 32];
                iconAnchor = [16, 16];
            }} else if (v.isStopped) {{
                vehicleIconHtml = `
                    <div style="font-size:11px;background:#ffffff;border:1.5px solid #be123c;border-radius:12px;padding:2px 6px;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,0.18);font-weight:800;color:#be123c;">
                        🛑 ${{v.icon}} Wait
                    </div>
                `;
                iconSize = [56, 22];
                iconAnchor = [28, 11];
            }} else {{
                vehicleIconHtml = `
                    <div style="font-size:16px;filter:drop-shadow(0 1px 3px rgba(0,0,0,0.3));">${{v.icon}}</div>
                `;
            }}

            const vIcon = L.divIcon({{
                html: vehicleIconHtml,
                className: '',
                iconSize: iconSize,
                iconAnchor: iconAnchor
            }});

            const tooltipText = v.isEmergency
                ? `🚨 <b>Priority Emergency Vehicle</b><br>Speed: 75 km/h (Preemption Wave)`
                : (v.isStopped ? `🛑 <b>${{v.id}}</b>: Idling at red signal` : `🟢 <b>${{v.id}}</b>: Moving at ${{v.speed}} km/h`);

            L.marker([v.lat, v.lon], {{ icon: vIcon, zIndexOffset: v.isEmergency ? 1000 : 100 }})
                .addTo(map)
                .bindTooltip(tooltipText);
        }});

        // Action Handlers
        function centerTiruppur() {{
            map.flyTo(defaultCenter, 14, {{ duration: 0.8 }});
            safeSet('qsignal_center_lat', defaultCenter[0]);
            safeSet('qsignal_center_lon', defaultCenter[1]);
            safeSet('qsignal_zoom', 14);
        }}

        function fitNetwork() {{
            if (allMarkers.length > 0) {{
                map.fitBounds(allMarkers, {{ padding: [40, 40] }});
            }}
        }}

        function locateEmergencyVehicle() {{
            if (ambulanceCoords) {{
                map.flyTo(ambulanceCoords, 16, {{ duration: 0.8 }});
            }}
        }}
    </script>
</body>
</html>"""
    return html_content



# ---------------------------------------------------------------------------
# Queue chart
# ---------------------------------------------------------------------------

def plot_queue_history(
    records: Dict[str, List],
    title: str = "Queue Length Over Time",
    theme: str = "mild",
) -> plt.Figure:
    """
    Plot time-series queue length for multiple controllers.
    records: {controller_name: [(time_s, mean_queue), ...]}
    """
    is_mild = (theme == "mild")
    fig_bg = "#ffffff" if is_mild else "#0f172a"
    ax_bg = "#f8fafc" if is_mild else "#1e293b"
    text_color = "#0f172a" if is_mild else "#f1f5f9"
    subtext_color = "#64748b" if is_mild else "#94a3b8"
    spine_color = "#e2e8f0" if is_mild else "#334155"
    grid_color = "#f1f5f9" if is_mild else "#334155"

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=fig_bg)
    ax.set_facecolor(ax_bg)
    ax.tick_params(colors=subtext_color)
    ax.spines[:].set_color(spine_color)
    ax.title.set_color(text_color)
    ax.xaxis.label.set_color(subtext_color)
    ax.yaxis.label.set_color(subtext_color)

    colors = {
        "fixed": "#dc2626" if is_mild else "#ef4444",
        "pressure": "#d97706" if is_mild else "#f59e0b",
        "qaoa": "#16a34a" if is_mild else "#22c55e",
        "unknown": "#6b7280",
    }

    for ctrl, data in records.items():
        if not data:
            continue
        times = [d[0] for d in data]
        values = [d[1] for d in data]
        ax.plot(times, values, label=ctrl.title(), color=colors.get(ctrl, "#64748b"),
                linewidth=2.2 if is_mild else 2, alpha=0.95)

    ax.set_xlabel("Simulation Time (s)", fontweight="500")
    ax.set_ylabel("Mean Queue Length", fontweight="500")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(facecolor=fig_bg, labelcolor=text_color, edgecolor=spine_color, framealpha=0.95)
    ax.grid(True, color=grid_color, linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    return fig


def plot_comparison_bar(
    summary: pd.DataFrame,
    metric: str = "mean_queue_mean",
    title: str = "Controller Comparison",
    theme: str = "mild",
) -> plt.Figure:
    """Bar chart comparing controllers on a single metric."""
    is_mild = (theme == "mild")
    fig_bg = "#ffffff" if is_mild else "#0f172a"
    ax_bg = "#f8fafc" if is_mild else "#1e293b"
    text_color = "#0f172a" if is_mild else "#f1f5f9"
    subtext_color = "#64748b" if is_mild else "#94a3b8"
    spine_color = "#e2e8f0" if is_mild else "#334155"
    grid_color = "#f1f5f9" if is_mild else "#334155"

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=fig_bg)
    ax.set_facecolor(ax_bg)
    ax.tick_params(colors=subtext_color)
    ax.spines[:].set_color(spine_color)

    if summary.empty or "controller" not in summary.columns or metric not in summary.columns:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=subtext_color)
        return fig

    controllers = summary["controller"].tolist()
    values = summary[metric].tolist()
    palette = ["#dc2626", "#d97706", "#16a34a", "#2563eb"] if is_mild else ["#ef4444", "#f59e0b", "#22c55e", "#3b82f6"]
    colors = [palette[i % len(palette)] for i in range(len(controllers))]

    bars = ax.bar(controllers, values, color=colors, edgecolor=fig_bg, linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 * max(values or [1]),
            f"{val:.2f}",
            ha="center", va="bottom", color=text_color, fontsize=10, fontweight="600"
        )

    ax.set_title(title, color=text_color, fontsize=12, fontweight="bold")
    ax.set_ylabel(metric.replace("_", " ").title(), color=subtext_color, fontweight="500")
    ax.grid(True, axis="y", color=grid_color, linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    return fig


def plot_metrics_panel(df: pd.DataFrame, theme: str = "mild") -> plt.Figure:
    """
    2×2 panel showing queue, delay, throughput, and CO2 over time
    for all controllers on the same experiment.
    """
    is_mild = (theme == "mild")
    fig_bg = "#ffffff" if is_mild else "#0f172a"
    ax_bg = "#f8fafc" if is_mild else "#1e293b"
    text_color = "#0f172a" if is_mild else "#f1f5f9"
    subtext_color = "#64748b" if is_mild else "#94a3b8"
    spine_color = "#e2e8f0" if is_mild else "#334155"
    grid_color = "#f1f5f9" if is_mild else "#334155"

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor=fig_bg)
    fig.suptitle("Simulation Metrics Comparison", color=text_color, fontsize=14, fontweight="bold")

    metrics = [
        ("mean_queue", "Mean Queue Length", False),
        ("mean_wait", "Mean Wait Time (s)", False),
        ("throughput_total", "Total Throughput", True),
        ("estimated_co2_kg", "Est. CO₂ (kg) [model estimate]", False),
    ]

    palette = {
        "fixed": "#dc2626" if is_mild else "#ef4444",
        "pressure": "#d97706" if is_mild else "#f59e0b",
        "qaoa": "#16a34a" if is_mild else "#22c55e"
    }

    for ax, (col, label, invert) in zip(axes.flat, metrics):
        ax.set_facecolor(ax_bg)
        ax.tick_params(colors=subtext_color, labelsize=8)
        ax.spines[:].set_color(spine_color)
        ax.set_title(label, color=subtext_color, fontsize=9.5, fontweight="600")

        if df.empty or col not in df.columns:
            continue

        for ctrl, grp in df.groupby("controller"):
            ax.plot(
                grp["time_s"], grp[col],
                label=ctrl.title(),
                color=palette.get(ctrl, "#6b7280"),
                linewidth=1.7, alpha=0.95,
            )
        ax.legend(facecolor=fig_bg, labelcolor=text_color, edgecolor=spine_color, fontsize=7.5)
        ax.grid(True, color=grid_color, linewidth=0.6, alpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def phase_legend_html(theme: str = "mild") -> str:
    """Return a clean, high-contrast HTML legend for signal phases with mild colors."""
    is_mild = (theme == "mild")
    card_bg = "#ffffff" if is_mild else "#0f172a"
    card_border = "#cbd5e1" if is_mild else "#334155"
    text_color = "#0f172a" if is_mild else "#f8fafc"
    ns_color = "#059669" if is_mild else "#10b981"  # Soft Sage Green
    ew_color = "#4338ca" if is_mild else "#6366f1"  # Mild Royal Slate / Indigo
    amber_color = "#d97706" if is_mild else "#f59e0b"  # Warm Honey Amber
    red_color = "#be123c" if is_mild else "#f43f5e"  # Soft Rose Terracotta

    items = f"""
    <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:7px;">
            <span style="display:inline-flex;align-items:center;justify-content:center;background:rgba(5,150,105,0.12);color:{ns_color};font-size:13px;font-weight:900;padding:2px 6px;border-radius:4px;border:1px solid {ns_color};">▲▼</span>
            <span style="font-size:13px;color:{text_color};font-weight:700;">NS Green (Vertical)</span>
        </div>
        <div style="display:flex;align-items:center;gap:7px;">
            <span style="display:inline-flex;align-items:center;justify-content:center;background:rgba(67,56,202,0.12);color:{ew_color};font-size:13px;font-weight:900;padding:2px 6px;border-radius:4px;border:1px solid {ew_color};">◀▶</span>
            <span style="font-size:13px;color:{text_color};font-weight:700;">EW Green (Horizontal)</span>
        </div>
        <div style="display:flex;align-items:center;gap:7px;">
            <span style="display:inline-block;width:13px;height:13px;background:{amber_color};border-radius:50%;border:1.5px solid {card_bg};box-shadow:0 0 4px {amber_color};"></span>
            <span style="font-size:13px;color:{text_color};font-weight:700;">Yellow (Caution)</span>
        </div>
        <div style="display:flex;align-items:center;gap:7px;">
            <span style="display:inline-block;width:13px;height:13px;background:{red_color};border-radius:3px;border:1.5px solid {card_bg};box-shadow:0 0 4px {red_color};"></span>
            <span style="font-size:13px;color:{text_color};font-weight:700;">All-Red (Clearance)</span>
        </div>
    </div>
    """
    return f'<div style="padding:8px 16px;background:{card_bg};border-radius:8px;margin-bottom:8px;border:1.5px solid {card_border};box-shadow:0 2px 8px rgba(15,23,42,0.06);">{items}</div>'


def build_network_svg(
    state: TrafficState,
    emergency_route: Optional[List[str]] = None,
    width: int = 720,
    height: int = 420,
    theme: str = "mild",
) -> str:
    """
    Generate a high-fidelity, flicker-free SVG network diagram for real-time rendering.
    Supports 'mild' (clean soft slate/white) and 'dark' themes.
    """
    import math

    is_mild = (theme == "mild")

    node_ids = list(state.intersections.keys())
    n_nodes = len(node_ids)
    if n_nodes == 0:
        return "<svg width='100%' height='200'></svg>"

    cols = math.ceil(math.sqrt(n_nodes))
    rows = math.ceil(n_nodes / cols)

    pad_x = 90
    pad_y = 70
    usable_w = width - 2 * pad_x
    usable_h = height - 2 * pad_y

    dx = usable_w / max(cols - 1, 1) if cols > 1 else usable_w / 2
    dy = usable_h / max(rows - 1, 1) if rows > 1 else usable_h / 2

    # Map node to SVG pixel coordinates
    node_coords: Dict[str, Tuple[float, float]] = {}
    for idx, nid in enumerate(node_ids):
        r = idx // cols
        c = idx % cols
        cx = pad_x + (c * dx if cols > 1 else usable_w / 2)
        cy = pad_y + (r * dy if rows > 1 else usable_h / 2)
        node_coords[nid] = (cx, cy)

    emergency_route_set = set()
    if emergency_route and len(emergency_route) >= 2:
        for i in range(len(emergency_route) - 1):
            pair = tuple(sorted([emergency_route[i], emergency_route[i+1]]))
            emergency_route_set.add(pair)

    svg_bg = "#f8fafc" if is_mild else "#0a0f1e"
    svg_border = "#cbd5e1" if is_mild else "#1e3a5f"
    road_base_stroke = "#cbd5e1" if is_mild else "#1e293b"
    road_center_stroke = "#94a3b8" if is_mild else "#475569"
    box_fill = "#ffffff" if is_mild else "#0b132b"
    box_default_stroke = "#94a3b8" if is_mild else "#334155"
    node_text_color = "#0f172a" if is_mild else "#f8fafc"
    timer_text_color = "#64748b" if is_mild else "#94a3b8"
    pill_fill = "#ffffff" if is_mild else "#0d1b2e"

    svg_parts = []
    svg_parts.append(
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:{svg_bg};border-radius:12px;'
        f'border:1px solid {svg_border};font-family:Inter,sans-serif;user-select:none;box-shadow:0 2px 10px rgba(0,0,0,0.04);">'
    )

    # Definitions (glow filters, gradients)
    if is_mild:
        svg_parts.append("""
        <defs>
          <filter id="soft-shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.08"/>
          </filter>
        </defs>
        """)
    else:
        svg_parts.append("""
        <defs>
          <filter id="glow-green" x="-30%" y="-30%" width="160%" height="160%">
            <feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="#22c55e" flood-opacity="0.9"/>
          </filter>
          <filter id="glow-blue" x="-30%" y="-30%" width="160%" height="160%">
            <feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="#38bdf8" flood-opacity="0.9"/>
          </filter>
          <filter id="glow-amber" x="-30%" y="-30%" width="160%" height="160%">
            <feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="#f59e0b" flood-opacity="0.9"/>
          </filter>
          <filter id="glow-red" x="-30%" y="-30%" width="160%" height="160%">
            <feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="#ef4444" flood-opacity="0.8"/>
          </filter>
          <filter id="glow-corridor" x="-30%" y="-30%" width="160%" height="160%">
            <feDropShadow dx="0" dy="0" stdDeviation="6" flood-color="#f97316" flood-opacity="1"/>
          </filter>
        </defs>
        """)

    # 1. Draw connecting road corridors
    drawn_edges = set()
    for idx, nid in enumerate(node_ids):
        c = idx % cols
        r = idx // cols
        x1, y1 = node_coords[nid]

        # Horizontal neighbor (East)
        if c + 1 < cols and (idx + 1) < n_nodes:
            nid_east = node_ids[idx + 1]
            x2, y2 = node_coords[nid_east]
            key = tuple(sorted([nid, nid_east]))
            if key not in drawn_edges:
                drawn_edges.add(key)
                is_em = key in emergency_route_set
                svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{road_base_stroke}" stroke-width="28" stroke-linecap="round"/>')
                if is_em:
                    em_stroke = "#ea580c" if is_mild else "#f97316"
                    glow_attr = "" if is_mild else 'filter="url(#glow-corridor)"'
                    svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{em_stroke}" stroke-width="7" stroke-dasharray="10,6" {glow_attr}/>')
                else:
                    svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{road_center_stroke}" stroke-width="2" stroke-dasharray="8,6"/>')

        # Vertical neighbor (South)
        if (idx + cols) < n_nodes:
            nid_south = node_ids[idx + cols]
            x2, y2 = node_coords[nid_south]
            key = tuple(sorted([nid, nid_south]))
            if key not in drawn_edges:
                drawn_edges.add(key)
                is_em = key in emergency_route_set
                svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{road_base_stroke}" stroke-width="28" stroke-linecap="round"/>')
                if is_em:
                    em_stroke = "#ea580c" if is_mild else "#f97316"
                    glow_attr = "" if is_mild else 'filter="url(#glow-corridor)"'
                    svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{em_stroke}" stroke-width="7" stroke-dasharray="10,6" {glow_attr}/>')
                else:
                    svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{road_center_stroke}" stroke-width="2" stroke-dasharray="8,6"/>')

    # 2. Draw intersections and status indicators
    for nid, inter in state.intersections.items():
        cx, cy = node_coords.get(nid, (width / 2, height / 2))
        phase = inter.phase

        # Phase signal colors
        if phase == PHASE_NS:
            ns_color = "#16a34a" if is_mild else "#22c55e"
            ns_filter = "" if is_mild else 'filter="url(#glow-green)"'
            ew_color, ew_filter = ("#dc2626" if is_mild else "#ef4444"), ""
        elif phase == PHASE_EW:
            ns_color, ns_filter = ("#dc2626" if is_mild else "#ef4444"), ""
            ew_color = "#2563eb" if is_mild else "#38bdf8"
            ew_filter = "" if is_mild else 'filter="url(#glow-blue)"'
        elif phase == PHASE_YELLOW:
            ns_color = "#d97706" if is_mild else "#f59e0b"
            ns_filter = "" if is_mild else 'filter="url(#glow-amber)"'
            ew_color, ew_filter = ns_color, ns_filter
        else:
            ns_color = "#dc2626" if is_mild else "#ef4444"
            ns_filter = "" if is_mild else 'filter="url(#glow-red)"'
            ew_color, ew_filter = ns_color, ns_filter

        # Get queues
        ns_q = next((a.queue for a in state.approaches.values() if a.intersection_id == nid and a.movement == "NS"), 0.0)
        ew_q = next((a.queue for a in state.approaches.values() if a.intersection_id == nid and a.movement == "EW"), 0.0)

        # Intersection box
        box_stroke = ("#ea580c" if is_mild else "#f97316") if inter.is_reserved else box_default_stroke
        box_width = "2.5" if inter.is_reserved else "1.5"
        box_filter = 'filter="url(#soft-shadow)"' if is_mild else ('filter="url(#glow-corridor)"' if inter.is_reserved else "")

        svg_parts.append(
            f'<rect x="{cx-36}" y="{cy-36}" width="72" height="72" rx="14" ry="14" '
            f'fill="{box_fill}" stroke="{box_stroke}" stroke-width="{box_width}" {box_filter}/>'
        )

        # Signal indicators: Directional arrow shapes (▲/▼ for NS, ◀/▶ for EW)
        # NS signals (North ▲ / South ▼ directional arrows)
        svg_parts.append(f'<polygon points="{cx},{cy-29} {cx-7},{cy-18} {cx+7},{cy-18}" fill="{ns_color}" {ns_filter}/>')
        svg_parts.append(f'<polygon points="{cx},{cy+29} {cx-7},{cy+18} {cx+7},{cy+18}" fill="{ns_color}" {ns_filter}/>')
        # EW signals (West ◀ / East ▶ directional arrows)
        svg_parts.append(f'<polygon points="{cx-29},{cy} {cx-18},{cy-7} {cx-18},{cy+7}" fill="{ew_color}" {ew_filter}/>')
        svg_parts.append(f'<polygon points="{cx+29},{cy} {cx+18},{cy-7} {cx+18},{cy+7}" fill="{ew_color}" {ew_filter}/>')

        # Center node label
        node_text = f"{nid}"
        if inter.is_reserved:
            node_text = f"🚨{nid}"
        svg_parts.append(
            f'<text x="{cx}" y="{cy+2}" fill="{node_text_color}" font-size="13" font-weight="700" '
            f'text-anchor="middle" dominant-baseline="middle">{node_text}</text>'
        )

        # Timer below node
        svg_parts.append(
            f'<text x="{cx}" y="{cy+14}" fill="{timer_text_color}" font-size="9" '
            f'text-anchor="middle">{inter.elapsed_green}s</text>'
        )

        # Queue badges (pills)
        def q_color(q: float) -> str:
            if q > 12: return "#dc2626" if is_mild else "#ef4444"
            if q > 6: return "#d97706" if is_mild else "#f59e0b"
            return "#2563eb" if is_mild else "#38bdf8"

        # NS queue badge (placed above)
        ns_badge_color = q_color(ns_q)
        svg_parts.append(
            f'<rect x="{cx-26}" y="{cy-55}" width="52" height="15" rx="5" fill="{pill_fill}" stroke="{ns_badge_color}" stroke-width="1"/>'
            f'<text x="{cx}" y="{cy-44}" fill="{ns_badge_color}" font-size="9" font-weight="600" text-anchor="middle">NS {ns_q:.1f}</text>'
        )

        # EW queue badge (placed to the side)
        ew_badge_color = q_color(ew_q)
        svg_parts.append(
            f'<rect x="{cx+42}" y="{cy-8}" width="52" height="15" rx="5" fill="{pill_fill}" stroke="{ew_badge_color}" stroke-width="1"/>'
            f'<text x="{cx+68}" y="{cy+3}" fill="{ew_badge_color}" font-size="9" font-weight="600" text-anchor="middle">EW {ew_q:.1f}</text>'
        )

    # 3. Draw active vehicles moving through intersection points and stopping at red lights
    for v in state.vehicles.values():
        if v.completed or not v.from_node or not v.to_node:
            continue
        if v.from_node not in node_coords or v.to_node not in node_coords:
            continue

        x1, y1 = node_coords[v.from_node]
        x2, y2 = node_coords[v.to_node]
        p = max(0.08, min(0.92, v.progress))

        # Position along road between intersection points
        vx = x1 + p * (x2 - x1)
        vy = y1 + p * (y2 - y1)

        # Right-hand lane offset for directional traffic
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            nx = -dy / length
            ny = dx / length
            vx += nx * 6.5
            vy += ny * 6.5

        if v.emergency:
            # Code 3 Emergency Ambulance
            badge_bg = "#ea580c" if is_mild else "#f97316"
            svg_parts.append(
                f'<g transform="translate({vx-18},{vy-18})">'
                f'<circle cx="18" cy="18" r="22" fill="none" stroke="{badge_bg}" stroke-width="2.5" opacity="0.8">'
                f'<animate attributeName="r" values="16;28;16" dur="0.9s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="0.9;0.1;0.9" dur="0.9s" repeatCount="indefinite"/>'
                f'</circle>'
                f'<circle cx="12" cy="7" r="3" fill="#dc2626">'
                f'<animate attributeName="opacity" values="1;0.1;1" dur="0.3s" repeatCount="indefinite"/>'
                f'</circle>'
                f'<circle cx="24" cy="7" r="3" fill="#2563eb">'
                f'<animate attributeName="opacity" values="0.1;1;0.1" dur="0.3s" repeatCount="indefinite"/>'
                f'</circle>'
                f'<text x="18" y="22" font-size="18" text-anchor="middle">🚑</text>'
                f'<rect x="1" y="-8" width="34" height="10" rx="3" fill="{badge_bg}"/>'
                f'<text x="18" y="-1" font-size="6.5" fill="#ffffff" font-weight="900" text-anchor="middle">PRIORITY</text>'
                f'</g>'
            )
        elif v.is_stopped:
            # Vehicle stopped at red signal or waiting in queue
            v_stop_fill = "#ffffff" if is_mild else "#0b132b"
            v_stop_stroke = "#dc2626" if is_mild else "#ef4444"
            svg_parts.append(
                f'<g transform="translate({vx-10},{vy-10})">'
                f'<rect width="20" height="20" rx="6" fill="{v_stop_fill}" stroke="{v_stop_stroke}" stroke-width="1.3" opacity="0.96"/>'
                f'<circle cx="4" cy="4" r="1.5" fill="{v_stop_stroke}"/>'
                f'<circle cx="16" cy="4" r="1.5" fill="{v_stop_stroke}"/>'
                f'<text x="10" y="14" font-size="11" text-anchor="middle">{v.icon}</text>'
                f'</g>'
            )
        else:
            # Moving vehicle clearing intersection on green
            headlight = "#ca8a04" if is_mild else "#fef08a"
            svg_parts.append(
                f'<g transform="translate({vx-8},{vy-8})">'
                f'<circle cx="3" cy="14" r="1.2" fill="{headlight}"/>'
                f'<circle cx="13" cy="14" r="1.2" fill="{headlight}"/>'
                f'<text x="8" y="12" font-size="12" text-anchor="middle">{v.icon}</text>'
                f'</g>'
            )

    svg_parts.append('</svg>')
    return "".join(svg_parts)


# ---------------------------------------------------------------------------
# Carbon & Monthly Traffic AI Visualizations
# ---------------------------------------------------------------------------

def plot_monthly_carbon_altair(
    daily_df: pd.DataFrame,
    stats: Optional[Dict[str, Any]] = None,
    theme: str = "mild",
) -> Any:
    """
    Client-side interactive Altair chart for 1-month carbon emissions & traffic.
    Zero-blink rendering with explicit day names, weekend indicators, and detailed
    Q-Signal emission reduction tooltips.
    """
    if not _ALTAIR_AVAILABLE:
        return None

    df = daily_df.copy()
    is_mild = (theme == "mild")

    red_pct_val = stats.get("reduction_pct", 35.0) if stats else 35.0
    saved_kg_val = stats.get("total_co2_saved_kg", df["co2_saved_kg"].sum()) if stats else df["co2_saved_kg"].sum()

    df["reduction_pct_str"] = f"-{red_pct_val:.1f}%"
    df["co2_saved_display"] = df["co2_saved_kg"].apply(lambda v: f"-{v:.1f} kg")
    df["weekend_label"] = df["is_weekend"].apply(lambda w: "Weekend (Light Flow)" if w else "Weekday (Commute)")

    # Green shaded gap representing carbon prevented by Q-Signal
    area = alt.Chart(df).mark_area(
        opacity=0.22,
        color="#16a34a" if is_mild else "#22c55e"
    ).encode(
        x=alt.X(
            "day:O",
            title="Calendar Days of Month (1 to 30)",
            axis=alt.Axis(
                labelAngle=-40,
                labelFontSize=9,
                labelColor="#64748b" if is_mild else "#94a3b8",
                titleColor="#0f172a" if is_mild else "#f1f5f9",
                labelExpr="datum.label + ' (' + ['Wed','Thu','Fri','Sat','Sun','Mon','Tue'][(datum.value-1)%7] + ')'"
            )
        ),
        y=alt.Y("optimized_co2_kg:Q", title="Waiting Carbon Emissions (kg CO₂ / day)"),
        y2="baseline_co2_kg:Q"
    )

    # Baseline line (Red)
    line_base = alt.Chart(df).mark_line(
        color="#dc2626" if is_mild else "#ef4444",
        strokeWidth=2.2,
        point=alt.OverlayMarkDef(color="#dc2626" if is_mild else "#ef4444", size=36)
    ).encode(
        x=alt.X("day:O"),
        y=alt.Y("baseline_co2_kg:Q"),
        tooltip=[
            alt.Tooltip("day_label:N", title="🗓️ Calendar Day"),
            alt.Tooltip("weekend_label:N", title="📅 Period"),
            alt.Tooltip("baseline_co2_kg:Q", title="🛑 Baseline CO₂ (kg)", format=".1f"),
            alt.Tooltip("optimized_co2_kg:Q", title="🟢 Q-Signal CO₂ (kg)", format=".1f"),
            alt.Tooltip("co2_saved_display:N", title="🍃 CO₂ Reduced by Q-Signal"),
            alt.Tooltip("reduction_pct_str:N", title="⚛️ Reduction %"),
            alt.Tooltip("avg_wait_baseline_s:Q", title="⏱️ Baseline Wait (s)", format=".1f"),
            alt.Tooltip("avg_wait_optimized_s:Q", title="⚡ Q-Signal Wait (s)", format=".1f"),
            alt.Tooltip("total_vehicles:Q", title="🚗 Traffic Volume", format=","),
            alt.Tooltip("trees_saved_equiv:Q", title="🌲 Trees Preserved", format=".1f"),
        ]
    )

    # Q-Signal Optimized line (Green)
    line_opt = alt.Chart(df).mark_line(
        color="#16a34a" if is_mild else "#22c55e",
        strokeWidth=2.6,
        point=alt.OverlayMarkDef(color="#16a34a" if is_mild else "#22c55e", size=36)
    ).encode(
        x=alt.X("day:O"),
        y=alt.Y("optimized_co2_kg:Q"),
        tooltip=[
            alt.Tooltip("day_label:N", title="🗓️ Calendar Day"),
            alt.Tooltip("weekend_label:N", title="📅 Period"),
            alt.Tooltip("baseline_co2_kg:Q", title="🛑 Baseline CO₂ (kg)", format=".1f"),
            alt.Tooltip("optimized_co2_kg:Q", title="🟢 Q-Signal CO₂ (kg)", format=".1f"),
            alt.Tooltip("co2_saved_display:N", title="🍃 CO₂ Reduced by Q-Signal"),
            alt.Tooltip("reduction_pct_str:N", title="⚛️ Reduction %"),
            alt.Tooltip("avg_wait_baseline_s:Q", title="⏱️ Baseline Wait (s)", format=".1f"),
            alt.Tooltip("avg_wait_optimized_s:Q", title="⚡ Q-Signal Wait (s)", format=".1f"),
            alt.Tooltip("total_vehicles:Q", title="🚗 Traffic Volume", format=","),
        ]
    )

    chart_title = alt.TitleParams(
        text="30-Day Carbon Emission Trajectory & Q-Signal Savings Gap (Zero-Blink)",
        subtitle=f"Q-Signal cuts -{red_pct_val:.1f}% carbon (-{saved_kg_val:,.1f} kg CO₂ prevented). Hover over any day to see exact savings.",
        color="#0f172a" if is_mild else "#f1f5f9",
        subtitleColor="#16a34a" if is_mild else "#22c55e",
        fontSize=13,
        subtitleFontSize=11,
        anchor="start"
    )

    chart = (area + line_base + line_opt).properties(
        title=chart_title,
        height=360
    ).configure_view(
        strokeOpacity=0
    ).interactive()

    return chart


def plot_monthly_carbon_analysis(
    daily_df: pd.DataFrame,
    stats: Optional[Dict[str, Any]] = None,
    theme: str = "mild",
) -> plt.Figure:
    """
    Innovative dual-axis 1-Month timeline comparing traffic volume,
    baseline waiting emissions, and optimized emissions with savings area.
    Shows explicit calendar days with day-of-week names, weekend shading,
    and prominent Q-Signal carbon reduction callouts.
    """
    is_mild = (theme == "mild")
    fig_bg = "#ffffff" if is_mild else "#0f172a"
    ax_bg = "#f8fafc" if is_mild else "#1e293b"
    text_color = "#0f172a" if is_mild else "#f1f5f9"
    subtext_color = "#64748b" if is_mild else "#94a3b8"
    spine_color = "#e2e8f0" if is_mild else "#334155"
    grid_color = "#f1f5f9" if is_mild else "#334155"

    fig, ax1 = plt.subplots(figsize=(11, 4.8), facecolor=fig_bg)
    ax1.set_facecolor(ax_bg)
    ax1.tick_params(colors=subtext_color)
    ax1.spines[:].set_color(spine_color)
    ax1.title.set_color(text_color)
    ax1.xaxis.label.set_color(subtext_color)
    ax1.yaxis.label.set_color(subtext_color)

    days = daily_df["day"].tolist()
    base_co2 = daily_df["baseline_co2_kg"].tolist()
    opt_co2 = daily_df["optimized_co2_kg"].tolist()
    vehicles = daily_df["total_vehicles"].tolist()

    # Weekend background shading to clearly tell days apart
    for _, row in daily_df.iterrows():
        if row.get("is_weekend"):
            ax1.axvspan(
                row["day"] - 0.45, row["day"] + 0.45,
                color="#e2e8f0" if is_mild else "#334155",
                alpha=0.40,
                zorder=0
            )

    # Left Axis: Carbon Emissions
    line1 = ax1.plot(
        days, base_co2,
        label="Baseline CO₂ (Fixed-Time, kg)",
        color="#dc2626" if is_mild else "#ef4444",
        linewidth=2.2,
        marker="o",
        markersize=4,
        alpha=0.95,
        zorder=3,
    )
    line2 = ax1.plot(
        days, opt_co2,
        label="Optimized CO₂ (Q-Signal QAOA, kg)",
        color="#16a34a" if is_mild else "#22c55e",
        linewidth=2.4,
        marker="s",
        markersize=4,
        alpha=0.95,
        zorder=3,
    )

    # Shaded savings region between curves
    ax1.fill_between(
        days, opt_co2, base_co2,
        color="#22c55e" if not is_mild else "#16a34a",
        alpha=0.18,
        label="CO₂ Prevented via Quantum Optimization",
        zorder=2,
    )

    # Explicit day of week labels on X axis
    day_labels = [f"D{r['day']}\n{r['day_name'][:3]}" for _, r in daily_df.iterrows()]
    tick_indices = list(range(1, 31, 2))
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels([day_labels[i-1] for i in tick_indices], fontsize=8.5)
    ax1.set_xlabel("Days of Month (with Day of Week · Shaded = Weekend)", fontweight="600", fontsize=10)
    ax1.set_ylabel("Waiting Carbon Emissions (kg CO₂ / day)", fontweight="600", fontsize=10)
    ax1.grid(True, color=grid_color, linewidth=0.7, alpha=0.9)

    # Right Axis: Traffic Volume (Vehicles)
    ax2 = ax1.twinx()
    ax2.tick_params(colors=subtext_color)
    ax2.spines[:].set_color(spine_color)
    line3 = ax2.plot(
        days, vehicles,
        label="Daily Vehicle Traffic Volume",
        color="#2563eb" if is_mild else "#38bdf8",
        linewidth=1.8,
        linestyle="--",
        alpha=0.8,
        zorder=3,
    )
    ax2.set_ylabel("Daily Vehicles Passing Junction", color="#2563eb" if is_mild else "#38bdf8", fontweight="600", fontsize=10)

    # Prominent Q-Signal Reduction Badge on the chart
    red_pct = stats.get("reduction_pct", 35.0) if stats else 35.0
    saved_kg = stats.get("total_co2_saved_kg", sum(daily_df["co2_saved_kg"])) if stats else sum(daily_df["co2_saved_kg"])
    saved_t = stats.get("total_co2_saved_tons", saved_kg / 1000.0) if stats else saved_kg / 1000.0
    trees_cnt = int(stats.get("trees_saved_month", saved_kg / (21.8 / 12.0))) if stats else int(saved_kg / (21.8 / 12.0))

    callout_text = (
        f"⚛️ Q-SIGNAL IMPACT:\n"
        f"• Carbon Cut: -{red_pct:.1f}%\n"
        f"• Prevented: -{saved_kg:,.0f} kg ({saved_t:.2f} t)\n"
        f"• Preserved: {trees_cnt:,} trees"
    )
    ax1.text(
        0.98, 0.95, callout_text,
        transform=ax1.transAxes,
        fontsize=8.5,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="#dcfce7" if is_mild else "#064e3b",
            edgecolor="#16a34a" if is_mild else "#10b981",
            linewidth=1.3,
            alpha=0.92
        ),
        color="#14532d" if is_mild else "#a7f3d0",
        zorder=10,
    )

    # Combined Legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    labels.append("CO₂ Emissions Prevented (Green Gap)")
    patch = mpatches.Patch(color="#16a34a" if is_mild else "#22c55e", alpha=0.25)
    handles = lines + [patch]

    ax1.legend(handles, labels, facecolor=fig_bg, labelcolor=text_color, edgecolor=spine_color, framealpha=0.95, loc="upper left", fontsize=8.5)

    ax1.set_title("30-Day Carbon Emission & Daily Traffic Volume Dynamics", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig



def plot_diurnal_hourly_emissions(
    hourly_df: pd.DataFrame,
    selected_day: int = 1,
    theme: str = "mild",
) -> plt.Figure:
    """
    24-hour diurnal profile of a selected day, showing morning/evening commute spikes
    and hourly emissions reduction.
    """
    is_mild = (theme == "mild")
    fig_bg = "#ffffff" if is_mild else "#0f172a"
    ax_bg = "#f8fafc" if is_mild else "#1e293b"
    text_color = "#0f172a" if is_mild else "#f1f5f9"
    subtext_color = "#64748b" if is_mild else "#94a3b8"
    spine_color = "#e2e8f0" if is_mild else "#334155"
    grid_color = "#f1f5f9" if is_mild else "#334155"

    day_data = hourly_df[hourly_df["day"] == selected_day].copy()
    if day_data.empty:
        day_data = hourly_df.head(24).copy()

    hours = day_data["hour"].tolist()
    labels = [f"{h:02d}:00" for h in hours]
    base_co2 = day_data["co2_baseline_kg"].tolist()
    opt_co2 = day_data["co2_optimized_kg"].tolist()
    vehicles = day_data["vehicles"].tolist()
    day_name = day_data.iloc[0]["day_name"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=fig_bg)

    # Plot 1: Hourly Carbon Emissions
    ax1.set_facecolor(ax_bg)
    ax1.tick_params(colors=subtext_color, labelsize=8)
    ax1.spines[:].set_color(spine_color)
    ax1.plot(hours, base_co2, label="Baseline Emissions", color="#dc2626" if is_mild else "#ef4444", linewidth=2.2, marker="o", markersize=3.5)
    ax1.plot(hours, opt_co2, label="Q-Signal Optimized", color="#16a34a" if is_mild else "#22c55e", linewidth=2.2, marker="s", markersize=3.5)
    day_saved = day_data["co2_saved_kg"].sum()
    day_pct = (day_saved / max(day_data["co2_baseline_kg"].sum(), 0.001)) * 100
    ax1.set_title(f"Day {selected_day} ({day_name}) — Q-Signal Cut -{day_saved:.1f} kg CO₂ (-{day_pct:.0f}%)", color="#16a34a" if is_mild else "#22c55e", fontsize=10.0, fontweight="bold")
    ax1.set_xlabel("Hour of Day (00:00 - 23:00)", color=subtext_color, fontsize=9)
    ax1.set_ylabel("Hourly Carbon (kg CO₂)", color=subtext_color, fontsize=9)
    ax1.set_xticks(range(0, 24, 3))
    ax1.legend(facecolor=fig_bg, labelcolor=text_color, edgecolor=spine_color, fontsize=8)
    ax1.grid(True, color=grid_color, linewidth=0.6)

    # Plot 2: Diurnal Traffic Flow Peak Cycle
    ax2.set_facecolor(ax_bg)
    ax2.tick_params(colors=subtext_color, labelsize=8)
    ax2.spines[:].set_color(spine_color)
    ax2.bar(hours, vehicles, color="#2563eb" if is_mild else "#38bdf8", alpha=0.85, edgecolor=fig_bg, width=0.7)
    ax2.set_title(f"Day {selected_day} ({day_name}) — Diurnal Vehicle Volume", color=text_color, fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Hour of Day (00:00 - 23:00)", color=subtext_color, fontsize=9)
    ax2.set_ylabel("Vehicles / Hour", color=subtext_color, fontsize=9)
    ax2.set_xticks(range(0, 24, 3))
    ax2.grid(True, axis="y", color=grid_color, linewidth=0.6)

    fig.tight_layout()
    return fig


def plot_vehicle_class_emissions(
    category_stats: Dict[str, Dict[str, float]],
    theme: str = "mild",
) -> plt.Figure:
    """
    Comparative chart of vehicle classes from co2.csv showing engine size
    and idling CO2 emission rates.
    """
    is_mild = (theme == "mild")
    fig_bg = "#ffffff" if is_mild else "#0f172a"
    ax_bg = "#f8fafc" if is_mild else "#1e293b"
    text_color = "#0f172a" if is_mild else "#f1f5f9"
    subtext_color = "#64748b" if is_mild else "#94a3b8"
    spine_color = "#e2e8f0" if is_mild else "#334155"
    grid_color = "#f1f5f9" if is_mild else "#334155"

    categories = list(category_stats.keys())
    rates = [category_stats[c]["idle_co2_g_s"] for c in categories]
    engines = [category_stats[c]["avg_engine_l"] for c in categories]

    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=fig_bg)
    ax.set_facecolor(ax_bg)
    ax.tick_params(colors=subtext_color, labelsize=8.5)
    ax.spines[:].set_color(spine_color)

    palette = ["#2563eb", "#0284c7", "#f59e0b", "#ef4444"] if is_mild else ["#38bdf8", "#60a5fa", "#fbbf24", "#f87171"]
    bars = ax.barh(categories, rates, color=palette[:len(categories)], edgecolor=fig_bg, height=0.55)

    for bar, rate, eng in zip(bars, rates, engines):
        ax.text(
            bar.get_width() + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.3f} g/s ({eng}L engine)",
            va="center", ha="left", color=text_color, fontsize=8.5, fontweight="600"
        )

    ax.set_xlim(0, max(rates) * 1.35)
    ax.set_xlabel("Idling CO₂ Emission Rate (g/s)", color=subtext_color, fontsize=9, fontweight="500")
    ax.set_title("Empirical Vehicle Idling CO₂ Rate by Class (co2.csv)", color=text_color, fontsize=10.5, fontweight="bold")
    ax.grid(True, axis="x", color=grid_color, linewidth=0.6)
    fig.tight_layout()
    return fig

