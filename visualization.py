import folium
from folium.plugins import MarkerCluster

def visualize_instructions_on_map(evacuation_zones, fire_perimeters, evacuation_scenarios, instructions_list, risk_levels, optimized_routes):
    map_center = [36.7783, -119.4179]  # Center of California
    map_zoom = 6
    map = folium.Map(location=map_center, zoom_start=map_zoom)

    # Add explanatory text and legend
    legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 250px; height: 160px; 
                    border:2px solid grey; z-index:9999; font-size:14px; background-color: white; opacity: 0.8;
                    font-family: Arial, sans-serif; padding: 10px;">
            <b>Wildfire Evacuation Map Legend</b><br>
            <i class="fa fa-circle" style="color:blue"></i> Evacuation Zone<br>
            <i class="fa fa-circle" style="color:red"></i> Fire Perimeter<br>
            <i class="fa fa-circle" style="color:green"></i> Recommended Evacuation Route<br>
            Click on markers for evacuation instructions.
        </div>
    '''
    map.get_root().html.add_child(folium.Element(legend_html))

    # Add risk heatmap
    risk_heatmap = folium.plugins.HeatMap(
        [[zone['centroid'].y, zone['centroid'].x, risk_levels[i]] for i, zone in enumerate(evacuation_zones)],
        name='Risk Heatmap',
        min_opacity=0.5,
        max_zoom=18,
        radius=25,
        blur=15,
        max_val=max(risk_levels),
    )
    risk_heatmap.add_to(map)

    evacuation_zone_markers = MarkerCluster(name="Evacuation Zones")
    fire_perimeter_markers = MarkerCluster(name="Fire Perimeters")

    for i, (scenario, instructions) in enumerate(zip(evacuation_scenarios, instructions_list)):
        evacuation_zone_index = i % len(evacuation_zones)
        fire_perimeter_index = i % len(fire_perimeters)

        evacuation_zone = evacuation_zones[evacuation_zone_index]
        fire_perimeter = fire_perimeters[fire_perimeter_index]

        evacuation_popup_content = f"<b>Evacuation Zone:</b> {evacuation_zone['location']}<br>"
        evacuation_popup_content += f"<b>Risk Level:</b> {risk_levels[evacuation_zone_index]:.2f}<br>"
        instruction_html = "<b>Generated Instructions:</b><br>"
        for j, instruction in enumerate(instructions):
            instruction_html += f"{j+1}. {instruction['instruction']} (Total Score: {instruction['total_score']:.2f})<br>"
            instruction_html += f"   Safety Score: {instruction['safety_score']:.2f}<br>"
            instruction_html += f"   Specificity Score: {instruction['specificity_score']:.2f}<br>"
            instruction_html += f"   Clarity Score: {instruction['clarity_score']:.2f}<br>"
            instruction_html += f"   Feedback Score: {instruction['feedback_score']:.2f}<br>"
        evacuation_popup_content += instruction_html

        evacuation_marker = folium.Marker(
            location=[evacuation_zone['centroid'].y, evacuation_zone['centroid'].x],
            icon=folium.Icon(color='blue', icon='info-sign'),
            popup=folium.Popup(evacuation_popup_content, max_width=300),
            tooltip=f"Evacuation Zone: {evacuation_zone['location']}"
        )
        evacuation_zone_markers.add_child(evacuation_marker)

        fire_marker = folium.Marker(
            location=[fire_perimeter['centroid'].y, fire_perimeter['centroid'].x],
            icon=folium.Icon(color='red', icon='fire'),
            popup=folium.Popup(f"<b>Fire:</b> {fire_perimeter['name']} ({fire_perimeter['year']})", max_width=300),
            tooltip=f"Fire: {fire_perimeter['name']} ({fire_perimeter['year']})"
        )
        fire_perimeter_markers.add_child(fire_marker)

        evacuation_polygon = folium.GeoJson(
            evacuation_zone['geometry'],
            style_function=lambda feature: {
                'fillColor': 'blue',
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.3
            },
            tooltip=f"Evacuation Zone: {evacuation_zone['location']}"
        )
        evacuation_polygon.add_to(map)

        fire_polygon = folium.GeoJson(
            fire_perimeter['geometry'],
            style_function=lambda feature: {
                'fillColor': 'red',
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.3
            },
            tooltip=f"Fire: {fire_perimeter['name']} ({fire_perimeter['year']})"
        )
        fire_polygon.add_to(map)

    # Add recommended evacuation routes
    for route in optimized_routes:
        folium.PolyLine(
            locations=route['coordinates'],
            color='green',
            weight=5,
            opacity=0.8,
            tooltip=f"Recommended Evacuation Route: {route['name']}",
        ).add_to(map)

    evacuation_zone_markers.add_to(map)
    fire_perimeter_markers.add_to(map)

    folium.LayerControl().add_to(map)

    return map