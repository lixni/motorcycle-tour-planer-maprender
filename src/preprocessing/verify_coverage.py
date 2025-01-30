import osmium
import folium
from pathlib import Path

class VerifyHandler(osmium.SimpleHandler):
    def __init__(self):
        super(VerifyHandler, self).__init__()
        self.nodes = []
        self.ways = []
        
    def way(self, w):
        if 'highway' in w.tags:
            coords = []
            for n in w.nodes:
                if n.location.valid():
                    coords.append((n.location.lat, n.location.lon))
            if coords:
                self.ways.append(coords)

def verify_osm_coverage(osm_file):
    """Verify OSM file coverage and create visualization"""
    handler = VerifyHandler()
    handler.apply_file(str(osm_file))
    
    # Create map centered on Faroe Islands
    m = folium.Map(location=[62.0, -6.8], zoom_start=9)
    
    # Add all roads
    for way in handler.ways:
        folium.PolyLine(
            way,
            weight=2,
            color='blue',
            opacity=0.8
        ).add_to(m)
    
    # Save map
    output_file = Path('coverage_map.html')
    m.save(str(output_file))
    print(f"Coverage map saved to {output_file}")
    print(f"Found {len(handler.ways)} roads")

if __name__ == "__main__":
    osm_file = Path("data/maps/raw/faroe-islands.osm.pbf")
    verify_osm_coverage(osm_file) 