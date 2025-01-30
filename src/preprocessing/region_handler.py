from pathlib import Path
import json
from shapely.geometry import shape, Point, LineString
import geopandas as gpd

class RegionHandler:
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent.parent
        self.regions_dir = self.root_dir / 'data' / 'maps' / 'regions'
        self.region_polygons = {}
        self.load_region_polygons()
    
    def load_region_polygons(self):
        """Load all region polygons from GeoJSON files"""
        if not self.regions_dir.exists():
            print(f"Creating regions directory at {self.regions_dir}")
            self.regions_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for geojson_file in self.regions_dir.glob('*.geojson'):
            region_name = geojson_file.stem
            try:
                with open(geojson_file, 'r', encoding='utf-8') as f:
                    geojson = json.load(f)
                    # Convert GeoJSON to Shapely geometry
                    self.region_polygons[region_name] = shape(geojson['features'][0]['geometry'])
                print(f"Loaded region polygon for {region_name}")
            except Exception as e:
                print(f"Error loading region {region_name}: {e}")
    
    def get_region_polygon(self, region_name):
        """Get polygon for a specific region"""
        if region_name not in self.region_polygons:
            geojson_path = self.regions_dir / f"{region_name}.geojson"
            if not geojson_path.exists():
                raise ValueError(f"No polygon data found for region: {region_name}")
            
            try:
                with open(geojson_path, 'r', encoding='utf-8') as f:
                    geojson = json.load(f)
                    self.region_polygons[region_name] = shape(geojson['features'][0]['geometry'])
            except Exception as e:
                raise ValueError(f"Error loading region {region_name}: {e}")
        
        return self.region_polygons[region_name]
    
    def point_in_region(self, point, region_name):
        """Check if a point is within a region"""
        polygon = self.get_region_polygon(region_name)
        return polygon.contains(Point(point[1], point[0]))  # Convert (lat, lon) to (x, y)
    
    def line_crosses_region(self, start, end, region_name):
        """Check if a line segment crosses a region"""
        polygon = self.get_region_polygon(region_name)
        line = LineString([(start[1], start[0]), (end[1], end[0])])  # Convert (lat, lon) to (x, y)
        return polygon.intersects(line) 