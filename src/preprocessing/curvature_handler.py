import os
from pathlib import Path
from fastkml import kml
from shapely.geometry import LineString, Point
import zipfile
from tqdm import tqdm
import numpy as np
from xml.etree import ElementTree as ET

class CurvatureHandler:
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent.parent
        self.curvature_dir = self.root_dir / 'data' / 'curvature'
        self.curvature_data = {}
        
        # Create curvature directory if it doesn't exist
        if not self.curvature_dir.exists():
            self.curvature_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created curvature directory at: {self.curvature_dir}")
        
        # Check for available curvature files on initialization
        self.available_regions = self._scan_curvature_files()
        if self.available_regions:
            print(f"Found curvature data for regions: {', '.join(self.available_regions)}")
        else:
            print(f"No curvature data files found in {self.curvature_dir}")
    
    def _scan_curvature_files(self):
        """Scan for available curvature files"""
        available = []
        for file in self.curvature_dir.glob("*.c_300.kmz"):
            region = file.name.split('.c_300.kmz')[0]
            available.append(region)
        return available
    
    def load_kmz_file(self, region):
        """Load and parse KMZ file for a region"""
        kmz_path = self.curvature_dir / f"{region}.c_300.kmz"
        
        if not kmz_path.exists():
            print(f"\nWarning: No curvature data found for {region}")
            print(f"Expected file: {kmz_path}")
            return {}
        
        print(f"\nLoading curvature data for {region}...")
        
        try:
            # Extract KML from KMZ
            with zipfile.ZipFile(kmz_path, 'r') as kmz:
                kml_content = None
                for name in kmz.namelist():
                    if name.endswith('.kml'):
                        kml_content = kmz.read(name)
                        break
                
                if not kml_content:
                    raise ValueError(f"No KML file found in {kmz_path}")
                
                # Parse KML using ElementTree
                root = ET.fromstring(kml_content)
                ns = {'kml': 'http://www.opengis.net/kml/2.2'}
                
                curvature_segments = {}
                
                # Find all Placemark elements
                for placemark in root.findall('.//kml:Placemark', ns):
                    # Get description (curvature value)
                    desc_elem = placemark.find('kml:description', ns)
                    if desc_elem is not None and desc_elem.text:
                        curvature = self._extract_curvature(desc_elem.text)
                        
                        # Get coordinates
                        coords_elem = placemark.find('.//kml:coordinates', ns)
                        if coords_elem is not None and coords_elem.text:
                            coords_text = coords_elem.text.strip()
                            coords = []
                            for coord in coords_text.split():
                                # Handle coordinates with or without elevation
                                parts = coord.split(',')
                                if len(parts) >= 2:
                                    lon, lat = map(float, parts[:2])
                                    coords.append((lon, lat))
                            
                            if coords:
                                segment_key = self._create_segment_key(coords)
                                curvature_segments[segment_key] = curvature
                
                print(f"Successfully loaded {len(curvature_segments):,} curved segments for {region}")
                return curvature_segments
                
        except Exception as e:
            print(f"Error loading curvature data for {region}: {e}")
            print("Full error:", str(e))
            return {}
    
    def _extract_curvature(self, description):
        """Extract curvature value from placemark description"""
        try:
            if "Curvature:" in description:
                return float(description.split("Curvature:")[1].strip().split()[0])
            return 0
        except:
            return 0
    
    def _create_segment_key(self, coords):
        """Create a unique key for a road segment"""
        start = coords[0]
        end = coords[-1]
        return f"{start[0]:.5f},{start[1]:.5f}_{end[0]:.5f},{end[1]:.5f}"
    
    def find_segment_curvature(self, start_coord, end_coord, region):
        """Find curvature for a road segment"""
        if region not in self.curvature_data:
            self.curvature_data[region] = self.load_kmz_file(region)
        
        # Create segment key
        segment_key = self._create_segment_key([(start_coord[1], start_coord[0]), 
                                              (end_coord[1], end_coord[0])])
        
        # Look for exact match
        if segment_key in self.curvature_data[region]:
            return self.curvature_data[region][segment_key]
        
        # Look for nearby segments
        return self._find_nearest_segment(start_coord, end_coord, region)
    
    def _find_nearest_segment(self, start, end, region):
        """Find nearest matching segment in curvature data"""
        if not self.curvature_data.get(region):
            return 0
        
        try:
            # Ensure start and end are properly formatted coordinates
            if isinstance(start, (int, float)) or isinstance(end, (int, float)):
                print(f"Warning: Invalid coordinates - start: {start}, end: {end}")
                return 0
            
            # Create LineString with proper coordinate order (lon, lat)
            search_line = LineString([(start[1], start[0]), (end[1], end[0])])
            min_distance = float('inf')
            best_curvature = 0
            
            for segment_key, segment_data in self.curvature_data[region].items():
                try:
                    coords = self._key_to_coords(segment_key)
                    if not coords or len(coords) < 2:
                        continue
                    
                    segment = LineString(coords)
                    
                    # Validate geometries before distance calculation
                    if not search_line.is_valid or not segment.is_valid:
                        continue
                    
                    # Check if segments are close and parallel
                    try:
                        dist = search_line.distance(segment)
                        if (dist < 0.001 and  # ~100m
                            self._are_parallel(search_line, segment)):
                            if dist < min_distance:
                                min_distance = dist
                                # Handle both old and new data structure
                                if isinstance(segment_data, dict):
                                    best_curvature = segment_data.get('curvature', 0)
                                else:
                                    best_curvature = segment_data
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Distance calculation failed for coords {coords}: {e}")
                        continue
                    
                except Exception as e:
                    print(f"Warning: Segment processing failed: {e}")
                    continue
            
            return best_curvature
            
        except Exception as e:
            print(f"Error in nearest segment search: {e}")
            return 0
    
    def _key_to_coords(self, key):
        """Convert segment key back to coordinates"""
        try:
            start, end = key.split('_')
            start_lon, start_lat = map(float, start.split(','))
            end_lon, end_lat = map(float, end.split(','))
            return [(start_lon, start_lat), (end_lon, end_lat)]
        except Exception as e:
            print(f"Warning: Failed to parse key {key}: {e}")
            return None
    
    def _are_parallel(self, line1, line2, threshold=20):
        """Check if two lines are roughly parallel"""
        try:
            # Get bearing of both lines
            def get_bearing(line):
                if not line.is_valid:
                    return None
                start = line.coords[0]
                end = line.coords[-1]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                return np.arctan2(dy, dx) * 180 / np.pi
            
            bearing1 = get_bearing(line1)
            bearing2 = get_bearing(line2)
            
            if bearing1 is None or bearing2 is None:
                return False
            
            diff = abs(bearing1 - bearing2)
            return min(diff, 360 - diff) < threshold
            
        except Exception as e:
            print(f"Warning: Parallel check failed: {e}")
            return False 