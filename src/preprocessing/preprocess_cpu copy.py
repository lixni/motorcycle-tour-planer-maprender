import osmium
import networkx as nx
import pickle
from pathlib import Path
import sys
import time
from math import radians, sin, cos, asin, sqrt, pi, atan2
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from tqdm import tqdm

# Define constants
RAW_MAPS_DIR = Path(__file__).parent.parent.parent / 'data' / 'maps' / 'raw'
PROCESSED_MAPS_DIR = Path(__file__).parent.parent.parent / 'data' / 'maps' / 'processed'

class CurvatureHandler:
    def __init__(self):
        self.curvature_data = {}

    def find_segment_curvature(self, start, end, region):
        # Simplified curvature calculation for CPU
        return 0.0

class CPUPreprocessHandler(osmium.SimpleHandler):
    def __init__(self, region_name=None):
        super(CPUPreprocessHandler, self).__init__()
        self.ways = defaultdict(dict)
        self.nodes = {}
        self.referenced_nodes = set()
        self.curvature_handler = CurvatureHandler()
        self.region = region_name
        self.road_types = None  # Accept all highway types
        self.count = 0
        self.batch_size = 1000000

    def node(self, n):
        """Store all nodes with valid locations"""
        if n.location.valid():
            self.nodes[n.id] = (n.location.lat, n.location.lon)
        self.count += 1
        
        if self.count % self.batch_size == 0:
            print(f"\rProcessed {self.count:,} nodes...", end='', flush=True)

    def way(self, w):
        """Store all ways that have the highway tag"""
        if 'highway' not in w.tags:
            return

        # Store node references
        for n in w.nodes:
            self.referenced_nodes.add(n.ref)

        # Collect all nodes for this way
        nodes = []
        for n in w.nodes:
            if n.ref in self.nodes:
                nodes.append(self.nodes[n.ref])
        
        if len(nodes) >= 2:
            self.ways[w.id] = {
                'nodes': nodes,
                'type': w.tags.get('highway', 'unclassified'),
                'name': w.tags.get('name', str(w.id)),
                'surface': w.tags.get('surface', 'unknown')
            }

def calculate_curviness_cpu(points_x, points_y):
    """Calculate curviness for a sequence of points using CPU"""
    results = np.zeros(len(points_x) - 2)
    
    for idx in range(len(points_x) - 2):
        # Calculate bearings between consecutive points
        lat1, lon1 = points_x[idx], points_y[idx]
        lat2, lon2 = points_x[idx + 1], points_y[idx + 1]
        lat3, lon3 = points_x[idx + 2], points_y[idx + 2]
        
        # Calculate angles using numpy
        dx1 = lon2 - lon1
        dy1 = lat2 - lat1
        dx2 = lon3 - lon2
        dy2 = lat3 - lat2
        
        # Calculate angles
        angle1 = np.arctan2(dy1, dx1)
        angle2 = np.arctan2(dy2, dx2)
        
        # Calculate bearing difference
        diff = abs(angle2 - angle1)
        if diff > np.pi:
            diff = 2 * np.pi - diff
            
        results[idx] = diff
    
    return results

def process_batch_cpu(batch):
    """Process a single batch of ways on CPU"""
    batch_results = {}
    
    try:
        # Prepare batch data
        points_arrays = [points for _, points in batch]
        total_points = sum(len(points) for points in points_arrays)
        
        # Skip if batch is too small
        if total_points < 3:
            return batch_results
        
        # Allocate arrays
        points_x = np.zeros(total_points)
        points_y = np.zeros(total_points)
        
        # Fill arrays
        current_idx = 0
        batch_indices = []
        for points in points_arrays:
            points_x[current_idx:current_idx + len(points)] = points[:, 0]
            points_y[current_idx:current_idx + len(points)] = points[:, 1]
            batch_indices.append((current_idx, current_idx + len(points)))
            current_idx += len(points)
        
        # Calculate curviness
        results = calculate_curviness_cpu(points_x, points_y)
        
        # Process results for each way in batch
        for (way_id, _), (start_idx, end_idx) in zip(batch, batch_indices):
            if end_idx - start_idx >= 3:
                way_results = results[start_idx:end_idx-2]
                batch_results[way_id] = float(np.mean(way_results))
        
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
    
    return batch_results

def process_ways_cpu(ways):
    """Process road ways using CPU with batching"""
    print("Processing ways on CPU...")
    
    # Batch sizes for CPU processing
    optimal_batch_size = 1000
    max_points_per_batch = 50000
    min_points_per_batch = 500
    
    processed_ways = {}
    way_items = list(ways.items())
    
    # Process in smart batches
    current_batch = []
    current_points = 0
    
    for way_id, way_info in tqdm(way_items, desc="Processing ways"):
        nodes = way_info['nodes']
        if len(nodes) >= 3:
            points = np.array(nodes)
            current_points += len(points)
            current_batch.append((way_id, points))
            
            if (len(current_batch) >= optimal_batch_size or 
                current_points >= max_points_per_batch):
                
                if current_points >= min_points_per_batch:
                    processed_ways.update(process_batch_cpu(current_batch))
                    current_batch = []
                    current_points = 0
    
    # Process remaining ways
    if current_batch and current_points >= min_points_per_batch:
        processed_ways.update(process_batch_cpu(current_batch))
    
    return processed_ways

def build_road_graph(ways, processed_ways):
    """Build road graph with both normal and curvy roads"""
    G = nx.Graph()
    
    print("Building road graph...")
    total_ways = len(ways)
    
    # Create a list of all nodes for spatial indexing
    all_nodes = []
    for way_data in ways.values():
        all_nodes.extend(way_data['nodes'])
    
    # Create a spatial index
    tree = cKDTree(all_nodes)
    
    # First pass: add all nodes and edges
    edges_to_add = []
    
    for i, (way_id, way_data) in enumerate(ways.items()):
        if i % 10000 == 0:
            print(f"\rProcessing way {i:,}/{total_ways:,} ({(i/total_ways)*100:.1f}%)", end='')
            
        try:
            nodes = way_data.get('nodes', [])
            if len(nodes) < 2:
                continue
                
            road_type = way_data.get('type', 'unclassified')
            
            for j in range(len(nodes) - 1):
                start = nodes[j]
                end = nodes[j + 1]
                
                if start and end:
                    start_lat, start_lon = start
                    end_lat, end_lon = end
                    distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
                    
                    segment_key = f"{start_lat:.5f},{start_lon:.5f}_{end_lat:.5f},{end_lon:.5f}"
                    segment_data = processed_ways.get(segment_key, {})
                    curvature = segment_data.get('curvature', 0) if isinstance(segment_data, dict) else segment_data
                    elevation_change = segment_data.get('elevation_change', 0) if isinstance(segment_data, dict) else 0
                    
                    edges_to_add.append((start, end, {
                        'distance': distance,
                        'type': road_type,
                        'curviness': curvature,
                        'elevation_change': elevation_change
                    }))
        
        except Exception as e:
            print(f"\nError processing way {way_id}: {e}")
            continue
    
    # Add edges in a batch
    G.add_edges_from(edges_to_add)
    
    print("\nAnalyzing graph connectivity...")
    components = list(nx.connected_components(G))
    print(f"Found {len(components)} connected components")
    
    if len(components) > 1:
        print("Attempting to connect components...")
        main_component = max(components, key=len)
        
        for component in components:
            if component == main_component:
                continue
            
            for node1 in component:
                distances, indices = tree.query(node1, k=1)
                if distances < 2.0:
                    closest_node = all_nodes[indices]
                    G.add_edge(node1, closest_node, distance=distances, type='connector', curviness=0, elevation_change=0)
                    print(f"Connected component of size {len(component)} to main component")
                    break
    
    print(f"\nFinal graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on earth in kilometers"""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def preprocess_osm_file_cpu(osm_file):
    """Preprocess OSM file with CPU"""
    start_time = time.time()
    
    input_path = RAW_MAPS_DIR / osm_file
    region_name = input_path.stem.split('.')[0]
    output_path = PROCESSED_MAPS_DIR / f"{region_name}_processed.pkl"
    
    print(f"\nStarting preprocessing of {input_path}")
    print(f"Output will be saved to {output_path}")
    
    step1_time = time.time()
    print("\nStep 1: Parsing OSM file...")
    handler = CPUPreprocessHandler(region_name)
    
    try:
        handler.apply_file(str(input_path), locations=True, idx='flex_mem')
        step1_duration = time.time() - step1_time
        print(f"\nStep 1 completed in {step1_duration:.1f} seconds")
        
        print(f"\nFound {len(handler.nodes):,} nodes and {len(handler.ways):,} relevant ways")
        
        step2_time = time.time()
        print("\nStep 2: Processing ways using CPU...")
        processed_ways = process_ways_cpu(handler.ways)
        step2_duration = time.time() - step2_time
        print(f"Step 2 completed in {step2_duration:.1f} seconds")
        
        step3_time = time.time()
        print("\nStep 3: Building road graph...")
        G = build_road_graph(handler.ways, processed_ways)
        step3_duration = time.time() - step3_time
        print(f"Step 3 completed in {step3_duration:.1f} seconds")
        
        step4_time = time.time()
        print(f"\nStep 4: Saving processed data to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(G, f)
        step4_duration = time.time() - step4_time
        print(f"Step 4 completed in {step4_duration:.1f} seconds")
        
        total_duration = time.time() - start_time
        print(f"\nPreprocessing complete!")
        print(f"Total time taken: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print(f"Step 1 (Parsing): {step1_duration:.1f}s ({(step1_duration/total_duration)*100:.1f}%)")
        print(f"Step 2 (CPU Processing): {step2_duration:.1f}s ({(step2_duration/total_duration)*100:.1f}%)")
        print(f"Step 3 (Graph Building): {step3_duration:.1f}s ({(step3_duration/total_duration)*100:.1f}%)")
        print(f"Step 4 (Saving): {step4_duration:.1f}s ({(step4_duration/total_duration)*100:.1f}%)")
        print(f"Processed data saved to: {output_path}")
        print(f"Graph contains {len(G.nodes):,} nodes and {len(G.edges):,} edges")
        
        return G
        
    except Exception as e:
        print(f"Error processing {region_name}: {e}")
        print(f"Full error: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess_cpu.py <osm_file_name>")
        print("The OSM file should be in the data/maps/raw directory")
        sys.exit(1)
    
    osm_file = sys.argv[1]
    if not (RAW_MAPS_DIR / osm_file).exists():
        print(f"Error: File {osm_file} not found in {RAW_MAPS_DIR}!")
        sys.exit(1)
    
    preprocess_osm_file_cpu(osm_file)