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
        self.ways = {}
        self.nodes = {}
        self.referenced_nodes = set()
        self.region = region_name
        self.count = 0
        self.batch_size = 1000000
        self.current_ways_batch = []
        self.max_ways_in_memory = 50000
        self.processed_nodes = 0
        self.processed_ways = 0

    def node(self, n):
        """Store nodes with valid locations"""
        if n.location.valid():
            self.nodes[n.id] = (n.location.lat, n.location.lon)
            self.processed_nodes += 1
        
        if self.processed_nodes % self.batch_size == 0:
            print(f"\rProcessed {self.processed_nodes:,} nodes...", end='', flush=True)

    def way(self, w):
        """Process ways that have the highway tag"""
        if 'highway' not in w.tags:
            return

        # Store node references
        nodes = []
        valid_way = False
        
        for n in w.nodes:
            self.referenced_nodes.add(n.ref)
            if n.ref in self.nodes:
                nodes.append(self.nodes[n.ref])
                valid_way = True

        if valid_way and len(nodes) >= 2:
            way_data = {
                'nodes': nodes,
                'type': w.tags.get('highway', 'unclassified'),
                'name': w.tags.get('name', str(w.id)),
                'surface': w.tags.get('surface', 'unknown')
            }
            
            self.ways[w.id] = way_data
            self.processed_ways += 1
            
            if self.processed_ways % 1000 == 0:
                print(f"\rProcessed {self.processed_ways:,} ways...", end='', flush=True)

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
    """Build road graph with memory-efficient processing"""
    G = nx.Graph()
    print("\nBuilding road graph...")
    total_ways = len(ways)
    
    # Process in batches
    batch_size = 5000
    edges_added = 0
    
    for i, (way_id, way_data) in enumerate(ways.items()):
        try:
            nodes = way_data.get('nodes', [])
            if len(nodes) < 2:
                continue
                
            road_type = way_data.get('type', 'unclassified')
            
            # Add nodes and edges for this way
            for j in range(len(nodes) - 1):
                start = nodes[j]
                end = nodes[j + 1]
                
                if start and end:
                    # Add nodes if they don't exist
                    if start not in G:
                        G.add_node(start)
                    if end not in G:
                        G.add_node(end)
                    
                    # Calculate distance
                    distance = haversine_distance(
                        start[0], start[1],
                        end[0], end[1]
                    )
                    
                    # Get curvature
                    curvature = processed_ways.get(way_id, 0)
                    
                    # Add edge
                    G.add_edge(
                        start, end,
                        distance=distance,
                        type=road_type,
                        curviness=curvature,
                        elevation_change=0
                    )
                    edges_added += 1
            
            # Print progress
            if (i + 1) % 1000 == 0:
                print(f"\rProcessed {i+1:,}/{total_ways:,} ways, {edges_added:,} edges added...", end='', flush=True)
        
        except Exception as e:
            print(f"\nError processing way {way_id}: {e}")
            continue
    
    print(f"\nFinal graph has {len(G.nodes):,} nodes and {len(G.edges):,} edges")
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
    """Memory-efficient preprocessing of OSM file"""
    start_time = time.time()
    
    input_path = RAW_MAPS_DIR / osm_file
    region_name = input_path.stem.split('.')[0]
    output_path = PROCESSED_MAPS_DIR / f"{region_name}_processed.pkl"
    
    print(f"\nStarting preprocessing of {input_path}")
    print(f"Output will be saved to {output_path}")
    
    try:
        # Step 1: Parse OSM file
        step1_time = time.time()
        print("\nStep 1: Parsing OSM file...")
        handler = CPUPreprocessHandler(region_name)
        handler.apply_file(str(input_path), locations=True, idx='flex_mem')
        step1_duration = time.time() - step1_time
        
        print(f"\nFound {len(handler.nodes):,} nodes and {len(handler.ways):,} relevant ways")
        
        # Step 2: Process ways
        step2_time = time.time()
        print("\nStep 2: Processing ways...")
        processed_ways = process_ways_cpu(handler.ways)
        step2_duration = time.time() - step2_time
        
        # Step 3: Build graph
        step3_time = time.time()
        print("\nStep 3: Building road graph...")
        G = build_road_graph(handler.ways, processed_ways)
        step3_duration = time.time() - step3_time
        
        # Step 4: Save processed data
        step4_time = time.time()
        print(f"\nStep 4: Saving processed data to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(G, f)
        step4_duration = time.time() - step4_time
        
        total_duration = time.time() - start_time
        print(f"\nPreprocessing complete!")
        print(f"Total time taken: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print(f"Step 1 (Parsing): {step1_duration:.1f}s ({(step1_duration/total_duration)*100:.1f}%)")
        print(f"Step 2 (Processing): {step2_duration:.1f}s ({(step2_duration/total_duration)*100:.1f}%)")
        print(f"Step 3 (Graph Building): {step3_duration:.1f}s ({(step3_duration/total_duration)*100:.1f}%)")
        print(f"Step 4 (Saving): {step4_duration:.1f}s ({(step4_duration/total_duration)*100:.1f}%)")
        
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
    
    preprocess_osm_file_cpu(osm_file)<