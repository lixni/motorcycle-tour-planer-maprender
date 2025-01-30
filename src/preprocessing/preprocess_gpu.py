import osmium
import networkx as nx
import pickle
from pathlib import Path
import sys
from tqdm import tqdm
import time
import numpy as np
import cupy as cp
from numba import cuda, jit
from math import radians, sin, cos, asin, sqrt, pi, atan2, acos
from collections import defaultdict
import os
from curvature_handler import CurvatureHandler
from scipy.spatial import cKDTree

# Define paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
RAW_MAPS_DIR = DATA_DIR / 'maps' / 'raw'
PROCESSED_MAPS_DIR = DATA_DIR / 'maps' / 'processed'
REGIONS_DIR = DATA_DIR / 'maps' / 'regions'

# Create directories if they don't exist
for dir_path in [RAW_MAPS_DIR, PROCESSED_MAPS_DIR, REGIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

@cuda.jit
def calculate_curviness_gpu(points_x, points_y, results):
    """CUDA kernel for calculating road curviness"""
    idx = cuda.grid(1)
    if idx < points_x.shape[0] - 2:
        # Calculate bearings between consecutive points
        lat1, lon1 = points_x[idx], points_y[idx]
        lat2, lon2 = points_x[idx + 1], points_y[idx + 1]
        lat3, lon3 = points_x[idx + 2], points_y[idx + 2]
        
        # Calculate bearings using direct math operations
        # We can't use math.atan2 in CUDA, so we'll use simpler angle calculations
        dx1 = lon2 - lon1
        dy1 = lat2 - lat1
        dx2 = lon3 - lon2
        dy2 = lat3 - lat2
        
        # Calculate angles using simplified method
        angle1 = cuda.libdevice.atan2(dy1, dx1)
        angle2 = cuda.libdevice.atan2(dy2, dx2)
        
        # Calculate bearing difference
        diff = abs(angle2 - angle1)
        if diff > 3.14159:  # pi
            diff = 6.28318 - diff  # 2*pi
            
        results[idx] = diff

class Preprocessor:
    def __init__(self, region_name):
        self.region = region_name
        self.road_types = None  # Accept all highway types
        self.count = 0
        self.batch_size = 1000000
        self.memory_pool = cp.get_default_memory_pool()
        self.pinned_memory_pool = cp.get_default_pinned_memory_pool()
        self.nodes = {}
        self.referenced_nodes = set()
        self.ways = {}

    def node(self, n):
        """Store all nodes with valid locations"""
        if n.location.valid():
            self.nodes[n.id] = (n.location.lat, n.location.lon)
        self.count += 1
        
        if self.count % self.batch_size == 0:
            print(f"\rProcessed {self.count:,} nodes...", end='', flush=True)
            self.clear_memory()

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
                'nodes': nodes
            }
            self.clear_memory()

    def clear_memory(self):
        """Clear GPU memory pools to avoid crashes"""
        self.memory_pool.free_all_blocks()
        self.pinned_memory_pool.free_all_blocks()

def process_ways_gpu(ways):
    """Process road ways using GPU acceleration with improved batching"""
    print("Processing ways on GPU...")
    
    # Adjust batch sizes for better GPU utilization
    optimal_batch_size = 2000  # Increased for better GPU utilization
    max_points_per_batch = 100000  # Increased maximum points per batch
    min_points_per_batch = 1000  # Minimum points to process in a batch
    
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
            
            # Process batch if it meets size criteria
            if (len(current_batch) >= optimal_batch_size or 
                current_points >= max_points_per_batch):
                
                # Only process if we have enough points for good GPU utilization
                if current_points >= min_points_per_batch:
                    processed_ways.update(process_batch_gpu(current_batch))
                    current_batch = []
                    current_points = 0
                    
                    # Clear GPU memory after each batch
                    cp.get_default_memory_pool().free_all_blocks()
    
    # Process remaining ways if we have enough points
    if current_batch and current_points >= min_points_per_batch:
        processed_ways.update(process_batch_gpu(current_batch))
    
    return processed_ways

def process_batch_gpu(batch):
    """Process a single batch of ways on GPU with optimized CUDA configuration"""
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
        
        # Transfer to GPU
        with cp.cuda.Stream() as stream:
            d_points_x = cp.array(points_x)
            d_points_y = cp.array(points_y)
            d_results = cp.zeros(total_points - 2)
            
            # Optimize CUDA kernel configuration
            threadsperblock = 256  # Keep thread block size optimal
            # Ensure minimum grid size of 32 blocks for better occupancy
            min_blocks = 32
            required_blocks = (total_points + (threadsperblock - 1)) // threadsperblock
            blockspergrid = max(min_blocks, required_blocks)
            
            # Launch kernel with optimized configuration
            calculate_curviness_gpu[blockspergrid, threadsperblock](
                d_points_x, d_points_y, d_results)
            
            # Get results back to CPU
            results = cp.asnumpy(d_results)
        
        # Process results for each way in batch
        for (way_id, _), (start_idx, end_idx) in zip(batch, batch_indices):
            if end_idx - start_idx >= 3:  # Need at least 3 points for curviness
                way_results = results[start_idx:end_idx-2]
                batch_results[way_id] = float(np.mean(way_results))
        
        # Clean up GPU memory
        del d_points_x, d_points_y, d_results
        
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
    
    return batch_results

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
                
            # Get road type and other attributes
            road_type = way_data.get('type', 'unclassified')
            
            # Add all road segments to the graph
            for j in range(len(nodes) - 1):
                start = nodes[j]
                end = nodes[j + 1]
                
                if start and end:  # Ensure both nodes exist
                    # Calculate segment properties
                    start_lat, start_lon = start
                    end_lat, end_lon = end
                    distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
                    
                    # Get curvature if available from processed data
                    segment_key = f"{start_lat:.5f},{start_lon:.5f}_{end_lat:.5f},{end_lon:.5f}"
                    segment_data = processed_ways.get(segment_key, {})
                    curvature = segment_data.get('curvature', 0) if isinstance(segment_data, dict) else segment_data
                    elevation_change = segment_data.get('elevation_change', 0) if isinstance(segment_data, dict) else 0
                    
                    # Collect edges to add in a batch
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
    G.add_edges_from((start, end, attrs) for start, end, attrs in edges_to_add)
    
    print("\nAnalyzing graph connectivity...")
    components = list(nx.connected_components(G))
    print(f"Found {len(components)} connected components")
    
    if len(components) > 1:
        print("Attempting to connect components...")
        main_component = max(components, key=len)
        
        # Use spatial indexing to connect components
        for component in components:
            if component == main_component:
                continue
            
            # Find the closest nodes between this component and the main component
            for node1 in component:
                distances, indices = tree.query(node1, k=1)  # Find the nearest node in the main component
                if distances < 2.0:  # Only connect if within 2km
                    closest_node = all_nodes[indices]
                    G.add_edge(node1, closest_node, distance=distances, type='connector', curviness=0, elevation_change=0)
                    print(f"Connected component of size {len(component)} to main component")
                    break  # Stop after connecting this component
    
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

def process_road_curvature(way_nodes, way_data):
    """Process road curvature using data from roadcurvature.com"""
    try:
        curvature_handler = CurvatureHandler()
        region = way_data.get('region', '')
        
        if not region:
            return calculate_basic_curvature(way_nodes)
        
        # Calculate curvature for each segment
        max_curvature = 0
        for i in range(len(way_nodes) - 1):
            start = way_nodes[i]
            end = way_nodes[i + 1]
            segment_curvature = curvature_handler.find_segment_curvature(start, end, region)
            max_curvature = max(max_curvature, segment_curvature)
        
        return max_curvature
        
    except Exception as e:
        print(f"Error processing curvature: {e}")
        return calculate_basic_curvature(way_nodes)

def calculate_basic_curvature(way_nodes):
    """Calculate basic curvature when no curvature data is available"""
    if len(way_nodes) < 3:
        return 0.0
    
    total_angle_change = 0.0
    for i in range(len(way_nodes) - 2):
        p1 = way_nodes[i]
        p2 = way_nodes[i + 1]
        p3 = way_nodes[i + 2]
        
        # Calculate vectors
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Calculate angle between vectors
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        v1_mag = sqrt(v1[0] ** 2 + v1[1] ** 2)
        v2_mag = sqrt(v2[0] ** 2 + v2[1] ** 2)
        
        if v1_mag * v2_mag == 0:
            continue
            
        cos_angle = dot_product / (v1_mag * v2_mag)
        # Handle floating point errors
        cos_angle = min(1.0, max(-1.0, cos_angle))
        angle = acos(cos_angle)
        
        total_angle_change += angle
    
    # Normalize by road length
    return (total_angle_change * 180 / pi) / len(way_nodes)

def preprocess_osm_file(osm_file):
    """Preprocess OSM file with GPU acceleration"""
    start_time = time.time()
    
    try:
        # Get input and output paths
        input_path = RAW_MAPS_DIR / osm_file
        region_name = input_path.stem.split('.')[0]  # Extract region name from filename
        output_path = PROCESSED_MAPS_DIR / f"{region_name}_processed.pkl"
        
        print(f"\nStarting preprocessing of {input_path}")
        print(f"Output will be saved to {output_path}")
        
        # Step 1: Parse OSM file
        step1_time = time.time()
        print("\nStep 1: Parsing OSM file...")
        handler = Preprocessor(region_name)
        
        handler.apply_file(str(input_path), locations=True, idx='flex_mem')
        step1_duration = time.time() - step1_time
        print(f"\nStep 1 completed in {step1_duration:.1f} seconds")
        
        print(f"\nFound {len(handler.nodes):,} nodes and {len(handler.ways):,} relevant ways")
        
        # Step 2: Process ways using GPU
        step2_time = time.time()
        print("\nStep 2: Processing ways using GPU...")
        processed_ways = process_ways_gpu(handler.ways)
        step2_duration = time.time() - step2_time
        print(f"Step 2 completed in {step2_duration:.1f} seconds")
        
        # Step 3: Build road graph
        step3_time = time.time()
        print("\nStep 3: Building road graph...")
        G = build_road_graph(handler.ways, processed_ways)
        step3_duration = time.time() - step3_time
        print(f"Step 3 completed in {step3_duration:.1f} seconds")
        
        if len(G.nodes) == 0 or len(G.edges) == 0:
            raise ValueError(f"Generated graph is empty: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
        # Step 4: Save processed data
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
        print(f"Step 2 (GPU Processing): {step2_duration:.1f}s ({(step2_duration/total_duration)*100:.1f}%)")
        print(f"Step 3 (Graph Building): {step3_duration:.1f}s ({(step3_duration/total_duration)*100:.1f}%)")
        print(f"Step 4 (Saving): {step4_duration:.1f}s ({(step4_duration/total_duration)*100:.1f}%)")
        print(f"Graph contains {len(G.nodes):,} nodes and {len(G.edges):,} edges")
        
        print(f"Successfully processed {region_name}")
        return G
        
    except Exception as e:
        print(f"Error processing {region_name}:")
        print(f"Full error: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess_gpu.py <osm_file_name>")
        print("The OSM file should be in the data/maps/raw directory")
        sys.exit(1)
    
    osm_file = sys.argv[1]
    if not (RAW_MAPS_DIR / osm_file).exists():
        print(f"Error: File {osm_file} not found in {RAW_MAPS_DIR}!")
        sys.exit(1)
    
    preprocess_osm_file(osm_file)