import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONMALLOC'] = 'debug'

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
import gc
import mmap
import tempfile

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
        self.nodes = {}  # Keep this for direct access
        self.referenced_nodes = set()
        self.region = region_name
        # Even smaller batch sizes
        self.batch_size = 1000
        self.node_batch_size = 100000
        self.max_ways_in_memory = 100
        self.processed_nodes = 0
        self.processed_ways = 0
        self.current_batch = []
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mmap')
        self.temp_nodes_file = self.temp_file.name
        print(f"Using temporary file: {self.temp_nodes_file}")
        
        self.node_buffer = None
        self.initialize_node_buffer()

    def initialize_node_buffer(self):
        """Initialize memory-mapped file for nodes"""
        try:
            # Create the memory map
            self.temp_file.seek(self.node_batch_size * 16 - 1)  # 16 bytes per record
            self.temp_file.write(b'\0')
            self.temp_file.flush()
            
            self.node_buffer = np.memmap(self.temp_nodes_file, 
                                       dtype=[('id', np.int64), ('lat', np.float32), ('lon', np.float32)],
                                       mode='r+', 
                                       shape=(self.node_batch_size,))
            
        except Exception as e:
            print(f"Error initializing node buffer: {e}")
            print(f"Attempted to create file at: {self.temp_nodes_file}")
            if os.path.exists(self.temp_nodes_file):
                print(f"File exists: True")
                print(f"File size: {os.path.getsize(self.temp_nodes_file)}")
                print(f"File is writable: {os.access(self.temp_nodes_file, os.W_OK)}")
            raise

    def node(self, n):
        """Store nodes both in memory map and dictionary for quick access"""
        if n.location.valid():
            # Store in memory map
            idx = self.processed_nodes % self.node_batch_size
            self.node_buffer[idx] = (n.id, n.location.lat, n.location.lon)
            
            # Also store in dictionary for quick access
            self.nodes[n.id] = (n.location.lat, n.location.lon)
            
            self.processed_nodes += 1

            if idx == self.node_batch_size - 1:
                self.node_buffer.flush()
                self.initialize_node_buffer()

        if self.processed_nodes % 100000 == 0:
            print(f"\rProcessed {self.processed_nodes:,} nodes...", end='', flush=True)
            gc.collect()

    def way(self, w):
        """Process ways and store their nodes"""
        if 'highway' not in w.tags:
            return

        nodes = []
        for n in w.nodes:
            self.referenced_nodes.add(n.ref)
            if n.ref in self.nodes:  # Check if we have this node
                nodes.append(self.nodes[n.ref])

        if len(nodes) >= 2:  # Only store ways with at least 2 nodes
            self.current_batch.append((w.id, nodes, dict(w.tags)))
            self.processed_ways += 1

            if len(self.current_batch) >= self.max_ways_in_memory:
                self._process_batch()

        if self.processed_ways % 1000 == 0:
            print(f"\rProcessed {self.processed_ways:,} ways...", end='', flush=True)
            gc.collect()

    def _process_batch(self):
        """Process and store the current batch of ways"""
        if self.current_batch:
            processed_count = 0
            for way_id, nodes, tags in self.current_batch:
                if len(nodes) >= 2:  # Double check we have enough nodes
                    self.ways[way_id] = {
                        'nodes': nodes,
                        'type': tags.get('highway', 'unclassified'),
                        'name': tags.get('name', str(way_id)),
                        'surface': tags.get('surface', 'unknown')
                    }
                    processed_count += 1
            
            if processed_count == 0:
                print("\nWarning: No ways processed in batch")
                print(f"Batch size: {len(self.current_batch)}")
                if self.current_batch:
                    sample_way = self.current_batch[0]
                    print(f"Sample way: id={sample_way[0]}, nodes={len(sample_way[1])}")
            
            self.current_batch = []
            gc.collect()

    def __del__(self):
        """Cleanup temporary files"""
        try:
            if hasattr(self, 'node_buffer'):
                del self.node_buffer
            if hasattr(self, 'temp_file'):
                self.temp_file.close()
            if hasattr(self, 'temp_nodes_file') and os.path.exists(self.temp_nodes_file):
                try:
                    os.unlink(self.temp_nodes_file)
                except Exception as e:
                    print(f"Warning: Could not remove temp file: {e}")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def get_stats(self):
        """Get processing statistics"""
        return {
            'processed_nodes': self.processed_nodes,
            'stored_nodes': len(self.nodes),
            'processed_ways': self.processed_ways,
            'stored_ways': len(self.ways),
            'referenced_nodes': len(self.referenced_nodes)
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
    """Process ways with memory mapping"""
    print("Processing ways on CPU...")
    
    # Very small batch sizes
    optimal_batch_size = 100
    max_points_per_batch = 1000
    min_points_per_batch = 50
    
    processed_ways = {}
    way_items = list(ways.items())
    
    for i in tqdm(range(0, len(way_items), optimal_batch_size)):
        batch = way_items[i:i + optimal_batch_size]
        current_batch = []
        current_points = 0
        
        for way_id, way_info in batch:
            nodes = way_info['nodes']
            if len(nodes) >= 3:
                points = np.array(nodes)
                if current_points + len(points) > max_points_per_batch:
                    if current_points >= min_points_per_batch:
                        processed_ways.update(process_batch_cpu(current_batch))
                        current_batch = []
                        current_points = 0
                        gc.collect()
                
                current_points += len(points)
                current_batch.append((way_id, points))
        
        if current_batch and current_points >= min_points_per_batch:
            processed_ways.update(process_batch_cpu(current_batch))
            gc.collect()
        
        if i % (optimal_batch_size * 5) == 0:
            gc.collect()
    
    return processed_ways

def build_road_graph(ways, processed_ways):
    """Build road graph with memory-efficient processing"""
    G = nx.Graph()
    print("\nBuilding road graph...")
    total_ways = len(ways)
    
    # Process in batches
    edges_added = 0
    errors = defaultdict(int)  # Track different types of errors
    
    for i, (way_id, way_data) in enumerate(ways.items()):
        try:
            nodes = way_data.get('nodes', [])
            if len(nodes) < 2:
                errors['too_few_nodes'] += 1
                continue
                
            road_type = way_data.get('type', 'unclassified')
            
            # Add nodes and edges for this way
            for j in range(len(nodes) - 1):
                try:
                    start = nodes[j]
                    end = nodes[j + 1]
                    
                    if not start or not end:
                        errors['invalid_node_pair'] += 1
                        continue
                        
                    if not isinstance(start, tuple) or not isinstance(end, tuple):
                        errors['invalid_node_type'] += 1
                        continue
                        
                    if len(start) != 2 or len(end) != 2:
                        errors['invalid_coordinates'] += 1
                        continue
                    
                    # Add nodes if they don't exist
                    if start not in G:
                        G.add_node(start, pos=start)
                    if end not in G:
                        G.add_node(end, pos=end)
                    
                    # Calculate distance
                    try:
                        distance = haversine_distance(
                            float(start[0]), float(start[1]),
                            float(end[0]), float(end[1])
                        )
                    except (ValueError, TypeError) as e:
                        errors['distance_calculation'] += 1
                        continue
                    
                    # Get curvature (with default of 0 if not found)
                    curvature = processed_ways.get(way_id, 0)
                    
                    # Add edge with all attributes
                    G.add_edge(
                        start, end,
                        distance=distance,
                        type=road_type,
                        curviness=curvature,
                        elevation_change=0,
                        way_id=way_id
                    )
                    edges_added += 1
                
                except Exception as e:
                    errors['edge_creation'] += 1
                    continue
            
            # Print progress
            if (i + 1) % 1000 == 0:
                print(f"\rProcessed {i+1:,}/{total_ways:,} ways, {edges_added:,} edges added...", end='', flush=True)
                # Print error summary periodically
                if errors:
                    print("\nCurrent error summary:")
                    for error_type, count in errors.items():
                        print(f"- {error_type}: {count:,}")
        
        except Exception as e:
            errors['way_processing'] += 1
            continue
    
    print(f"\nFinal graph has {len(G.nodes):,} nodes and {len(G.edges):,} edges")
    if errors:
        print("\nFinal error summary:")
        for error_type, count in errors.items():
            print(f"- {error_type}: {count:,}")
    
    if len(G.nodes) == 0:
        print("\nDebug information:")
        print(f"Total ways processed: {total_ways}")
        print(f"Total edges added: {edges_added}")
        print(f"Error counts: {dict(errors)}")
        print(f"Sample way data: {next(iter(ways.items())) if ways else 'No ways'}")
    
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
    G = None  # Initialize graph variable
    
    try:
        input_path = RAW_MAPS_DIR / osm_file
        region_name = input_path.stem.split('.')[0]
        output_path = PROCESSED_MAPS_DIR / f"{region_name}_processed.pkl"
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
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
            
            if len(G.nodes) == 0 or len(G.edges) == 0:
                raise ValueError(f"Generated graph is empty: {len(G.nodes)} nodes, {len(G.edges)} edges")
            
            # Step 4: Save processed data
            step4_time = time.time()
            print(f"\nStep 4: Saving processed data to {output_path}")
            with open(output_path, 'wb') as f:
                pickle.dump(G, f)
                
            if not output_path.exists():
                raise IOError(f"Failed to create output file: {output_path}")
                
            file_size = output_path.stat().st_size
            if file_size == 0:
                raise IOError(f"Output file is empty: {output_path}")
                
            print(f"Successfully saved graph to {output_path} ({file_size/1024/1024:.1f} MB)")
            step4_duration = time.time() - step4_time
            
            total_duration = time.time() - start_time
            print(f"\nPreprocessing complete!")
            print(f"Total time taken: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
            print(f"Step 1 (Parsing): {step1_duration:.1f}s ({(step1_duration/total_duration)*100:.1f}%)")
            print(f"Step 2 (Processing): {step2_duration:.1f}s ({(step2_duration/total_duration)*100:.1f}%)")
            print(f"Step 3 (Graph Building): {step3_duration:.1f}s ({(step3_duration/total_duration)*100:.1f}%)")
            print(f"Step 4 (Saving): {step4_duration:.1f}s ({(step4_duration/total_duration)*100:.1f}%)")
            print(f"Graph contains {len(G.nodes):,} nodes and {len(G.edges):,} edges")
            print(f"\nSuccessfully processed {region_name}")
            return G
            
        except Exception as inner_e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\nError processing {region_name}:")
            print(f"Error details: {str(inner_e)}")
            print(f"Full traceback:\n{error_details}")
            
            # Try to save partial results if we have a graph
            if G is not None and len(G.nodes) > 0:
                try:
                    partial_output = output_path.parent / f"{region_name}_partial.pkl"
                    with open(partial_output, 'wb') as f:
                        pickle.dump(G, f)
                    print(f"Saved partial results to {partial_output}")
                except Exception as save_error:
                    print(f"Failed to save partial results: {str(save_error)}")
            return None
            
    except Exception as outer_e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\nOuter error processing {region_name}:")
        print(f"Error details: {str(outer_e)}")
        print(f"Full traceback:\n{error_details}")
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
    
    # Ensure output directory exists
    PROCESSED_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    
    preprocess_osm_file_cpu(osm_file)