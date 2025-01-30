from pathlib import Path
import sys
import os
#from preprocess_gpu import preprocess_osm_file
from preprocess_cpu import preprocess_osm_file_cpu

def process_all_new_files(processor='gpu'):
    """Process all new OSM files that haven't been processed yet"""
    # Get absolute paths
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    raw_dir = root_dir / 'data' / 'maps' / 'raw'
    processed_dir = root_dir / 'data' / 'maps' / 'processed'

    print(f"Looking for OSM files in: {raw_dir}")

    # Create directories if they don't exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Get list of all OSM files
    osm_files = list(raw_dir.glob('*.osm.pbf'))
    
    if not osm_files:
        print(f"No OSM files found in {raw_dir}")
        print("Please place .osm.pbf files in the raw directory")
        # List all files in the directory to help debug
        print("\nFiles found in directory:")
        for file in raw_dir.iterdir():
            print(f"- {file.name}")
        return

    print(f"\nFound {len(osm_files)} OSM files:")
    for f in osm_files:
        print(f"- {f.name}")
    
    # Process each file that hasn't been processed yet
    for osm_file in osm_files:
        region_name = osm_file.stem.split('.')[0]
        processed_file = processed_dir / f"{region_name}_processed.pkl"
        
        if processed_file.exists():
            print(f"Skipping {region_name} - already processed")
            continue
            
        print(f"\nProcessing {region_name} using {processor.upper()}...")
        try:
            if processor == 'gpu':
                #preprocess_osm_file(osm_file.name)
                print("gpu")
            else:
                preprocess_osm_file_cpu(osm_file.name)
            print(f"Successfully processed {region_name}")
        except Exception as e:
            print(f"Error processing {region_name}: {e}")
            print(f"Full error: {str(e)}")

if __name__ == "__main__":
    processor = 'gpu'  # Default to GPU
    if len(sys.argv) > 1 and sys.argv[1] in ['cpu', 'gpu']:
        processor = sys.argv[1]
    
    process_all_new_files(processor) 