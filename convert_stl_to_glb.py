import trimesh
import os
from pathlib import Path

def convert_stl_to_glb(stl_path, glb_path):
    """Converts a single STL file to a GLB file."""
    try:
        mesh = trimesh.load_mesh(stl_path)
        mesh.export(glb_path, file_type='glb')
        print(f"Successfully converted {stl_path} to {glb_path}")
    except Exception as e:
        print(f"Could not convert {stl_path}. Error: {e}")

def batch_convert_in_directory(directory):
    """Batch converts all STL files in a given directory to GLB."""
    path = Path(directory)
    stl_files = list(path.rglob('*.stl'))
    
    if not stl_files:
        print(f"No STL files found in {directory}")
        return

    for stl_file in stl_files:
        glb_file = stl_file.with_suffix('.glb')
        convert_stl_to_glb(str(stl_file), str(glb_file))
        # Optional: remove original stl file
        # os.remove(stl_file)

if __name__ == '__main__':
    # Define the base directory for the orca_hand assets
    base_dir = Path('dex-retargeting/assets/robots/hands/orca_hand')
    
    # Define subdirectories to process
    visual_dir = base_dir / 'visual'
    collision_dir = base_dir / 'collision'
    
    print("Starting conversion for visual meshes...")
    batch_convert_in_directory(visual_dir)
    
    print("\nStarting conversion for collision meshes...")
    batch_convert_in_directory(collision_dir)
    
    print("\nBatch conversion process finished.") 