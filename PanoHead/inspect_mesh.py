import trimesh
import numpy as np

def load_and_triangulate_mesh(file_path):
    # Load the original mesh from an OBJ file
    mesh = trimesh.load(file_path, process=False)
    
    # Ensure the mesh is triangulated.
    # Trimesh automatically triangulates faces when loading,
    # but this step ensures that any non-triangular faces are converted.
    if not all(len(face) == 3 for face in mesh.faces):
        mesh = mesh.subdivide_to_size(max_edge=mesh.scale / 2, max_iter=1)
    
    # Convert to a triangle mesh if it's not already
    if not mesh.is_watertight:
        mesh = mesh.convex_hull

    # Extract vertices and faces as NumPy arrays
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    return vertices, faces

# Path to your OBJ file
file_path = '/home/beckmann/Projects/PanoHead/data/seed0003_mesh.obj'

# Load the mesh and triangulate
vertices, faces = load_and_triangulate_mesh(file_path)

print("Vertices:", vertices[:10])
print("Faces:", faces[:10])