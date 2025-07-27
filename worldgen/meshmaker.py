import meshlib.mrmeshpy as mrmeshpy

def generate_mesh():
    # Placeholder just loading an existing mesh
    try:
        mesh = mrmeshpy.loadMesh("resource\Ball OBJ.obj")
        return mesh
    except Exception as e:
        print(e)
