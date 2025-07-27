from meshmaker import generate_mesh

import meshlib.mrmeshpy as mrmeshpy
import OpenGL.GL as GL
import OpenGL.GLUT as GLUT


def repack_tripoint(point: mrmeshpy.Vector3f):
    return (point.x, point.y, point.z)

def display(mesh: mrmeshpy.Mesh):
    GL.glBegin(GL.GL_LINES)
    for face in mesh.topology.getValidFaces():
        for point in mesh.getTriPoints(face):
            GL.glVertex3fv(repack_tripoint(point))
    GL.glEnd()

    GLUT.glutSwapBuffers()

if __name__ == "__main__":
    mesh = generate_mesh()
    display(mesh)
    print(mesh.points)
