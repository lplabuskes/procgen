import meshlib.mrmeshpy as mrmeshpy
import OpenGL.GL as GL
import OpenGL.GLU as GLU

from PySide6.QtWidgets import QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtCore import QTimer


def repack_tripoint(point: mrmeshpy.Vector3f):
    return (point.x, point.y, point.z)

class MeshViewer(QOpenGLWidget):
    def __init__(self, parent=None, mesh: mrmeshpy.Mesh=None):
        super(MeshViewer, self).__init__(parent)
        self.mesh: mrmeshpy.Mesh = mesh
        self.angle = 0

    def initializeGL(self):
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(45.0, w / h if h != 0 else 1.0, 0.1, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def paintGL(self):
        if self.mesh is None:
            return
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        GL.glTranslatef(0.0, 0.0, -5.0)
        GL.glRotatef(self.angle, 1.0, 1.0, 0.0)

        GL.glBegin(GL.GL_TRIANGLES)
        # for face in self.mesh.topology.getValidFaces():
        #     for point in self.mesh.getTriPoints(face):
        #         GL.glVertex3fv(repack_tripoint(point))
        GL.glVertex3f(-1.0, -1.0, 0.0)
        GL.glVertex3f(1.0, -1.0, 0.0)
        GL.glVertex3f(0.0, 1.0, 0.0)
        GL.glEnd()

        self.angle += 0.5


class MainWindow(QMainWindow):
    def __init__(self, mesh: mrmeshpy.Mesh=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Mesh Viewer")
        self.resize(800, 600)

        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
        QSurfaceFormat.setDefaultFormat(fmt)

        self.viewer = MeshViewer(self, mesh)
        self.viewer.setFormat(fmt)
        self.setCentralWidget(self.viewer)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.viewer.update)
        self.timer.start(16)  # ~60 FPS

    def set_mesh(self, mesh: mrmeshpy.Mesh):
        self.viewer.mesh = mesh