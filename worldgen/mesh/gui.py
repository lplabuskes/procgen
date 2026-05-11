from typing import Optional

import OpenGL.GL as GL
import OpenGL.GLU as GLU
from OpenGL.arrays import vbo

from PySide6.QtWidgets import QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtCore import QTimer

from mesh.mymesh import PolygonMesh

class MeshViewer(QOpenGLWidget):
    def __init__(self, parent=None, mesh: Optional[PolygonMesh]=None):
        super(MeshViewer, self).__init__(parent)
        self.mesh: Optional[PolygonMesh] = mesh
        self.vertex_buffer: Optional[vbo.VBO] = None
        self.color_buffer: Optional[vbo.VBO] = None
        self.angle = 0

    def initializeGL(self):
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        if self.mesh is not None:
            self.vertex_buffer = vbo.VBO(self.mesh.render_vertices)
            self.color_buffer = vbo.VBO(self.mesh.render_colors)

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(45.0, w / h if h != 0 else 1.0, 0.1, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def paintGL(self):
        if self.mesh is None or self.vertex_buffer is None or self.color_buffer is None:
            return
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        GL.glTranslatef(0.0, 0.0, -5.0)
        GL.glRotatef(self.angle, 1.0, 1.0, 0.0)

        self.vertex_buffer.bind()
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, None)

        self.color_buffer.bind()
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glColorPointer(4, GL.GL_FLOAT, 0, None)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self.mesh.render_vertices))

        self.color_buffer.unbind()
        self.vertex_buffer.unbind()
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

        self.angle += 0.5

    def cleanup(self):
        self.makeCurrent()
        if self.vertex_buffer is not None:
            self.vertex_buffer.delete()
            self.vertex_buffer = None
        if self.color_buffer is not None:
            self.color_buffer.delete()
            self.color_buffer = None
        self.doneCurrent()


class MainWindow(QMainWindow):
    def __init__(self, mesh: Optional[PolygonMesh]=None):
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

    def closeEvent(self, event):
        self.timer.stop()
        self.viewer.cleanup()
        super().closeEvent(event)

    def set_mesh(self, mesh: PolygonMesh):
        self.viewer.mesh = mesh
        self.viewer.vertex_buffer = vbo.VBO(mesh.render_vertices)
        self.viewer.color_buffer = vbo.VBO(mesh.render_colors)