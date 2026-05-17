import math
from typing import Optional, Tuple

import numpy as np
import OpenGL.GL as GL
import OpenGL.GLU as GLU
from OpenGL.arrays import vbo

from PySide6.QtWidgets import QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtCore import QTimer

from mesh.mymesh import PolygonMesh


class MeshViewer(QOpenGLWidget):
    def __init__(self, parent=None, mesh: Optional[PolygonMesh] = None):
        super(MeshViewer, self).__init__(parent)
        self.mesh: Optional[PolygonMesh] = mesh
        self.vertex_buffer: Optional[vbo.VBO] = None
        self.color_buffer: Optional[vbo.VBO] = None
        self.uv_buffer: Optional[vbo.VBO] = None
        self.texture_id: Optional[int] = None
        self.angle = 0

    def initializeGL(self):
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        if self.mesh is not None:
            self._setup_buffers(self.mesh)

    def _setup_buffers(self, mesh: PolygonMesh) -> None:
        self.vertex_buffer = vbo.VBO(mesh.render_vertices)
        if mesh.face_textures:
            self.color_buffer = None
            atlas_bytes, atlas_uvs, atlas_h, atlas_w = self._pack_texture_atlas(mesh)
            self.uv_buffer = vbo.VBO(atlas_uvs)
            self.texture_id = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, atlas_w, atlas_h, 0,
                            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, atlas_bytes)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        else:
            self.color_buffer = vbo.VBO(mesh.render_colors)
            self.uv_buffer = None
            self.texture_id = None

    def _pack_texture_atlas(self, mesh: PolygonMesh) -> Tuple[np.ndarray, np.ndarray, int, int]:
        assert mesh.face_textures
        textures = next(iter(mesh.face_textures.values()))  # (N, H, W, 4) float32
        N, H, W = textures.shape[:3]
        cols = math.ceil(math.sqrt(N))
        rows = math.ceil(N / cols)
        atlas = np.zeros((rows * H, cols * W, 4), dtype=np.float32)
        for i in range(N):
            r, c = divmod(i, cols)
            atlas[r * H:(r + 1) * H, c * W:(c + 1) * W] = textures[i]
        atlas_bytes = np.ascontiguousarray((atlas * 255).clip(0, 255).astype(np.uint8))

        face_idx = mesh.render_face_indices          # (M,)
        local_uvs = mesh.render_local_uvs            # (M, 2)
        col_off = (face_idx % cols).astype(np.float32)
        row_off = (face_idx // cols).astype(np.float32)
        atlas_us = (col_off + local_uvs[:, 0]) / cols
        # Flip v: numpy row 0 is at the top; OpenGL v=0 is at the bottom.
        atlas_vs = (row_off + (1.0 - local_uvs[:, 1])) / rows
        atlas_uvs = np.stack([atlas_us, atlas_vs], axis=1).astype(np.float32)
        return atlas_bytes, atlas_uvs, rows * H, cols * W

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(45.0, w / h if h != 0 else 1.0, 0.1, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def paintGL(self):
        if self.mesh is None or self.vertex_buffer is None:
            return
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        GL.glTranslatef(0.0, 0.0, -5.0)
        GL.glRotatef(self.angle, 1.0, 1.0, 0.0)

        self.vertex_buffer.bind()
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, None)

        if self.texture_id is not None and self.uv_buffer is not None:
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            GL.glColor4f(1.0, 1.0, 1.0, 1.0)
            self.uv_buffer.bind()
            GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)
            GL.glTexCoordPointer(2, GL.GL_FLOAT, 0, None)
        else:
            if self.color_buffer is None:
                return
            self.color_buffer.bind()
            GL.glEnableClientState(GL.GL_COLOR_ARRAY)
            GL.glColorPointer(4, GL.GL_FLOAT, 0, None)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self.mesh.render_vertices))

        if self.texture_id is not None and self.uv_buffer is not None:
            self.uv_buffer.unbind()
            GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)
            GL.glDisable(GL.GL_TEXTURE_2D)
        else:
            if self.color_buffer is not None:
                self.color_buffer.unbind()
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)

        self.vertex_buffer.unbind()
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
        if self.uv_buffer is not None:
            self.uv_buffer.delete()
            self.uv_buffer = None
        if self.texture_id is not None:
            GL.glDeleteTextures([self.texture_id])
            self.texture_id = None
        self.doneCurrent()


class MainWindow(QMainWindow):
    def __init__(self, mesh: Optional[PolygonMesh] = None):
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
        self.viewer.makeCurrent()
        self.viewer.mesh = mesh
        self.viewer._setup_buffers(mesh)
        self.viewer.doneCurrent()
