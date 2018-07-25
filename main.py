import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class StaticVector():
    def __init__(self, ax, origin, vector):
        self.origin = origin
        self.vector = vector
        self.arrow = ax.arrow(self.origin[0], self.origin[1],
                              self.vector[0], self.vector[1], head_width=1e-1)
    def bounds(self):
        coords = np.stack([self.origin, self.origin + self.vector], axis=-1)
        return coords
    def render_obj(self):
        return self.arrow

class AnimatedElement():
    def bounds(self):
        pass
    def render_obj(self):
        pass
    def update(self, alpha):
        pass

class LinearInterpVector(AnimatedElement):
    def __init__(self, ax, origin, vector, end_vector):
        self.origin = origin
        self.vector = vector 
        self.end_vector = end_vector 
        self.arrow = ax.arrow(self.origin[0], self.origin[1],
                              self.vector[0], self.vector[1], head_width=1e-1)
    def bounds(self):
        coords = np.stack([self.origin, self.origin + self.vector], axis=0)
        end_coords = np.stack([self.origin,
                               self.origin + self.end_vector], axis=0)
        return np.vstack([coords, end_coords])
    def render_obj(self):
        return self.arrow
    def update(self, alpha):
        interp_vec = (1 - alpha) * self.vector + alpha * self.end_vector
        self.arrow.set_xy([tuple(self.origin), tuple(interp_vec)])

class LinearInterpQuad(AnimatedElement):
    def __init__(self, ax, xs, ys, end_xs, end_ys, alpha=0.5):
        self.pts = np.stack([xs, ys], axis=-1)
        self.end_pts = np.stack([end_xs, end_ys], axis=-1)
        self.quad, = ax.fill(np.zeros(4), np.zeros(4), alpha=alpha)
    def bounds(self):
        return np.vstack([self.pts, self.end_pts])
    def render_obj(self):
        return self.quad
    def update(self, alpha):
        interm_pts = (1 - alpha) * pts + alpha * end_pts
        self.quad.set_xy(interm_pts)

class Animation():
    def __init__(self, fig, ax, static_elems, animated_elems, num_disp_pts):
        self.fig, self.ax = fig, ax
        self.static_elems = static_elems
        self.animated_elems = animated_elems
        self.num_disp_pts = num_disp_pts
        self.alpha_range = np.linspace(0, 1, num_disp_points)
    def animate(self, outpath='out.mp4'):
        ani = FuncAnimation(self.fig, self.update, frames=self.alpha_range,
                            init_func=self.init, interval=1, blit=True)
        ani.save(outpath, fps=60)
        plt.show()
    def init(self):
        # Initialize the viewing window to the maximum of all points.
        bounding_coords = []
        for elem in self.static_elems:
            bounding_coords.append(elem.bounds())
        for elem in self.animated_elems:
            bounding_coords.append(elem.bounds())
        bounding_coords = np.vstack(bounding_coords)
        print(bounding_coords)
        self.ax.set_xlim([np.min(bounding_coords[:, 0]),
                          np.max(bounding_coords[:, 0])])
        self.ax.set_ylim([np.min(bounding_coords[:, 1]),
                          np.max(bounding_coords[:, 1])])
        return [x.render_obj() for x in self.static_elems + self.animated_elems]
    def update(self, alpha):
        for elem in self.animated_elems:
            elem.update(alpha)
        return [x.render_obj() for x in self.animated_elems]

num_disp_points = 1e2
# matrix = np.array([[5, 2],
#                    [4, 3]])
matrix = np.array([[8, -1],
                   [1, 4]])
vals, vecs = np.linalg.eig(matrix)
end_vecs = vecs * vals
xs, ys = [0, 0, 1, 1], [0, 1, 1, 0]
pts = np.stack([xs, ys], axis=-1)
end_pts = np.dot(matrix, pts.T).T

fig, ax = plt.subplots()
eigenvec1 = StaticVector(ax, (0, 0), vecs[:, 0])
eigenvec2 = StaticVector(ax, (0, 0), vecs[:, 1])
grid = LinearInterpQuad(ax, xs, ys, end_pts[:, 0], end_pts[:, 1])
eigenvec1_anim = LinearInterpVector(ax, (0, 0), vecs[:, 0], end_vecs[:, 0])
eigenvec2_anim = LinearInterpVector(ax, (0, 0), vecs[:, 1], end_vecs[:, 1])
static_elems = [eigenvec1, eigenvec2]
animated_elems = [grid, eigenvec1_anim, eigenvec2_anim]

anim = Animation(fig, ax, static_elems, animated_elems, num_disp_points)
anim.animate()
