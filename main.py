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
        interm_pts = (1 - alpha) * self.pts + alpha * self.end_pts
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
fig, ax = plt.subplots()

def eigenvec_mult(fig, ax, matrix):
    vals, vecs = np.linalg.eig(matrix)
    end_vecs = vecs * vals
    xs, ys = [0, 0, 1, 1], [0, 1, 1, 0]
    pts = np.stack([xs, ys], axis=-1)
    end_pts = np.dot(matrix, pts.T).T

    eigenvec1 = StaticVector(ax, (0, 0), vecs[:, 0])
    eigenvec2 = StaticVector(ax, (0, 0), vecs[:, 1])
    grid = LinearInterpQuad(ax, xs, ys, end_pts[:, 0], end_pts[:, 1])
    eigenvec1_anim = LinearInterpVector(ax, (0, 0), vecs[:, 0], end_vecs[:, 0])
    eigenvec2_anim = LinearInterpVector(ax, (0, 0), vecs[:, 1], end_vecs[:, 1])

    static_elems = [eigenvec1, eigenvec2]
    animated_elems = [grid, eigenvec1_anim, eigenvec2_anim]
    return static_elems, animated_elems

def diagonal_addition_eigenvecs(fig, ax, matrix, rowspace=False):
    vals, vecs = np.linalg.eig(matrix)
    xs, ys = [0, 0, 1, 1], [0, 1, 1, 0]
    pts = np.stack([xs, ys], axis=-1)
    end_pts = np.dot(matrix, pts.T).T
    print("Eigenvalues: %s" % str(vals))
    lower_range = min(0, np.min(vals))
    upper_range = max(0, np.max(vals))
    lower_cols = matrix - np.eye(2) * lower_range
    upper_cols = matrix - np.eye(2) * upper_range
    if rowspace:
        lower_cols, upper_cols = lower_cols.T, upper_cols.T

    eigenvec1 = StaticVector(ax, (0, 0), vecs[:, 0])
    eigenvec2 = StaticVector(ax, (0, 0), vecs[:, 1])
    grid = LinearInterpQuad(ax, xs, ys, end_pts[:, 0], end_pts[:, 1])
    column1_anim = LinearInterpVector(ax, (0, 0), lower_cols[:, 0], upper_cols[:, 0])
    column2_anim = LinearInterpVector(ax, (0, 0), lower_cols[:, 1], upper_cols[:, 1])

    static_elems = [eigenvec1, eigenvec2]
    animated_elems = [grid, column1_anim, column2_anim]
    return static_elems, animated_elems

# static_elems, animated_elems = eigenvec_mult(fig, ax, matrix)
static_elems, animated_elems = diagonal_addition_eigenvecs(fig, ax, matrix)
anim = Animation(fig, ax, static_elems, animated_elems, num_disp_points)
anim.animate()
