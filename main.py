import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
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

class StaticContour():
    def __init__(self, ax, xs, ys, zs, N, fill=True):
        self.xs, self.ys, self.zs = xs, ys, zs
        self.N = N
        if fill:
            self.contour = ax.contourf(xs, ys, zs, N)
        else:
            self.contour = ax.contour(xs, ys, zs, N)
    def bounds(self):
        x_bounds = [np.min(self.xs), np.max(self.xs)]
        y_bounds = [np.min(self.ys), np.max(self.ys)]
        return np.stack([x_bounds, y_bounds], axis=1)
    def render_obj(self):
        return self.contour

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

class LinearMatrixInterpEigenvector(AnimatedElement):
    def __init__(self, ax, index, matrix, end_matrix):
        self.vals, self.vecs = np.linalg.eig(matrix)
        self.end_vals, self.end_vecs = np.linalg.eig(matrix)
        self.index = index
        self.matrix = matrix
        self.end_matrix = end_matrix
        self.arrow = ax.arrow(0, 0, self.vecs[0, index], self.vecs[1, index],
                              head_width=1e-1)
    def bounds(self):
        coords = np.stack([self.vecs[:, self.index],
                           self.end_vecs[:, self.index]], axis=0)
        return coords
    def render_obj(self):
        return self.arrow
    def update(self, alpha):
        interm_matrix = (1 - alpha) * self.matrix + alpha * self.end_matrix
        vals, vecs = np.linalg.eig(interm_matrix)
        print(vecs)
        self.arrow.set_xy([(0, 0), tuple(vecs[:, self.index])])

class AnimatedContour(AnimatedElement):
    def __init__(self, ax, xs, ys, zs_list):
        self.xs, self.ys, self.zs_list = xs, ys, zs_list
        norm = matplotlib.colors.Normalize(vmin=np.min(zs_list),
                                           vmax=np.max(zs_list))
        # Implement smooth contour using imshow as contour and contourf
        # cannot be animated.
        self.contour = ax.imshow(
            zs_list[0][::-1], extent=[np.min(xs), np.max(xs),
                                      np.min(ys), np.max(ys)],
            norm=norm, aspect='auto')
    def bounds(self):
        x_bounds = [np.min(self.xs), np.max(self.xs)]
        y_bounds = [np.min(self.ys), np.max(self.ys)]
        return np.stack([x_bounds, y_bounds], axis=1)
    def render_obj(self):
        return self.contour
    def update(self, alpha):
        idx = int(np.round(alpha * (len(self.xs) - 1)))
        self.contour.set_data(self.zs_list[idx][::-1])

class AnimatedPlot(AnimatedElement):
    def __init__(self, ax, xs, ys):
        self.xs, self.ys = xs, ys
        self.plot, = ax.plot(xs[0], ys[0])
    def bounds(self):
        x_bounds = [np.min(self.xs), np.max(self.xs)]
        y_bounds = [np.min(self.ys), np.max(self.ys)]
        return np.stack([x_bounds, y_bounds], axis=1)
    def render_obj(self):
        return self.plot
    def update(self, alpha):
        idx = int(np.round(alpha * (len(self.xs) - 1)))
        self.plot.set_xdata(self.xs[:idx])
        self.plot.set_ydata(self.ys[:idx])

class AnimatedVectorField(AnimatedElement):
    def __init__(self, ax, xs, ys, zs_list):
        self.xs, self.ys, self.zs_list = xs, ys, zs_list
        norm = matplotlib.colors.Normalize(vmin=np.min(zs_list),
                                           vmax=np.max(zs_list))
        self.field = ax.quiver(xs, ys, zs_list[0][:, :, 0],
                                       zs_list[0][:, :, 1])
        self.field.set_array(zs_list[0])
    def bounds(self):
        x_bounds = [np.min(self.xs), np.max(self.xs)]
        y_bounds = [np.min(self.ys), np.max(self.ys)]
        return np.stack([x_bounds, y_bounds], axis=1)
    def render_obj(self):
        return self.field
    def update(self, alpha):
        idx = int(np.round(alpha * (len(self.zs_list) - 1)))
        zs = self.zs_list[idx]
        self.field.set_UVC(zs[:, :, 0], zs[:, :, 1], zs[:, :, 2])

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
# matrix = np.array([[8, -1], [1, 4]])
# matrix = np.array([[2, 1],
#                    [1, 4]])
matrix = np.array([[2, 1, 3],
                   [1, 4, 2],
                   [3, 2, 1]])
fig, ax = plt.subplots()
def eigenvec_mult(fig, ax, matrix):
    vals, vecs = np.linalg.eig(matrix)
    print("Eigenvalues: %s" % str(vals))
    print("Eigenvectors: %s" % str(vecs))
    print(np.linalg.norm(matrix - vals[1] * np.outer(vecs[:, 1], vecs[:, 1])))
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

def diagonal_addition_eigenvec_cols(fig, ax, matrix, rowspace=False):
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

def diagonal_addition_eigenvecs(fig, ax, matrix, upper_range=5):
    end_matrix = matrix + np.eye(2) * upper_range
    xs, ys = [0, 0, 1, 1], [0, 1, 1, 0]
    pts = np.stack([xs, ys], axis=-1)
    end_pts1 = np.dot(matrix, pts.T).T
    end_pts2 = np.dot(end_matrix, pts.T).T

    grid = LinearInterpQuad(ax, end_pts1[:, 0], end_pts1[:, 1],
                                end_pts2[:, 0], end_pts2[:, 1])
    eigenvec1_anim = LinearMatrixInterpEigenvector(ax, 0, matrix, end_matrix)
    eigenvec2_anim = LinearMatrixInterpEigenvector(ax, 1, matrix, end_matrix)

    static_elems, animated_elems = [], [grid, eigenvec1_anim, eigenvec2_anim]
    return static_elems, animated_elems

def solve_1d_dyad_approx(matrix, tol=1e-10):
    # Initialize a unit length vector randomly on the unit sphere.
    a, b = np.random.randn(matrix.shape[0]), np.random.randn(matrix.shape[1])
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    matrix_col_norm = np.linalg.norm(matrix, axis=0)
    while True:
        prev_a, prev_b = np.copy(a), np.copy(b)
        # Fix b, solve for minimal .
        w = np.square(b)
        a = np.dot((1 / b) * matrix, w) / np.sum(w)
        # Project each column onto a.
        b = np.dot(a / np.linalg.norm(a), matrix) / np.linalg.norm(a)
        if np.allclose(a, prev_a, tol) and np.allclose(b, prev_b, tol):
            break
    return a, b

def solve_1d_symm_dyad_approx(matrix, lr=1e-2, tol=1e-10,
                              max_iter=10000, history=False):
    # Initialize a unit length vector randomly on the unit sphere.
    a = np.random.randn(matrix.shape[0])
    a /= np.linalg.norm(a)
    a_hist = [np.copy(a)]
    for i in range(max_iter):
        prev_a = np.copy(a)
        partial = np.tile(np.expand_dims(a, axis=1), [1, len(a)])
        grad = -partial * (matrix - np.outer(a, a))
        grad = np.sum(grad, axis=0)
        a -= lr * grad
        a_hist.append(np.copy(a))
        if np.allclose(a, prev_a, tol):
            b = a
            if history:
                return a, b, a_hist
            else:
                return a, b
    a = np.zeros(matrix.shape[0])
    b = a
    return a, b

def solve_nd_dyad_approx(matrix, n):
    curr_matrix = matrix
    vecs, scalars = [], []
    for i in range(n):
        a, b = solve_1d_dyad_approx(curr_matrix)
        curr_matrix = curr_matrix - np.outer(a, b)
        vecs.append(a)
        scalars.append(b)
    vecs = np.stack(vecs, axis=1)
    scalars = np.stack(scalars, axis=0)
    return vecs, scalars

def solve_nd_symm_dyad_approx(matrix, n):
    curr_matrix = matrix
    vecs, scalars = [], []
    # Find dyad corresponding to positive eigenvalues.
    for i in range(n):
        a, b = solve_1d_symm_dyad_approx(curr_matrix)
        if np.allclose(a, np.zeros(len(a)), 1e-10):
            break
        curr_matrix = curr_matrix - np.outer(a, b)
        vecs.append(a)
        scalars.append(b)
    # Find dyad corresponding to negative eigenvalues.
    curr_matrix = -curr_matrix 
    for i in range(n - len(vecs)):
        a, b = solve_1d_symm_dyad_approx(curr_matrix)
        if np.allclose(a, np.zeros(len(a)), 1e-10):
            break
        curr_matrix = curr_matrix - np.outer(a, b)
        vecs.append(a)
        scalars.append(-b)
    vecs = np.stack(vecs, axis=1)
    scalars = np.stack(scalars, axis=0)
    return vecs, scalars

def dyad_1d_em(fig, ax, matrix):
    a, b = solve_nd_dyad_approx(matrix, 2)

def symm_dyad_1d_em(fig, ax, matrix):
    a, b = solve_nd_symm_dyad_approx(matrix, 3)
    vals, vecs = np.linalg.eig(matrix)

def symm_dyad_loss_path(fig, ax, matrix):
    a, b, a_hist = solve_1d_symm_dyad_approx(matrix, history=True)
    if np.allclose(a, np.zeros(len(a)), 1e-10):
        a, b, a_hist = solve_1d_symm_dyad_approx(-matrix, history=True)
        b = -a
    a_hist = np.array(a_hist)
    if len(a) == 2:
        xs, ys = np.meshgrid(np.linspace(-4, 4, 100),
                             np.linspace(-4, 4, 100))
        xy = np.stack([xs, ys], axis=-1)
        dyad = xy[:, :, :, None] * xy[:, :, None, :]
        loss = np.linalg.norm(dyad - matrix, axis=(2, 3))
        contour = StaticContour(ax, xs, ys, loss, 20)
        plot = AnimatedPlot(ax, a_hist[:, 0], a_hist[:, 1])
        static_elems = [contour]
        animated_elems = [plot]
        vals, vecs = np.linalg.eig(matrix)
    elif len(a) == 3:
        losses = []
        xs, ys, zs = np.meshgrid(np.linspace(-2, 2, 100),
                                 np.linspace(-2, 2, 100),
                                 np.linspace(-2, 2, 100))
        xyz = np.stack([xs, ys, zs], axis=-1)
        dyad = xyz[:, :, :, :, None] * xyz[:, :, :, None, :]
        loss = np.linalg.norm(dyad - matrix, axis=(3, 4))
        j = np.argmin(loss)
        v = np.array([xs.flatten()[j], ys.flatten()[j], zs.flatten()[j]])
        xs, ys = np.meshgrid(np.linspace(-2, 2, 100),
                             np.linspace(-2, 2, 100))
        xy = np.stack([xs, ys], axis=-1)
        for i, vec in enumerate(a_hist):
            z = vec[-1]
            xyz = np.append(xy, z * np.ones(xs.shape + (1,)), axis=-1)
            dyad = xyz[:, :, :, None] * xyz[:, :, None, :]
            loss = np.linalg.norm(dyad - matrix, axis=(2, 3))
            losses.append(loss)
        contour = AnimatedContour(ax, xs, ys, losses)
        plot = AnimatedPlot(ax, a_hist[:, 0], a_hist[:, 1])
        static_elems = []
        animated_elems = [contour, plot]
    return static_elems, animated_elems

def symm_dyad_loss_slice(fig, ax, matrix):
    xs, ys, zs = np.meshgrid(np.linspace(-2, 2, 100),
                             np.linspace(-2, 2, 100),
                             np.linspace(-2, 2, 100))
    xyz = np.stack([xs, ys, zs], axis=-1)
    dyad = xyz[:, :, :, :, None] * xyz[:, :, :, None, :]
    losses = np.linalg.norm(dyad - matrix, axis=(3, 4))
    gradients = \
        np.tile(xyz[:, :, :, :, None], [1, 1, 1, 1, 3]) * (-dyad + matrix)
    gradients = np.transpose(np.sum(gradients, axis=3), (2, 0, 1, 3))
    contour = AnimatedContour(ax, xs, ys, losses)
    grad_field = AnimatedVectorField(ax, xs[::5, ::5, 0], ys[::5, ::5, 0],
                                     gradients[:, ::5, ::5])
    fig.colorbar(contour.contour)
    # fig.colorbar(grad_field.field)
    static_elems = []
    animated_elems = [contour, grad_field]
    return static_elems, animated_elems

test_fns = {'eigenvec_mult': eigenvec_mult,
            'diag_add_eigenvec': diagonal_addition_eigenvecs,
            'diag_add_eigenvec_cols': diagonal_addition_eigenvec_cols,
            'symm_dyad_loss_path': symm_dyad_loss_path,
            'symm_dyad_loss_slice': symm_dyad_loss_slice,
            'symm_dyad': symm_dyad_1d_em}

option = sys.argv[1]
static_elems, animated_elems = test_fns[option](fig, ax, matrix)
if len(animated_elems) == 0:
    plt.show()
else:
    anim = Animation(fig, ax, static_elems, animated_elems, num_disp_points)
    anim.animate()
