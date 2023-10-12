import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy import sparse
from scipy.spatial import cKDTree

L = 32.0
rho = 3.0  # 3.0
N = int(rho * L**2)
print(" N", N)

r0 = 1.0
deltat = 1.0
factor = 0.5
v0 = r0 / deltat * factor
iterations = 110  # 10000
loop_n = 10
# eta = 0.1  # 0.15
seed = 0
for eta_n in range(1, 11):
    eta = eta_n * 0.1
    print("eta", eta)

    np.random.seed(seed)
    pos = np.random.uniform(0, L, size=(N, 2))
    orient = np.random.uniform(-np.pi, np.pi, size=N)

    # fig, ax = plt.subplots(figsize=(6, 6))
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("eta = {}".format(eta), fontsize=16)

    ax1 = fig.add_subplot(121)
    ax1.set_xlim(0, L)
    ax1.set_ylim(0, L)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    qv = ax1.quiver(
        pos[:, 0],
        pos[:, 1],
        np.cos(orient),
        np.sin(orient),
        orient,
        clim=[-np.pi, np.pi],
    )
    mf_array = np.zeros(iterations * loop_n, dtype=complex)
    abs_mf_array = np.zeros(iterations * loop_n, dtype=float)
    angle_mf_array = np.zeros(iterations * loop_n, dtype=float)
    ax2 = fig.add_subplot(222)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("abs of mf")
    ax2.set_ylim(0, 1)
    plot2 = ax2.plot(abs_mf_array, color="black")[0]
    ax3 = fig.add_subplot(224)
    ax3.set_ylabel("angle of mf")
    ax3.set_ylim(-np.pi, np.pi)
    ax3.set_xlabel("iteration")
    plot3 = ax3.plot(angle_mf_array, color="black")[0]

    def animate(pre_i):
        global orient

        for loop_i in range(loop_n):
            i = pre_i * 10 + loop_i
            tree = cKDTree(pos, boxsize=[L, L])
            dist = tree.sparse_distance_matrix(
                tree, max_distance=r0, output_type="coo_matrix"
            )

            # important 3 lines: we evaluate a quantity for every column j
            data = np.exp(orient[dist.col] * 1j)
            # construct  a new sparse marix with entries in the same places ij of the dist matrix
            neigh = sparse.coo_matrix(
                (data, (dist.row, dist.col)), shape=dist.get_shape()
            )
            # and sum along the columns (sum over j)
            S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))

            orient = np.angle(S) + eta * np.random.uniform(-np.pi, np.pi, size=N)

            cos, sin = np.cos(orient), np.sin(orient)
            pos[:, 0] += cos * v0
            pos[:, 1] += sin * v0

            pos[pos > L] -= L
            pos[pos < 0] += L

            mf = np.average(np.exp(1j * orient))
            abs_mf = np.abs(mf)
            angle_mf = np.angle(mf)
            mf_array[i] = mf
            abs_mf_array[i] = abs_mf
            angle_mf_array[i] = angle_mf

            if i % 100 == 99:
                print(i, abs_mf, angle_mf)

            if i == iterations * loop_n - 1:
                average_mf = np.average(mf_array[-100:])
                abs_average_mf = np.abs(average_mf)
                angle_average_mf = np.angle(average_mf)
                print([eta, abs_average_mf], angle_average_mf)

        qv.set_offsets(pos)
        qv.set_UVC(cos, sin, orient)
        ax1.set_title("iteration = {}".format(i))
        plot2.set_data(np.arange(i), abs_mf_array[:i])
        plot3.set_data(np.arange(i), angle_mf_array[:i])

    anim = FuncAnimation(
        fig,
        animate,
        np.arange(iterations),
        interval=1,
        # blit=True
        repeat=False,
    )

    anim.save(f"data/eta={eta}.gif", writer="imagemagick")
    # plt.show()

"""
seed = 0
[
    [0.0, 0.9998309378153204],
    [0.1, 0.971666862511216],
    [0.2, 0.9001647603014683],
    [0.3, 0.7963392365056187],
    [0.4, 0.6408669574596928],
    [0.5, 0.3076182810920998],
    [0.6, 0.08008597791291498],
    [0.7, 0.026328736364920346],
    [0.8, 0.004054810190902796],
    [0.9, 0.002602742712273006],
    [1.0, 0.001018604861399327],
]
"""
