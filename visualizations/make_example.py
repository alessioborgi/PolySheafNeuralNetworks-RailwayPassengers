import matplotlib.pyplot as plt
import numpy as np

# Node positions
v_pos = np.array([0, 0])
u_pos = np.array([1, 0.5])

# Two vectors in node v
vecs = [np.array([0.6, 0.2]), np.array([0.3, -0.4])]

# Edge direction
r = u_pos - v_pos
r_hat = r / np.linalg.norm(r)
t_hat = np.array([-r_hat[1], r_hat[0]])

# radial/tangential scaling
a, b = 1.2, 0.6

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Draw nodes and edge
ax.scatter(*v_pos, s=500, color='darkblue')
ax.scatter(*u_pos, s=500, color='darkred')
ax.plot([v_pos[0], u_pos[0]], [v_pos[1], u_pos[1]], 'k-', linewidth=2)

# Draw vectors
colors = ['blue', 'green']
for i, vec in enumerate(vecs):
    rad = a * np.dot(vec, r_hat) * r_hat
    tan = b * np.dot(vec, t_hat) * t_hat
    transported = rad + tan
    # radial component
    ax.arrow(v_pos[0], v_pos[1], rad[0], rad[1], color='red', width=0.01)
    # tangential component
    ax.arrow(v_pos[0], v_pos[1], tan[0], tan[1], color='orange', width=0.01)
    # final transported vector
    ax.arrow(v_pos[0], v_pos[1], transported[0], transported[1], color=colors[i], width=0.015, alpha=0.7)

plt.show()
