import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_3d_cnn():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Helper function to draw 3D cuboids
    def draw_cuboid(ax, x, y, z, dx, dy, dz, label, color):
        vertices = [
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz]
        ]
        # Edges are defined as tuples of vertex indices
        edges = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
            [vertices[3], vertices[0], vertices[4], vertices[7]],  # Left face
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
        ]
        ax.add_collection3d(Poly3DCollection(edges, alpha=0.7, facecolor=color, edgecolor="black"))
        ax.text(x + dx / 2, y + dy / 2, z + dz + 2, label, ha='center', fontsize=10)

    # Define layers
    layers = [
        {"label": "Input\n20x5", "x": 0, "y": 0, "z": 0, "dx": 5, "dy": 20, "dz": 1, "color": "skyblue"},
        {"label": "Conv2D\n32 filters\n20x32", "x": 10, "y": 0, "z": 0, "dx": 5, "dy": 20, "dz": 32, "color": "lightgreen"},
        {"label": "MaxPool\n10x32", "x": 20, "y": 0, "z": 0, "dx": 5, "dy": 10, "dz": 32, "color": "lightcoral"},
        {"label": "Conv2D\n64 filters\n10x64", "x": 30, "y": 0, "z": 0, "dx": 5, "dy": 10, "dz": 64, "color": "lightgreen"},
        {"label": "MaxPool\n5x64", "x": 40, "y": 0, "z": 0, "dx": 5, "dy": 5, "dz": 64, "color": "lightcoral"},
        {"label": "Flatten\n320", "x": 50, "y": 5, "z": 0, "dx": 5, "dy": 1, "dz": 1, "color": "khaki"},
        {"label": "Dense\n128 neurons", "x": 60, "y": 5, "z": 0, "dx": 5, "dy": 1, "dz": 1, "color": "lightblue"},
        {"label": "Output\n2 neurons", "x": 70, "y": 5, "z": 0, "dx": 5, "dy": 1, "dz": 1, "color": "lightblue"}
    ]

    # Draw layers
    for layer in layers:
        draw_cuboid(ax, x=layer["x"], y=layer["y"], z=layer["z"],
                    dx=layer["dx"], dy=layer["dy"], dz=layer["dz"],
                    label=layer["label"], color=layer["color"])

    # Add connections (arrows)
    for i in range(len(layers) - 1):
        start = layers[i]
        end = layers[i + 1]
        ax.quiver(
            start["x"] + start["dx"], start["y"] + start["dy"] / 2, start["z"] + start["dz"] / 2,
            end["x"] - start["x"] - start["dx"], 0, 0,
            arrow_length_ratio=0.2, color="black"
        )

    # Set the limits and labels
    ax.set_xlim(0, 80)
    ax.set_ylim(-10, 30)
    ax.set_zlim(0, 70)
    ax.set_box_aspect([2, 1, 1])  # Aspect ratio for better visualization
    ax.axis("off")
    plt.title("3D Visualization of CNN Model", fontsize=16)
    plt.show()


# Draw the 3D CNN
draw_3d_cnn()
