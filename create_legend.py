import matplotlib.pyplot as plt
import numpy as np

color_dict = {}
with open("color.txt", "r") as file:
    for row in file:
        prof, r, g, b, a = [b for b in row.replace("\n", "").split(" ") if b != ""]
        if prof in ['-9999', 'nan', '0']: continue
        color_dict[prof] = (int(r), int(g), int(b), int(a))


# Define the class labels and colors (RGBA format)
legend_data = {
    "1": "Acropora Branching", 
    "2": "Acropora Tabular",
    "3": "Acropora SubMassive",
    "4": "Other Coral",
    "5": "Sand / Rubble",
}

# Create a figure
fig, ax = plt.subplots(figsize=(5, len(legend_data) * 0.5))

# Generate legend elements
for i, (index, label) in enumerate(legend_data.items()):
    color = np.array(color_dict.get(index)) / 255.0  # Normalize color to [0,1] range
    ax.add_patch(plt.Rectangle((0, i), 0.5, 1, color=color, ec="black"))
    ax.text(0.6, i + 0.5, label, va='center', fontsize=12)

# Formatting
ax.set_xlim(0, 2)
ax.set_ylim(0, len(legend_data))
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

# Save legend as an image
plt.savefig("legend.png", dpi=300, bbox_inches="tight")