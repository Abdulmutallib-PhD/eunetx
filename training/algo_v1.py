import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow

def draw_box(ax, text, xy, width=1.5, height=0.5, fc='lightblue', ec='black'):
    box = FancyBboxPatch(
        (xy[0]-width/2, xy[1]-height/2),
        width, height,
        boxstyle="round,pad=0.2",
        linewidth=1,
        facecolor=fc,
        edgecolor=ec
    )
    ax.add_patch(box)
    ax.text(xy[0], xy[1], text, ha='center', va='center', fontsize=10)

def draw_arrow(ax, start, end, text=None):
    ax.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(arrowstyle='->', lw=1.5)
    )
    if text:
        mid_x = (start[0] + end[0])/2
        mid_y = (start[1] + end[1])/2
        ax.text(mid_x, mid_y+0.2, text, ha='center', fontsize=9)

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)

# Encoder
draw_box(ax, 'Input CT Image\n$X$', (2, 7))
draw_arrow(ax, (2,6.75), (2,6.25))
draw_box(ax, 'Encoder Path\n$E_0 \\to E_n$', (2, 5.5))
draw_arrow(ax, (2,5.25), (5,4.75), text='$E_n$')

# Bottleneck
draw_box(ax, 'Bottleneck\n$B$', (5, 4.5))
draw_arrow(ax, (5,4.25), (8,3.75), text='$B$')

# Decoder
draw_box(ax, 'Decoder Path\n$D_0 \\to D_n$', (8, 3.5))
draw_arrow(ax, (8,3.25), (8,2.75), text='$D_n$')

# Output
draw_box(ax, 'Final Output\n$\hat{Y}$', (8, 2))

# LFF (skip connections)
draw_arrow(ax, (2,5), (8,3.75), text='LFF $S_i$')

# Annotations
ax.text(2,6, 'Weights: $w_{enc}$', fontsize=9, ha='left')
ax.text(5,4, 'Weights: -', fontsize=9, ha='center')
ax.text(8,3, 'Weights: $w_{dec}$', fontsize=9, ha='right')
ax.text(5,2, 'LFF Weights: $w_{LFF}$', fontsize=9, ha='center')

plt.title("UNetX Segmentation Pipeline", fontsize=12)
plt.tight_layout()
plt.show()
