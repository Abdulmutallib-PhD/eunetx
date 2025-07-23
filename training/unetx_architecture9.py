import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, ConnectionPatch
import matplotlib.image as mpimg

fig, ax = plt.subplots(figsize=(22, 12))

encoder_y = [0.9, 0.7, 0.5, 0.3]
bottleneck_y = 0.1
decoder_y = encoder_y[::-1]
x_left = 0.1
x_right = 0.7

# (a) Input Image
ax.text(x_left-0.1, encoder_y[0]+0.1, '(a) Input Image\n[256x256]', fontsize=10, ha='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
arrow_in = FancyArrowPatch((x_left-0.05, encoder_y[0]+0.05), (x_left, encoder_y[0]+0.03),
                           arrowstyle='->', mutation_scale=15, linewidth=2)
ax.add_patch(arrow_in)

# (b) Encoder Path
for i, y in enumerate(encoder_y):
    rect = Rectangle((x_left, y-0.05), 0.1, 0.1, color='lightblue', ec='k', lw=1.5)
    ax.add_patch(rect)
    ax.text(x_left+0.05, y, f'(b{i+1}) Encoder {i+1}\nConv Block', ha='center', va='center', fontsize=8)
    if i < len(encoder_y)-1:
        arrow = FancyArrowPatch((x_left+0.05, y-0.07), (x_left+0.05, encoder_y[i+1]+0.07),
                                arrowstyle='-|>', mutation_scale=15, linewidth=1.5)
        ax.add_patch(arrow)

# (c) Bottleneck
rect = Rectangle((x_left+0.35, bottleneck_y-0.05), 0.1, 0.1, color='salmon', ec='k', lw=1.5)
ax.add_patch(rect)
ax.text(x_left+0.4, bottleneck_y, '(c) Bottleneck\nDeep Features', ha='center', va='center', fontsize=8)

# (d) Decoder Path
for i, y in enumerate(decoder_y):
    rect = Rectangle((x_right, y-0.05), 0.1, 0.1, color='lightgreen', ec='k', lw=1.5)
    ax.add_patch(rect)
    ax.text(x_right+0.05, y, f'(d{i+1}) Decoder {i+1}\nUpsample', ha='center', va='center', fontsize=8)
    if i < len(decoder_y)-1:
        arrow = FancyArrowPatch((x_right+0.05, y-0.07), (x_right+0.05, decoder_y[i+1]+0.07),
                                arrowstyle='-|>', mutation_scale=15, linewidth=1.5)
        ax.add_patch(arrow)

arrow = FancyArrowPatch((x_left+0.45, bottleneck_y+0.05), (x_right+0.05, decoder_y[0]-0.07),
                        arrowstyle='-|>', mutation_scale=15, linewidth=1.5)
ax.add_patch(arrow)

# (e) LFF Modules & explicit skip connections
for i in range(len(encoder_y)):
    lff_x = (x_left+x_right)/2
    lff_y = (encoder_y[i]+decoder_y[i])/2
    lff = Circle((lff_x, lff_y), 0.035, color='purple', ec='k', lw=1.5)
    ax.add_patch(lff)
    ax.text(lff_x, lff_y, f'(e{i+1})\nLFF', ha='center', va='center', fontsize=7, color='w')

    conn1 = ConnectionPatch((x_left+0.1, encoder_y[i]), (lff_x-0.035, lff_y), "data", "data",
                            arrowstyle='-|>', color='gray', lw=1)
    conn2 = ConnectionPatch((lff_x+0.035, lff_y), (x_right, decoder_y[i]), "data", "data",
                            arrowstyle='-|>', color='gray', lw=1)
    ax.add_patch(conn1)
    ax.add_patch(conn2)

# (f) Deep Supervision with example heatmaps & aggregation arrows
for i, y in enumerate(decoder_y):
    # Connection from decoder block to deep supervision point
    arrow_ds = FancyArrowPatch((x_right+0.1, y), (x_right+0.2, y),
                               arrowstyle='-|>', color='red', mutation_scale=15, lw=1.5)
    ax.add_patch(arrow_ds)

    # Heatmap thumbnail placeholder
    thumb_x = x_right+0.25
    thumb_y = y
    rect_thumb = Rectangle((thumb_x, thumb_y-0.03), 0.06, 0.06, color='pink', ec='k', lw=1)
    ax.add_patch(rect_thumb)
    ax.text(thumb_x+0.03, thumb_y, f'(f{i+1}) Prediction', ha='center', va='center', fontsize=6)

    # Connection from heatmap to final output with aggregation arrow
    arrow_agg = FancyArrowPatch((thumb_x+0.06, thumb_y), (x_right+0.45, decoder_y[-1]-0.1),
                                arrowstyle='fancy', color='red', mutation_scale=12, lw=1.2, linestyle='--')
    ax.add_patch(arrow_agg)

# (g) Output Segmentation
ax.text(x_right+0.5, decoder_y[-1]-0.12, '(g) Output\nFinal Annotated Segmentation', fontsize=10, ha='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))

ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.title("Figure 3.X: The Proposed U-NetX Architecture\nDeep Supervision predictions with example heatmaps aggregated into final output.", fontsize=12)
plt.tight_layout()
plt.savefig("unetx_academic_with_heatmaps_and_aggregation.png")
plt.show()

print("Final U-NetX diagram with heatmaps & aggregation saved as 'unetx_academic_with_heatmaps_and_aggregation.png'")
