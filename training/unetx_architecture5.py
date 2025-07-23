import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

fig, ax = plt.subplots(figsize=(18, 9))

encoder_y = [0.9, 0.7, 0.5, 0.3]
bottleneck_y = 0.1
decoder_y = encoder_y[::-1]
x_left = 0.1
x_right = 0.7

# (a) Input Image
ax.text(x_left-0.05, encoder_y[0]+0.12, '(a) Input Image\n[256x256]', fontsize=10, ha='center',
        bbox=dict(boxstyle="round", facecolor='lightgray'))
arrow_in = FancyArrowPatch((x_left-0.02, encoder_y[0]+0.1), (x_left, encoder_y[0]+0.05), arrowstyle='->', mutation_scale=10)
ax.add_patch(arrow_in)

# (b) Encoder Path
for i, y in enumerate(encoder_y):
    rect = Rectangle((x_left, y-0.05), 0.1, 0.1, color='lightblue', ec='k')
    ax.add_patch(rect)
    ax.text(x_left+0.05, y, f'(b{i+1}) Encoder {i+1}\n(Conv Block)', ha='center', va='center', fontsize=8)
    if i < len(encoder_y)-1:
        arrow = FancyArrowPatch((x_left+0.05, y-0.05), (x_left+0.05, encoder_y[i+1]+0.05),
                                arrowstyle='->', mutation_scale=10)
        ax.add_patch(arrow)

# (c) Bottleneck
rect = Rectangle((x_left+0.3, bottleneck_y-0.05), 0.1, 0.1, color='salmon', ec='k')
ax.add_patch(rect)
ax.text(x_left+0.35, bottleneck_y, '(c) Bottleneck\n(Deep Features)', ha='center', va='center', fontsize=8)

# (d) Decoder Path
for i, y in enumerate(decoder_y):
    rect = Rectangle((x_right, y-0.05), 0.1, 0.1, color='lightgreen', ec='k')
    ax.add_patch(rect)
    ax.text(x_right+0.05, y, f'(d{i+1}) Decoder {i+1}\n(Upsample)', ha='center', va='center', fontsize=8)
    if i < len(decoder_y)-1:
        arrow = FancyArrowPatch((x_right+0.05, y-0.05), (x_right+0.05, decoder_y[i+1]+0.05),
                                arrowstyle='->', mutation_scale=10)
        ax.add_patch(arrow)

arrow = FancyArrowPatch((x_left+0.4, bottleneck_y+0.05), (x_right+0.05, decoder_y[0]-0.05),
                        arrowstyle='->', mutation_scale=10)
ax.add_patch(arrow)

# (e) LFF Modules with full-scale skip connections
for i in range(len(encoder_y)):
    lff_x = (x_left+x_right)/2
    lff_y = (encoder_y[i]+decoder_y[i])/2
    lff = Circle((lff_x, lff_y), 0.03, color='purple', ec='k')
    ax.add_patch(lff)
    ax.text(lff_x, lff_y, f'(e{i+1}) LFF\n{i+1}', ha='center', va='center', fontsize=6, color='w')

    arrow1 = FancyArrowPatch((x_left+0.2, encoder_y[i]), (lff_x-0.03, lff_y),
                             arrowstyle='->', color='gray', mutation_scale=8)
    arrow2 = FancyArrowPatch((lff_x+0.03, lff_y), (x_right, decoder_y[i]),
                             arrowstyle='->', color='gray', mutation_scale=8)
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)

# (f) Deep Supervision
for i, y in enumerate(decoder_y):
    arrow = FancyArrowPatch((x_right+0.1, y), (x_right+0.2, y),
                            arrowstyle='->', color='red', mutation_scale=10)
    ax.add_patch(arrow)
    ax.text(x_right+0.23, y, f'(f{i+1}) Deep Supervision', va='center', fontsize=7, color='red')

# (g) Output Segmentation
arrow_out = FancyArrowPatch((x_right+0.2, decoder_y[-1]-0.05), (x_right+0.3, decoder_y[-1]-0.1), arrowstyle='->', mutation_scale=10)
ax.add_patch(arrow_out)
ax.text(x_right+0.35, decoder_y[-1]-0.12, '(g) Output\nAnnotated Segmentation', fontsize=10, ha='center',
        bbox=dict(boxstyle="round", facecolor='lightgray'))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.title("Figure 3.X: The Proposed U-NetX Architecture\n(a) Input [256x256] → (b) Encoder → (c) Bottleneck → (d) Decoder → (e) LFF & Full-Scale Skip → (f) Deep Supervision → (g) Annotated Output", fontsize=12)
plt.tight_layout()
plt.savefig("unetx_academic_detailed_architecture.png")
plt.show()

print("Academic U-NetX architecture diagram saved as 'unetx_academic_detailed_architecture.png'")
