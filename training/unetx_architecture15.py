import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

fig, ax = plt.subplots(figsize=(20, 12))

encoder_y = [0.9, 0.7, 0.5, 0.3]
bottleneck_y = 0.1
decoder_y = encoder_y[::-1]
x_left = 0.1
x_right = 0.7

# (a) Input Image
rect_input = Rectangle((x_left-0.15, encoder_y[0]-0.05), 0.1, 0.1, color='lightgray', ec='k')
ax.add_patch(rect_input)
ax.text(x_left-0.1, encoder_y[0]+0.08, '(a) Input\n$X \\in \\mathbb{R}^{H\\times W}$', fontsize=10, ha='center', clip_on=False)
arrow_in = FancyArrowPatch((x_left-0.05, encoder_y[0]), (x_left, encoder_y[0]), arrowstyle='->', mutation_scale=15)
ax.add_patch(arrow_in)

# (b) Encoder Path
for i, y in enumerate(encoder_y):
    rect = Rectangle((x_left, y-0.05), 0.1, 0.1, color='lightblue', ec='k')
    ax.add_patch(rect)
    ax.text(x_left+0.05, y+0.09, f'(b{i+1}) Encoder {i+1}', ha='center', fontsize=8, clip_on=False)
    ax.text(x_left+0.05, y-0.09, f'$E_{{{i}}} = f(E_{{{i-1}}})$', ha='center', fontsize=8, clip_on=False)
    if i < len(encoder_y)-1:
        arrow = FancyArrowPatch((x_left+0.05, y-0.07), (x_left+0.05, encoder_y[i+1]+0.07),
                                arrowstyle='-|>', mutation_scale=15, color='blue')
        ax.add_patch(arrow)

    # LFF modules
    lff_x = (x_left + x_right) / 2
    lff_y = (y + decoder_y[i]) / 2
    lff = Circle((lff_x, lff_y), 0.03, color='purple', ec='k', zorder=5)
    ax.add_patch(lff)
    ax.text(lff_x, lff_y+0.05, f'(e{i+1}) LFF', ha='center', fontsize=7, color='w', clip_on=False)
    ax.text(lff_x, lff_y-0.05, f'$S_{{{i}}} = LFF(E_{{{i}}})$', ha='center', fontsize=7, color='w', clip_on=False)

    conn_lff = FancyArrowPatch((x_left+0.1, y), (lff_x-0.03, lff_y), arrowstyle='-|>', color='purple', lw=1, linestyle='--')
    conn_to_decoder = FancyArrowPatch((lff_x+0.03, lff_y), (x_right, decoder_y[i]), arrowstyle='-|>', color='purple', lw=1, linestyle='--')
    ax.add_patch(conn_lff)
    ax.add_patch(conn_to_decoder)

# (c) Bottleneck
rect = Rectangle((x_left+0.35, bottleneck_y-0.05), 0.1, 0.1, color='salmon', ec='k')
ax.add_patch(rect)
ax.text(x_left+0.4, bottleneck_y+0.07, '(c) Bottleneck', ha='center', fontsize=8, clip_on=False)
ax.text(x_left+0.4, bottleneck_y-0.08, '$B = f(E_n)$', ha='center', fontsize=8, clip_on=False)

arrow_bottleneck = FancyArrowPatch((x_left+0.05, encoder_y[-1]-0.07), (x_left+0.4, bottleneck_y+0.05),
                                   connectionstyle="arc3,rad=-0.5", arrowstyle='-|>', mutation_scale=15, color='blue')
ax.add_patch(arrow_bottleneck)

arrow_bottleneck_to_decoder = FancyArrowPatch((x_left+0.45, bottleneck_y+0.05), (x_right+0.05, decoder_y[0]-0.07),
                                              connectionstyle="arc3,rad=0.5", arrowstyle='-|>', mutation_scale=15, color='green')
ax.add_patch(arrow_bottleneck_to_decoder)

# (d) Decoder Path
for i, y in enumerate(decoder_y):
    rect = Rectangle((x_right, y-0.05), 0.1, 0.1, color='lightgreen', ec='k')
    ax.add_patch(rect)
    ax.text(x_right+0.05, y+0.09, f'(d{i+1}) Decoder {i+1}', ha='center', fontsize=8, clip_on=False)
    ax.text(x_right+0.05, y-0.09, f'$D_{{{i}}} = f(D_{{{i-1}}}, S)$', ha='center', fontsize=8, clip_on=False)
    if i < len(decoder_y)-1:
        arrow = FancyArrowPatch((x_right+0.05, y-0.07), (x_right+0.05, decoder_y[i+1]+0.07),
                                arrowstyle='-|>', mutation_scale=15, color='green')
        ax.add_patch(arrow)

# (f) Deep Supervision
for i, y in enumerate(decoder_y):
    arrow_ds = FancyArrowPatch((x_right+0.1, y), (x_right+0.2, y),
                               arrowstyle='-|>', color='red', mutation_scale=15, lw=1)
    ax.add_patch(arrow_ds)

    thumb_x = x_right+0.25
    thumb_y = y
    rect_thumb = Rectangle((thumb_x, thumb_y-0.025), 0.05, 0.05, color='pink', ec='k')
    ax.add_patch(rect_thumb)
    ax.text(thumb_x+0.025, thumb_y+0.045, f'(f{i+1}) $\\hat{{Y}}_{{{i}}}$', ha='center', fontsize=7, clip_on=False)

    arrow_agg = FancyArrowPatch((thumb_x+0.05, thumb_y), (x_right+0.7, decoder_y[-1]-0.15),
                                arrowstyle='fancy', color='red', mutation_scale=12, lw=1, linestyle='--')
    ax.add_patch(arrow_agg)

    ax.text((thumb_x+x_right+0.7)/2, (thumb_y+decoder_y[-1]-0.15)/2+0.02, f'$w_{{{i}}}$', fontsize=7, color='darkred', ha='center', clip_on=False)

# (g) Output Segmentation — moved much farther left and down a bit
output_x = x_right + 0.75
output_y = decoder_y[-1]-0.2
rect_output = Rectangle((output_x, output_y), 0.1, 0.1, color='lightgray', ec='k')
ax.add_patch(rect_output)
ax.text(output_x+0.05, output_y+0.13, '(g) Output\n$\\hat{{Y}} = \\sum w_i \\hat{{Y}}_i$', fontsize=10, ha='center', clip_on=False)

# Final adjustments: widen xlim further
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1)
ax.axis('off')
plt.title("UNetX Architecture: U-Shape with Bottleneck & Deep Supervision", fontsize=14)
plt.tight_layout()
plt.savefig("unetx_architecture_output_fixed.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Final diagram saved as 'unetx_architecture_output_fixed.png' — Output text fully visible.")
