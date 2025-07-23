import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, ConnectionPatch

fig, ax = plt.subplots(figsize=(20, 12))

encoder_y = [0.9, 0.7, 0.5, 0.3]
bottleneck_y = 0.1
decoder_y = encoder_y[::-1]
x_left = 0.1
x_right = 0.7

# (a) Input Image
rect_input = Rectangle((x_left-0.15, encoder_y[0]-0.05), 0.1, 0.1, color='lightgray', ec='k', lw=1)
ax.add_patch(rect_input)
ax.text(x_left-0.1, encoder_y[0], '(a) Input\n$X \in \\mathbb{R}^{H\\times W}$', fontsize=9, ha='center')
arrow_in = FancyArrowPatch((x_left-0.05, encoder_y[0]), (x_left, encoder_y[0]),
                           arrowstyle='->', mutation_scale=15, linewidth=1)
ax.add_patch(arrow_in)

# (b) Encoder Path
for i, y in enumerate(encoder_y):
    rect = Rectangle((x_left, y-0.05), 0.1, 0.1, color='lightblue', ec='k', lw=1)
    ax.add_patch(rect)
    ax.text(x_left+0.05, y+0.07, f'(b{i+1}) Encoder {i+1}', ha='center', fontsize=7)
    ax.text(x_left+0.05, y-0.08, f'$E_{{{i}}}=f(E_{{{i-1}}})$', ha='center', fontsize=7)
    if i < len(encoder_y)-1:
        arrow = FancyArrowPatch((x_left+0.05, y-0.07), (x_left+0.05, encoder_y[i+1]+0.07),
                                arrowstyle='-|>', mutation_scale=15, linewidth=1, color='blue')
        ax.add_patch(arrow)

    # Parallel LFF arrow from encoder to LFF
    lff_x = (x_left + x_right) / 2
    lff_y = (y + decoder_y[i]) / 2
    lff = Circle((lff_x, lff_y), 0.03, color='purple', ec='k', lw=1)
    ax.add_patch(lff)
    ax.text(lff_x, lff_y+0.05, f'(e{i+1}) LFF', ha='center', fontsize=6, color='w')
    ax.text(lff_x, lff_y-0.05, f'$S_{{{i}}}=LFF(E_{{{i}}})$', ha='center', fontsize=6, color='w')

    conn_lff = FancyArrowPatch((x_left+0.1, y), (lff_x-0.03, lff_y),
                               arrowstyle='-|>', color='purple', lw=1, linestyle='--')
    conn_to_decoder = FancyArrowPatch((lff_x+0.03, lff_y), (x_right, decoder_y[i]),
                                      arrowstyle='-|>', color='purple', lw=1, linestyle='--')
    ax.add_patch(conn_lff)
    ax.add_patch(conn_to_decoder)

# (c) Bottleneck — now connected from last encoder block directly
rect = Rectangle((x_left+0.35, bottleneck_y-0.05), 0.1, 0.1, color='salmon', ec='k', lw=1)
ax.add_patch(rect)
ax.text(x_left+0.4, bottleneck_y+0.07, '(c) Bottleneck', ha='center', fontsize=7)
ax.text(x_left+0.4, bottleneck_y-0.08, '$B=f(E_n)$', ha='center', fontsize=7)

# U-shaped connection from last encoder to bottleneck
arrow_bottleneck = FancyArrowPatch((x_left+0.05, encoder_y[-1]-0.07), (x_left+0.4, bottleneck_y+0.05),
                                   connectionstyle="arc3,rad=-0.5", arrowstyle='-|>',
                                   mutation_scale=15, linewidth=1, color='blue')
ax.add_patch(arrow_bottleneck)

# New arrow: Bottleneck to first decoder
arrow_bottleneck_to_decoder = FancyArrowPatch((x_left+0.45, bottleneck_y+0.05), (x_right+0.05, decoder_y[0]-0.07),
                                              connectionstyle="arc3,rad=0.5", arrowstyle='-|>',
                                              mutation_scale=15, linewidth=1, color='green')
ax.add_patch(arrow_bottleneck_to_decoder)

# (d) Decoder Path
for i, y in enumerate(decoder_y):
    rect = Rectangle((x_right, y-0.05), 0.1, 0.1, color='lightgreen', ec='k', lw=1)
    ax.add_patch(rect)
    ax.text(x_right+0.05, y+0.07, f'(d{i+1}) Decoder {i+1}', ha='center', fontsize=7)
    ax.text(x_right+0.05, y-0.08, f'$D_{{{i}}}=f(D_{{{i-1}}},S)$', ha='center', fontsize=7)
    if i < len(decoder_y)-1:
        arrow = FancyArrowPatch((x_right+0.05, y-0.07), (x_right+0.05, decoder_y[i+1]+0.07),
                                arrowstyle='-|>', mutation_scale=15, linewidth=1, color='green')
        ax.add_patch(arrow)

# (f) Deep Supervision
for i, y in enumerate(decoder_y):
    arrow_ds = FancyArrowPatch((x_right+0.1, y), (x_right+0.2, y),
                               arrowstyle='-|>', color='red', mutation_scale=15, lw=1)
    ax.add_patch(arrow_ds)

    thumb_x = x_right+0.25
    thumb_y = y
    rect_thumb = Rectangle((thumb_x, thumb_y-0.025), 0.05, 0.05, color='pink', ec='k', lw=1)
    ax.add_patch(rect_thumb)
    ax.text(thumb_x+0.025, thumb_y+0.04, f'(f{i+1}) $\\hat{{Y}}_{{{i}}}$', ha='center', fontsize=6)

    arrow_agg = FancyArrowPatch((thumb_x+0.05, thumb_y), (x_right+0.5, decoder_y[-1]-0.1),
                                arrowstyle='fancy', color='red', mutation_scale=12, lw=1, linestyle='--')
    ax.add_patch(arrow_agg)

    ax.text((thumb_x+0.25+x_right)/2, (thumb_y+decoder_y[-1]-0.1)/2+0.02, '$w_{{{i}}}$', fontsize=6, color='darkred', ha='center')

# (g) Output Segmentation
rect_output = Rectangle((x_right+0.5, decoder_y[-1]-0.12), 0.1, 0.1, color='lightgray', ec='k', lw=1)
ax.add_patch(rect_output)
ax.text(x_right+0.55, decoder_y[-1]-0.1, '(g) Output\n$\\hat{Y}=\\sum w_i \\hat{Y}_i$', fontsize=9, ha='center')

ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1)
ax.axis('off')
plt.title("U-NetX Architecture with U-Shape and Bottleneck-to-Decoder Path", fontsize=12)
plt.tight_layout()
plt.savefig("unetx_academic_u_shape_b2d.png", dpi=300)
plt.show()

print("Diagram with U-shaped encoder→bottleneck→decoder path saved as 'unetx_academic_u_shape_b2d.png'")
