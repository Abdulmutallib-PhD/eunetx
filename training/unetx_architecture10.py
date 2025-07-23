import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, ConnectionPatch

fig, ax = plt.subplots(figsize=(26, 16))

encoder_y = [0.9, 0.7, 0.5, 0.3]
bottleneck_y = 0.1
decoder_y = encoder_y[::-1]
x_left = 0.1
x_right = 0.7

# (a) Input Image
rect_input = Rectangle((x_left-0.15, encoder_y[0]-0.05), 0.1, 0.1, color='lightgray', ec='k', lw=1.5)
ax.add_patch(rect_input)
ax.text(x_left-0.1, encoder_y[0], '(a) Input\n$X \in \mathbb{R}^{H\\times W}$', fontsize=10, ha='center')
arrow_in = FancyArrowPatch((x_left-0.05, encoder_y[0]), (x_left, encoder_y[0]), arrowstyle='->', mutation_scale=20, linewidth=2)
ax.add_patch(arrow_in)

# (b) Encoder Path with math under boxes
for i, y in enumerate(encoder_y):
    rect = Rectangle((x_left, y-0.05), 0.1, 0.1, color='lightblue', ec='k', lw=1.5)
    ax.add_patch(rect)
    ax.text(x_left+0.05, y+0.08, f'(b{i+1}) Encoder {i+1}', ha='center', va='center', fontsize=8)
    ax.text(x_left+0.05, y-0.09, f'$E_{{{i}}}=f(E_{{{i-1}}})$', ha='center', fontsize=8)
    if i < len(encoder_y)-1:
        arrow = FancyArrowPatch((x_left+0.05, y-0.07), (x_left+0.05, encoder_y[i+1]+0.07),
                                arrowstyle='-|>', mutation_scale=20, linewidth=2, color='blue')
        ax.add_patch(arrow)

# (c) Bottleneck
rect = Rectangle((x_left+0.35, bottleneck_y-0.05), 0.1, 0.1, color='salmon', ec='k', lw=1.5)
ax.add_patch(rect)
ax.text(x_left+0.4, bottleneck_y+0.08, '(c) Bottleneck', ha='center', fontsize=8)
ax.text(x_left+0.4, bottleneck_y-0.09, '$B=f(E_n)$', ha='center', fontsize=8)

arrow_b2d = FancyArrowPatch((x_left+0.45, bottleneck_y+0.05), (x_right+0.05, decoder_y[0]-0.07),
                            arrowstyle='-|>', mutation_scale=20, linewidth=2, color='blue')
ax.add_patch(arrow_b2d)

# (d) Decoder Path
for i, y in enumerate(decoder_y):
    rect = Rectangle((x_right, y-0.05), 0.1, 0.1, color='lightgreen', ec='k', lw=1.5)
    ax.add_patch(rect)
    ax.text(x_right+0.05, y+0.08, f'(d{i+1}) Decoder {i+1}', ha='center', fontsize=8)
    ax.text(x_right+0.05, y-0.09, f'$D_{{{i}}}=f(D_{{{i-1}}},S)$', ha='center', fontsize=8)
    if i < len(decoder_y)-1:
        arrow = FancyArrowPatch((x_right+0.05, y-0.07), (x_right+0.05, decoder_y[i+1]+0.07),
                                arrowstyle='-|>', mutation_scale=20, linewidth=2, color='green')
        ax.add_patch(arrow)

# (e) LFF Modules
for i in range(len(encoder_y)):
    lff_x = (x_left+x_right)/2
    lff_y = (encoder_y[i]+decoder_y[i])/2
    lff = Circle((lff_x, lff_y), 0.035, color='purple', ec='k', lw=1.5)
    ax.add_patch(lff)
    ax.text(lff_x, lff_y+0.06, f'(e{i+1}) LFF', ha='center', fontsize=7, color='w')
    ax.text(lff_x, lff_y-0.06, f'$S_{{{i}}}=LFF(E_{{{i}}})$', ha='center', fontsize=7, color='w')

    conn1 = ConnectionPatch((x_left+0.1, encoder_y[i]), (lff_x-0.035, lff_y), "data", "data",
                            arrowstyle='-|>', color='gray', lw=1.5)
    conn2 = ConnectionPatch((lff_x+0.035, lff_y), (x_right, decoder_y[i]), "data", "data",
                            arrowstyle='-|>', color='gray', lw=1.5)
    ax.add_patch(conn1)
    ax.add_patch(conn2)

# (f) Deep Supervision
for i, y in enumerate(decoder_y):
    arrow_ds = FancyArrowPatch((x_right+0.1, y), (x_right+0.2, y),
                               arrowstyle='-|>', color='red', mutation_scale=20, lw=2)
    ax.add_patch(arrow_ds)

    thumb_x = x_right+0.25
    thumb_y = y
    rect_thumb = Rectangle((thumb_x, thumb_y-0.03), 0.06, 0.06, color='pink', ec='k', lw=1.5)
    ax.add_patch(rect_thumb)
    ax.text(thumb_x+0.03, thumb_y+0.05, f'(f{i+1}) $\\hat{{Y}}_{{{i}}}$', ha='center', fontsize=6)

    arrow_agg = FancyArrowPatch((thumb_x+0.06, thumb_y), (x_right+0.55, decoder_y[-1]-0.1),
                                arrowstyle='fancy', color='red', mutation_scale=15, lw=1.5, linestyle='--')
    ax.add_patch(arrow_agg)

    ax.text((thumb_x+0.3+x_right)/2, (thumb_y+decoder_y[-1]-0.1)/2+0.02, '$w_{{{i}}}$', fontsize=7, color='darkred', ha='center')

# (g) Output Segmentation
rect_output = Rectangle((x_right+0.55, decoder_y[-1]-0.15), 0.1, 0.1, color='lightgray', ec='k', lw=1.5)
ax.add_patch(rect_output)
ax.text(x_right+0.6, decoder_y[-1]-0.12, '(g) Output\n$\\hat{Y}=\\sum w_i \\hat{Y}_i$', fontsize=10, ha='center')

ax.set_xlim(0, 1.3)
ax.set_ylim(0, 1)
ax.axis('off')
plt.title("Figure 3.X: U-NetX Architecture with Mathematical Expressions Annotated at Steps", fontsize=12)
plt.tight_layout()
plt.savefig("unetx_academic_with_math_at_steps.png")
plt.show()

print("Diagram with mathematical expressions annotated at corresponding steps saved as 'unetx_academic_with_math_at_steps.png'")