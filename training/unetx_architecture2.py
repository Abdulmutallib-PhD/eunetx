import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

fig, ax = plt.subplots(figsize=(12, 8))

# U-shape coordinates
encoder_y = [0.9, 0.7, 0.5, 0.3]
bottleneck_y = 0.1
decoder_y = encoder_y[::-1]
x_left = 0.1
x_right = 0.7

# Plot Encoder blocks
for i, y in enumerate(encoder_y):
    rect = Rectangle((x_left, y-0.05), 0.1, 0.1, color='lightblue', ec='k')
    ax.add_patch(rect)
    ax.text(x_left+0.05, y, f'Encoder {i+1}', ha='center', va='center')
    if i < len(encoder_y)-1:
        arrow = FancyArrowPatch((x_left+0.05, y-0.05), (x_left+0.05, encoder_y[i+1]+0.05),
                                arrowstyle='->', mutation_scale=10)
        ax.add_patch(arrow)

# Bottleneck
rect = Rectangle((x_left+0.3, bottleneck_y-0.05), 0.1, 0.1, color='salmon', ec='k')
ax.add_patch(rect)
ax.text(x_left+0.35, bottleneck_y, 'Bottleneck', ha='center', va='center')

# Plot Decoder blocks
for i, y in enumerate(decoder_y):
    rect = Rectangle((x_right, y-0.05), 0.1, 0.1, color='lightgreen', ec='k')
    ax.add_patch(rect)
    ax.text(x_right+0.05, y, f'Decoder {i+1}', ha='center', va='center')
    if i < len(decoder_y)-1:
        arrow = FancyArrowPatch((x_right+0.05, y-0.05), (x_right+0.05, decoder_y[i+1]+0.05),
                                arrowstyle='->', mutation_scale=10)
        ax.add_patch(arrow)

# Connect Bottleneck to first decoder
arrow = FancyArrowPatch((x_left+0.4, bottleneck_y+0.05), (x_right+0.05, decoder_y[0]-0.05),
                        arrowstyle='->', mutation_scale=10)
ax.add_patch(arrow)

# Skip connections
for i in range(len(encoder_y)):
    arrow = FancyArrowPatch((x_left+0.2, encoder_y[i]), (x_right, decoder_y[i]),
                            connectionstyle="arc3,rad=-0.5", arrowstyle='->', color='gray')
    ax.add_patch(arrow)
    ax.text((x_left+0.2+x_right)/2, (encoder_y[i]+decoder_y[i])/2+0.03, f'LFF {i+1}', ha='center', fontsize=8)

# Input & Output
ax.text(x_left, encoder_y[0]+0.1, 'Input', fontsize=12, ha='center')
ax.text(x_right+0.2, decoder_y[-1]-0.1, 'Output', fontsize=12, ha='center')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.title("UNetX U-shaped Architecture with Interconnections", fontsize=14)
plt.tight_layout()
plt.savefig("unetx_u_shape_diagram.png")
plt.show()

print("UNetX U-shaped architecture diagram saved as 'unetx_u_shape_diagram.png'")
