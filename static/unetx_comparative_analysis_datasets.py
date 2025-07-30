import matplotlib.pyplot as plt

metrics = ['DSC', 'IoU', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
internal = [0.9544, 0.9128, 0.9374, 0.9980, 0.9721, 0.9954]
external = [0.9232, 0.8573, 0.9276, 0.9692, 0.9188, 0.9727]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10,6))
plt.bar([i - width/2 for i in x], internal, width=width, label='Internal (CT Brain)')
plt.bar([i + width/2 for i in x], external, width=width, label='External (MRI Lung)')
plt.xticks(x, metrics)
plt.ylabel("Score")
plt.ylim(0.80, 1.01)
plt.title("UNetX Performance Comparison Across Datasets")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("results/unetx_comparative_analysis_datasets.png")
plt.show()
