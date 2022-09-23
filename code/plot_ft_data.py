import matplotlib.pyplot as plt
import numpy as np

ft = np.load('ft_data.npy')
labels = (
    'Jacket',
    'Button-Down',
    'Jersey',
    'Hoodie',
    'Sweater',
    'Tee',
    'Jeans',
    'Sweatpants',
    'Shorts',
    'Dress',
    'Skirt',
    'Top')
ax = plt.subplot()
plt.figure(figsize=(8,3))

# ax.set_yticks(list(range(len(labels))), labels)
# ax.scatter(ft[:, 4], ft[:, 0])
l = []
for i in range(len(labels)):
    l.append(np.array(ft[ft[:, 0] == i][:, 4]).astype(np.float32))
flierprops = dict(marker='.')# , markerfacecolor='green', markersize=12, markeredgecolor='none')
medianprops = dict(color='dodgerblue', linewidth=1.5)
meanprops = dict(color='orange')
ax.boxplot(l, labels=labels, vert=False, meanprops=meanprops, flierprops=flierprops, medianprops=medianprops, showmeans=True, meanline=True)
plt.show()