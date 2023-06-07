import numpy as np
from matplotlib import pyplot as plt

s=np.load('samples.npz', allow_pickle=True)
# print(s.files)
# print(s['arr_0'])

all_results=s['arr_0'][()]
# print(all_results.items())

counter = all_results['counterfactual_sample'][0]
orig = all_results['original'][0]['image']
x = 0

o1 = orig[x][0][:][:]
fig = plt.figure(figsize=(12., 12.))
plt.imshow(o1, cmap='gray')
plt.axis("off")
plt.show()

l1 = all_results['original'][0]['y']
print(l1[x])

c1 = counter[x][0][:][:]
fig = plt.figure(figsize=(12., 12.))
plt.imshow(c1, cmap='gray')
plt.axis("off")
plt.show()

# sample_list = np.einsum("ibwh -> bihw", all_results)
# sample_list = sample_list[:16]
# grid = np.concatenate(np.concatenate(sample_list, axis=2), axis=0)
#
# fig = plt.figure(figsize=(12., 40.))
# plt.imshow(grid, cmap='gray')
# plt.axis("off")
# plt.show()

# diff = abs(c1) - abs(o1.numpy())
# print(diff)

# fig = plt.figure(figsize=(12., 12.))
# plt.imshow(diff, cmap='viridis')
# plt.axis("off")
# plt.show()