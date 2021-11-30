import numpy as np

AB = np.load('matrix.npy')
BC = np.load('BC.npy')
bias = np.load('bias.npy')
scale = np.load('scale.npy')
ref = np.load('ref.npy')
result = np.load('cpu_output.npy')
AC = np.dot(AB, BC) + np.expand_dims(bias, 1)
AC = AC.astype(np.float32) * np.expand_dims(scale, 1)
print('AC')
print(AC)
print('\nref')
print(ref)
print('\nresult', '\t#result == ref:', np.array_equal(ref, result))
print(result)

