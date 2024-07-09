import sys
import numpy as np

n=1024

a = [[0]*n for _ in range(n)]
a = np.array(a, dtype=np.int16)

print(sys.getsizeof(a))
print(sys.getsizeof(a.tolist()))