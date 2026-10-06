"""Convert MNIST idx.gz files into the CSV layout WISE's LoadData_MNIST expects:
one row per image, label first, then 784 pixel values, no header."""
import gzip, struct, sys
import numpy as np

def read_idx(path):
    with gzip.open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        ndim = magic & 0xFF
        dims = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)

raw, out = sys.argv[1], sys.argv[2]
for split, prefix in [('train', 'train'), ('test', 't10k')]:
    img = read_idx(f'{raw}/{prefix}-images-idx3-ubyte.gz').reshape(-1, 784)
    lab = read_idx(f'{raw}/{prefix}-labels-idx1-ubyte.gz')
    data = np.concatenate([lab[:, None], img], axis=1).astype(np.int64)
    np.savetxt(f'{out}/mnist_{split}.csv', data, fmt='%d', delimiter=',')
    print(split, data.shape)
