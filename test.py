import numpy as np

from numba import guvectorize, float64, int32, vectorize, jit

@guvectorize(['void(float64[:], intp, float64[:])'],
             '(n),()->(n)', writable_args=('a',))
def move_mean(a: list, window_width: int, out: list):
    asum = 0.0
    count = 0
    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / count
    np.delete(a, obj=-1)

arr = np.arange(20, dtype=np.float64).reshape(2, 10)
# print(arr)
# print(move_mean(arr, 3))
test_out = move_mean(arr, 3)
# print(test_out)

@guvectorize(['void(float64[:, :], int32[:])'], '(m,n)->()')
def test_function(input, output):
    output[0] = input[1, 2]


arr2 = np.arange(20, dtype=np.float64)

@jit
def test_function2(arr: list):
    for elem in arr:
        print(elem)
    return arr

def report(report: np.array, output: np.array):
    def test(entry1: np.array, entry2: np.array):
        return abs(entry2 - entry1) <= 3

    for entry_id, entry in enumerate(report):
        print(np.prod(entry))
        output.append([np.prod(np.array(list(map(test, entry[:-1], entry[1:]))))])
    
    return output

reportarr = np.array([np.array([1, 4, 8])])

print("---TEST---")
result = report(reportarr, [])
print(result)