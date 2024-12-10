from numba import jit, int32, guvectorize, vectorize, njit, boolean, types, uint8
from numba.typed import List
import numpy as np

def load_files():
    fContent1 = []

    with open("testinput2.txt") as f:

        for (lineIndex, line) in enumerate(f):  #loading the file into an np.array
            if bool(line) and line != "\n":
                report = line.replace("\n", "").split()
                fContent1.append(np.array(report, dtype=np.int32))

    return(fContent1)

@guvectorize(["void(int32[:,:], int64[:])"], "(m, n)->()")
def task1(report: np.array, output: np.array):

    def test1(entry1: np.array, entry2: np.array):
        return int(1 <= abs(np.int32(entry2) - np.int32(entry1)) <= 3)
    
    for entry_id, entry in enumerate(report):
        entry_comparator = sorted(list(set(list(entry))))
        test2 = int(list(entry) ==  entry_comparator or list(entry) == sorted(entry_comparator, reverse=True))
        output[entry_id] = np.prod(np.array(list(map(test1, entry[:-1], entry[1:])))) * test2


@jit(["(int32[:])(int32[:])"])
def test1(entry: np.array):
    res = np.zeros(1, dtype=np.int32)
    task1([entry], res)
    return res

@guvectorize(["void(int32[:], int64[:])"], "(m)->()")
def task2(report: np.array, output: np.array):

    def test2(entry1: np.array, entry2: np.array):
        return int(1 <= abs(np.int32(entry2) - np.int32(entry1)) <= 3) * np.sign(np.int32(entry2) - np.int32(entry1))
    
    def test3(test_array: np.array):
        condition1 = (test_array == 0).astype(np.int32)
        condition2 = (test_array == 1).astype(np.int32)
        condition3 = (test_array == -1).astype(np.int32)
        if np.sum(condition1) + np.sum(condition2) + np.sum(condition3) <= 1:
            return np.where((condition1 + condition2 + condition3) == 1)[0][0]



    test_array = np.array(list(map(test2, report[:-1], report[1:])))

    if not (index := int(test3(test_array))) is None:
        subtest_array1 = np.delete(test_array, index).astype(int32)
        subtest_array2 = np.delete(test_array, index + 1).astype(int32)

        out = test1(subtest_array1) or test1(subtest_array2)


    output[0] = out


def main():
    try:
        report = np.array(load_files(), dtype=np.int32)
    except:
        report = np.array(load_files(), dtype=object)
    out1 = np.sum([task1([entry]) for entry in report])
    out2 = np.sum([task2(entry) if task1([entry]) == 0 else 0 for entry in report])

    return out1, out2

if __name__ == "__main__":
    print(main())
