from numba import jit, int32, guvectorize, vectorize, njit, boolean, types, uint8, int64
from numba.typed import List
import numpy as np
import numba as nb

def load_files():
    fContent1 = []

    with open("testinput2.txt") as f:

        for (lineIndex, line) in enumerate(f):  #loading the file into an np.array
            if bool(line) and line != "\n":
                report = line.replace("\n", "").split()
                fContent1.append(np.array(report, dtype=np.int32))

    return(fContent1)

@njit("int64(int32[:])")
def task1(report: np.array):

    def test1(entry1: np.array, entry2: np.array):
        return 1 * (1 <= np.abs(entry2 - entry1) <= 3)

    entry_comparator = sorted(list(np.unique(report)))
    test2 = int(list(report) == entry_comparator or list(report) == sorted(entry_comparator, reverse=True))

    test_map = np.empty_like(report, dtype=np.int64)
    mapped_list = list(map(test1, report[:-1], report[1:]))

    for entry_id, _ in enumerate(mapped_list):
        test_map[entry_id] = mapped_list[entry_id]

    output = np.prod(test_map) * test2
    return output


@njit(["int32(int32[:])"])
def test1(entry: np.array):
    return task1(entry)

@njit(["int64(int32[:])"])
def task2(report: np.array):
    out = 0
    test_array = np.empty_like(report)
    subtest_array1 = np.empty_like(report)
    subtest_array2 = np.empty_like(report)

    def test2(entry1: np.int32, entry2: np.int32):
        return (1 <= np.absolute(entry2 - entry1) <= 3) * np.sign(np.int32(entry2) - np.int32(entry1))
    
    def test3(test_array: np.array):
        condition1 = (test_array == 0).astype(np.int32)
        condition2 = (test_array == 1).astype(np.int32)
        condition3 = (test_array == -1).astype(np.int32)
        if np.sum(condition1) + np.sum(condition2) + np.sum(condition3) <= 1:
            return np.where((condition1 + condition2 + condition3) == 1)[0][0]



    test_map = list(map(test2, report[:-1], report[1:]))

    for entry_id, _ in enumerate(test_map):
        test_array[entry_id] = test_map[entry_id]

    index = int(test3(test_array))

    if not (index is None):  # ERROR IN THIS FLOW CONTROL
        subtest_array1 = np.delete(test_array, index).astype(int32)
        subtest_array2 = np.delete(test_array, index + 1).astype(int32)

        out = test1(subtest_array1) or test1(subtest_array2)

    return out


def main():
    try:
        report = np.array(load_files(), dtype=np.int32)
    except:
        report = np.array(load_files(), dtype=object)

    out1 = np.sum([task1(entry) for entry in report])

    out2 = np.sum([0 if task1(entry) != 0 else task2(entry) for entry in report])

    return out1, out2

if __name__ == "__main__":
    print(main())
