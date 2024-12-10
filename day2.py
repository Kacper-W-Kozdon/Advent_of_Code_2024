from numba import jit, int32, guvectorize, vectorize, njit, boolean, types, uint8
import numpy as np

def load_files():
    fContent1 = []

    with open("input2.txt") as f:

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

def main():
    report = np.array(load_files(), dtype=object)
    out1 = np.sum([task1([entry]) for entry in report])

    return out1

if __name__ == "__main__":
    print(main())
