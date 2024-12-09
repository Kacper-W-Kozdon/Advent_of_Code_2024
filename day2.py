from numba import jit, int32, guvectorize, vectorize, njit, boolean
import numpy as np

def load_files():
    fContent1 = []
    fContent2 = []
    with open("testinput2.txt") as f:

        for (lineIndex, line) in enumerate(f):  #loading the file into an np.array
            if bool(line) and line != "\n":
                report = line.strip("\n").split()
                report = list(map(int, report))
                fContent1.append(report)

    return(fContent1)

@guvectorize(["void(int32[:, :], int64[:])"], "(m,n)->(n)")
def task1(report: np.array, output: np.array):
    def test(entry1: np.array, entry2: np.array):
        return abs(entry2 - entry1) <= 3

    for entry_id, entry in enumerate(report):
        output[entry_id] = np.prod(np.array(list(map(test, entry[:-1], entry[1:]))))

if __name__ == "__main__":
    print(load_files())