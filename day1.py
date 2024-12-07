from numba import jit, int32, guvectorize, vectorize, njit
import numpy as np

def load_files():
    fContent1 = []
    fContent2 = []
    with open("input1.txt") as f:

        for (lineIndex, line) in enumerate(f):  #loading the file into an np.array
            if bool(line) and line != "\n":
                position1, position2 = line.strip("\n").split()
                fContent1.append(int(position1))
                fContent2.append(int(position2))

    return(fContent1, fContent2)


# @guvectorize(['void(int32[:], int32[:])'], '(n)->()', writable_args=('np_array',), nopython=False)
@jit(['Tuple((int32, int32[:]))(int32[:])'])
def np_pop_min(np_array: list) -> int:
    '''
    Pop the "index" from np_array and return the value. 
    Default value for index is the last element.
    '''
    # read the value of the given array at the given index
    index: int = np.where(np_array == np.min(np_array))[0][0]
    value = np_array[index]
    # remove value from array
    np_array = np.delete(np_array, index)

    return value, np_array

@njit(['int32(int32[:], int32[:])'])
def task1(list1: list, list2: list) -> int:
    total: int = 0
    assert len(list1) == len(list2)
    list_length: int = len(list1)
    min1: int
    min2: int
    for _ in range(list_length):
        min1, list1 = np_pop_min(list1)
        min2, list2 = np_pop_min(list2)

        total += abs(min2 - min1)

    return total

@njit(['int32(int32[:], int32[:])'])
def task2(list1: list, list2: list) -> int:
    total: int = 0
    assert len(list1) == len(list2)
    list_length: int = len(list1)
    min1: int
    min2: int
    for _ in range(list_length):
        min1, list1 = np_pop_min(list1)
        repetitions = len(np.where(list2 == min1)[0])
        total += min1 * repetitions

    return total

def main() -> int:
    list1, list2 = load_files()
    list1_arr = np.array(list1)
    list2_arr = np.array(list2)
    out1 = task1(list1_arr, list2_arr)
    out2 = task2(list1_arr, list2_arr)

    return out1, out2

if __name__ == "__main__":
    print(main())
