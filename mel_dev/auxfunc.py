"""
Auxiliary functions for Mel computation
"""

def binsearch(arr, target) -> int:
    """
    Finds the index of the item in the array closest to the value of the target
    :param arr: The array to search
    :param target: The target to search for
    """
    if arr.size == 0:
        raise IndexError("The array is empty.")
    elif arr[0] >= target:
        return 0
    elif arr[-1] <= target:
        return arr.shape[-1] - 1
    else:
        low_idx = 0
        mid_idx = arr.size // 2
        high_idx = arr.size - 1
        while True:
            if arr[mid_idx] == target:
                return mid_idx
            elif high_idx - low_idx < 2:
                if arr[high_idx] - target >= target - arr[low_idx]:
                    return low_idx
                else:
                    return high_idx
            elif target < arr[mid_idx]:
                high_idx = mid_idx
            else:
                low_idx = mid_idx
            mid_idx = low_idx + (high_idx - low_idx) // 2

def binsearch_le(arr, target) -> int:
    """
    Finds the index of the item in the array less than or equal to the target
    :param arr: The array to search
    :param target: The target to search for
    """
    if arr.size == 0:
        raise IndexError("The array is empty.")
    elif arr[0] > target:
        raise IndexError("The target is less than the first item in the array.")
    elif arr[0] == target:
        return 0
    elif arr[-1] <= target:
        return arr.shape[-1] - 1
    else:
        low_idx = 0
        mid_idx = arr.size // 2
        high_idx = arr.size - 1
        while True:
            if arr[mid_idx] == target:
                return mid_idx
            elif high_idx - low_idx < 2:
                return low_idx
            elif target < arr[mid_idx]:
                high_idx = mid_idx
            else:
                low_idx = mid_idx
            mid_idx = low_idx + (high_idx - low_idx) // 2
