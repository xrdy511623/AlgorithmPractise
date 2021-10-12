package sort

/*
1.0 二分查找实现，递归版与非递归版,返回要查找元素在数组中的索引下标，若数组中不存在该元素，返回-1
*/

// 二分查找非递归版
func BinarySearch(array []int, item int) int {
	n := len(array)
	if n == 0 {
		return -1
	}
	start, stop := 0, n-1
	for start <= stop {
		mid := start + (stop - start) / 2
		if item == array[mid] {
			return mid
		} else if item > array[mid] {
			start = mid + 1
		} else {
			stop = mid - 1
		}
	}
	return -1
}

// 二分查找递归版
func BinarySearchUseRecursion(array []int, item int) int {
	n := len(array)
	if n == 0 {
		return -1
	}
	mid := n / 2
	if item == array[mid] {
		return mid
	} else if item > array[mid] {
		return BinarySearchUseRecursion(array[mid+1:], item)
	} else {
		return BinarySearchUseRecursion(array[:mid], item)
	}
}

/*
1.1 二分查找变形之一:在排序数组中查找元素的第一个和最后一个位置
定一个按照升序排列的整数数组nums，和一个目标值target。找出给定目标值在数组中的开始位置和结束位置。
如果数组中不存在目标值target，返回[-1, -1]。
*/

// 将问题分解为在有序数组中查找第一个等于目标值的元素以及查找最后一个等于目标值的元素
func SearchRange(nums []int, target int) []int {
	left := BinarySearchFirstEqualTarget(nums, target)
	right := BinarySearchLastEqualTarget(nums, target)
	return []int{left, right}
}


func BinarySearchFirstEqualTarget(array []int, item int)int{
	n := len(array)
	if n == 0 {
		return -1
	}
	low, high := 0, n-1
	for low <= high{
		mid := low + (high - low) / 2
		if array[mid] > item{
			high = mid - 1
		} else if array[mid] < item{
			low = mid + 1
		} else{
			if (mid == 0) || (mid != 0 && array[mid-1]<item){
				return mid
			} else{
				high = mid -1
			}
		}
	}
	return -1
}

func BinarySearchLastEqualTarget(array []int, item int)int{
	n := len(array)
	if n == 0 {
		return -1
	}
	low, high := 0, n-1
	for low <= high{
		mid := low + (high - low) / 2
		if array[mid] > item{
			high = mid - 1
		} else if array[mid] < item{
			low = mid + 1
		} else{
			if (mid == n-1) ||(mid != n-1 && array[mid+1] > item){
				return mid
			} else{
				low = mid + 1
			}
		}
	}
	return -1
}