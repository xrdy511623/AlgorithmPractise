package sort

/*
1.0 二分查找实现，递归版与非递归版,返回要查找元素在数组中的索引下标，若数组中不存在该元素，返回-1
*/

// BinarySearch 二分查找非递归版
func BinarySearch(array []int, target int) int {
	n := len(array)
	if n == 0 {
		return -1
	}
	start, stop := 0, n-1
	for start <= stop {
		mid := start + (stop - start) / 2
		if target == array[mid] {
			return mid
		} else if target > array[mid] {
			start = mid + 1
		} else {
			stop = mid - 1
		}
	}
	return -1
}

// BinarySearchUseRecursion 二分查找递归版
func BinarySearchUseRecursion(array []int, target int) int {
	n := len(array)
	if n == 0 {
		return -1
	}
	mid := n / 2
	if target == array[mid] {
		return mid
	} else if target > array[mid] {
		return BinarySearchUseRecursion(array[mid+1:], target)
	} else {
		return BinarySearchUseRecursion(array[:mid], target)
	}
}

/*
1.1 二分查找变形之一:在排序数组中查找元素的第一个和最后一个位置
定一个按照升序排列的整数数组nums，和一个目标值target。找出给定目标值在数组中的开始位置和结束位置。
如果数组中不存在目标值target，返回[-1, -1]。
*/

// SearchRange 将问题分解为在有序数组中查找第一个等于目标值的元素以及查找最后一个等于目标值的元素
func SearchRange(nums []int, target int) []int {
	left := BinarySearchFirstEqualTarget(nums, target)
	right := BinarySearchLastEqualTarget(nums, target)
	return []int{left, right}
}


func BinarySearchFirstEqualTarget(array []int, target int)int{
	n := len(array)
	if n == 0 {
		return -1
	}
	low, high := 0, n-1
	for low <= high{
		mid := low + (high - low) / 2
		if array[mid] > target{
			high = mid - 1
		} else if array[mid] < target{
			low = mid + 1
		} else{
			// 此时array[mid] = target, 因为是有序数组，如果mid=0说明就是第一个元素就是数组中第一个等于target的元素
			// 或者mid!=0但是它的前一个元素小于target,也证明它是第一个等于target的元素，因为之前的元素都小于target
			if (mid == 0) || (mid != 0 && array[mid-1]<target){
				return mid
			} else{
				// 否则证明mid之前还有等于target的元素，所以我们应该在[low,mid-1]区间寻找第一个等于target的元素
				high = mid -1
			}
		}
	}
	return -1
}

func BinarySearchLastEqualTarget(array []int, target int)int{
	n := len(array)
	if n == 0 {
		return -1
	}
	low, high := 0, n-1
	for low <= high{
		mid := low + (high - low) / 2
		if array[mid] > target{
			high = mid - 1
		} else if array[mid] < target{
			low = mid + 1
		} else{
			// 此时array[mid] = target, 因为是有序数组，如果mid=n-1说明数组末尾元素就是最后一个等于target的元素
			// 或者mid!=n-1但是它的前一个元素大于target,也证明它是最后一个等于target的元素，因为它之后的元素都大于target
			if (mid == n-1) ||(mid != n-1 && array[mid+1] > target){
				return mid
			} else{
				// 否则证明mid之后还有等于target的元素，所以我们应该在[mid+1， high]区间寻找最后一个等于target的元素
				low = mid + 1
			}
		}
	}
	return -1
}

/*
1.2 在排序数组中统计一个数在数组中出现的次数
示例 1:
输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
输入: nums = [5,7,7,8,8,10], target = 6
输出: 0

思路: 在排序数组中找出等于目标值target的起始位置index，如果index为-1,证明数组中无此元素，返回0，
否则从数组index位置开始向后遍历数组元素，只要数组元素等于target，则使用map将其出现次数累加1,如果遇到不等于
target的元素，说明后面的元素都大于target，此时退出循环，最后返回map中target的对应值即可
 */

func Search(nums []int, target int) int {
	index := BinarySearchFirstEqualTarget(nums, target)
	if index == -1 {
		return 0
	}
	m := make(map[int]int, 0)
	for _, num := range nums[index:]{
		if num == target{
			m[target]++
		} else {
			break
		}
	}
	return m[target]
}
