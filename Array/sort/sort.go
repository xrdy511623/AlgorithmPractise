package sort

import "AlgorithmPractise/Utils"

/*
1.0 实现冒泡，快排和归并排序
*/

// BubbleSort 最坏时间复杂度O(n`2),最好时间复杂度O(n) 稳定算法
func BubbleSort(array []int) []int {
	n := len(array)
	for j := 0; j < n-1; j++ {
		for i := 0; i < n-1-j; i++ {
			if array[i] > array[i+1] {
				array[i], array[i+1] = array[i+1], array[i]
			}
		}
	}

	return array
}

/*
快速排序步骤:
1 从数列中挑出一个元素,称为基准
2 重新排序数列,所有元素比基准值小的摆放在基准前面,所有元素比基准值大的摆在基准后面(相同的数
可以放在任意一边).在这次分区结束后,该基准值就处于它在有序数列中的正确位置了.这个称为分区操作
3 递归地将小于基准值元素的子序列和大于基准值元素的子序列进行分区操作.
时间复杂度O(nlogn),最坏时间复杂度O(n`2)
*/

func QuickSort(array []int, start, stop int) {
	if start >= stop {
		return
	}
	mid := array[start]
	left := start
	right := stop
	for left < right {
		for left < right && array[right] >= mid {
			right -= 1
		}
		array[left] = array[right]
		for left < right && array[left] < mid {
			left += 1
		}
		array[right] = array[left]
	}

	array[right] = mid
	QuickSort(array, start, right-1)
	QuickSort(array, right+1, stop)

}

/*
分治法 (Divide and Conquer)
很多有用的算法结构上是递归的，为了解决一个特定问题，算法一次或者多次递归调用其自身以解决
若干子问题。
这些算法典型地遵循分治法的思想：将原问题分解为几个规模较小但是类似于原问题的子问题，
递归求解这些子问题，
然后再合并这些问题的解来建立原问题的解。

分治法在每层递归时有三个步骤：
- **分解**原问题为若干子问题，这些子问题是原问题的规模最小的实例
- **解决**这些子问题，递归地求解这些子问题。当子问题的规模足够小，就可以直接求解
- **合并**这些子问题的解成原问题的解
归并排序:最坏时间复杂度O(nlogn),最好时间复杂度O(nlogn) 稳定
*/

func MergeSort(array []int) []int {
	if len(array) <= 1 {
		return array
	}
	mid := len(array) / 2
	left := MergeSort(array[:mid])
	right := MergeSort(array[mid:])
	return Merge(left, right)
}

func Merge(left, right []int) (res []int) {
	l, r := 0, 0
	for l < len(left) && r < len(right) {
		if left[l] < right[r] {
			res = append(res, left[l])
			l += 1
		} else {
			res = append(res, right[r])
			r += 1
		}
	}

	leftPlus := left[l:]
	rightPlus := right[r:]
	if len(leftPlus) >= 1 {
		res = append(res, leftPlus...)
	}

	if len(rightPlus) >= 1 {
		res = append(res, rightPlus...)
	}
	return res
}


/*
1.1 二分查找实现，递归版与非递归版,返回要查找元素在数组中的索引下标，若数组中不存在该元素，返回-1
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

/*
1.3 寻找两个有序数组的中位数
给定两个大小分别为 m 和 n 的正序（从小到大）数组nums1和nums2。请你找出并返回这两个正序数组的中位数 。
算法的时间复杂度应该为O(log(m+n)) 。
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2
输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
*/

/*
解题思路:二分法
如果对时间复杂度的要求有log，通常都需要用到二分查找，这道题也可以通过二分查找实现。
根据中位数的定义，当 m+nm+n 是奇数时，中位数是两个有序数组中的第(m+n)/2 个元素，当m+n是偶数时，中位数是两个有序数组中的第(m+n)/2个
元素和第(m+n)/2+1 个元素的平均值。因此，这道题可以转化成寻找两个有序数组中的第k小的数，其中k为(m+n)/2或(m+n)/2+1。

假设两个有序数组分别是A和B。要找到第k个元素，我们可以比较A[k/2−1]和B[k/2−1]，其中/表示整数除法。由于A[k/2−1]和B[k/2−1]的前面分别有
A[0..k/2−2]和B[0..k/2−2]，即k/2−1 个元素，对于A[k/2−1]和B[k/2−1]中的较小值，最多只会有 (k/2-1)+(k/2-1) ≤ k−2 个元素比它小，那么
它就不能是第k小的数了。
因此我们可以归纳出三种情况：
如果A[k/2−1]<B[k/2−1]，则比A[k/2−1] 小的数最多只有A的前 k/2−1个数和B的前k/2−1个数，即比A[k/2−1] 小的数最多只有k-2个，因此A[k/2−1]
不可能是第k个数，A[0] 到A[k/2−1]也都不可能是第k个数，可以全部排除。
如果A[k/2−1]>B[k/2−1]，则可以排除B[0]到B[k/2−1]。
如果A[k/2−1]=B[k/2−1]，则可以归入第一种情况处理。

可以看到，比较A[k/2−1]和 B[k/2−1] 之后，可以排除k/2个不可能是第k小的数，查找范围缩小了一半。同时，我们将在排除后的新数组上继续进行二分
查找，并且根据我们排除数的个数，减少k的值，这是因为我们排除的数都不大于第k小的数。

有以下三种情况需要特殊处理：
如果A[k/2−1]或者B[k/2−1]越界，那么我们可以选取对应数组中的最后一个元素。在这种情况下，我们必须根据排除数的个数减少k的值，而不能直接将
k减去k/2。

如果一个数组为空，说明该数组中的所有元素都被排除，我们可以直接返回另一个数组中第k小的元素。
如果k=1，我们只要返回两个数组首元素的最小值即可。

用一个例子说明上述算法。假设两个有序数组如下：

A: 1 3 4 9
B: 1 2 3 4 5 6 7 8 9
两个有序数组的长度分别是4和9，长度之和是13，中位数是两个有序数组中的第7个元素，因此需要找到第k=7个元素。
比较两个有序数组中下标为k/2−1=2 的数，即A[2]和B[2]，如下面所示：

A: 1 3 4 9
       ↑
B: 1 2 3 4 5 6 7 8 9
       ↑
由于A[2]>B[2]，因此排除B[0]到B[2]，即数组B的下标偏移（offset）变为3，同时更新k的值：k=k-k/2=4。

下一步寻找，比较两个有序数组中下标为 k/2−1=1 的数，即A[1]和B[4]，如下面所示，其中方括号部分表示已经被排除的数。

A: 1 3 4 9
     ↑
B: [1 2 3] 4 5 6 7 8 9
             ↑
由于A[1]<B[4]，因此排除A[0]到A[1]，即数组A的下标偏移变为2，同时更新k的值：k=k−k/2=2。

下一步寻找，比较两个有序数组中下标为k/2−1=0 的数，即比较A[2]和B[3]，如下面所示，其中方括号部分表示已经被排除的数。

A: [1 3] 4 9
         ↑
B: [1 2 3] 4 5 6 7 8 9
           ↑
由于A[2]=B[3]，根据之前的规则，排除A中的元素，因此排除A[2]，即数组A的下标偏移变为3，同时更新k的值：k=k−k/2=1。
由于k的值变成1，因此比较两个有序数组中的未排除下标范围内的第一个数，其中较小的数即为第kk个数，由于A[3]>B[3]，因此第k个数是B[3]=4。

A: [1 3 4] 9
           ↑
B: [1 2 3] 4 5 6 7 8 9
           ↑
*/

// FindMedianSortedArrays 二分法解决可以将时间复杂度降为O(log(m+n))，空间复杂度O(1)
func FindMedianSortedArrays(nums1, nums2 []int) float64 {
	totalLength := len(nums1) + len(nums2)
	if totalLength%2 == 1 {
		midIndex := totalLength/2 + 1
		return float64(getKthElement(nums1, nums2, midIndex))
	} else {
		midIndex1, midIndex2 := totalLength/2, totalLength/2+1
		return float64(getKthElement(nums1, nums2, midIndex1)+getKthElement(nums1, nums2, midIndex2)) / 2.0
	}
}

func getKthElement(nums1, nums2 []int, k int) int {
	index1, index2 := 0, 0
	m, n := len(nums1), len(nums2)
	for {
		// 特殊情形
		if index1 == m {
			return nums2[index2+k-1]
		}
		if index2 == n {
			return nums1[index1+k-1]
		}
		if k == 1 {
			return Utils.Min(nums1[index1], nums2[index2])
		}
		// 正常情况
		newIndex1 := Utils.Min(index1+k/2-1, m-1)
		newIndex2 := Utils.Min(index2+k/2-1, n-1)
		pivot1, pivot2 := nums1[newIndex1], nums2[newIndex2]
		if pivot1 <= pivot2 {
			k -= newIndex1 - index1 + 1
			index1 = newIndex1 + 1
		} else {
			k -= newIndex2 - index2 + 1
			index2 = newIndex2 + 1
		}
	}
}
