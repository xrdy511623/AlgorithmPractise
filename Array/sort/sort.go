package sort

import (
	"AlgorithmPractise/Array/feature"
	"AlgorithmPractise/Utils"
	"sort"
)

/*
1.0 实现冒泡，插入排序，快排和归并排序
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

func InsertSort(array []int) []int {
	n := len(array)
	if n == 0 {
		return array
	}
	for i := 1; i < n; i++ {
		for j := i; j >= 0; j-- {
			if array[j] < array[j-1] {
				array[j], array[j-1] = array[j-1], array[j]
			} else {
				break
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
- 分解原问题为若干子问题，这些子问题是原问题的规模最小的实例
- 解决这些子问题，递归地求解这些子问题。当子问题的规模足够小，就可以直接求解
- 合并这些子问题的解成原问题的解
归并排序:最坏时间复杂度O(nlogn),最好时间复杂度O(nlogn) 稳定
*/

func MergeSort(array []int) []int {
	// 递归终止条件
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
	res = append(res, left[l:]...)
	res = append(res, right[r:]...)
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
		mid := start + (stop-start)/2
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

// BinarySearchUseRecursion 二分查找递归版, 判断目标值在有序数组中是否存在
func BinarySearchUseRecursion(array []int, target int) bool {
	n := len(array)
	if n == 0 {
		return false
	}
	mid := n / 2
	if target == array[mid] {
		return true
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

func BinarySearchFirstEqualTarget(array []int, target int) int {
	n := len(array)
	if n == 0 {
		return -1
	}
	low, high := 0, n-1
	for low <= high {
		mid := low + (high-low)/2
		if array[mid] > target {
			high = mid - 1
		} else if array[mid] < target {
			low = mid + 1
		} else {
			// 此时array[mid] = target, 因为是有序数组，如果mid=0说明就是第一个元素就是数组中第一个等于target的元素
			// 或者mid!=0但是它的前一个元素小于target,也证明它是第一个等于target的元素，因为之前的元素都小于target
			if (mid == 0) || (mid != 0 && array[mid-1] < target) {
				return mid
			} else {
				// 否则证明mid之前还有等于target的元素，所以我们应该在[low,mid-1]区间寻找第一个等于target的元素
				high = mid - 1
			}
		}
	}
	return -1
}

func BinarySearchLastEqualTarget(array []int, target int) int {
	n := len(array)
	if n == 0 {
		return -1
	}
	low, high := 0, n-1
	for low <= high {
		mid := low + (high-low)/2
		if array[mid] > target {
			high = mid - 1
		} else if array[mid] < target {
			low = mid + 1
		} else {
			// 此时array[mid] = target, 因为是有序数组，如果mid=n-1说明数组末尾元素就是最后一个等于target的元素
			// 或者mid!=n-1但是它的前一个元素大于target,也证明它是最后一个等于target的元素，因为它之后的元素都大于target
			if (mid == n-1) || (mid != n-1 && array[mid+1] > target) {
				return mid
			} else {
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
	for _, num := range nums[index:] {
		if num == target {
			m[target]++
		} else {
			break
		}
	}
	return m[target]
}

/*
1.3 搜索旋转排序数组
整数数组nums按升序排列，数组中的值互不相同 。
在传递给函数之前，nums在预先未知的某个下标k（0 <= k < nums.length）上进行了旋转，使数组变为[nums[k], nums[k+1], ...,
nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标从0开始计数）。例如，[0,1,2,4,5,6,7]在下标3处经旋转后可能变为
[4,5,6,7,0,1,2] 。

给你旋转后的数组nums和一个整数target ，如果nums中存在这个目标值target ，则返回它的下标，否则返回-1。

示例 1：
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4

示例2：
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1

示例3：
输入：nums = [1], target = 0
输出：-1
*/

/*
解决思路:虽然进行旋转后的数组整体上不再是有序的了，但是从中间某个位置将数组分成左右两部分时，一定有一部分是有序的，所以我们仍然可以
通过二分查找来解决此问题，只不过需要根据有序的部分确定应该如何改变二分查找的上下限，因为我们可以根据有序的部分判断出target是否在
这个部分。
如果 [l, mid - 1]是有序数组，且target的大小满足 [nums[l],nums[mid]]，则我们应该将搜索范围缩小至[l, mid-1]，否则在
[mid+1, r]中寻找。
如果[mid, r] 是有序数组，且target的大小满足[nums[mid+1],nums[r]]，则我们应该将搜索范围缩小至[mid+1, r]，否则在
[l, mid-1]中寻找。
时间复杂度O(logN),空间复杂度O(1)
*/

func RevolveArraySearch(nums []int, target int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	l, r := 0, n-1
	for l <= r {
		mid := (l + r) / 2
		if target == nums[mid] {
			return mid
		}
		if nums[0] <= nums[mid] {
			if nums[0] <= target && target < nums[mid] {
				r = mid - 1
			} else {
				l = mid + 1
			}
		} else {
			if nums[mid] < target && target <= nums[n-1] {
				l = mid + 1
			} else {
				r = mid - 1
			}
		}
	}
	return -1
}

/*
1.4 寻找旋转排序数组中的最小值
已知一个长度为 n 的数组，预先按照升序排列，经由1到n次旋转后，得到输入数组。例如，原数组nums = [0,1,2,4,5,6,7]
在变化后可能得到：
若旋转4次，则可以得到 [4,5,6,7,0,1,2]
若旋转7次，则可以得到 [0,1,2,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]]旋转一次的结果为数组[a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
给你一个元素值互不相同的数组nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的
最小元素。

示例 1：
输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转3次得到输入数组。

示例2：
输入：nums = [4,5,6,7,0,1,2]
输出：0
解释：原数组为 [0,1,2,4,5,6,7] ，旋转4次得到输入数组。

示例3：
输入：nums = [11,13,15,17]
输出：11
解释：原数组为 [11,13,15,17] ，旋转4次得到输入数组。

提示：
n == nums.length
1 <= n <= 5000
-5000 <= nums[i] <= 5000
nums 中的所有整数互不相同
nums 原来是一个升序排序的数组，并进行了1至n次旋转
*/

/*
思路:若原升序数组长度为n,则根据旋转次数会有以下三种情况：
若只旋转一次，则最小值为数组第二个元素；
譬如7，0，1，2，4，5，6
若旋转次数为n的整数倍，那么数组会恢复原样，最小值自然就是数组第一个元素。
否则旋转后的数组一定是前半部分升序，后半部分降序，譬如
4，5，6，7，0，1，2
那么最小值一定是升序部分末尾元素的下一位。
所以我们算法可以设计为从下标0开始，寻找升序部分末尾元素下标index, 如果index为0，那么它是第一种情况，只旋转一次，
导致最大值出现在数组起始位置，所以我们返回nums[1]，也就是nums[index+1];
如果index==n-1,说明它一定是第二种情况，旋转次数为n的整数倍，数组恢复原样，导致index移动到了数组末尾，所以返回
nums[0]。
否则，说明它一定是第三种情况，最小值一定是升序部分末尾元素的下一位，所以我们返回nums[i+1]
特别的，如果数组长度为1，那么我们直接返回nums[0]
*/

// FindMin 时间复杂度为O(N)，空间复杂度O(1)
func FindMin(nums []int) int {
	n := len(nums)
	// 特殊情况处理
	if n == 1 {
		return nums[0]
	}
	i := 0
	// 寻找数组升序部分的末尾元素
	for i < n-1 && nums[i] < nums[i+1] {
		i++
	}
	// 如果i移动到数组末尾位置，证明是旋转次数为n的整数倍，数组恢复原样
	// 所以返回nums[0]
	if i == n-1 {
		return nums[0]
	}
	// 否则，其他两种情况，我们返回nums[i+1]
	return nums[i+1]
}

/*
思路:二分查找
一个不包含重复元素的升序数组在经过旋转之后，可以得到下面可视化的折线图：
旋转排序数组.png
其中横轴表示数组元素的下标，纵轴表示数组元素的值。图中标出了最小值的位置，是我们需要查找的目标。
我们考虑数组中的最后一个元素x：在最小值右侧的元素（不包括最后一个元素本身），它们的值一定都严格小于x；而在最小值
左侧的元素，它们的值一定都严格大于x。因此，我们可以根据这一条性质，通过二分查找的方法找出最小值。

在二分查找的每一步中，左边界为low，右边界为high，区间的中点为pivot，最小值就在该区间内。我们将中轴元素nums[pivot]
与右边界元素nums[high]进行比较，可能会有以下的三种情况：

第一种情况是nums[pivot]<nums[high]。如下图所示，这说明nums[pivot]是最小值右侧的元素，因此我们可以忽略二分查找
区间的右半部分, 我们需要在pivot左侧区间查找，也就是[low, pivot]区间查找。
第一种情况.png

第二种情况是nums[pivot]>nums[high]。如下图所示，这说明nums[pivot] 是最小值左侧的元素，因此我们可以忽略二分查找
区间的左半部分，我们需要在pivot右侧区间查找，也就是[pivot+1, high]区间查找。
第二种情况.png

由于数组不包含重复元素，并且只要当前的区间长度不为 1，pivot就不会与high重合；而如果当前的区间长度为1，这说明我们
已经可以结束二分查找了。因此不会存在nums[pivot]=nums[high] 的情况。
当二分查找结束时，我们就得到了最小值所在的位置。
*/

// FindMinSimple 时间复杂度为O(logN)，空间复杂度O(1)
func FindMinSimple(nums []int) int {
	low, high := 0, len(nums)-1
	for low < high {
		mid := (low + high) / 2
		if nums[mid] < nums[high] {
			high = mid
		} else {
			low = mid + 1
		}
	}
	return nums[low]
}

/*
1.5 数组中的第K个最大元素
给定整数数组nums和整数k，请返回数组中第k个最大的元素。
请注意，你需要找的是数组排序后的第k个最大的元素，而不是第k个不同的元素。

示例 1:
输入: [3,2,1,5,6,4] 和 k = 2
输出: 5

示例2:
输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
*/

// FindKthLargest 用最大堆排序解决
func FindKthLargest(nums []int, k int) int {
	n := len(nums)
	mh := Utils.NewMaxHeap(n)
	for _, num := range nums {
		mh.Add(num)
	}
	var sortedArray []int
	for i := 0; i < n; i++ {
		value, _ := mh.Extract()
		sortedArray = append(sortedArray, value)
	}
	return sortedArray[k-1]
}

/*
思路:快排可以解决问题，但是它需要确定数组中所有元素的正确位置，对于本题而言，我们只需要确定第k大元素的位置pos,
我们只需要确保pos左边的元素都比它小，pos右边的元素都比它大即可，不需要关心其左边和右边的集合是否有序，所以，我们
需要对快排进行改进，将目标值的位置pos与分区函数Partition求得的位置index进行比对，如果两值相等，说明index对应的元素即为
所求值，如果index<pos，则递归的在[index+1, right]范围求解；否则则在[left, index-1]范围求解，如此便可大幅缩小求解范围。
*/

func FindKthLargestElement(nums []int, k int) int {
	TopkSplit(nums, len(nums)-k, 0, len(nums)-1)
	return nums[len(nums)-k]
}

func Partition(array []int, start, stop int) int {
	if start >= stop {
		return -1
	}
	pivot := array[start]
	left, right := start, stop

	for left < right {
		for left < right && array[right] >= pivot {
			right -= 1
		}
		array[left] = array[right]
		for left < right && array[left] < pivot {
			left += 1
		}
		array[right] = array[left]
	}
	// 循环结束，left与right相等
	// 确定基准元素pivot在数组中的位置
	array[right] = pivot
	return right
}

// TopkSplit topK切分
func TopkSplit(nums []int, k, left, right int) {
	if left < right {
		index := Partition(nums, left, right)
		if index == k {
			return
		} else if index < k {
			TopkSplit(nums, k, index+1, right)
		} else {
			TopkSplit(nums, k, left, index-1)
		}
	}
}

// 以下是利用快排解决topK类问题的总结

/*
1.5.1 获得前k小的数
*/

func TopkSmallest(nums []int, k int) []int {
	TopkSplit(nums, k, 0, len(nums)-1)
	return nums[:k]
}

/*
1.5.2 获得前k大的数
*/

func TopkLargest(nums []int, k int) []int {
	TopkSplit(nums, len(nums)-k, 0, len(nums)-1)
	return nums[len(nums)-k:]
}

/*
1.5.3 获取第k小的数
*/

func TopkSmallestElement(nums []int, k int) int {
	TopkSplit(nums, k, 0, len(nums)-1)
	return nums[k-1]
}

/*
1.5.4 获取第k大的数
*/

func TopkLargestElement(nums []int, k int) int {
	TopkSplit(nums, len(nums)-k, 0, len(nums)-1)
	return nums[len(nums)-k]
}

/*
1.6 寻找两个有序数组的中位数
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

/*
1.7 三数之和
给你一个包含n个整数的数组nums，判断nums中是否存在三个元素 a，b，c ，使得a + b + c = 0 ？请你找出所有和为0且不重复的三元组。
注意：答案中不可以包含重复的三元组。
*/

// ThreeSum 排序+双指针解决,难点是去重, 时间复杂度O(N^2),空间复杂度O(logN)
func ThreeSum(nums []int) [][]int {
	var res [][]int
	n := len(nums)
	if n < 3 {
		return res
	}
	// 对数组进行排序
	sort.Ints(nums)
	for i := 0; i < len(nums)-2; i++ {
		// 因为nums是升序数组，所以nums[i]之后的数都会大于0，三个正数之和不可能等于0，所以此时要break
		if nums[i] > 0 {
			break
		}
		// nums[i] == nums[i-1], 去重
		if i >= 1 && nums[i] == nums[i-1] {
			continue
		}
		// 左右指针初始值分别为i+1,len(nums)-1
		l, r := i+1, n-1
		for l < r {
			// 判断三数之和是否等于0
			sum := nums[i] + nums[l] + nums[r]
			if sum == 0 {
				res = append(res, []int{nums[i], nums[l], nums[r]})
				// 只要nums[l] == nums[l+1]，左指针向右移动一位
				for l < r && nums[l] == nums[l+1] {
					l++
				}
				// nums[r] == nums[r-1]，右指针向左移动一位
				for l < r && nums[r] == nums[r-1] {
					r--
				}
				// 如果sum == 0, l, r分别+1，-1
				l++
				r--
			} else if sum > 0 {
				// 此时说明sum过大，所以右指针应该向左移动，寻找更小的值
				r--
			} else {
				// 此时说明sum过小，所以左指针应该向右移动，寻找更大的值
				l++
			}
		}
	}
	return res
}

// ThreeSumUseHashTable 第二种思路是双层循环+哈希表, 麻烦的地方是去重，很难搞，同时不可避免的导致效率下降
func ThreeSumUseHashTable(nums []int) [][]int {
	n := len(nums)
	res := make([][]int, 0)
	if n < 3 {
		return res
	}
	// 对数组进行排序
	sort.Ints(nums)
	for i, v := range nums[:n-2] {
		// nums[i] == nums[i-1], 去重
		if i >= 1 && v == nums[i-1] {
			continue
		}
		d := make(map[int]int, 0)
		for _, x := range nums[i+1:] {
			if _, ok := d[x]; !ok {
				d[-v-x]++
			} else {
				// 此时说明找到了第三个数:-(v+x)
				res = append(res, []int{v, x, -v - x})
			}
		}
	}
	return DropDuplicates(res)
}

type Set struct {
	K1, K2, K3 int
}

// DropDuplicates 利用结构体作为key来给二维数组去重
func DropDuplicates(src [][]int) (dst [][]int) {
	m := make(map[Set]int, 0)
	for _, array := range src {
		// 排序仍然免不了
		sort.Ints(array)
		key := Set{
			array[0],
			array[1],
			array[2],
		}
		if m[key] == 0 {
			m[key]++
		} else {
			continue
		}
	}
	for key := range m {
		dst = append(dst, []int{key.K1, key.K2, key.K3})
	}
	return dst
}

/*
1.8 求x的n次方
用O(N)的时间复杂度解决是很容易，你能在O(logN)时间复杂度内解决吗？
*/

// MyPow 简单递归解决, 递归调用n次，所以时间复杂度为O(N)，空间复杂度O(1)
func MyPow(x float64, n int) float64 {
	var helper func(float64, int) float64
	helper = func(x float64, n int) float64 {
		if n == 0 {
			return 1
		}
		return helper(x, n-1) * x
	}
	if n >= 0 {
		return helper(x, n)
	}
	return 1.0 / helper(x, -n)
}

// MyPowSimple 要想将时间复杂度降低为O(logN),就要使用二分法.
func MyPowSimple(x float64, n int) float64 {
	var helper func(float64, int) float64
	helper = func(x float64, n int) float64 {
		if n == 0 {
			return 1
		}
		t := helper(x, n/2)
		if n%2 == 1 {
			return t * t * x
		}
		return t * t
	}
	if n >= 0 {
		return helper(x, n)
	}
	return 1.0 / helper(x, -n)
}

/*
1.9 二维数组中的查找
在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个
高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

示例:

现有矩阵 matrix 如下：

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定target=5，返回true。

给定target=20，返回false。
*/

// FindNumberIn2DArrayBinary 每一行进行二分查找，时间复杂度为O(N*logM)，空间复杂度O(1)
func FindNumberIn2DArrayBinary(matrix [][]int, target int) bool {
	if matrix == nil || len(matrix) == 0 || len(matrix[0]) == 0 {
		return false
	}
	for _, nums := range matrix {
		if existed := BinarySearchUseRecursion(nums, target); existed {
			return existed
		} else {
			continue
		}
	}
	return false
}

/*
由于给定的二维数组具备每行从左到右递增以及每列从上到下递增的特点，当访问到一个元素时，可以排除数组中的部分元素。
从二维数组的右上角开始查找。如果当前元素等于目标值，则返回true。如果当前元素大于目标值，则移到左边一列。如果当前
元素小于目标值，则移到下边一行。

可以证明这种方法不会错过目标值。如果当前元素大于目标值，说明当前元素的下边的所有元素都一定大于目标值，因此往下查找
不可能找到目标值，往左查找可能找到目标值。如果当前元素小于目标值，说明当前元素的左边的所有元素都一定小于目标值，
因此往左查找不可能找到目标值，往下查找可能找到目标值。

若数组为空，返回false
初始化行下标为0，列下标为二维数组的列数减1
重复下列步骤，直到行下标或列下标超出边界
获得当前下标位置的元素 num
如果 num 和 target 相等，返回 true
如果 num 大于 target，列下标减 1
如果 num 小于 target，行下标加 1
循环体执行完毕仍未找到元素等于 target ，说明不存在这样的元素，返回 false
*/

// FindNumberIn2DArray 时间复杂度为O(N+M)，空间复杂度O(1)
func FindNumberIn2DArray(matrix [][]int, target int) bool {
	if matrix == nil || len(matrix) == 0 || len(matrix[0]) == 0 {
		return false
	}
	rows, columns := len(matrix), len(matrix[0])
	row, column := 0, columns-1
	for row < rows && column >= 0 {
		if matrix[row][column] == target {
			return true
		} else if matrix[row][column] > target {
			column--
		} else {
			row++
		}
	}
	return false
}

/*
1.10 有序数组的平方
给你一个按非递减顺序排序的整数数组nums，返回每个数字的平方组成的新数组，要求也按非递减顺序排序。

示例1：
输入：nums = [-4,-1,0,3,10]
输出：[0,1,9,16,100]
解释：平方后，数组变为 [16,1,0,9,100]
排序后，数组变为 [0,1,9,16,100]

示例2：
输入：nums = [-7,-3,2,3,11]
输出：[4,9,9,49,121]

提示：
1 <= nums.length <= 104
-104 <= nums[i] <= 104
nums已按非递减顺序 排序
*/

/*
双指针法，创建一个result数组，长度与nums数组相等，令i=0, j, k=len(nums)-1。
由于是平方数，那么绝对值越大的平方数肯定就越大，而nums数组是有序的，所以平方数最大的要么是数组
的起始元素，要么是末尾元素。由于新数组也需要是升序排列，所以我们从尾到头填充新数组result.
在for循环遍历中(i<=j)如果nums[i]的平方数大于nums[j]的平方数，那么nums[k]=nums[i]的平方数，i++
否则，nums[k]=nums[j]的平方数，j--, 同时无论何种情况，k均向前移动一位，也就是k--。这样遍历结束，
新数组result便填充完毕了。
*/

func SortedSquares(nums []int) []int {
	n := len(nums)
	result := make([]int, n)
	i, j, k := 0, n-1, n-1
	for i <= j {
		if nums[i]*nums[i] > nums[j]*nums[j] {
			result[k] = nums[i] * nums[i]
			i++
		} else {
			result[k] = nums[j] * nums[j]
			j--
		}
		k--
	}
	return result
}

/*
1.11 下一个排列
实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列（即，组合出下一个更大的整数）。
如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。
必须原地修改，只允许使用额外常数空间。


示例1：
输入：nums = [1,2,3]
输出：[1,3,2]

示例2：
输入：nums = [3,2,1]
输出：[1,2,3]

示例3：
输入：nums = [1,1,5]
输出：[1,5,1]

示例4：
输入：nums = [1]
输出：[1]
*/

/*
对于长度为n的排列a：
首先从后向前查找第一个顺序对(i,i+1)，满足a[i]<a[i+1]。这样「较小数」即为a[i]。此时[i+1,n)必然是下降序列。
如果找到了顺序对，那么在区间[i+1,n) 中从后向前查找第一个元素j满足a[i] < a[j]。这样「较大数」即为a[j]。
交换a[i]与a[j]，此时可以证明区间[i+1,n) 必为降序。我们可以直接使用双指针反转区间[i+1,n)使其变为升序，
而无需对该区间进行排序。
*/

func NextPermutation(nums []int) {
	n := len(nums)
	i, j := n-2, n-1
	for i >= 0 && nums[i] >= nums[i+1] {
		i--
	}
	if i >= 0 {
		for j >= 0 && nums[i] >= nums[j] {
			j--
		}
		nums[i], nums[j] = nums[j], nums[i]
	}
	feature.ReverseArray(nums[i+1:])
}