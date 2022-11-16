package sort

import (
	"AlgorithmPractise/Utils"
	"sort"
)

/*
1.0 实现冒泡，插入排序，快排和归并排序
*/

/*
将无序数组中的元素从左向右依次迭代,(n=len(array)比较相邻两个元素的大小,若左边的大于右边的元素,那么它们俩
互换位置,经过第一次迭代,数组中最大的元素便会冒泡似地层层向上传递到数组末尾位置;
第二次迭代前面n-1个元素,以同样的方式进行比较交换,那么数组中第二大的元素便会上浮冒泡到倒数
第二个位置....
重复上面的动作,直到没有两个元素可以比较为止(也就是剩下一个元素),此时这个元素就是数组中最小的
最后形成的就是一个从小到大的有序数组
最坏时间复杂度O(n`2),最好时间复杂度O(n) 稳定
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
每次挑选一个元素插入到已排序数组中,初始时已排序数组中只有一个元素.
最坏时间复杂度O(n`2),最好时间复杂度O(n) 稳定
*/

func InsertSort(array []int) []int {
	n := len(array)
	if n == 0 {
		return array
	}
	// 从第二个位置,即下标为1的元素开始向前插入
	for i := 1; i < n; i++ {
		for j := i; j > 0; j-- {
			// 从第i个元素开始向前比较,如果小于前一个元素,交换位置
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
1 从数组中挑出一个元素,称为基准
2 重新排序数组,所有元素比基准值小的摆放在基准前面,所有元素比基准值大的摆在基准后面(相同的数
可以放在任意一边).在这次分区结束后,该基准值就处于它在有序数组中的正确位置了.这个称为分区操作
3 递归地将小于基准值元素的子序列和大于基准值元素的子序列进行分区操作.
时间复杂度O(NlogN),最坏时间复杂度O(n`2),空间复杂度O(1)
相比归并排序，快排的优点在于它是原地排序，空间复杂度更低。
*/

func QuickSort(nums []int, start, stop int) {
	if start >= stop {
		return
	}
	pivot := nums[start]
	left, right := start, stop
	for left < right {
		for left < right && nums[right] >= pivot {
			right--
		}
		nums[left] = nums[right]
		for left < right && nums[left] < pivot {
			left++
		}
		nums[right] = nums[left]
	}
	// 此时left==right,我们找到了基准值pivot的正确位置left/right
	nums[left] = pivot
	// 递归地将小于基准值pivot的子序列和大于基准值pivot的子序列进行分区操作
	QuickSort(nums, start, left-1)
	QuickSort(nums, left+1, stop)
}

/*
分治法 (Divide and Conquer)
很多有用的算法结构上是递归(递归的问题是递归过深可能导致堆栈溢出)的，为了解决一个特定问题，算法一次或者
多次递归调用其自身以解决若干子问题。
这些算法典型地遵循分治法的思想：将原问题分解为几个规模较小但是类似于原问题的子问题，
递归求解这些子问题，
然后再合并这些问题的解来建立原问题的解。
分治法在每层递归时有三个步骤：
- 分解原问题为若干子问题，这些子问题是原问题的规模最小的实例
- 解决这些子问题，递归地求解这些子问题。当子问题的规模足够小，就可以直接求解
- 合并这些子问题的解成原问题的解
归并排序:最坏时间复杂度O(NlogN),最好时间复杂度O(NlogN) 稳定,空间复杂度O(N)
*/

func MergeSort(array []int) []int {
	// 递归终止条件
	n := len(array)
	if n <= 1 {
		return array
	}
	mid := n / 2
	left := MergeSort(array[:mid])
	right := MergeSort(array[mid:])
	return Merge(left, right)
}

func Merge(left, right []int) (res []int) {
	l, r := 0, 0
	ll, rl := len(left), len(right)
	for l < ll && r < rl {
		if left[l] < right[r] {
			res = append(res, left[l])
			l++
		} else {
			res = append(res, right[r])
			r++
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
		mid := (start + stop) / 2
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
Leetcode 34. 在排序数组中查找元素的第一个和最后一个位置
1.1 二分查找变形之一:在排序数组中查找元素的第一个和最后一个位置。
给定一个按照升序排列的整数数组nums，和一个目标值target。找出给定目标值在数组中的开始位置和结束位置。
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
剑指Offer 53 - I. 在排序数组中查找数字I
1.2 在排序数组中统计一个数在数组中出现的次数
要求时间复杂度为O(logN),空间复杂度为O(1)
示例 1:
输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
输入: nums = [5,7,7,8,8,10], target = 6
输出: 0

思路: 在排序数组中找出等于目标值target的起始位置index，如果index为-1,证明数组中无此元素，返回0，
否则从数组index位置开始向后遍历数组元素，只要数组元素等于target，则将其出现次数count累加1,如果遇到不等于
target的元素，说明后面的元素都大于target，此时退出循环，最后返回count即可
*/

func Search(nums []int, target int) int {
	index := BinarySearchFirstEqualTarget(nums, target)
	if index == -1 {
		return 0
	}
	count := 0
	for _, num := range nums[index:] {
		if num == target {
			count++
		} else {
			break
		}
	}
	return count
}

/*
旋转数组专题
*/

/*
leetcode 189. 轮转数组
1.3 给你一个数组，将数组中的元素向右轮转k个位置，其中k是非负数。

示例1:
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]

示例2:
输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释:
向右轮转 1 步: [99,-1,-100,3]
向右轮转 2 步: [3,99,-1,-100]

提示：
1 <= nums.length <= 105
-231 <= nums[i] <= 231 - 1
0 <= k <= 105
*/

// Rotate 时间复杂度O(2N),空间复杂度O(1)
func Rotate(nums []int, k int) {
	k %= len(nums)
	// k为0或者旋转次数为n的整数倍，那么数组会恢复原样
	// 所以不做任何操作
	if k == 0 {
		return
	}
	// 先反转整个数组
	Utils.ReverseArray(nums)
	// 然后反转数组nums[:k]部分
	Utils.ReverseArray(nums[:k])
	// 最后反转数组nums[k:]部分
	Utils.ReverseArray(nums[k:])
}

/*
leetcode 153. 寻找旋转排序数组中的最小值
1.4 已知一个长度为n的数组，预先按照升序排列，经由1到n次旋转后，得到输入数组。例如，原数组nums = [0,1,2,4,5,6,7]
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
否则旋转后的数组一定是由两个升序部分组合而成([4,5,6,7]和[0,1,2])，譬如
4，5，6，7，0，1，2
那么最小值一定是第一个升序部分末尾元素的下一位。
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

由于数组不包含重复元素，并且只要当前的区间长度不为1，pivot就不会与high重合；而如果当前的区间长度为1，这说明我们
已经可以结束二分查找了。因此不会存在nums[pivot]=nums[high]的情况。
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
leetcode 154. 寻找旋转排序数组中的最小值II
1.5 已知一个长度为n的数组，预先按照升序排列，经由1到n次旋转后，得到输入数组。例如，原数组nums = [0,1,4,4,5,6,7]
在变化后可能得到：
若旋转 4 次，则可以得到 [4,5,6,7,0,1,4]
若旋转 7 次，则可以得到 [0,1,4,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
给你一个可能存在重复元素值的数组nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组
中的最小元素 。

示例1：
输入：nums = [1,3,5]
输出：1

示例2：
输入：nums = [2,2,2,0,1]
输出：0
*/

func FindMinTwo(nums []int) int {
	n := len(nums)
	// 特殊情况处理
	if n == 1 {
		return nums[0]
	}
	i := 0
	// 寻找数组升序部分的末尾元素
	for i < n-1 && nums[i] <= nums[i+1] {
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

func FindMinTwoSimple(nums []int) int {
	low, high := 0, len(nums)-1
	for low < high {
		mid := (low + high) / 2
		if nums[mid] < nums[high] {
			high = mid
		} else if nums[mid] > nums[high] {
			low = mid + 1
			// 因为数组中有重复元素，所以此时指针high向左移动
		} else {
			high--
		}
	}
	return nums[low]
}

/*
leetcode 33. 搜索旋转排序数组
1.6 整数数组nums按升序排列，数组中的值互不相同。
在传递给函数之前，nums在预先未知的某个下标k（0 <= k < nums.length）上进行了旋转，使数组变为[nums[k],
nums[k+1], ...,nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标从0开始计数）。例如，
[0,1,2,4,5,6,7]在下标3处经旋转后可能变为[4,5,6,7,0,1,2] 。

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

提示：
1 <= nums.length <= 5000
-10^4 <= nums[i] <= 10^4
nums中的每个值都独一无二
题目数据保证 nums 在预先未知的某个下标上进行了旋转
-10^4 <= target <= 10^4
*/

/*
思路:虽然进行旋转后的数组整体上不再是有序的了，但是从中间某个位置将数组分成左右两部分时，一定有一部分是有序的，
所以我们仍然可以通过二分查找来解决此问题，只不过需要根据有序的部分确定应该如何改变二分查找的上下限，因为我们可以
根据有序的部分判断出target是否在这个部分。
如果 [l, mid - 1]是有序数组，且target的大小满足 [nums[l],nums[mid]]，则我们应该将搜索范围缩小至[l, mid-1]，
否则在 [mid+1, r]中寻找。
如果[mid, r] 是有序数组，且target的大小满足[nums[mid+1],nums[r]]，则我们应该将搜索范围缩小至[mid+1, r]，
否则在[l, mid-1]中寻找。
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
		// 中间值即为target,直接返回mid
		if target == nums[mid] {
			return mid
		}
		// 此时前半部分有序
		if nums[l] <= nums[mid] {
			// 此时target落在前半部分有序区间内
			if nums[l] <= target && target < nums[mid] {
				r = mid - 1
			} else {
				// 此时target落在后半部分无序区间内
				l = mid + 1
			}
		} else {
			// 此时后半部分有序
			// 此时target落在后半部分有序区间内
			if nums[mid] < target && target <= nums[r] {
				l = mid + 1
			} else {
				// 此时target落在前半部分无序区间内
				r = mid - 1
			}
		}
	}
	// 循环结束没有找到目标值target，返回-1
	return -1
}

/*
leetcode 81. 搜索旋转排序数组II
1.7 已知存在一个按非降序排列的整数数组nums，数组中的值不必互不相同。
在传递给函数之前，nums在预先未知的某个下标k（0 <= k < nums.length）上进行了旋转，使数组变为
[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标从0开始计数）。
例如，[0,1,2,4,4,4,5,6,6,7] 在下标5处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。

给你旋转后的数组nums和一个整数target，请你编写一个函数来判断给定的目标值是否存在于数组中。
如果nums中存在这个目标值target，则返回true，否则返回false。

示例1：
输入：nums = [2,5,6,0,0,1,2], target = 0
输出：true

示例2：
输入：nums = [2,5,6,0,0,1,2], target = 3
输出：false


提示：
1 <= nums.length <= 5000
-104 <= nums[i] <= 104
题目数据保证nums在预先未知的某个下标上进行了旋转
-104 <= target <= 104
*/

func SearchTarget(nums []int, target int) bool {
	n := len(nums)
	if n == 0 {
		return false
	}
	l, r := 0, len(nums)-1
	for l <= r {
		// 关键在于处理重复元素
		// 若左边有重复数字，将左边界l右移
		for l < r && nums[l] == nums[l+1] {
			l++
		}
		// 若右边有重复数字，将右边界r左移
		for l < r && nums[r] == nums[r-1] {
			r--
		}
		mid := (l + r) / 2
		// 中间值即为target,返回true
		if nums[mid] == target {
			return true
		}
		// 此时数组前半部分有序
		if nums[l] <= nums[mid] {
			// 此时target落在前半部分有序区间内
			if nums[l] <= target && target < nums[mid] {
				r = mid - 1
			} else {
				// 此时target落在后半部分无序区间内
				l = mid + 1
			}
		} else {
			// 此时数组后半部分有序
			// 此时target落在后半部分有序区间内
			if nums[mid] < target && target <= nums[r] {
				l = mid + 1
			} else {
				// 此时target落在前半部分无序区间内
				r = mid - 1
			}
		}
	}
	return false
}

/*
leetcode 面试题 10.03. 搜索旋转数组
1.8 搜索旋转数组。给定一个排序后的数组，包含n个整数，但这个数组已被旋转过很多次了，次数不详。请编写代码找出数组
中的某个元素，假设数组元素原先是按升序排列的。若有多个相同元素，返回索引值最小的一个。

示例1:
输入: arr = [15, 16, 19, 20, 25, 1, 3, 4, 5, 7, 10, 14], target = 5
输出: 8（元素5在该数组中的索引）

示例2:
输入：arr = [15, 16, 19, 20, 25, 1, 3, 4, 5, 7, 10, 14], target = 11
输出：-1 （没有找到）

提示:
arr 长度范围在[1, 1000000]之间
*/

func SearchRotateArray(arr []int, target int) int {
	n := len(arr)
	if n == 0 {
		return -1
	}
	l, r := 0, n-1
	for l <= r {
		// 当左边界l对应元素为target时直接返回l,因为题目要求返回最小索引
		if arr[l] == target {
			return l
		}
		mid := (l + r) / 2
		// 当中间值等于target时，将右边界r左移到mid，因为mid左边可能还有等于target的元素
		if arr[mid] == target {
			r = mid
		} else if arr[l] < arr[mid] {
			if arr[l] <= target && target < arr[mid] {
				r = mid - 1
			} else {
				l = mid + 1
			}
		} else if arr[l] > arr[mid] {
			if arr[mid] < target && target <= arr[r] {
				l = mid + 1
			} else {
				r = mid - 1
			}
		} else {
			// 当中间值与左边界l对应的元素相等时，将左边界l右移
			// 因为此时arr[l]==arr[mid]但是arr[mid] != target,即arr[l] != target
			// 所以target一定落在[l+1:r]区间内
			l++
		}
	}
	return -1
}

/*
leetcode 215. 数组中的第K个最大元素
1.9 给定整数数组nums和整数k，请返回数组中第k个最大的元素。
请注意，你需要找的是数组排序后的第k个最大的元素，而不是第k个不同的元素。
你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。
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
		value := mh.Extract()
		sortedArray = append(sortedArray, value)
	}
	return sortedArray[k-1]
}

/*
思路:快排可以解决问题，但是它需要确定数组中所有元素的正确位置，对于本题而言，我们只需要确定第k大元素的位置pos,
我们只需要确保pos左边的元素都比它小，pos右边的元素都比它大即可，不需要关心其左边和右边的集合是否有序，所以，我们
需要对快排进行改进，将目标值的位置pos与分区函数Partition求得的位置index进行比对，如果两值相等，说明index对应的
元素即为所求值，如果index<pos，则递归的在[index+1, right]范围求解；否则则在[left, index-1]范围求解，如此便
可大幅缩小求解范围。
*/

func FindKthLargestElement(nums []int, k int) int {
	n := len(nums)
	TopkSplit(nums, n-k, 0, n-1)
	return nums[n-k]
}

func Partition(array []int, start, stop int) int {
	if start >= stop {
		return -1
	}
	pivot := array[start]
	left, right := start, stop
	for left < right {
		for left < right && array[right] >= pivot {
			right--
		}
		array[left] = array[right]
		for left < right && array[left] < pivot {
			left++
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
1.9.1 获得前k小的数
*/

func TopkSmallest(nums []int, k int) []int {
	TopkSplit(nums, k, 0, len(nums)-1)
	return nums[:k]
}

/*
1.9.2 获得前k大的数
*/

func TopkLargest(nums []int, k int) []int {
	TopkSplit(nums, len(nums)-k, 0, len(nums)-1)
	return nums[len(nums)-k:]
}

/*
1.9.3 获取第k小的数
*/

func TopkSmallestElement(nums []int, k int) int {
	TopkSplit(nums, k, 0, len(nums)-1)
	return nums[k-1]
}

/*
1.9.4 获取第k大的数
*/

func TopkLargestElement(nums []int, k int) int {
	TopkSplit(nums, len(nums)-k, 0, len(nums)-1)
	return nums[len(nums)-k]
}

/*
1.10 leetcode 4 寻找两个有序数组的中位数
给定两个大小分别为m和n的正序（从小到大）数组nums1和nums2。请你找出并返回这两个正序数组的中位数 。
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
根据中位数的定义，当m+n是奇数时，中位数是两个有序数组中的第(m+n)/2 个元素，当m+n是偶数时，中位数是两个
有序数组中的第(m+n)/2个元素和第(m+n)/2+1 个元素的平均值。因此，这道题可以转化成寻找两个有序数组中的第k小的数，
其中k为(m+n)/2或(m+n)/2+1。

假设两个有序数组分别是A和B。要找到第k个元素，我们可以比较A[k/2−1]和B[k/2−1]，其中/表示整数除法。由于A[k/2−1]
和B[k/2−1]的前面分别有A[0..k/2−2]和B[0..k/2−2]，即k/2−1 个元素，对于A[k/2−1]和B[k/2−1]中的较小值，最多
只会有 (k/2-1)+(k/2-1) ≤ k−2 个元素比它小，那么它就不能是第k小的数了。
因此我们可以归纳出三种情况：
如果A[k/2−1]<B[k/2−1]，则比A[k/2−1] 小的数最多只有A的前 k/2−1个数和B的前k/2−1个数，即比A[k/2−1] 小的数
最多只有k-2个，因此A[k/2−1]不可能是第k个数，A[0] 到A[k/2−1]也都不可能是第k个数，可以全部排除。
如果A[k/2−1]>B[k/2−1]，则可以排除B[0]到B[k/2−1]。
如果A[k/2−1]=B[k/2−1]，则可以归入第一种情况处理。

可以看到，比较A[k/2−1]和 B[k/2−1] 之后，可以排除k/2个不可能是第k小的数，查找范围缩小了一半。同时，我们将在
排除后的新数组上继续进行二分查找，并且根据我们排除数的个数，减少k的值，这是因为我们排除的数都是不大于第k小的数。

有以下三种情况需要特殊处理：
如果A[k/2−1]或者B[k/2−1]越界，那么我们可以选取对应数组中的最后一个元素。在这种情况下，我们必须根据排除数的个数
减少k的值，而不能直接将k减去k/2。

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
由于k的值变成1，因此比较两个有序数组中的未排除下标范围内的第一个数，其中较小的数即为第k个数，由于A[3]>B[3]，
因此第k个数是B[3]=4。

A: [1 3 4] 9
           ↑
B: [1 2 3] 4 5 6 7 8 9
           ↑
*/

// FindMedianSortedArrays 二分法解决可以将时间复杂度降为O(log(m+n))，空间复杂度O(1)
func FindMedianSortedArrays(nums1, nums2 []int) float64 {
	length := len(nums1) + len(nums2)
	if length%2 == 1 {
		return float64(getKthElement(nums1, nums2, length/2+1))
	} else {
		m1, m2 := length/2, length/2+1
		return float64(getKthElement(nums1, nums2, m1)+getKthElement(nums1, nums2, m2)) / 2.0
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
直观的解法是归并排序，将这两个有序数组合并后找中位数。实际上也可以不用合并数组，用双指针也行。
*/

func FindMidInSortedArrays(nums1, nums2 []int) float64 {
	m, n := len(nums1), len(nums2)
	totalLength := m + n
	l1, l2 := 0, 0
	front, target := 0, 0
	// 如果totalLength为奇数，那问题转换为求第k小的数(k=totalLength/2+1)
	// 所以循环k-1次，迭代right即可得到第k小的数
	// 如果totalLength为偶数，那么问题转换为求第k-1小的数和第k小的数两个数的平均值
	for i := 0; i <= totalLength/2; i++ {
		front = target
		if l1 < m && (l2 >= n || nums1[l1] < nums2[l2]) {
			target = nums1[l1]
			l1++
		} else {
			target = nums2[l2]
			l2++
		}
	}
	if totalLength%2 == 1 {
		return float64(target)
	}
	return float64(front+target) / 2.0
}

/*
leetcode 15. 三数之和
1.11 给你一个包含n个整数的数组nums，判断nums中是否存在三个元素 a，b，c ，使得a + b + c = 0 ？请你找出所有和
为0且不重复的三元组。
注意：答案中不可以包含重复的三元组。
*/

// ThreeSum 排序+双指针解决,难点是去重, 时间复杂度O(N^2),空间复杂度O(logN)
func ThreeSum(nums []int) [][]int {
	res := [][]int{}
	n := len(nums)
	if n < 3 {
		return res
	}
	// 对数组进行排序
	sort.Ints(nums)
	for i := 0; i < n-2; i++ {
		// 因为nums是升序数组，所以nums[i]之后的数都会大于0，三个正数之和不可能等于0，所以此时要break
		if nums[i] > 0 {
			break
		}
		// nums[i] == nums[i-1], 去重
		if i >= 1 && nums[i] == nums[i-1] {
			continue
		}
		if nums[i]+nums[i+1]+nums[i+2] > 0 {
			break
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
leetcode 50. Pow(x, n)
1.12 用O(N)的时间复杂度解决是很容易，你能在O(logN)时间复杂度内解决吗？
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
剑指Offer 04. 二维数组中的查找
1.13 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

示例:

现有矩阵matrix如下：

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
		if BinarySearchUseRecursion(nums, target) {
			return true
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
获得当前下标位置的元素num
如果num和target 相等，返回true
如果num大于target，列下标减 1
如果num小于target，行下标加 1
循环执行完毕仍未找到元素等于target ，说明不存在这样的元素，返回false
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
leetcode 977. 有序数组的平方
1.14 给你一个按非递减顺序排序的整数数组nums，返回每个数字的平方组成的新数组，要求也按非递减顺序排序。

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
	res := make([]int, n)
	l, r, index := 0, n-1, n-1
	for l <= r && index >= 0 {
		if nums[l]*nums[l] > nums[r]*nums[r] {
			res[index] = nums[l] * nums[l]
			l++
		} else {
			res[index] = nums[r] * nums[r]
			r--
		}
		index--
	}
	return res
}

/*
1.15 下一个排列
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
我们希望下一个数比当前数大，这样才满足“下一个排列”的定义。因此只需要将后面的「大数」与前面的「小数」交换，就能得到
一个更大的数。比如123456，将5和6交换就能得到一个更大的数123465。
我们还希望下一个数增加的幅度尽可能地小，这样才满足“下一个排列与当前排列紧邻“的要求。为了满足这个要求，我们需要：
在尽可能靠右的低位进行交换，需要从后向前查找。
将一个尽可能小的大数与前面的小数交换。比如123465，下一个排列应该把5和4交换而不是把6和4交换。
将大数换到前面后，需要将大数后面的所有数重置为升序，升序排列就是最小的排列。以123465为例：
首先按照上一步，交换5和4，得到123564；然后需要将5之后的数重置为升序，得到123546。显然123546比123564更小，
123546就是123465的下一个排列。
以上就是求“下一个排列”的分析过程。

算法过程
标准的“下一个排列”算法可以描述为：
1 从后向前查找第一个相邻升序的元素对(i,j)，满足 A[i] < A[j]。此时 [j,end) 必然是降序
2 在[j,end) 从后向前查找第一个满足 A[i] < A[k]的 k。A[i]、A[k] 分别就是上文所说的小数、大数
3 将A[i]与A[k]交换
4 可以断定这时 [j,end) 必然是降序，逆置 [j,end)，使其升序
如果在步骤1找不到符合的升序相邻元素对，说明当前 [begin,end) 为一个降序顺序，则直接跳到步骤4
*/

func NextPermutation(nums []int) {
	n := len(nums)
	i := n - 2
	for i >= 0 && nums[i] >= nums[i+1] {
		i--
	}
	if i >= 0 {
		j := n - 1
		for j >= 0 && nums[i] >= nums[j] {
			j--
		}
		nums[i], nums[j] = nums[j], nums[i]
	}
	Utils.ReverseArray(nums[i+1:])
}

/*
leetcode 88. 合并两个有序数组
1.16 给你两个按非递减顺序排列的整数数组nums1 和 nums2，另有两个整数m和n ，分别表示nums1和nums2中的元素数目。
请你合并nums2到nums1中，使合并后的数组同样按非递减顺序排列。
注意：最终，合并后数组不应由函数返回，而是存储在数组nums1 中。为了应对这种情况，nums1的初始长度为 m + n，
其中前m个元素表示应合并的元素，后n个元素为0 ，应忽略。nums2的长度为n 。

示例1：
输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]
解释：需要合并 [1,2,3] 和 [2,5,6] 。
合并结果是 [1,2,2,3,5,6]
*/

// merge 逆向双指针，时间复杂度O(M+N), 空间复杂度O(1)
func merge(nums1 []int, m int, nums2 []int, n int) {
	p1, p2, tail := m-1, n-1, m+n-1
	for p1 >= 0 || p2 >= 0 {
		cur := 0
		if p1 == -1 {
			cur = nums2[p2]
			p2--
		} else if p2 == -1 {
			cur = nums1[p1]
			p1--
		} else if nums1[p1] > nums2[p2] {
			cur = nums1[p1]
			p1--
		} else {
			cur = nums2[p2]
			p2--
		}
		nums1[tail] = cur
		tail--
	}
}

/*
如果数组是单调递增或单调递减的，那么它是单调的。

如果对于所有 i <= j，nums[i] <= nums[j]，那么数组nums是单调递增的。
如果对于所有 i <= j，nums[i]> = nums[j]，那么数组nums是单调递减的。

当给定的数组nums是单调数组时返回true，否则返回false。

示例1：
输入：nums = [1,2,2,3]
输出：true

示例2：
输入：nums = [6,5,4,4]
输出：true

示例3：
输入：nums = [1,3,2]
输出：false
*/

func isMonotonic(nums []int) bool {
	if checkAscend(nums) || checkDescend(nums) {
		return true
	} else {
		return false
	}
}

func checkAscend(nums []int) bool {
	flag := true
	for i, n := 0, len(nums)-1; i < n; i++ {
		if nums[i] > nums[i+1] {
			flag = false
			break
		}
	}
	return flag
}

func checkDescend(nums []int) bool {
	flag := true
	for i, n := 0, len(nums)-2; i < n; i++ {
		if nums[i] < nums[i+1] {
			flag = false
			break
		}
	}
	return flag
}

func isMonotonicSimple(nums []int) bool {
	asc, desc := true, true
	for i, n := 0, len(nums)-1; i < n; i++ {
		if nums[i] > nums[i+1] {
			asc = false
		}
		if nums[i] < nums[i+1] {
			desc = false
		}
	}
	return asc || desc
}
