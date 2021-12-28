package feature

import "AlgorithmPractise/Utils"

/*
1.1 反转数组
*/

// ReverseArraySimple 可以创建新数组
func ReverseArraySimple(nums []int) []int {
	n := len(nums)
	var res []int
	for i := n - 1; i >= 0; i-- {
		res = append(res, nums[i])
	}
	return res
}

// ReverseArray 原地反转
func ReverseArray(nums []int) []int {
	n := len(nums)
	for i := 0; i < n/2; i++ {
		temp := nums[n-1-i]
		nums[n-1-i] = nums[i]
		nums[i] = temp
	}
	return nums
}

/*
1.2 找众数
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
你可以假设数组是非空的，并且给定的数组总是存在多数元素。
*/

/*
第一种思路，最简单的办法就是遍历数组，用哈希表记录数组中每个元素出现的次数，如果哪个元素的出现次数大于数组长度的一半，
那么这个元素就是众数，这样做时间复杂度O(n)，空间复杂度O(n/2),显然空间复杂度较高，不是最优解。
第二种思路:如果我们把众数的值记为+1，把其他数记为−1，将它们全部加起来，显然和大于0，从结果本身我们可以看出众数比其他数多。
我们维护一个候选众数candidate和它出现的次数count。初始时candidate可以为任意值，count为0；
我们遍历数组nums中的所有元素，对于每个元素x，在判断x之前，如果count的值为0，我们先将x的值赋予candidate，随后我们判断x：
如果x与candidate相等，那么计数器count的值增加1；
如果x与candidate不等，那么计数器count的值减少1。
在遍历完成后，candidate 即为整个数组的众数。为什么？因为非众数在遍历过程中一定会遇到出现次数比它多的众数，这样count值会被减到0，
从而引发candidate的重新赋值，同理，只有众数赋值给candidate后，count才不会减到0，因为所有非众数出现的次数加起来都没有它多，
这样candidate的值就会一直是众数，直到遍历数组结束。

时间复杂度O(n)，空间复杂度O(1)
*/

func MajorityElement(nums []int) int {
	candidate, count := 0, 0
	for _, v := range nums {
		if count == 0 {
			candidate = v
		}
		if v == candidate {
			count++
		} else {
			count--
		}
	}
	return candidate
}

/*
1.3 删除有序数组中的重复项
给你一个有序数组nums，请你原地删除重复出现的元素，使每个元素只出现一次 ，返回删除后数组的新长度。
不要使用额外的数组空间，你必须在原地修改输入数组 并在使用O(1) 额外空间的条件下完成。
*/

/*
数组是有序的，那么重复的元素一定会相邻。在同一个数组里面操作，也就是不重复的元素移到数组的左侧，
最后取左侧的数组的值。
 */

func RemoveDuplicates(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	slow := 0
	for fast := 1; fast < n; fast++ {
		if nums[fast] != nums[fast-1] {
			slow++
			nums[slow] = nums[fast]
		}
	}
	return slow + 1
}

/*
1.4 进阶 移除元素
给你一个数组 nums和一个值val，你需要原地移除所有数值等于val的元素，并返回移除后数组的新长度。
不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并原地修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]

输入：nums = [0,1,2,2,3,0,4,2], val = 2
输出：5, nums = [0,1,4,0,3]
*/

func RemoveElement(nums []int, val int) int {
	for i := 0; i < len(nums); i++ {
		if nums[i] == val {
			nums = append(nums[:i], nums[i+1:]...)
			i--
		}
	}
	return len(nums)
}


/*
双指针法，快慢指针fast,slow初始位置均为0，fast指向当前要处理的元素，slow指向下一个将要赋值的位置，
如果fast指向的元素不等于val，那它一定是新数组的一个元素，我们将它赋值给slow指向的位置，同时快慢
指针同时向右移动，若fast指向的元素等于val，那它不是新数组想要的元素，此时slow指针不动，fast向后移动一位。
 */

func RemoveElementsTwo(nums []int, val int)int{
	fast, slow := 0, 0
	for fast < len(nums){
		if nums[fast] != val{
			nums[slow] = nums[fast]
			slow++
		}
		fast++
	}
	return slow
}

// 双指针法可以写得更简单一些
func RemoveElementSimple(nums []int, val int)int{
	index := 0
	// 此时i就是快指针fast
	for i:=0;i<len(nums);i++{
		if nums[i] != val{
			nums[index] = nums[i]
			index++
		}
	}
	return index
}

/*
1.5 移动零
给定一个数组nums，编写一个函数将所有0移动到数组的末尾，同时保持非零元素的相对顺序。

示例:
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]

说明:
必须在原数组上操作，不能拷贝额外的数组。
尽量减少操作次数。
 */

/*
思路:先处理非0元素，最后处理0
index指针指向非0元素，count统计数组中零的个数，index的位置和count的数量初始值均为0，
在for循环遍历过程中，如果遇到非0元素，则将其赋值给nums[index]，同时index指针向右移动一位。
若遇到0，则count累加1。遍历结束后，index即指向所有非0元素最后一位的右边。意味着
nums[index:index+count]区间内的元素都应该是0，那就循环count次，将0赋值给nums[index]就可以了，
当然，每次赋值后index还得向后移动一位。
 */

func MoveZeroes(nums []int)  {
	index, count := 0, 0
	for i:=0;i<len(nums);i++{
		if nums[i] != 0{
			nums[index] = nums[i]
			index++
		}else{
			count++
		}
	}
	for i:=0;i<count;i++{
		nums[index] = 0
		index++
	}
}


/*
1.5 最长公共前缀
编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串""。

示例 1：
输入：strs = ["flower","flow","flight"]
输出："fl"

示例 2：
输入：strs = ["dog","racecar","car"]
输出：""
解释：输入不存在公共前缀。
*/

/*
依次遍历字符串数组中的每个字符串，对于每个遍历到的字符串，更新最长公共前缀，当遍历完所有的字符串以后，即可得到字符串数组中的最长公共前缀。
如果在尚未遍历完所有的字符串时，最长公共前缀已经是空串，则最长公共前缀一定是空串，因此不需要继续遍历剩下的字符串，直接返回空串即可。
复杂度分析
时间复杂度：O(mn)，其中m是字符串数组中的字符串的平均长度，n是字符串的数量。最坏情况下，字符串数组中的每个字符串的每个字符都会被比较一次。
空间复杂度：O(1)。使用的额外空间复杂度为常数。
*/

func LongestCommonPrefix(strs []string) string {
	count := len(strs)
	if count == 0 {
		return ""
	}
	prefix := strs[0]
	for i := 1; i < count; i++ {
		prefix = lcp(prefix, strs[i])
		if len(prefix) == 0 {
			break
		}
	}
	return prefix
}

func lcp(str1, str2 string) string {
	length := Utils.Min(len(str1), len(str2))
	index := 0
	for index < length && str1[index] == str2[index] {
		index++
	}
	return str1[:index]
}

/*
1.6 最长连续递增子序列
给定一个未经排序的整数数组，找到最长且连续递增的子序列，并返回该序列的长度。
连续递增的子序列，可以由两个下标l和r（l < r）确定，如果对于每个l <= i < r，都有nums[i] < nums[i + 1] ，
那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。

输入：nums = [1,3,5,4,7]
输出：3
解释：最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。

输入：nums = [2,2,2,2,2]
输出：1
解释：最长连续递增序列是 [2], 长度为1。
*/

func FindLengthOfLCIS(nums []int) int {
	maxLength, start := 0, 0
	for i, v := range nums {
		if i > 0 && v <= nums[i-1] {
			start = i
		}
		maxLength = Utils.Max(maxLength, i-start+1)
	}

	return maxLength
}

/*
1.7 最大子序列和
给定一个整数数组 nums，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

示例 1：
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组[4,-1,2,1] 的和最大，为6 。

示例 2：
输入：nums = [1]
输出：1
示例 3：

输入：nums = [0]
输出：0
示例 4：

输入：nums = [-1]
输出：-1
*/

func MaxSubArray(nums []int) int {
	max := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i]+nums[i-1] > nums[i] {
			nums[i] += nums[i-1]
		}
		if nums[i] > max {
			max = nums[i]
		}
	}
	return max
}
