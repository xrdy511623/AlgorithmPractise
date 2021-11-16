package feature

import "AlgorithmPractise/Utils"

/*
1.1 反转数组
 */

// ReverseArraySimple 可以创建新数组
func ReverseArraySimple(nums []int)[]int{
	n := len(nums)
	var res []int
	for i:=n-1;i>=0;i--{
		res = append(res, nums[i])
	}
	return res
}

// ReverseArray 原地反转
func ReverseArray(nums []int)[]int{
	n := len(nums)
	for i:=0;i<n/2;i++{
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
第一种思路:如果我们把众数的值记为+1，把其他数记为−1，将它们全部加起来，显然和大于0，从结果本身我们可以看出众数比其他数多。
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

func RemoveDuplicates(nums []int) int {
	n := len(nums)
	if n == 0{
		return 0
	}
	slow := 1
	for fast:=1;fast<n;fast++{
		if nums[fast] != nums[fast-1]{
			nums[slow] = nums[fast]
			slow++
		}
	}
	return slow
}

/*
1.4 最长公共前缀
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
1.5 最长连续递增子序列
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
1.6 全排列
给定一个不含重复数字的数组nums，返回其所有可能的全排列。你可以按任意顺序返回答案。
示例 1：
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

示例 2：
输入：nums = [0,1]
输出：[[0,1],[1,0]]

示例 3：
输入：nums = [1]
输出：[[1]]
 */

func Permute(nums []int)[][]int{
	var res [][]int
	visited := make(map[int]bool, len(nums))
	var dfs func(path []int)
	dfs = func(path []int){
		if len(path) == len(nums){
			temp := make([]int, len(path))
			copy(temp, path)
			res = append(res, path)
			return
		}
		for _, v := range nums{
			if visited[v]{
				continue
			}
			path = append(path, v)
			visited[v] = true
			dfs(path[:len(path)-1])
			visited[v] = false
		}
	}
	dfs([]int{})
	return res
}