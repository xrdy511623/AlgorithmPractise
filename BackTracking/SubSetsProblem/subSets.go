package SubSetsProblem

import "sort"

/*
leetcode 78. 子集
1.1 给定一组不含重复元素的整数数组nums，返回该数组所有可能的子集（幂集）。
说明：解集不能包含重复的子集。

示例:
输入: nums = [1,2,3]
输出: [ [3],   [1],   [2],   [1,2,3],   [1,3],   [2,3],   [1,2],   [] ]
*/

/*
我们要弄清楚子集问题和组合问题、分割问题的的区别，子集是收集树形结构中树的所有节点的结果。
而组合问题、分割问题是收集树形结构中叶子节点的结果。
所以，求取子集问题，不需要任何剪枝！因为子集就是要遍历整棵树。
*/

func Subsets(nums []int) [][]int {
	var res [][]int
	var path []int
	n := len(nums)
	var backTrack func(int)
	backTrack = func(start int) {
		// 递归终止条件，当start移动到数组nums末尾元素后
		if start > n {
			return
		}
		// 收集树中所有节点的结果
		temp := make([]int, len(path))
		copy(temp, path)
		// 此处不能return，因为子集要收集所有树节点的结果
		res = append(res, temp)
		for i := start; i < n; i++ {
			// 处理每一个元素
			path = append(path, nums[i])
			// 递归
			backTrack(i + 1)
			// 回溯
			path = path[:len(path)-1]
		}
	}
	backTrack(0)
	return res
}

/*
leetcode 90. 子集II
1.2 给定一个可能包含重复元素的整数数组nums，返回该数组所有可能的子集（幂集）。
说明：解集不能包含重复的子集。

示例:
输入: [1,2,2]
输出: [ [2], [1], [1,2,2], [2,2], [1,2], [] ]
*/

/*
本题与上题唯一的区别是数组中有重复元素，但解集中不能有重复的子集，所以需要去重。这里的去重
是树的同一层级的去重，不是树枝上的去重，所以只要对数组进行排序后再去重，便可以分分钟AC了。
*/

func SubsetsWithDup(nums []int) [][]int {
	var res [][]int
	var path []int
	// 对nums数组进行排序
	sort.Ints(nums)
	var backTrack func(int)
	backTrack = func(start int) {
		// 递归终止条件，当start移动到数组nums末尾元素后
		if start > len(nums) {
			return
		}
		// 收集树中所有节点的结果
		temp := make([]int, len(path))
		copy(temp, path)
		// 此处不能return，因为子集要收集所有树节点的结果
		res = append(res, temp)
		for i := start; i < len(nums); i++ {
			// 同一树层去重
			if i > start && nums[i] == nums[i-1] {
				continue
			}
			// 处理每一个元素
			path = append(path, nums[i])
			// 递归
			backTrack(i + 1)
			// 回溯
			path = path[:len(path)-1]
		}
	}
	backTrack(0)
	return res
}

/*
leetcode 491. 递增子序列
1.3 给定一个整型数组, 你的任务是找到所有该数组的递增子序列，递增子序列的长度至少是2。

示例:
输入: [4, 6, 7, 7]
输出: [[4, 6], [4, 7], [4, 6, 7], [4, 6, 7, 7], [6, 7], [6, 7, 7], [7,7], [4,7,7]]
说明:

给定数组的长度不会超过15。
数组中的整数范围是 [-100,100]。
给定数组中可能包含重复数字，相等的数字应该被视为递增的一种情况。
*/

/*
本题要求取得所有的升序子序列，所以不能像组合问题 1.5组合总和II那样，对数组先排序然后去重。
极端情况下，如果数组是降序序列，那么符合要求的子序列为0，如果先排序会得到很多不满足题意的解。
只能是同一层循环遍历时跳过已经选取过的重复元素以及无法作为升序子序列path最大值的元素。
*/

func FindSubsequences(nums []int) [][]int {
	var res [][]int
	var path []int
	size := len(nums)
	var backTrack func(int)
	backTrack = func(start int) {
		// 剪枝，当start移动到数组nums末尾位置后
		if start > size {
			return
		}
		// 递归终止条件
		if len(path) >= 2 {
			temp := make([]int, len(path))
			copy(temp, path)
			res = append(res, temp)
		}
		// 每层都新建一个哈希表用于去重
		used := make(map[int]bool)
		for i := start; i < size; i++ {
			// 剪枝，如果待加入path的元素nums[i]小于path中最大的数，那么nums[i]加入后便无法
			// 成为升序序列，所以此种情况要跳过；或者nums[i]在同一层已经使用过也需要跳过
			if (len(path) > 0 && nums[i] < path[len(path)-1]) || used[nums[i]] {
				continue
			}
			used[nums[i]] = true
			// 处理每一个元素
			path = append(path, nums[i])
			// 递归
			backTrack(i + 1)
			// 回溯
			path = path[:len(path)-1]
		}
	}
	backTrack(0)
	return res
}
