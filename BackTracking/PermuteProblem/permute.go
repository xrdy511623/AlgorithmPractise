package PermuteProblem

import "sort"

/*
leetcode 46. 全排列
1.1 给定一个没有重复数字的序列，返回其所有可能的全排列。

示例:
输入: [1,2,3]
输出: [ [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1] ]
*/

func Permute(nums []int) [][]int {
	var res [][]int
	var path []int
	visited := make(map[int]bool, 0)
	size := len(nums)
	var backTrack func()
	backTrack = func() {
		// 递归终止条件
		if len(path) == size {
			temp := make([]int, size)
			copy(temp, path)
			res = append(res, temp)
			return
		}
		for i := 0; i < size; i++ {
			// 一个排列结果(path)里一个元素只能使用一次
			if visited[nums[i]] {
				continue
			}
			visited[nums[i]] = true
			path = append(path, nums[i])
			// 递归
			backTrack()
			// 回溯
			visited[nums[i]] = false
			path = path[:len(path)-1]
		}
	}
	backTrack()
	return res
}

/*
Leetcode 47. 全排列 II
1.2 给定一个可包含重复数字的序列nums，按任意顺序返回所有不重复的全排列。

示例1：
输入：nums = [1,1,2]
输出： [[1,1,2], [1,2,1], [2,1,1]]

示例2：
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

提示：
1 <= nums.length <= 8
-10 <= nums[i] <= 10
*/

func PermuteUnique(nums []int) [][]int {
	var res [][]int
	var path []int
	// 排序后方便去重
	sort.Ints(nums)
	size := len(nums)
	visited := make([]bool, size)
	var backTrack func()
	backTrack = func() {
		// 递归终止条件
		if len(path) == size {
			temp := make([]int, size)
			copy(temp, path)
			res = append(res, temp)
			return
		}
		for i := 0; i < size; i++ {
			// 对原数组排序后，相同的数字都相邻，然后每次填入的数一定是这个数所在重复数集合中
			// 从左往右第一个未被填过的数字
			if visited[i] || i > 0 && nums[i] == nums[i-1] && visited[i-1] {
				continue
			}
			// 处理每一个元素
			visited[i] = true
			path = append(path, nums[i])
			// 递归
			backTrack()
			// 回溯
			visited[i] = false
			path = path[:len(path)-1]
		}
	}
	backTrack()
	return res
}