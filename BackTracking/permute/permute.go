package permute

import (
	"sort"
)

/*
leetcode 46. 全排列
1.1 给定一个没有重复数字的序列，返回其所有可能的全排列。

示例:
输入: [1,2,3]
输出: [ [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1] ]
*/

func Permute(nums []int) [][]int {
	res := [][]int{}
	path := []int{}
	visited := make(map[int]bool)
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
			if visited[i] {
				continue
			}
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
	res := [][]int{}
	path := []int{}
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

/*
leetcode 22 括号生成
数字n代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且有效的括号组合。

示例 1：
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]

示例 2：
输入：n = 1
输出：["()"]

提示：
1 <= n <= 8
*/

func generateParenthesis(n int) []string {
	res := []string{}
	var dfs func(int, int, string)
	// l和r分别代表当前字符串中左括号的数量和右括号的数量
	dfs = func(l, r int, path string) {
		// 终止条件：当前字符串长度达到 2*n
		if len(path) == 2*n {
			res = append(res, path)
			return
		}
		// 分支 1：如果左括号数量小于 n，可以添加左括号
		if l < n {
			// 递归调用，在当前字符串后添加 "("，左括号数量加 1
			dfs(l+1, r, path+"(")
		}
		// 分支 2：如果右括号数量小于左括号数量，可以添加右括号
		if r < l {
			// 递归调用，在当前字符串后添加 ")"，右括号数量加 1
			dfs(l, r+1, path+")")
		}
	}
	dfs(0, 0, "")
	return res
}

/*
输入一个字符串，打印出该字符串中字符的所有排列。
你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。
注意:字符串里可能有重复字符，但要求输出排列不重复

示例:
输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]
*/

func permuteOfString(s string) []string {
	runes := []rune(s)
	sort.Slice(runes, func(i, j int) bool {
		return runes[i] < runes[j]
	})
	n := len(runes)
	res := []string{}
	path := make([]rune, 0, n)
	visited := make(map[int]bool, n)
	var backTrack func()
	backTrack = func() {
		if len(path) == n {
			res = append(res, string(path))
			return
		}
		for i := 0; i < n; i++ {
			if visited[i] {
				continue
			}
			if i > 0 && runes[i] == runes[i-1] && visited[i-1] {
				continue
			}
			visited[i] = true
			path = append(path, runes[i])
			backTrack()
			visited[i] = false
			path = path[:len(path)-1]
		}
	}
	backTrack()
	return res
}
