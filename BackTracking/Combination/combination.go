package Combination

/*
1.1 组合
给定两个整数n和k，返回范围 [1, n]中所有可能的k个数的组合。
你可以按任何顺序返回答案。

示例1：
输入：n = 4, k = 2
输出：
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]

示例2：

输入：n = 1, k = 1
输出：[[1]]

提示：
1 <= n <= 20
1 <= k <= n
*/

// Combine 回溯
func Combine(n, k int) [][]int {
	var res [][]int
	if n <= 0 || k <= 0 || k > n {
		return res
	}
	var path []int
	var backTrack func(int, int, int)
	backTrack = func(n, k, start int) {
		// 剪枝：path长度加上区间 [start, n]的长度小于k，不可能构造出长度为k的path
		if len(path)+n-start+1 < k {
			return
		}
		// 递归终止条件
		// 如果已经找到了长度为k的path,便将path添加到结果集res中，结束递归
		if len(path) == k {
			temp := make([]int, k)
			copy(temp, path)
			res = append(res, temp)
			return
		}
		// for循环遍历
		for i := start; i <= n-(k-len(path))+1; i++ {
			// 处理每一个节点(元素)
			path = append(path, i)
			// 因为是组合，元素不能重复，所以下一层递归从i+1开始
			backTrack(n, k, i+1)
			// 回溯
			path = path[:len(path)-1]
		}
	}
	backTrack(n, k, 1)
	return res
}