package SubSetsProblem

/*
1.1 子集
给定一组不含重复元素的整数数组nums，返回该数组所有可能的子集（幂集）。
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
	var backTrack func(int)
	backTrack = func(start int) {
		// 递归终止条件，当start移动到数组nums末尾元素时
		if start > len(nums) {
			return
		}
		temp := make([]int, len(path))
		copy(temp, path)
		res = append(res, temp)
		for i := start; i < len(nums); i++ {
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