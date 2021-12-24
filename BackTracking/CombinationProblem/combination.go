package CombinationProblem

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

/*
回溯算法其实是暴力搜索，要降低算法时间复杂度，必须尽可能的剪枝，也就是尽可能的避免不必要的遍历
本题剪枝的关键是探讨搜索起点start的最大值，也就是上限。
例如：n = 6 ，k = 4。
path.size() == 1 时，接下来要选择3个数，搜索起点最大是4，最后一个被选的组合是 [4, 5, 6]；
path.size() == 2 时，接下来要选择2个数，搜索起点最大是5，最后一个被选的组合是 [5, 6]；
path.size() == 3 时，接下来要选择1个数，搜索起点最大是6，最后一个被选的组合是 [6]；

再如：n = 15 ，k = 4。
path.size() == 1 时，接下来要选择3个数，搜索起点最大是13，最后一个被选的是 [13, 14, 15]；
path.size() == 2 时，接下来要选择2个数，搜索起点最大是14，最后一个被选的是 [14, 15]；
path.size() == 3 时，接下来要选择1个数，搜索起点最大是15，最后一个被选的是 [15]；

可以归纳出：
搜索起点的上限 + 接下来要选择的元素个数 - 1 = n
而接下来要选择的元素个数 = k - len(path).
所以有: 搜索起点的上限 = n - (k - len(path)) + 1
所以，我们的剪枝过程就是：把 i <= n 改成 i <= n - (k - len(path)) + 1
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


/*
1.2 组合总和III
找出所有相加之和为n的k个数的组合。组合中只允许含有1-9的正整数，并且每种组合中不存在重复的数字。

说明：
所有数字都是正整数。
解集不能包含重复的组合。

示例1:
输入: k = 3, n = 7 输出: [[1,2,4]]

示例2:
输入: k = 3, n = 9 输出: [[1,2,6], [1,3,5], [2,3,4]]
 */

/*
本题与1.1 组合本质上并无不同，仍然是找k个数的组合，只是多了一个(组合)和为n的限制，而且组合中的元素题目已经
限定是[1,9]范围内的正整数，所以解决起来是很容易的。搜索起点start的上限与上题也是一样的。
 */

// CombinationSumThird 回溯法解决
func CombinationSumThird(k int, n int) [][]int {
	var res [][]int
	var path []int
	var backTrack func(int, int)
	backTrack = func(sum, start int){
		// 剪枝，如果当前路径和已经大于n,后续遍历就没有意义了
		if sum > n{
			return
		}
		// 递归终止条件
		// 如果当前路径长度为k,且路径和等于n，便找到了一条满足要求的路径
		if len(path) == k && sum == n{
			temp := make([]int, k)
			copy(temp, path)
			res = append(res, temp)
			return
		}
		// for循环遍历(剪枝)
		for i:=start;i<=9-(k-len(path))+1;i++{
			// 处理每一个节点
			sum += i
			path = append(path, i)
			// 递归
			backTrack(sum, i+1)
			// 回溯
			sum -= i
			path = path[:len(path)-1]
		}
	}
	backTrack(0, 1)
	return res
}