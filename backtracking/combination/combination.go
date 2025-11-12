package combination

import (
	"math"
	"sort"

	"algorithmpractise/utils"
)

/*
回溯算法题目的解题套路就是:
for循环横向遍历，递归纵向遍历，回溯不断调整结果集。
*/

/*
leetcode 77. 组合
1.1 给定两个整数n和k，返回范围 [1, n]中所有可能的k个数的组合。
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
回溯算法其实是暴力搜索，要降低算法时间复杂度，必须尽可能的剪枝，也就是尽可能避免不必要的遍历
本题剪枝的关键是探讨搜索起点start的最大值，也就是上限。
例如：n = 6，k = 4。
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
	var backTrack func(int)
	backTrack = func(start int) {
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
			// 递归，因为是组合，元素不能重复，所以下一层递归从i+1开始
			backTrack(i + 1)
			// 回溯
			path = path[:len(path)-1]
		}
	}
	backTrack(1)
	return res
}

/*
leetcode 216. 组合总和III
1.2 找出所有相加之和为n的k个数的组合。组合中只允许含有1-9的正整数，并且每种组合中不存在重复的数字。

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
	if k >= n {
		return [][]int{}
	}
	var res [][]int
	var path []int
	sum := 0
	var backTrack func(int)
	backTrack = func(start int) {
		// 剪枝，如果当前路径长度已经超过k或当前路径和已经大于n,后续遍历就没有意义了
		if len(path) > k || sum > n {
			return
		}
		// 递归终止条件
		// 如果当前路径长度为k,且路径和等于n，便找到了一条满足要求的路径
		if len(path) == k && sum == n {
			temp := make([]int, k)
			copy(temp, path)
			res = append(res, temp)
			return
		}
		// for循环遍历(剪枝)
		for i := start; i <= 9-(k-len(path))+1; i++ {
			// 处理每一个节点
			sum += i
			path = append(path, i)
			// 递归
			backTrack(i + 1)
			// 回溯
			sum -= i
			path = path[:len(path)-1]
		}
	}
	backTrack(1)
	return res
}

/*
leetcode 17. 电话号码的字母组合
1.3 给定一个仅包含数字2-9的字符串，返回所有它能表示的字母组合。答案可以按任意顺序返回。
给出数字到字母的映射如下（与电话按键相同）。注意1不对应任何字母。

示例1：
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]

示例2：
输入：digits = ""
输出：[]

示例3：
输入：digits = "2"
输出：["a","b","c"]

提示：
0 <= digits.length <= 4
digits[i] 是范围 ['2', '9'] 的一个数字。
*/

/*
因为题目中每一个数字(电话号码)代表的是不同集合，所以本题其实是求不同集合之间的组合，不像前面两题是求同一个
集合中的组合，但大体思路是没差的。
*/

// LetterCombinations 回溯算法解决
func LetterCombinations(digits string) []string {
	res := []string{}
	length := len(digits)
	if length == 0 {
		return res
	}
	// 建立电话号码与字符串的映射关系
	m := make(map[byte]string)
	m['2'] = "abc"
	m['3'] = "def"
	m['4'] = "ghi"
	m['5'] = "jkl"
	m['6'] = "mno"
	m['7'] = "pqrs"
	m['8'] = "tuv"
	m['9'] = "wxyz"
	path := []byte{}
	var backTrack func(int)
	backTrack = func(index int) {
		// 递归终止条件
		// 当拼接的临时字符串temp长度等于digits长度时，表明已经找到了一个符合条件的字符串
		if len(path) == length {
			temp := make([]byte, length)
			copy(temp, path)
			res = append(res, string(temp))
			return
		}
		// 找到digits[index]对应数字所代表的字符串
		letter := m[digits[index]]
		for i := 0; i < len(letter); i++ {
			// for循环遍历letters，拼接temp字符串
			path = append(path, letter[i])
			// 递归，从index+1下标继续拼接
			backTrack(index + 1)
			// 回溯
			path = path[:len(path)-1]
		}
	}
	// 从digits的0下标开始处理
	backTrack(0)
	return res
}

/*
leetcode 39. 组合总和
1.4 给定一个无重复元素的数组candidates和一个目标数target ，找出candidates中所有可以使数字和为target的组合。
candidates中的数字可以无限制重复被选取。

说明：
所有数字（包括target）都是正整数。
解集不能包含重复的组合。

示例1：
输入：candidates = [2,3,6,7], target = 7, 所求解集为：[[7], [2,2,3] ]

示例2：
输入：candidates = [2,3,5], target = 8, 所求解集为：[[2,2,2,2],[2,3,3],[3,5] ]
*/

/*
本题与前面题目唯一的区别在于集合中的元素可以无限制重复使用，解题思路大差不差，值得注意的是，如果题目改为
求组合数，那么这就是一个完全背包问题，求得的组合数肯定与本题求得的组合集合的长度相等。
*/

func CombinationSum(candidates []int, target int) [][]int {
	var res [][]int
	var path []int
	sum, n := 0, len(candidates)
	var backTrack func(int)
	backTrack = func(start int) {
		// 剪枝优化，如果sum已经大于目标和target，就可以返回了，继续遍历没有意义
		if sum > target {
			return
		}
		// 递归终止条件
		// 若当前路径和等于目标和target，将此路径添加到结果集中
		if sum == target {
			temp := make([]int, len(path))
			copy(temp, path)
			res = append(res, temp)
			return
		}
		for i := start; i < n; i++ {
			sum += candidates[i]
			path = append(path, candidates[i])
			// 因为candidates中的元素可以无限制重复选取，所以这里不再是i+1了
			// 递归
			backTrack(i)
			// 回溯
			sum -= candidates[i]
			path = path[:len(path)-1]
		}
	}
	backTrack(0)
	return res
}

// CompletePack 若是求组合数，可用此函数求解，结果可与上面的求得的组合集合的长度相互验证
func CompletePack(candidates []int, target int) int {
	dp := make([]int, target+1)
	dp[0] = 1
	for i := 0; i < len(candidates); i++ {
		for j := candidates[i]; j <= target; j++ {
			dp[j] += dp[j-candidates[i]]
		}
	}
	return dp[target]
}

/*
leetcode 40. 组合总和 II
1.5 给定一个数组candidates和一个目标数target，找出candidates中所有可以使数字和为target的组合。
candidates中的每个数字在每个组合中只能使用一次。
数组candidates中的元素可能有重复。

注意：解集不能包含重复的组合。

示例1:
输入: candidates = [10,1,2,7,6,1,5], target = 8,
输出:
[[1,1,6],[1,2,5],[1,7],[2,6]]

示例2:
输入: candidates =[2,5,2,1,2], target =5,
输出:
[[1,2,2],[5]]

提示:
1 <=candidates.length <= 100
1 <=candidates[i] <= 50
1 <= target <= 30
*/

/*
本题的难点在于集合（数组candidates）有重复元素，但还不能有重复的组合，所以去重特别关键。
所谓去重，其实就是使用过的元素不能重复选取。一个维度是同一树枝上使用过，一个维度是同一树层上使用过。
那么问题来了，我们是要同一树层上使用过，还是同一树枝上使用过呢？
回看一下题目，元素在同一个组合内是可以重复的，怎么重复都没事，但两个组合不能相同。
所以我们要去重的是同一树层上的“使用过”，同一树枝上的都是一个组合里的元素，不用去重。

在求和问题中，排序之后加剪枝是常见的套路！
*/

func CombinationSumTwo(candidates []int, target int) [][]int {
	var res [][]int
	var path []int
	sum := 0
	n := len(candidates)
	sort.Ints(candidates)
	var backTrack func(int)
	backTrack = func(start int) {
		// 剪枝
		if sum > target {
			return
		}
		// 递归终止条件
		if sum == target {
			temp := make([]int, len(path))
			copy(temp, path)
			res = append(res, temp)
		}
		for i := start; i < n; i++ {
			// 同一树层去重
			if i > start && candidates[i] == candidates[i-1] {
				continue
			}
			// 处理每一个元素
			sum += candidates[i]
			path = append(path, candidates[i])
			// 递归
			backTrack(i + 1)
			// 回溯
			sum -= candidates[i]
			path = path[:len(path)-1]
		}
	}
	backTrack(0)
	return res
}

/*
游戏分组
部门准备举办一场王者荣耀表演赛，有 10 名游戏爱好者参与，分为两队，每队 5 人。
每位参与者都有一个评分，代表着他的游戏水平。为了表演赛尽可能精彩，我们需要把 10 名参赛者分为实力尽量相近的两队。
一队的实力可以表示为这一队 5 名队员的评分总和。
现在给你 10 名参与者的游戏水平评分，请你根据上述要求分队，最后输出这两组的实力差绝对值。

输入描述
10 个整数，表示 10 名参与者的游戏水平评分。范围在 [1,10000] 之间。

输出描述
实力最相近两队的实力差绝对值。

示例1
输入：
1 2 3 4 5 6 7 8 9 10

输出：
1

说明：
10 名队员分为两组，两组实力差绝对值最小为 1
*/

func ArrangeGameGroup(scores []int) int {
	n := len(scores)
	length := n / 2
	group := []int{}
	curSum := 0
	// 计算数组和totalSum
	totalSum := 0
	for _, score := range scores {
		totalSum += score
	}
	minDiff := math.MaxInt32
	var backTrack func(int)
	backTrack = func(start int) {
		// 剪枝
		if start > n || len(group) > length {
			return
		}
		// 迭代最小差值
		if len(group) == length {
			// totalSum - curSum为另一分组的分数和，减去当前分组分数和，即为两个分组的实力差
			diff := utils.Abs(totalSum - curSum - curSum)
			if diff < minDiff {
				minDiff = diff
			}
		}
		for i := start; i < n; i++ {
			curSum += scores[i]
			group = append(group, scores[i])
			// 递归
			backTrack(i + 1)
			// 回溯
			curSum -= scores[i]
			group = group[:len(group)-1]
		}
	}
	backTrack(0)
	return minDiff
}
