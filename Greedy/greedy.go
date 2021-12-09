package Greedy

import "sort"

/*
token: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
username: 2437341470@qq.com
github token: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
url = https://github.com/xrdy511623/AlgorithmPractise.git
*/

/*
修改1.12 完全平方数
初看此问题感觉无从下手，其实这就是一个完全背包问题，物品就是完全平方数，背包就是目标值n, 翻译过来就是最少需要多少个物品(完全平方数)能够凑够
背包的容量n，由于你可以重复使用完全平方数(物品),所以这实际上是一个完全背包问题。

1 确定dp数组及其下标含义
dp[j]表示和为j所需要的最少完全平方数个数

2 确定递推公式
dp[j]明显只能由dp[j-i^2]推出，要凑够和为j-i*i所需的最少完全平方数为dp[j-i^2],那么只要再来一个完全平方数i^2,就能得到j,所以
dp[j] = dp[j-i^2] + 1，由于是求最小值，所以dp[j] = min(dp[j], dp[j-i^2]+1)

3 初始化dp数组
dp[0] = 0, 这个完全是为了递推公式，非0下标的值应该初始化为最大整数，以便遍历时可以被最小值迭代替换掉。

4 确定遍历顺序，完全背包，且本题是求最小数，不是求排列，也不睡求组合，所以可以外层遍历物品，内层遍历背包，或者反过来都行，但是完全背包
遍历背包必须是正序遍历
*/

// medium增加单词拆分

/*
1.13 单词拆分
给定一个非空字符串s和一个包含非空单词的列表wordDict，判定s是否可以被空格拆分为一个或多个在字典中出现的单词。

说明：
拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。

示例1：
输入: s = "leetcode", wordDict = ["leet", "code"] 输出: true 解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。

示例2：
输入: s = "applepenapple", wordDict = ["apple", "pen"] 输出: true 解释: 返回 true 因为 "applepenapple" 可以被拆分成
"apple pen apple"。  注意你可以重复使用字典中的单词。

示例3：
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"] 输出: false
*/

/*
思路:动态规划
初看此题感觉无从下手，但其实本题是完全背包问题。
首先将本题转化为背包问题单词就是物品，字符串s就是背包，单词列表wordDict中的单词就是物品，单词能否组成字符串s，就是问物品能不能把背包装满。
由于可以重复使用字典中的单词，说明这是一个完全背包问题。

动规五部曲分析如下：

1 确定dp数组以及下标的含义
dp[i]表示字符串长度为i的话，dp[i]为true，表示可以拆分为一个或多个在字典中出现的单词。

2 确定递推公式
如果确定dp[j]是true，且 [j, i] 这个区间的子串出现在字典里，那么dp[i]一定是true。（j < i ）。
所以递推公式是如果([j, i]这个区间的子串出现在wordDict中并且dp[j]是true，那么dp[i] = true。

3 dp数组如何初始化
从递归公式中可以看出，dp[i] 的状态依靠 dp[j]是否为true，那么dp[0]就是递归的根基，dp[0]一定要为true，否则递归下去后面都都是false了。
那么dp[0]初始为true完全就是为了推导公式，不必深究。

下标非0的dp[i]初始化为false，只要没有被覆盖说明都是不可拆分为一个或多个在字典中出现的单词。

4 确定遍历顺序
题目中说是拆分为一个或多个在字典中出现的单词，所以这是完全背包。
但本题有特殊性，因为是要求子串，最好是遍历背包放在外循环，将遍历物品放在内循环。

如果要是外层for循环遍历物品，内层for遍历背包，就需要把所有的子串都预先放在一个容器里。（如果不理解的话，可以自己尝试这么写一写就理解了）
所以最终的遍历顺序为：遍历背包放在外循环，将遍历物品放在内循环。因为是完全背包，所以内循环从前到后。

5 举例推导dp数组
略
*/

// WordBreak 时间复杂度O(N*N),空间复杂度O(N),N为字符串s的长度
func WordBreak(s string, wordDict []string) bool {
	n := len(s)
	// 记录wordDict中出现的单词
	hashTable := make(map[string]bool)
	for _, word := range wordDict {
		hashTable[word] = true
	}
	dp := make([]bool, n+1)
	dp[0] = true
	for i := 1; i <= n; i++ {
		for j := 0; j < i; j++ {
			if dp[j] && hashTable[s[j:i]] {
				dp[i] = true
				break
			}
		}
	}
	return dp[n]
}

// 修改1.14 最小路径和
/*

if len(grid) == 0 || len(grid[0]) == 0{
   return 0
}
*/

// subSequence.go 增加1.4 最长公共子序列的注释
// 本题与1.3 最长重复子数组最大的不同是公共子序列不要求是连续的了,只需保持相对顺序即可

// 开始攻击贪心

/*
1.1 发放饼干
假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将
这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

示例1:
输入: g = [1,2,3], s = [1,1]
输出: 1 解释:你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。所以你
应该输出1。

示例2:
输入: g = [1,2], s = [1,2,3]
输出: 2
解释:你有两个孩子和三块小饼干，2个孩子的胃口值分别是1,2。你拥有的饼干数量和尺寸都足以让所有孩子满足。所以你应该输出2.

提示：
1 <= g.length <= 3 * 10^4
0 <= s.length <= 3 * 10^4
1 <= g[i], s[j] <= 2^31 - 1
*/

/*
思路:
为了满足更多的小孩，要尽量避免饼干尺寸的浪费。
大尺寸的饼干既可以满足胃口大的孩子也可以满足胃口小的孩子，那么就应该优先满足胃口大的。
这里的局部最优就是大饼干喂给胃口大的，充分利用饼干尺寸喂饱一个，全局最优就是喂饱尽可能多的小孩。
可以尝试使用贪心策略，先将饼干数组和小孩数组排序。
然后从后向前遍历小孩数组，用大饼干优先满足胃口大的，并统计满足小孩数量。
*/

// FindContentChildren 先满足大胃口的孩子
func FindContentChildren(g []int, s []int) int {
	sort.Ints(g)
	sort.Ints(s)
	index, count := len(s)-1, 0
	for i := len(g) - 1; i >= 0; i-- {
		if index >= 0 && g[i] <= s[index] {
			count++
			index--
		}
	}
	return count
}

// FindContentChildrenTwo 也可以先满足胃口小的孩子
func FindContentChildrenTwo(g []int, s []int) int {
	sort.Ints(g)
	sort.Ints(s)
	count, index := 0, 0
	for i := 0; i < len(s); i++ {
		if index < len(g) && s[i] >= g[index] {
			count++
			index++
		}
	}
	return count
}