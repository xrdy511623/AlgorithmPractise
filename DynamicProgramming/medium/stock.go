package medium

/*
medium contains middle level problems
*/

import  "AlgorithmPractise/Utils"


/*
1.1 leetcode 121 买卖股票的最佳时机
给定一个数组prices ，它的第i个元素prices[i]表示一支给定股票第 i 天的价格。
你只能选择某一天买入这只股票，并选择在未来的某一个不同的日子卖出该股票。设计一个算法来计算你所能获取的最大利润。
返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

示例1：
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。

示例2：
输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 没有交易完成, 所以最大利润为 0。
*/

/*
本题之前用贪心算法做过，这次用动态规划的思路来解决。

1 确定dp数组以及下标含义
dp[i][0]表示第i天持有股票所得最大现金，一开始现金为0，那么第i天买入买入股票所得现金就是-prices[i]，这是一个负数

2 确定递推公式
如果第i天持有股票即dp[i][0]，那么可以由两个状态推出来:
第i-1天持有股票，那就是保持现状，所得现金就是第i-1天，也就是昨天持有股票的所得现金，即dp[i-1][0]
第i-1天不持有股票，第i天买入股票，所得现金就是-prices[i]
所以dp[i][0] = max(dp[i-1][0], -prices[i])

如果第i天不持有股票所得最大现金，也可以由两个状态推出来
第i-1天就不持有股票，那么就保持现状，所得现金就是第i-1天也就是昨天不持有股票所得现金，即dp[i-1][1]
第i-1天持有股票，第i天不持有股票，也就是第i天卖出，所得现金即为昨天不持有股票所得现金+今天卖出后所得现金，
即为dp[i-1][0]+prices[i]
所以dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])

3 初始化dp数组
由递推公式 dp[i][0] = max(dp[i-1][0], -prices[i]); 和 dp[i][1] = max(dp[i-1][1], prices[i] + dp[i-1][0]);
可以看出其基础都是要从dp[0][0]和dp[0][1]推导出来。

那么dp[0][0]表示第0天持有股票，此时的持有股票就一定是买入股票了，因为不可能有前一天推出来，所以dp[0][0] = -prices[0];
dp[0][1]表示第0天不持有股票，不持有股票那么现金就是0，所以dp[0][1] = 0;

4确定遍历顺序
从递推公式可以看出dp[i]都是由dp[i - 1]推导出来的，那么一定是从前向后遍历。

5 举例推导dp数组
以示例1，输入：[7,1,5,3,6,4]为例，dp数组状态如下：
下标  dp[i][0]  dp[i][1]
0    -7        0
1    -1        0
2    -1        4
3    -1        4
4    -1        5
5    -1        5

dp[5][1]就是最终结果
为什么不是dp[5][0]呢？
因为本题中不持有股票状态所得金钱一定比持有股票状态得到的多！
*/

func maxProfit(prices []int) int {
	n := len(prices)
	if n == 0 {
		return 0
	}
	dp := make([][]int, len(prices))
	for i := 0; i < n; i++ {
		dp[i] = make([]int, 2)
	}
	dp[0][0] = -prices[0]
	dp[0][1] = 0
	for i := 1; i < n; i++ {
		dp[i][0] = Utils.Max(dp[i-1][0], -prices[i])
		dp[i][1] = Utils.Max(dp[i-1][1], dp[i-1][0]+prices[i])
	}
	return dp[n-1][1]
}

// MaxProfitSimple 贪心的解法
func MaxProfitSimple(prices []int) int {
	n := len(prices)
	if n == 0 {
		return 0
	}
	minPrice, maxProfit := prices[0], 0
	for i := 1; i < len(prices); i++ {
		profit := prices[i] - minPrice
		if profit > 0 {
			maxProfit = Utils.Max(maxProfit, profit)
		}
		minPrice = Utils.Min(minPrice, prices[i])
	}
	return maxProfit
}

/*
1.2 leetcode 122 买卖股票的最佳时机
给定一个数组，它的第i个元素是一支给定股票第i天的价格。
设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

示例 1:
输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4。随后，
在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。

示例 2:
输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。

示例 3:
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

提示：
1 <= prices.length <= 3 * 10 ^ 4
0 <= prices[i] <= 10 ^ 4
*/

/*
思路:
1 确定dp数组以及下标含义
dp[i][0]表示第i天持有股票所得最大现金，一开始现金为0，那么第i天买入买入股票所得现金就是-prices[i]，这是一个负数

2 确定递推公式
如果第i天持有股票即dp[i][0]，那么可以由两个状态推出来:
第i-1天就持有股票，那就是保持现状，所得现金就是第i-1天，也就是昨天持有股票的所得现金，即dp[i-1][0]
第i-1天不持有股票，第i天买入股票，所得现金就是昨天不持有股票的所得减去今天的股票价格，即dp[i-1][1]-prices[i]
注意这里和上题唯一不同的地方，就是推导dp[i][0]的时候，第i天买入股票的情况。
在上一道题，因为股票全程只能买卖一次，所以如果买入股票，那么第i天持有股票即dp[i][0]一定就是 -prices[i]。
而本题，因为一只股票可以买卖多次，所以当第i天买入股票的时候，所持有的现金可能有之前买卖过的利润。
那么第i天持有股票即dp[i][0]，如果是第i天买入股票，所得现金就是昨天不持有股票的所得现金减去今天的股票价格 即：
dp[i-1][1]-prices[i]。
所以dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[i])

如果第i天不持有股票所得最大现金，也可以由两个状态推出来
第i-1天就不持有股票，那么就保持现状，所得现金就是第i-1天也就是昨天不持有股票所得现金，即dp[i-1][1]
第i-1天持有股票，第i天不持有股票，也就是第i天卖出，所得现金即为昨天不持有股票所得现金+今天卖出后所得现金，
即为dp[i-1][0]+prices[i]
所以dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])

3 初始化dp数组
由递推公式 dp[i][0] = max(dp[i-1][0], -prices[i]); 和 dp[i][1] = max(dp[i-1][1], prices[i] + dp[i-1][0]);
可以看出其基础都是要从dp[0][0]和dp[0][1]推导出来。

那么dp[0][0]表示第0天持有股票，此时的持有股票就一定是买入股票了，因为不可能有前一天推出来，所以dp[0][0] = -prices[0];
dp[0][1]表示第0天不持有股票，不持有股票那么现金就是0，所以dp[0][1] = 0;

4 确定遍历顺序
从递推公式可以看出dp[i]都是由dp[i - 1]推导出来的，那么一定是从前向后遍历。

5 举例推导dp数组
以示例1，输入：[7,1,5,3,6,4]为例，dp数组状态如下：
下标  dp[i][0]  dp[i][1]
0    -7        0
1    -1        0
2    -1        4
3     1        4
4     1        7
5     3        7
*/

// maxProfitOne 时间复杂度O(N),空间复杂度O(2*N)
func maxProfitOne(prices []int) int {
	n := len(prices)
	if n == 0 {
		return 0
	}
	dp := make([][]int, len(prices))
	for i := 0; i < n; i++ {
		dp[i] = make([]int, 2)
	}
	dp[0][0] = -prices[0]
	dp[0][1] = 0
	for i := 1; i < n; i++ {
		dp[i][0] = Utils.Max(dp[i-1][0], dp[i-1][1]-prices[i])
		dp[i][1] = Utils.Max(dp[i-1][1], dp[i-1][0]+prices[i])
	}
	return dp[n-1][1]
}

// maxProfitOneSimple 更简单的写法 时间复杂度O(N), 空间复杂度O(1)
func maxProfitOneSimple(prices []int) int {
	n := len(prices)
	if n == 0 {
		return 0
	}
	sell, buy := 0, -prices[0]
	for i := 1; i < n; i++ {
		sell = Utils.Max(sell, buy+prices[i])
		buy = Utils.Max(buy, sell-prices[i])
	}
	return sell
}

/*
1.3 leetcode 123 买卖股票的最佳时机III
给定一个数组，它的第i个元素是一支给定的股票在第i天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成两笔交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

示例1:
输入：prices = [3,3,5,0,0,3,1,4] 输出：6 解释：在第4天（股票价格 = 0）的时候买入，在第6天
（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。随后，在第7天（股票价格 = 1）的时候买入，
在第8天 （股票价格 = 4）的时候卖出,这笔交易所能获得利润 = 4-1 = 3。

示例 2：
输入：prices = [1,2,3,4,5] 输出：4 解释：在第1天（股票价格 = 1）的时候买入，在第5天（股票价格 = 5）的
时候卖出,这笔交易所能获得利润 = 5-1 = 4。注意你不能在第1天和第2天接连购买股票，之后再将它们卖出。因为这样属
于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。

示例 3：
输入：prices = [7,6,4,3,1] 输出：0 解释：在这个情况下, 没有交易完成, 所以最大利润为0。

示例 4：
输入：prices = [1] 输出：0

提示：
1 <= prices.length <= 10^5
0 <= prices[i] <= 10^5
*/

/*
思路:
本题相对前两题复杂了不少
关键在于最多买卖两次，这意味着可以买卖一次，可以买卖两次，也可以不买卖。
所以每一天的状态一共有五种

1 确定dp数组以及下标含义
dp[i][j]表示第i天第j种状态时所得最大现金

2 确定递推公式
首先讨论dp[i][1]表示第i天，(第一次)买入股票所得最大现金，它可以由以下两种情况推出
第i-1天第一次买入股票，第i天不操作，维持买入状态，此时所得现金为dp[i-1][1]
第i-1天未操作，第i天第一次买入股票，此时所得现金为dp[i-1][0]-prices[i]

接着讨论dp[i][2]表示第i天，(第一次)卖出股票所得最大现金，它可以由以下两种情况推出
第i-1天第一次卖出股票，第i天不操作，维持卖出状态，此时所得现金为dp[i-1][2]
第i-1天为(第一次)买入状态，第i天第一次卖出股票，此时所得现金为dp[i-1][1]+prices[i]

接下来讨论dp[i][3]表示第i天，(第二次)买入股票所得最大现金，它可以由以下两种情况推出
第i-1天第二次买入股票，第i天不操作，维持买入状态，此时所得现金为dp[i-1][3]
第i-1天为(第一次)卖出状态，第i天第二次买入股票，此时所得现金为dp[i-1][2]-prices[i]

最后讨论dp[i][4]表示第i天，(第二次)卖出股票所得最大现金，它可以由以下两种情况推出
第i-1天第二次卖出股票，第i天不操作，维持卖出状态，此时所得现金为dp[i-1][4]
第i-1天为(第二次)买入状态，第i天第二次卖出股票，此时所得现金为dp[i-1][3]+prices[i]

3 初始化dp数组
dp[0][0] = 0
dp[0][1] = -prices[0]
dp[0][2] = 0
// 此时不存在第二次买入或卖出，但是为了初始化仍按照第一次买入和卖出赋值
dp[0][3] = -prices[0]
dp[0][4] = 0

4 确定遍历顺序
从递推公式可以看出dp[i]都是由dp[i - 1]推导出来的，那么一定是从前向后遍历。

5 举例推导dp数组
略
*/

// maxProfitTwo 时间复杂度O(N),空间复杂度O(5*N)
func maxProfitTwo(prices []int) int {
	n := len(prices)
	if n == 0 {
		return 0
	}
	dp := make([][]int, len(prices))
	for i := 0; i < n; i++ {
		dp[i] = make([]int, 5)
	}
	dp[0][0] = 0
	dp[0][1] = -prices[0]
	dp[0][2] = 0
	dp[0][3] = -prices[0]
	dp[0][4] = 0
	for i := 1; i < n; i++ {
		dp[i][1] = Utils.Max(dp[i-1][1], dp[i-1][0]-prices[i])
		dp[i][2] = Utils.Max(dp[i-1][2], dp[i-1][1]+prices[i])
		dp[i][3] = Utils.Max(dp[i-1][3], dp[i-1][2]-prices[i])
		dp[i][4] = Utils.Max(dp[i-1][4], dp[i-1][3]+prices[i])
	}
	return dp[n-1][4]
}

// maxProfitTwoSimple 更简单的写法 时间复杂度O(N),空间复杂度O(1)
func maxProfitTwoSimple(prices []int) int {
	n := len(prices)
	if n == 0 {
		return 0
	}
	buy1, sell1 := -prices[0], 0
	buy2, sell2 := -prices[0], 0
	for i := 1; i < n; i++ {
		buy1 = Utils.Max(buy1, -prices[i])
		sell1 = Utils.Max(sell1, buy1+prices[i])
		buy2 = Utils.Max(buy2, sell1-prices[i])
		sell2 = Utils.Max(sell2, buy2+prices[i])
	}
	return sell2
}

/*
1.4 leetcode 188 买卖股票的最佳时机IV
给定一个整数数组 prices ，它的第i个元素 prices[i] 是一支给定的股票在第i天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成k笔交易。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

示例1：
输入：k = 2, prices = [2,4,1] 输出：2 解释：在第1天 (股票价格 = 2) 的时候买入，在第2天 (股票价格 = 4) 的时候卖出，
这笔交易所能获得利润 = 4-2 = 2。

示例2：
输入：k = 2, prices = [3,2,6,5,0,3] 输出：7 解释：在第2天 (股票价格 = 2) 的时候买入，在第3天 (股票价格 = 6) 的
时候卖出, 这笔交易所能获得利润 = 6-2 = 4。随后，在第5天 (股票价格 = 0) 的时候买入，在第6天 (股票价格 = 3) 的时候卖出,
这笔交易所能获得利润 = 3-0 = 3 。

提示：
0 <= k <= 100
0 <= prices.length <= 1000
0 <= prices[i] <= 1000
*/

/*
思路:
本题在上一题的基础上又增加了难度，这次是求完成k笔交易的通用解。

1 确定dp数组及其下标含义
使用二维数组dp[i][j] ：第i天的状态为j，所获得的最大现金为dp[i][j]
j的状态表示为：
0 表示不操作
1 第一次买入
2 第一次卖出
3 第二次买入
4 第二次卖出
......

从上面可以很容易的发现一个规律,除了0以外，奇数代表买入，偶数代表卖出。
题目中要求是最多有k笔交易，那么一共有2*k+1种状态，j的范围就可以定义为[0, 2*k+1]

2 确定递推公式
当j为奇数，也就是此时为买入状态时，dp[i][j]可以由以下两种情况推出
第i-1天买入股票，第i天不操作，维持买入状态，此时所得现金为dp[i-1][j]
第i-1天卖出状态，第i天买入股票，此时所得现金为dp[i-1][j-1]-prices[i]

所以当j%2==1时,递推公式为:
dp[i][j] = max(dp[i-1][j], dp[i-1][j-1]-prices[i])

当j为偶数，也就是此时为卖出状态时，dp[i][j]可以由以下两种情况推出
第i-1天卖出股票，第i天不操作，维持卖出状态，此时所得现金为dp[i-1][j]
第i-1天买入状态，第i天卖出股票，此时所得现金为dp[i-1][j-1]+prices[i]

所以当j%2==0时,递推公式为:
dp[i][j] = max(dp[i-1][j], dp[i-1][j-1]+prices[i])

3 初始化dp数组
第0天没有操作，这个最容易想到，就是0，即：dp[0][0] = 0;
第0天无论第几次买入，也就是j为奇数时，dp[0][j]=-prices[i]
第0天无论第几次卖出，也就是j为偶数时，dp[0][j] = 0

4 确定遍历顺序
从递归公式其实已经可以看出，一定是从前向后遍历，因为dp[i]是由依靠dp[i - 1]推出的。

5 举例推导dp数组
略
*/

// maxProfitK 时间复杂度O((2*K+1)*N+K+N)，空间复杂度O((2*K+1)*N)
func maxProfitK(prices []int, k int) int {
	n := len(prices)
	if n == 0 {
		return 0
	}
	dp := make([][]int, n)
	// 可以买卖k次，则一共有2*k+1种情况
	for i := 0; i < n; i++ {
		dp[i] = make([]int, 2*k+1)
	}
	// 奇数代表买入，第一天无论是第几次买入，所获得的最大价值都是-prices[0]
	for i := 1; i < 2*k+1; i += 2 {
		dp[0][i] = -prices[0]
	}
	for i := 1; i < n; i++ {
		for j := 1; j < 2*k+1; j++ {
			// j为奇数，代表买入
			if j%2 == 1 {
				dp[i][j] = Utils.Max(dp[i-1][j], dp[i-1][j-1]-prices[i])
			} else {
				// j为偶数，代表卖出
				dp[i][j] = Utils.Max(dp[i-1][j], dp[i-1][j-1]+prices[i])
			}
		}
	}
	return dp[n-1][2*k]
}

/*
1.5 leetcode 309 买卖股票的最佳时机含冷冻期
给定一个整数数组，其中第i个元素代表了第i天的股票价格 。
设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
卖出股票后，你无法在第二天买入股票 (即冷冻期为1天)。
示例: 输入: [1,2,3,0,2] 输出: 3 解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
*/

/*
思路:

1 确定dp数组及其下标含义
dp[i][j]表示第i天结束之后的累计最大收益，根据题意，由于我们最多只能同时买入(持有)一支股票，并且卖出股票后有冷冻期的限制，
因此我们会有三种不同的状态(j的取值范围为[0,2]):
我们目前持有一支股票，对应的累计最大收益为dp[i][0]
我们目前不持有股票，并且处于冷冻期，对应的累计最大收益为dp[i][1]
我们目前不持有股票，且不处于冷冻期，对应的累计最大收益为dp[i][2]

这里的「处于冷冻期」指的是在第i天结束之后的状态。也就是说：如果第i天结束之后处于冷冻期，那么第i+1天无法买入股票。

2 确定递推公式(状态转移方程)
首先讨论dp[i][0],它可以由以下两种情况推出
第i-1天持有股票，第i天不操作，保持持有状态，此时dp[i][0] = dp[i-1][0]
第i天持有(买入)股票，那么第i-1天就不能持有股票且不处于冷冻期中，此时dp[i][0]=dp[i-1][2]-prices[i]
所以dp[i][0] = max(dp[i-1][0],dp[i-1][2]-prices[i])

接下来讨论dp[i][1],它可以由以下一种情况推出
既然第i+1天无法买入股票，说明正好第i天卖出了股票，则第i-1天是持有股票的，则有dp[i][1]=dp[i-1][0]+prices[i]

最后讨论dp[i][2]，它可以由以下两种情况推出
第i-1天不持有(卖出)股票，且第i-1天处于冷冻期，此时dp[i][2] = dp[i-1][1]；另一种情况是第i-1天不处于冷冻期，
此时dp[i][2] = dp[i-1][2]
所以dp[i][2] = max(dp[i-1][1],dp[i-1][2])

这样我们就得到了所有的递推公式，如果一共有n天，那么最终的答案为max(dp[n-1][0],dp[n-1][1],dp[n-1][2])
注意到如果在最后一天（第 n-1）结束之后，手上仍然持有股票，那么显然是没有任何意义的。因此更加精确地，最终的答案实际上是
dp[n-1][1],dp[n-1][2]中的较大值，即：max(dp[n-1][1],dp[n-1][2])

3 dp数组初始化
dp[0][0] = -prices[0]
dp[0][1] = 0
dp[0][0] = 0

4 确定遍历顺序
从递归公式其实已经可以看出，一定是从前向后遍历，因为dp[i]是由依靠dp[i - 1]推出的。

5 举例推导dp数组
略
*/

func maxProfitIncludeFreeze(prices []int) int {
	n := len(prices)
	if n == 0 {
		return 0
	}
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, 3)
	}
	dp[0][0] = -prices[0]
	for i := 1; i < n; i++ {
		dp[i][0] = Utils.Max(dp[i-1][0], dp[i-1][2]-prices[i])
		dp[i][1] = dp[i-1][0] + prices[i]
		dp[i][2] = Utils.Max(dp[i-1][1], dp[i-1][2])
	}
	return Utils.Max(dp[n-1][1], dp[n-1][2])
}

/*
第二种思路:将每天的状态划分为四种其实更好理解，如果只划分为以上三种其实比较容易绕进去。
1 确定dp数组及其下标含义
dp[i][j]表示在第i天处于状态j时所获取的最大收益为dp[i][j]
本次我们将每天的状态划分为四种，其实严格来讲是五种，不操作也是一种，但是不操作每天的最大收益都是0，就不讨论了。
状态0为持有股票状态；
状态1为卖出股票且已过冷冻期状态；
状态2为今天卖出股票；
状态3为今天是冷冻期；

2 确定递归公式
首先看状态0，即持有股票状态，它明显可以由以下三种状态推出:
a 第i-1天就是持有股票状态(状态0)，今天也就是第i天继续保持此种状态，即dp[i][0]=dp[i-1][0]
b 第i-1天是卖出股票且已过冷冻期状态(状态1)，今天买入，那么dp[i][0]=dp[i-1][1]-prices[i]
c 第i-1天冷冻期状态(状态3)，今天买入，即dp[i][0]=dp[i-1][3]-prices[i]
要求最大收益，所以有:
dp[i][0]=max(dp[i-1][0], max(dp[i-1][1],dp[i-1][3])-prices[i])

接下来看状态1，即卖出股票且已过冷冻期状态，它明显可以由以下两种状态推出:
a 第i-1天就是卖出股票且已过冷冻期状态(状态1)，今天也就是第i天继续保持此种状态，即dp[i][1]=dp[i-1][1]
b 第i-1天是冷冻期状态(状态3)，今天正好就是卖出股票且已过冷冻期状态，那么dp[i][1]=dp[i-1][3]
要求最大收益，所以有:
dp[i][1]=max(dp[i-1][1], dp[i-1][3])

再看状态2，即今天卖出股票，它明显只能由以下一种状态推出:
那就是第i-1天是持有股票状态(状态0)，今天也就是第i天卖出，所以有dp[i][2]=dp[i-1][0]+prices[i]

最后看状态3，即今天是冷冻期状态，它明显只能由以下一种状态推出:
那就是昨天，也就是第i-1天是刚刚卖出股票状态，所以有dp[i][3]=dp[i-1][2]

3 dp数组初始化
明显有dp[0][0]=-prices[0];dp[0][1]=0;dp[0][2]=0;dp[0][3]=0

4 确定遍历顺序
由递推公式可知，第1天状态由第i-1天推出，所以必然是从前向后遍历。

5 举例推到dp数组
略
*/

func maxProfitIncludeFreezePeriod(prices []int) int {
	n := len(prices)
	if n == 0 {
		return 0
	}
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, 4)
	}
	dp[0][0] = -prices[0]
	for i := 1; i < n; i++ {
		dp[i][0] = Utils.Max(dp[i-1][0], Utils.Max(dp[i-1][1], dp[i-1][3])-prices[i])
		dp[i][1] = Utils.Max(dp[i-1][1], dp[i-1][3])
		dp[i][2] = dp[i-1][0] + prices[i]
		dp[i][3] = dp[i-1][2]
	}
	return Utils.Max(dp[n-1][1], Utils.Max(dp[n-1][2], dp[n-1][3]))
}

/*
1.6 leetcode714 买卖股票的最佳时机含手续费
给定一个整数数组prices，其中第i个元素代表了第i天的股票价格；非负整数fee代表了交易股票的手续费用。
你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。
返回获得利润的最大值。

注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要支付一次手续费。

示例 1:
输入: prices = [1, 3, 2, 8, 4, 9], fee = 2 输出: 8

解释: 能够达到的最大利润: 在此处买入 prices[0] = 1 在此处卖出 prices[3] = 8 在此处买入 prices[4] = 4
在此处卖出 prices[5] = 9 总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8.

注意:
0 < prices.length <= 50000.
0 < prices[i] < 50000.
0 <= fee < 50000.
*/

/*
思路:
本题跟1.2相比，没有任何变化，只是多了一个手续费，所以解决起来异常容易
*/

func maxProfitIncludeFee(prices []int, fee int) int {
	n := len(prices)
	if n == 0 {
		return 0
	}
	dp := make([][]int, len(prices))
	for i := 0; i < n; i++ {
		dp[i] = make([]int, 2)
	}
	dp[0][0] = -prices[0]
	for i := 1; i < n; i++ {
		dp[i][0] = Utils.Max(dp[i-1][0], dp[i-1][1]-prices[i])
		dp[i][1] = Utils.Max(dp[i-1][1], dp[i-1][0]+prices[i]-fee)
	}
	return dp[n-1][1]
}

func maxProfitIncludeFeeSimple(prices []int, fee int) int {
	n := len(prices)
	if n == 0 {
		return 0
	}
	sell, buy := 0, -prices[0]
	for i := 1; i < n; i++ {
		sell = Utils.Max(sell, buy+prices[i]-fee)
		buy = Utils.Max(buy, sell-prices[i])
	}
	return sell
}