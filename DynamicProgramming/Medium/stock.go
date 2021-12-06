package Medium

import "AlgorithmPractise/Utils"

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
 */

func MaxProfit(prices []int)int{
	n := len(prices)
	if n == 0{
		return 0
	}
	dp := make([][]int, len(prices))
	for i:=0;i<n;i++{
		dp[i] = make([]int, 2)
	}
	dp[0][0] = -prices[0]
	dp[0][1] = 0
	for i:=1;i<n;i++{
		dp[i][0] = Utils.Max(dp[i-1][0], -prices[i])
		dp[i][1] = Utils.Max(dp[i-1][1], dp[i-1][0]+prices[i])
	}
	return dp[n-1][1]
}