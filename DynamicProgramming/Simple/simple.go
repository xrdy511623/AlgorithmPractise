package Simple

import "AlgorithmPractise/Utils"

/*
1.0 斐波拉契数
斐波那契数，通常用F(n) 表示，形成的序列称为斐波那契数列 。该数列由0和1开始，后面的每一项数字都是前面两项数字的和。也就是：

F(0)= 0，F(1)= 1
F(n) = F(n-1) + F(n-2)，其中 n > 1
给你n ，请计算 F(n) 。
 */

// Fib 递推公式题目已经给出:F(n) = F(n-1) + F(n-2). 时间复杂度O(N), 空间复杂度O(N)
func Fib(n int)int{
	if n <= 1{
		return n
	}
	dp := make([]int, n+1, n+1)
	dp[0] = 0
	dp[1] = 1
	for i:=2;i<=n;i++{
		dp[i] = dp[i-1]+dp[i-2]
	}
	return dp[n]
}

// 当然，上面的方法还有优化的空间，其实我们只需要维护两个数值就可以了，不需要记录整个序列
// 时间复杂度O(N), 空间复杂度O(1)

func FibSimple(n int)int{
	if n <= 1{
		return n
	}
	dp := make([]int, 2, 2)
	dp[0] = 0
	dp[1] = 1
	for i:=2;i<=n;i++{
		sum := dp[0]+dp[1]
		dp[0] = dp[1]
		dp[1] = sum
	}
	return dp[1]
}

// FibUseRecursion 当然，本题还可以用递归解决，但是会有大量重复计算，时间复杂度O(2^N), 空间复杂度O(N)
func FibUseRecursion(n int)int{
	if n < 2{
		return n
	}
	return FibUseRecursion(n-1)+FibUseRecursion(n-2)
}

// FibUseRecursionAndCache 要降低时间复杂度，需要使用缓存, 时间复杂度O(N), 空间复杂度O(N)
func FibUseRecursionAndCache(n int)int{
	cache := make(map[int]int, n+1)
	var helper func(int)int
	helper = func(n int)int{
		if v, ok := cache[n];ok{
			return v
		}
		if n < 2 {
			cache[n] = n
			return n
		}
		cache[n] = helper(n-1)+helper(n-2)
		return cache[n]
	}
	return helper(n)
}


/*
1.1 爬楼梯
假设你正在爬楼梯。需要n阶你才能到达楼顶。
每次你可以爬1或2个台阶。你有多少种不同的方法可以爬到楼顶呢？
注意：给定n是一个正整数。

示例 1：
输入：2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶

示例 2：
输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶
 */


/*
本题与求斐波拉契数原理相同，只是这里递推公式需要自己推导。dp[i]代表爬到第i层楼梯，有dp[i]种方法，那么问题的关键在于如何推导出
dp[i]呢？从dp[i]的定义可以看出，dp[i]可以由两个方向推出来。
首先是dp[i-1]，上i-1层楼梯，有dp[i-1]种方法，那么再一步跳一个台阶不就是dp[i]了么。
还有就是dp[i-2]，上i-2层楼梯，有dp[i-2]种方法，那么再一步跳两个台阶不就是dp[i]了么。
那么dp[i]就是 dp[i-1]与dp[i-2]之和！
所以递推公式就是dp[i]=dp[i-1]+dp[i-2]。
dp数组如何初始化？
dp[1]=1,dp[2]=2
 */

// ClimbStairs 时间复杂度O(N), 空间复杂度O(N)
func ClimbStairs(n int)int{
	if n <= 1{
		return n
	}
	dp := make([]int, n+1, n+1)
	dp[1] = 1
	dp[2] = 2
	for i:=3;i<=n;i++{
		dp[i] = dp[i-1]+dp[i-2]
	}
	return dp[n]
}

// ClimbStairsSimple 同样的，可以优化一下空间复杂度.时间复杂度O(N), 空间复杂度O(1)
func ClimbStairsSimple(n int)int{
	if n <= 1{
		return n
	}
	dp := make([]int, 2, 2)
	dp[0] = 1
	dp[1] = 2
	for i:=3;i<=n;i++{
		sum := dp[0]+dp[1]
		dp[0] = dp[1]
		dp[1] = sum
	}
	return dp[1]
}

/*
1.2 使用最小花费爬楼梯
数组的每个下标作为一个阶梯，第i个阶梯对应着一个非负数的体力花费值cost[i]（下标从0开始）。
每当你爬上一个阶梯你都要花费对应的体力值，一旦支付了相应的体力值，你就可以选择向上爬一个阶梯或者爬两个阶梯。
请你找出达到楼层顶部的最低花费。在开始时，你可以选择从下标为0或1的元素作为初始阶梯。

示例1：
输入：cost = [10, 15, 20]
输出：15
解释：最低花费是从 cost[1] 开始，然后走两步即可到阶梯顶，一共花费15。

示例 2：
输入：cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
输出：6
解释：最低花费方式是从cost[0]开始，逐个经过那些1 ，跳过cost[3]，一共花费6 。
 */

/*
1 确定dp数组以及下标含义
dp[i]的定义:到达第i个台阶所花费的最小体力为dp[i](注意这里认为是第一步一定是要花费)
2 确定递推公式
可以有两个途径得到dp[i],dp[i-1]和dp[i-2],那么究竟是选择dp[i-1]还是dp[i-2]呢？
从题意来看，一定是选最小的，所以dp[i] = min(dp[i-1], dp[i-2]) + cost[i]
注意这里为什么是加cost[i]，而不是cost[i-1],cost[i-2]之类的，因为题目中说了：每当你爬上一个阶梯你都要花费对应的体力值
3 dp数组如何初始化
根据dp数组的定义，dp数组初始化其实是比较难的，因为不可能初始化为第i台阶所花费的最少体力。
那么看一下递归公式，dp[i]由dp[i-1]，dp[i-2]推出，既然初始化所有的dp[i]是不可能的，那么只初始化dp[0]和dp[1]就够了，其他的最终
都是dp[0]和dp[1]推出。显然dp:=make([]int, len(cost), len(cost)];dp[0]=cost[0],dp[1]=cost[1]
4 确定遍历顺序，由于是爬台阶,dp[i]又是由dp[i-1]，dp[i-2]推出，所以从前到后遍历cost数组就可以了。
5 举例推导dp数组。以示例二为例，根据cost数组，举例推导一下dp数组。
dp[0] = 1;dp[1] = 100;dp[2] = 2;dp[3] = 1;dp[4] = 3;dp[5] = 103;dp[6] = 4;dp[7] = 5;dp[8] = 104;dp[9] = 6;
注意最后一步可以理解为不用花费，所以取倒数第一步，第二步的最小值,所以示例2的minCost := min(dp[8],dp[9])=min(104,6)=6
 */

// MinCostClimbingStairs 时间复杂度O(N), 空间复杂度O(N)
func MinCostClimbingStairs(cost []int)int{
	n := len(cost)
	dp := make([]int, n, n)
	dp[0] = cost[0]
	dp[1] = cost[1]
	for i:=2;i<n;i++{
		dp[i] = Utils.Min(dp[i-1], dp[i-2]) + cost[i]
	}
	// 注意最后一步可以理解为不用花费，所以取倒数第一步，第二步的最小值
	return Utils.Min(dp[n-1], dp[n-2])
}

// MinCostClimbingStairsSimple 同样的，可以优化一下空间复杂度.时间复杂度O(N), 空间复杂度O(1)
func MinCostClimbingStairsSimple(cost []int)int{
	n := len(cost)
	dp := make([]int, 2, 2)
	dp[0] = cost[0]
	dp[1] = cost[1]
	for i:=2;i<n;i++{
		minCost := Utils.Min(dp[0], dp[1])+cost[i]
		dp[0] = dp[1]
		dp[1] = minCost
	}
	// 注意最后一步可以理解为不用花费，所以取倒数第一步，第二步的最小值
	return Utils.Min(dp[0], dp[1])
}