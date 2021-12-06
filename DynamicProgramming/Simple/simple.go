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
func Fib(n int) int {
	if n <= 1 {
		return n
	}
	dp := make([]int, n+1, n+1)
	dp[0] = 0
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

// 当然，上面的方法还有优化的空间，其实我们只需要维护两个数值就可以了，不需要记录整个序列
// 时间复杂度O(N), 空间复杂度O(1)

func FibSimple(n int) int {
	if n <= 1 {
		return n
	}
	dp := make([]int, 2, 2)
	dp[0] = 0
	dp[1] = 1
	for i := 2; i <= n; i++ {
		sum := dp[0] + dp[1]
		dp[0] = dp[1]
		dp[1] = sum
	}
	return dp[1]
}

// FibUseRecursion 当然，本题还可以用递归解决，但是会有大量重复计算，时间复杂度O(2^N), 空间复杂度O(N)
func FibUseRecursion(n int) int {
	if n < 2 {
		return n
	}
	return FibUseRecursion(n-1) + FibUseRecursion(n-2)
}

// FibUseRecursionAndCache 要降低时间复杂度，需要使用缓存, 时间复杂度O(N), 空间复杂度O(N)
func FibUseRecursionAndCache(n int) int {
	cache := make(map[int]int, n+1)
	var helper func(int) int
	helper = func(n int) int {
		if v, ok := cache[n]; ok {
			return v
		}
		if n < 2 {
			cache[n] = n
			return n
		}
		cache[n] = helper(n-1) + helper(n-2)
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
func ClimbStairs(n int) int {
	if n <= 1 {
		return n
	}
	dp := make([]int, n+1, n+1)
	dp[1], dp[2] = 1, 2
	for i := 3; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

// ClimbStairsSimple 同样的，可以优化一下空间复杂度.时间复杂度O(N), 空间复杂度O(1)
func ClimbStairsSimple(n int) int {
	if n <= 1 {
		return n
	}
	dp := make([]int, 2, 2)
	dp[0], dp[1] = 1, 2
	for i := 3; i <= n; i++ {
		sum := dp[0] + dp[1]
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
func MinCostClimbingStairs(cost []int) int {
	n := len(cost)
	dp := make([]int, n, n)
	dp[0] = cost[0]
	dp[1] = cost[1]
	for i := 2; i < n; i++ {
		dp[i] = Utils.Min(dp[i-1], dp[i-2]) + cost[i]
	}
	// 注意最后一步可以理解为不用花费，所以取倒数第一步，第二步的最小值
	return Utils.Min(dp[n-1], dp[n-2])
}

// MinCostClimbingStairsSimple 同样的，可以优化一下空间复杂度.时间复杂度O(N), 空间复杂度O(1)
func MinCostClimbingStairsSimple(cost []int) int {
	n := len(cost)
	dp := make([]int, 2, 2)
	dp[0] = cost[0]
	dp[1] = cost[1]
	for i := 2; i < n; i++ {
		minCost := Utils.Min(dp[0], dp[1]) + cost[i]
		dp[0] = dp[1]
		dp[1] = minCost
	}
	// 注意最后一步可以理解为不用花费，所以取倒数第一步，第二步的最小值
	return Utils.Min(dp[0], dp[1])
}

/*
1.3 不同路径
一个机器人位于一个mxn网格的左上角（起始点在下图中标记为 “Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
问总共有多少条不同的路径？

示例1:
输入：m = 3, n = 7
输出：28

示例2：
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有3条路径可以到达右下角。
1. 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右
3. 向下 -> 向右 -> 向下
*/

/*
机器人从(0, 0) 位置出发，到(m - 1, n - 1)终点。
按照动规五部曲来分析：
1 确定dp数组以及下标的含义
dp[i][j]:表示从（0,0）出发，到(i, j) 有dp[i][j]条不同的路径。

2 确定递推公式
想要求dp[i][j]，只能有两个方向来推导出来，即dp[i - 1][j] 和 dp[i][j - 1]。
此时再回顾一下 dp[i - 1][j]表示啥，是从(0, 0)的位置到(i - 1, j)有几条路径，dp[i][j - 1]同理。
那么很自然，dp[i][j] = dp[i - 1][j] + dp[i][j - 1]，因为dp[i][j]只有这两个方向过来。

3 dp数组的初始化
如何初始化呢，首先dp[i][0]一定都是1，因为从(0, 0)的位置到(i, 0)的路径只有一条，那么dp[0][j]也同理。

4 确定遍历顺序
这里要看一下递归公式dp[i][j] = dp[i - 1][j] + dp[i][j - 1]，dp[i][j]都是从其上方和左方推导而来，那么从左到右一层一层遍历就可以了。
这样就可以保证推导dp[i][j]的时候，dp[i - 1][j] 和 dp[i][j - 1]一定是有数值的。

5 举例推导dp数组
见图不同路径.png
*/

// UniquePath 时间复杂度O(m*n),空间复杂度O(m*n)
func UniquePath(m, n int) int {
	// dp数组初始化
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
		dp[i][0] = 1
	}
	for j := 0; j < n; j++ {
		dp[0][j] = 1
	}
	// 遍历顺序
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			// 递推公式
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}

// UniquePathSimple 其实用一个一维数组就可以了，但是不太好理解，可以优化空间复杂度。时间复杂度O(m*n),空间复杂度O(n)
func UniquePathSimple(m, n int) int {
	dp := make([]int, n)
	for j := 0; j < n; j++ {
		dp[j] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[j] += dp[j-1]
		}
	}
	return dp[n-1]
}

/*
进阶 1.4 有障碍的不同路径
一个机器人位于一个m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

示例: 见有障碍的不同路径图.png
输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
输出：2
解释：
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右
*/

/*
本题与1.3基本相同，只是因为有了障碍，(i, j)如果就是障碍的话应该就保持初始状态（初始状态为0），即dp[i][j]=0
另外就是dp数组初始化时，如果(i, 0) 这条边有了障碍之后，障碍之后（包括障碍）都是走不到的位置了，所以障碍之后的dp[i][0]应该还是初始值0。
也就是这样:
1,1,1,障碍(0),0,0,0
下标(0, j)的初始化情况同理。
最后就是dp数组遍历时，如果obstacleGrid[i][j]这个点是障碍物, 那么我们的dp[i][j]保持为0
*/

// UniquePathsWithObstacles 时间复杂度O(m*n),空间复杂度O(m*n)
func UniquePathsWithObstacles(obstacleGrid [][]int) int {
	m := len(obstacleGrid)
	n := len(obstacleGrid[0])
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	for i := 0; i < m; i++ {
		// 如果是障碍物, 后面的就都是0, 不用循环了
		if obstacleGrid[i][0] == 1 {
			break
		}
		dp[i][0] = 1
	}
	for j := 0; j < n; j++ {
		// 如果是障碍物, 后面的就都是0, 不用循环了
		if obstacleGrid[0][j] == 1 {
			break
		}
		dp[0][j] = 1
	}
	// dp数组推导过程
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			// 如果obstacleGrid[i][j]这个点是障碍物, 那么我们的dp[i][j]保持为0
			if obstacleGrid[i][j] != 1 {
				// 否则我们需要计算当前点可以到达的路径数
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}
		}
	}
	return dp[m-1][n-1]
}

/*
1.5 整数拆分
给定一个正整数n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

示例 1:
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1。

示例2:
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 ×3 ×4 = 36。
说明: 你可以假设n不小于2且不大于58。
*/

/*
解题思路:
1 确定dp数组以及下标的含义
dp[i]：分拆数字i，可以得到的最大乘积为dp[i]。

2 确定递推公式
可以想dp[i]最大乘积是怎么得到的呢？
其实可以从1遍历j，然后有两种渠道得到dp[i].
一个是j * (i - j) 直接相乘。
一个是j * dp[i - j]，相当于是拆分(i - j).
至于为什么不拆分j,因为j是从1开始遍历，拆分j的情况，在遍历j的过程中其实都计算过了。
那么从1遍历j，比较(i - j) * j和dp[i - j] * j 取最大的。递推公式：dp[i] = max(dp[i], max((i - j) * j, dp[i - j] * j));
也可以这么理解，j * (i - j) 是单纯的把整数拆分为两个数相乘，而j * dp[i - j]是拆分成两个以及两个以上的个数相乘。
所以递推公式：dp[i] = max({dp[i], (i - j) * j, dp[i - j] * j});
那么在取最大值的时候，为什么还要比较dp[i]呢？
因为在递推公式推导的过程中，每次计算dp[i]，取最大的而已。

3 dp的初始化
dp[0] dp[1]应该初始化多少呢？
有的题解里会给出dp[0] = 1，dp[1] = 1的初始化，但解释比较牵强，主要还是因为这么初始化可以把题目过了。
严格从dp[i]的定义来说，dp[0] dp[1] 就不应该初始化，也就是没有意义的数值。
拆分0和拆分1的最大乘积是多少？
这是无解的。
这里只初始化dp[2] = 1，从dp[i]的定义来说，拆分数字2，得到的最大乘积是1，这个没有任何异议！

4 确定遍历顺序
确定遍历顺序，先来看看递归公式：dp[i] = max(dp[i], max((i - j) * j, dp[i - j] * j));
dp[i]是依靠dp[i - j]的状态，所以遍历i一定是从前向后遍历，先有dp[i - j]再有dp[i]。
枚举j的时候，是从1开始的。i是从3开始，这样dp[i - j]就是dp[2]正好可以通过我们初始化的数值求出来。
所以遍历顺序为：
for (int i = 3; i <= n ; i++) {
    for (int j = 1; j < i - 1; j++) {
        dp[i] = max(dp[i], max((i - j) * j, dp[i - j] * j));
    }
}

5 举例推导dp数组
举例当n为10的时候，dp数组里的数值，如下：
下标i    2  3  4  5  6  7  8  9  10
dp[i]   1  2  4  6  9  12 18 27 36
*/

func IntegerBreak(n int) int {
	dp := make([]int, n+1)
	dp[2] = 1
	for i := 3; i <= n; i++ {
		for j := 1; j < i-1; j++ {
			dp[i] = Utils.Max(dp[i], Utils.Max((i-j)*j, dp[i-j]*j))
		}
	}
	return dp[n]
}
