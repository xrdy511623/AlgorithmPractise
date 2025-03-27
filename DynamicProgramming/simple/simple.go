package simple

/*
simple package contains easy items
*/

import (
	"algorithm-practise/utils"
	"strconv"
)

/*
leetcode 509
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
	dp := make([]int, n+1)
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

// 当然，上面的方法还有优化的空间，其实我们只需要维护两个数值就可以了，不需要记录整个序列
// 时间复杂度O(N), 空间复杂度O(1)

func fibSimple(n int) int {
	if n <= 1 {
		return n
	}
	dp := make([]int, 2)
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
leetcode 70
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
	if n <= 2 {
		return n
	}
	dp := make([]int, n+1)
	dp[1], dp[2] = 1, 2
	for i := 3; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

// ClimbStairsSimple 同样的，可以优化一下空间复杂度。时间复杂度O(N), 空间复杂度降低到O(1)
func ClimbStairsSimple(n int) int {
	if n <= 2 {
		return n
	}
	dp := make([]int, 2)
	dp[0], dp[1] = 1, 2
	for i := 3; i <= n; i++ {
		sum := dp[0] + dp[1]
		dp[0] = dp[1]
		dp[1] = sum
	}
	return dp[1]
}

/*
剑指offer 10 矩形覆盖
我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
*/

/*
思路:从最简单的开始推导，可以发现这又是一个斐波拉契数列
显然，当n = 1时，只有一种方法;
当n = 2时，有两种方法;
当n = 3时，有三种方法;
当n = 4时，有五种方法;
于是有dp[n]=dp[n-1]+dp[n-2]
*/

func coverMatrix(n int) int {
	if n <= 2 {
		return n
	}
	l1, l2 := 1, 2
	for i := 3; i <= n; i++ {
		sum := l1 + l2
		l1 = l2
		l2 = sum
	}
	return l2
}

/*
leetcode 746
1.2 使用最小花费爬楼梯
给你一个整数数组cost，其中cost[i]是从楼梯第i个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个
或者两个台阶。
你可以选择从下标为0或下标为1的台阶开始爬楼梯。
请你计算并返回达到楼梯顶部的最低花费。


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
dp[i]的定义:到达第i个台阶所花费的最小体力为dp[i]
2 确定递推公式
可以有两个途径得到dp[i],dp[i-1]和dp[i-2],那么究竟是选择dp[i-1]还是dp[i-2]呢？
从题意来看，一定是选最小的。如果你到了第i-1个台阶所花费的最小体力是dp[i-1]，此时你要到达第i个台阶，根据题意
cost[i]是从楼梯第i个台阶向上爬需要支付的费用，所以你从第i-1个台阶向上爬1个台阶到第i个台阶，还需要花费cost[i-1]体力
总共需要花费dp[i-1]+cost[i-1],同理如果从第i-2个台阶向上爬2个台阶到第i个台阶，需花费dp[i-2]+cost[i-2]
因此有dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]) + cost[i-2]

3 dp数组如何初始化
根据dp数组的定义，dp数组初始化其实是比较难的，因为不可能初始化为第i台阶所花费的最少体力。
那么看一下递归公式，dp[i]由dp[i-1]，dp[i-2]推出，既然初始化所有的dp[i]是不可能的，那么只初始化dp[0]和dp[1]就够了，其他的最终
都是dp[0]和dp[1]推出。显然dp:=make([]int, len(cost)];dp[0]=0,dp[1]=0,因为根据题意可以选择从下标为0或
下标为1的台阶开始爬楼梯，所以第0个和第1个台阶都可以是起点，花费为0。
4 确定遍历顺序，由于是爬台阶,dp[i]又是由dp[i-1]，dp[i-2]推出，所以从前到后遍历cost数组就可以了。
5 举例推导dp数组。以示例二为例，根据cost数组，举例推导一下dp数组。
dp[0] = 0;dp[1] = 0;dp[2] = 1;dp[3] = 2;dp[4] = 2;dp[5] = 3;dp[6] = 3;dp[7] = 4;dp[8] = 4;dp[9] = 5;
爬到楼顶，也就是第n个台阶，可以从第n-1个台阶向上爬1个台阶到达，也可以从第n-2个台阶向上爬2个台阶到达，花费的体力分别为
dp[n-1]+cost[n-1], dp[n-2]+cost[n-2]，我们求最小花费，自然是求两值中的较小值。
所以最后返回min(dp[8]+cost[8], dp[9]+cost[9])=min(4+100,5+1)=min(104,6)=6
*/

// MinCostClimbingStairs 时间复杂度O(N), 空间复杂度O(N)
func MinCostClimbingStairs(cost []int) int {
	n := len(cost)
	dp := make([]int, n)
	// dp数组make初始化时默认值就是0,所以dp[0]和dp[1]不用再赋值了
	for i := 2; i < n; i++ {
		dp[i] = utils.Min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2])
	}
	return utils.Min(dp[n-1]+cost[n-1], dp[n-2]+cost[n-2])
}

// MinCostClimbingStairsSimple 同样的，可以优化一下空间复杂度.时间复杂度O(N), 空间复杂度O(1)
func MinCostClimbingStairsSimple(cost []int) int {
	n := len(cost)
	dp := make([]int, 2)
	for i := 2; i < n; i++ {
		minCost := utils.Min(dp[0]+cost[i-2], dp[1]+cost[i-1])
		dp[0] = dp[1]
		dp[1] = minCost
	}
	return utils.Min(dp[0]+cost[n-2], dp[1]+cost[n-1])
}

/*
leetcode 62
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
dp[i][j]表示从（0,0）出发，到(i, j) 有dp[i][j]条不同的路径。

2 确定递推公式
想要求dp[i][j]，只能有两个方向来推导出来，即dp[i - 1][j] 和 dp[i][j - 1]。
此时再回顾一下 dp[i - 1][j]表示啥，是从(0, 0)的位置到(i - 1, j)有几条路径，dp[i][j - 1]同理。
那么很自然，dp[i][j] = dp[i - 1][j] + dp[i][j - 1]，因为dp[i][j]只能从这两个方向过来。

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
leetcode 63
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
leetcode 64
进阶 1.5 最小路径和
给定一个包含非负整数的m x n网格grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
说明：每次只能向下或者向右移动一步。

示例一
参见:网格.png
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。

示例 2：
输入：grid = [[1,2,3],[4,5,6]]
输出：12
*/

/*
1 确定dp数组以及下标含义
dp[i][j]表示从grid[0,0]到grid[i,j]的最小路径和为dp[i][j]

2 确定递推公式
问题的关键在于推导出递推公式
由于题目规定只能向右或者向下移动，那么要走到grid[i][j]，只能从grid[i-1][j]和grid[i][j-1]而来，前者是向下走一步，后者是向右走一步。
而走到grid[i-1][j]和grid[i][j-1]的最小路径和分别为dp[i-1][j]和dp[i][j-1]，所以到grid[i,j]的最小路径和dp[i][j]其实就是
min(dp[i-1][j], dp[i][j-1]) + grid[i][j]就好了。
所以递推公式如下:
dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

3 初始化dp数组
dp[0][0] = grid[0][0]
m, n := len(grid), len(grid[0])
当1<=i<m时，dp[i][0] = dp[i-1][0] + grid[i][0](此时只能从上边走下来，因为它是最左边)
当1<=j<n时，dp[0][j] = dp[0][j-1] + grid[0][j](此时只能从左边走过来，因为它是最上边的那一行)

4 确定遍历顺序
按照二维数组正序遍历即可
*/

func minPathSum(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	if m == 0 || n == 0 {
		return 0
	}
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	dp[0][0] = grid[0][0]
	for i := 1; i < m; i++ {
		dp[i][0] = dp[i-1][0] + grid[i][0]
	}
	for j := 1; j < n; j++ {
		dp[0][j] = dp[0][j-1] + grid[0][j]
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = utils.Min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
		}
	}
	return dp[m-1][n-1]
}

/*
雷同题
剑指 Offer 47. 礼物的最大价值
在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿
格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多
能拿到多少价值的礼物？


示例 1:

输入:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物

0 < grid.length <= 200
0 < grid[0].length <= 200
*/

/*
本题与上一题几乎完全雷同，只不过换成了求最大值
*/

func maxValue(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	dp[0][0] = grid[0][0]
	for i := 1; i < m; i++ {
		dp[i][0] = dp[i-1][0] + grid[i][0]
	}
	for j := 1; j < n; j++ {
		dp[0][j] = dp[0][j-1] + grid[0][j]
	}

	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = utils.Max(dp[i-1][j], dp[i][j-1]) + grid[i][j]
		}
	}
	return dp[m-1][n-1]
}

/*
leetcode 343
1.6 整数拆分
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

提示:
2 <= n <= 58
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
for (i：= 3; i <= n ; i++) {
    for (j := 1; j < i - 1; j++) {
        dp[i] = max(dp[i], max((i - j) * j, dp[i - j] * j));
    }
}

5 举例推导dp数组
举例当n为10的时候，dp数组里的数值，如下：
下标i    2  3  4  5  6  7  8  9  10
dp[i]   1  2  4  6  9  12 18 27 36
*/

func integerBreak(n int) int {
	dp := make([]int, n+1)
	dp[2] = 1
	for i := 3; i <= n; i++ {
		// 因为初始化dp[2] = 1, 所以j的取值最大不能超过i-2,这样i-j才会>=2
		for j := 1; j < i-1; j++ {
			dp[i] = utils.Max(dp[i], utils.Max((i-j)*j, dp[i-j]*j))
		}
	}
	return dp[n]
}

/*
Leetcode 118. 杨辉三角
1.7 给定一个非负整数numRows，生成杨辉三角的前numRows行。
在杨辉三角中，每个数是它左上方和右上方的数的和。

示例1:
输入: numRows = 5
输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
*/

func generate(numRows int) [][]int {
	dp := make([][]int, numRows)
	for i := 0; i < numRows; i++ {
		// 每一行数组的长度等于行数
		dp[i] = make([]int, i+1)
		dp[i][0], dp[i][i] = 1, 1
		for j := 1; j < i; j++ {
			dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
		}
	}
	return dp
}

/*
leetcode 119. 杨辉三角II
给定一个非负索引rowIndex，返回「杨辉三角」的第rowIndex行。
在「杨辉三角」中，每个数是它左上方和右上方的数的和。
示例1:
输入: rowIndex = 3
输出: [1,3,3,1]
*/

func getRowComplex(rowIndex int) []int {
	var pre, cur []int
	for i := 0; i <= rowIndex; i++ {
		cur = make([]int, i+1)
		cur[0], cur[i] = 1, 1
		for j := 1; j < i; j++ {
			cur[j] = pre[j-1] + pre[j]
		}
		pre = cur
	}
	return pre
}

func getRow(rowIndex int) []int {
	row := make([]int, rowIndex+1)
	row[0] = 1
	for i := 1; i <= rowIndex; i++ {
		for j := i; j > 0; j-- {
			row[j] += row[j-1]
		}
	}
	return row
}

func getRowSimple(rowIndex int) []int {
	row := make([]int, rowIndex+1)
	row[0] = 1
	for i := 1; i <= rowIndex; i++ {
		row[i] = row[i-1] * (rowIndex - i + 1) / i
	}
	return row
}

/*
剑指offer 46 把数字翻译成字符串
现有一串神秘的密文 code，经调查，密文的特点和规则如下：

密文由非负整数组成
数字 0-25 分别对应字母 a-z
请根据上述规则将密文 code 解密为字符串，并返回共有多少种解密结果。


示例 1：
输入：code = 216612
输出：6
解释：216612 解密后有 6 种不同的形式，分别是 "cbggbc"，"vggbc"，"vggm"，"cbggm"，"cqgbc" 和 "cqgm"


提示：
0 <= code < 231
*/

func crackNumber(code int) int {
	// 处理小于等于 9 的情况：只有一个数字，只能解码为一个字母，所以方法数为 1
	if code <= 9 {
		return 1
	}
	// 处理 10 到 25 的情况：可以解码为两个单独的字母或一个组合字母，所以方法数为 2
	if code >= 10 && code <= 25 {
		return 2
	}
	// 将整数 code 转换为字符串，便于按位处理
	s := strconv.Itoa(code)
	n := len(s)
	// 创建一个动态规划数组 dp，dp[i] 表示前 i+1 个数字的解码方法数
	dp := make([]int, n)
	// 初始化 dp[0]：第一个数字（s[0]）只能单独解码，所以方法数为 1
	dp[0] = 1
	// 计算前两个数字（s[0:2]）的解码方法数
	num, _ := strconv.Atoi(s[:2])
	// 如果前两个数字在 10 到 25 之间，可以单独解码（两个字母）或组合解码（一个字母），方法数为 2
	if num >= 10 && num <= 25 {
		dp[1] = 2
	} else {
		// 如果前两个数字不在 10 到 25 之间，只能单独解码为两个字母，方法数为 1，比如27只能各自分开单独解码，得到ch
		dp[1] = 1
	}
	// 从第 3 个数字开始（i=2），通过循环计算每个位置的解码方法数
	for i := 2; i < n; i++ {
		// 取出当前数字和前一个数字，组成一个两位数
		newNum, _ := strconv.Atoi(s[i-1 : i+1])
		if newNum >= 10 && newNum <= 25 {
			// 如果这个两位数在 10 到 25 之间：
			// 1. 可以单独解码当前数字，方法数为 dp[i-1]
			// 2. 可以与前一个数字组合解码，方法数为 dp[i-2]
			// 总方法数为 dp[i-1] + dp[i-2]
			dp[i] = dp[i-1] + dp[i-2]
		} else {
			// 如果这个两位数不在 10 到 25 之间，只能单独解码当前数字
			// 方法数与前一个位置相同，即 dp[i-1]
			dp[i] = dp[i-1]
		}
	}
	// 返回整个数字的解码方法数，即 dp 数组的最后一个值
	return dp[n-1]
}
