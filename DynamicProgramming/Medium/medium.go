package Medium

import (
	"AlgorithmPractise/Utils"
)

/*
1.1  不同的二叉搜索树
给你一个整数n，求由n个节点组成且节点值从1到n互不相同的二叉搜索树有多少种？返回满足题意的二叉搜索树的种数。
*/

/*
解题思路:由于1,2...n这个数列是递增的，所以我们从任意一个位置“提起”这棵树，都满足二叉搜索
树的这个条件：左子树小于根节点，右子树大于根节点从1,2,...n数列构建搜索树，实际上只是一个不断细分的过程
例如，我要用[1,2,3,4,5,6]构建。首先，提起"2"作为树根，[1]为左子树，[3,4,5,6]为右子树。现在就变成了一个更小的问题：
如何用[3,4,5,6]构建搜索树？比如，我们可以提起"5"作为树根，[3,4]是左子树，[6]是右子树
现在就变成了一个更更小的问题：如何用[3,4]构建搜索树？那么这里就可以提起“3”作为树根，[4]是右子树;或"4"作为树根，[3]是左子树
可见n=6时的问题是可以不断拆分成更小的问题的
假设f(n) = 我们有n个数字时可以构建几种搜索树
我们可以很容易得知几个简单情况 f(0) = 1, f(1) = 1, f(2) = 2
(注：这里的f(0)可以理解为=1也可以理解为=0，这个不重要，我们这里理解为=1,即没有数字时
只有一种情况，就是空的情况）
那n=3时呢？我们来看[1,2,3]
如果提起1作为树根，左边有f(0)种情况，右边f(2)种情况，左右搭配一共有f(0)*f(2)种情况
如果提起2作为树根，左边有f(1)种情况，右边f(1)种情况，左右搭配一共有f(1)*f(1)种情况
如果提起3作为树根，左边有f(2)种情况，右边f(0)种情况，左右搭配一共有f(2)*f(0)种情况
容易得知f(3) = f(0)*f(2) + f(1)*f(1) + f(2)*f(0)
同理,
f(4) = f(0)*f(3) + f(1)*f(2) + f(2)*f(1) + f(3)*f(0)
f(5) = f(0)*f(4) + f(1)*f(3) + f(2)*f(2) + f(3)*f(1) + f(4)*f(0)
......
发现了吗？
对于每一个n，其式子都是有规律的:每一项两个f()的数字加起来都等于n-1
既然我们已知f(0) = 1, f(1) = 1
那么就可以先算出f(2),再算出f(3),然后f(4)也可以算了...
计算过程中可以把这些存起来，方便随时使用
最后得到的f(n)就是我们需要的解了.

所以按照动态规划五部曲分解就是：
1 确定dp数组以及下标含义
dp := make([]int, n+1),dp[i]代表由1...i为节点组成的所有不同的二叉搜索树的种数
2 确定递推公式
dp[i] = 所有的dp[i-j] * dp[i-1-j]累加之和(j介于[0,i-1]之间)
3 dp数组初始化
dp[0], dp[1] = 1, 1
4 确定遍历顺序
i从2开始遍历，下面嵌套一层j的遍历，从0开始
5 举例推导dp数组
下标i    0  1  2  3  4   5   6    7
dp[i]   1  1  2  5  14  42  132  429
*/

func NumOfBST(n int) int {
	dp := make([]int, n+1)
	dp[0], dp[1] = 1, 1
	if n <= 1 {
		return dp[n]
	}
	// dp[0]和dp[1]都已经初始化好了，所以外层循环遍历从2开始
	for i := 2; i < n+1; i++ {
		for j := 0; j < i; j++ {
			// 两个下标和为i-1
			dp[i] += dp[j] * dp[i-j-1]
		}
	}
	return dp[n]
}

/*
1.2 01背包理论
有N件物品和一个最多能背重量为W的背包。第i件物品的重量是weight[i]，得到的价值是value[i] 。每件物品只能用一次，求解将哪些物品装入背包里
物品价值总和最大。

示例:weight:[1,3,4];value:[15,20,30],w=4
输出35

1 确定dp数组及其下标含义
dp[i][j]表示从下标为[0, i]的物品里任意取，放进容量为j的背包，所得到的最大价值总和。

2 确定递推公式
根据dp数组定义，我们可以从两个方向推导出dp[i][j]
不放物品i:由dp[i-1][j]推出，即背包容量为j，里面不放物品i的最大价值，此时dp[i][j]就是dp[i-1][j]。(此时物品i无法放进背包中，因为背包容量
已经是j,再放物品背包就装不下了，weight[i]+j>j, 所以背包内的价值依然和前面相同。)
放物品i,由dp[i-1][j-weight[i]]推出, dp[i-1][j-weight[i]]为背包容量为j-weight[i]时不放物品i的最大价值,那么
dp[i-1][j-weight[i]]+value[i](物品i的价值),就是容量为j(此时容量为j-weight[i]+weight[i]=j)的背包放入物品i后的最大价值

所以递推公式为: dp[i][j] = max(dp[i-1][j],dp[i-1][j-weight[i]]+value[i])

3 dp数组初始化
从dp数组定义出发，如果背包容量j为0的话，即dp[i][0]，无论选取哪些物品，背包价值总和一定为0，因为背包无法容纳任何物品。
再看其他情况
从状态转移方程 dp[i][j] = max(dp[i-1][j],dp[i-1][j-weight[i]]+value[i]) 可以看出i是由i-1推导出来，那么i为0时一定要初始化。
dp[0][j]即只选取下标为0的物品时，容量为j的背包所能得到的最大物品价值。
那么很明显，当j<weight[0]时,dp[0][j]=0，因为此时下标为0的物品重量超过了背包容量j，无法放入物品0,价值自然为0
当j>=weight[0]时,dp[0][j]应该是value[0],因为背包容量足以容纳下标为0的物品。
所以dp数组初始化为:
n := len(weight)
dp := make([][]int, n)
for i:=0;i<n;i++{
	dp[i] = make([]int, bagWeight+1)
}

for j:=weight[0];j<=bagWeight;j++{
	dp[0][j] = value[0]
}

dp[0][j]和dp[i][0] 都已经初始化了，那么其他下标应该初始化多少呢？
其实从递归公式:dp[i][j] = max(dp[i-1][j], dp[i-1][j-weight[i]]+value[i]); 可以看出dp[i][j]是由左上方数值推导出来了，
那么其他下标初始为什么数值都可以，因为都会被覆盖。这里为了方便统一初始化为0

4 确定遍历顺序
由递推公式可知，dp[i][j]是由左上方数值推导出来，所以先遍历物品，还是先遍历背包都可以，先遍历物品更容易理解一些。

5 举例推导dp数组
参见01背包.png
*/

// BagProblem 时间复杂度O(m*n)，空间复杂度O(m*n),m为weight数组长度，n为capacity
func BagProblem(weight, value []int, capacity int) int {
	n := len(weight)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, capacity+1)
	}
	for j := weight[0]; j <= capacity; j++ {
		dp[0][j] = value[0]
	}
	for i := 1; i < n; i++ {
		for j := 0; j <= capacity; j++ {
			if j < weight[i] {
				dp[i][j] = dp[i-1][j]
			} else {
				dp[i][j] = Utils.Max(dp[i-1][j], dp[i-1][j-weight[i]]+value[i])
			}
		}
	}
	return dp[n-1][capacity]
}

// BagProblemSimple 用一维数组空间复杂度还有优化空间，时间复杂度O(m*n)，空间复杂度O(n),m为weight数组长度，n为capacity
func BagProblemSimple(weight, value []int, capacity int) int {
	n := len(weight)
	dp := make([]int, capacity+1)
	for i := 0; i < n; i++ {
		// 必须逆序遍历背包, 确保元素不会被重复放入
		for j := capacity; j >= weight[i]; j-- {
			// dp一维数组递推公式
			dp[j] = Utils.Max(dp[j], dp[j-weight[i]]+value[i])
		}
	}
	return dp[capacity]
}

/*
1.3 分割等和子集
给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
注意: 每个数组中的元素不会超过100，数组的大小不会超过 200

示例1:
输入: [1, 5, 11, 5] 输出: true 解释: 数组可以分割成 [1, 5, 5] 和 [11].

示例2:
输入: [1, 2, 3, 5] 输出: false 解释: 数组不能分割成两个元素和相等的子集.

提示：
1 <= nums.length <= 200
1 <= nums[i] <= 100
*/

/*
思路:本题是要找是否可以将数组分割成两个子集，使得两个子集的元素和相等。
那么只要找到集合里能够出现sum/2的子集总和，就算是可以分割成两个相同元素和的子集了。
下面套用一维数组的动归五部曲解决
背包的体积为sum(数组和)/2
背包要放入的物品就是（集合里的元素），其重量为元素的数值，价值也为元素的数值
背包如何正好装满，说明找到了总和为sum/2的子集。
背包中每一个元素都不可重复放入。

1 确定dp数组以及定义
dp[j]表示背包总容量为j,其所容纳的所有物品(集合元素)的最大价值(子集元素和)，
由于数组元素容量不超过200，而元素最大值不超过100，所以数组元素和最大不会超过20000，所以背包的最大
体积也就是sum/2不会超过10000，故dp数组的长度可定为10001; 当然，最精确的做法是遍历nums数组，累加数组元素得到数组元素和sum,
长度就等于sum/2+1(整除是向下取整，所以要+1)

2 确定递推公式
dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])

3 dp数组如何初始化
dp[0] = 0 这一点是可以确定的。至于其他下标根据题目条件都初始化为0就可以了，因为题目说集合中的元素都是正整数，那么初始化为0
就好了，这样可已经足以确保dp数组在递归过程中所取的最大值不会被初始值覆盖掉。如果题目给定的元素有负数，那么其他下标就需要初始化为
负无穷大，以确保递归得到的最大值不会被初始值覆盖。

4 确定遍历顺序
一维数组遍历顺序，是先遍历物品，再遍历背包，且遍历背包时必须是倒序

5 举例推导dp数组
下标j  0  1  2  3  4  5  6  7  8  9  10  11
dp[j] 0  1  1  1  1  5  6  6  6  6  10  11
target := sum(array) / 2 = 11
dp[target] = target, 返回true
*/

// CanPartition 时间复杂度O(n^2)，空间复杂度O(n)
func CanPartition(nums []int) bool {
	sum := Utils.SumOfArray(nums)
	// 如果数组nums元素之和sum为奇数则不可能平分为两个子集
	if sum%2 == 1 {
		return false
	}
	target := sum / 2
	dp := make([]int, target+1)
	for i := 0; i < len(nums); i++ {
		// 必须逆序遍历背包, 确保元素不会被重复放入
		for j := target; j >= nums[i]; j-- {
			// 递推公式 dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])
			// 写成下面这种方式效率更高，因为只有满足条件时才会给dp[j]赋值,完全按递推公式写每次都会比较和重新赋值
			if dp[j] < dp[j-nums[i]]+nums[i] {
				dp[j] = dp[j-nums[i]] + nums[i]
			}
		}
	}
	return dp[target] == target
}

/*
1.4 最后一块石头的重量
有一堆石头，每块石头的重量都是正整数。
每一回合，从中选出任意两块石头，然后将它们一起粉碎。假设石头的重量分别为x和 y，且x <= y。那么粉碎的可能结果如下：
如果x == y，那么两块石头都会被完全粉碎；如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。 最后，
最多只会剩下一块石头。返回此石头最小的可能重量。如果没有石头剩下，就返回 0。

示例：输入：[2,7,4,1,8,1] 输出：1 解释： 组合 2 和 4，得到 2，所以数组转化为 [2,7,1,8,1]， 组合 7 和 8，得到 1，
所以数组转化为 [2,1,1,1]， 组合 2 和 1，得到 1，所以数组转化为 [1,1,1]， 组合 1 和 1，得到 0，所以数组转化为 [1]，
这就是最优值。

提示：
1 <= stones.length <= 30
1 <= stones[i] <= 1000
*/

/*
思路:
本题其实就是尽量让石头分成重量相等的两堆，相撞之后剩下的石头最小，这样就化解成01背包问题了。
其实和1.3 分割等和子集非常像了。

本题物品的重量为store[i]，物品的价值也为store[i]。
对应着01背包里的物品重量weight[i]和物品价值value[i]。

1 确定dp数组以及下标含义
dp[j]表示重量为j的背包，最多可以容纳重量为dp[j]的石头

2 确定递推公式
dp[j] = max(dp[j], dp[j-stones[i]] + stones[i])

3 初始化dp数组
dp数组的长度，最精确的做法是遍历stones数组，累加数组元素得到数组元素和sum, 长度就等于sum/2+1(整除是向下取整，所以要+1)
由于重量都不会是负数，所以统一初始化为0即可。

4 确定遍历顺序
一维数组遍历顺序，是先遍历物品，再遍历背包，且遍历背包时必须是倒序

5 举例推导dp数组
输入[2,4,1,1] target=4
参见最后一块石头.png
*/

// LastStoneWeight 时间复杂度O(sum/2 * n), 空间复杂度为O(n), n为stones数组长度，sum/2为stones数组之和的一半
func LastStoneWeight(stones []int) int {
	sum := Utils.SumOfArray(stones)
	target := sum / 2
	dp := make([]int, target+1)
	for i := 0; i < len(stones); i++ {
		for j := target; j >= stones[i]; j-- {
			if dp[j] < dp[j-stones[i]]+stones[i] {
				dp[j] = dp[j-stones[i]] + stones[i]
			}
		}
	}
	// 最后stones石块被分成了dp[target]和sum-dp[target]两堆
	// 由于sum/2是向下取整，所以sum-dp[target]一定比dp[target]大(因为dp[target]<=sum/2)，故相撞粉碎的结果就是
	// dp[target]这一堆没了，sum-dp[target]还剩下sum-dp[target]-dp[target]
	return sum - dp[target] - dp[target]
}

/*
1.5 目标和
给你一个整数数组 nums 和一个整数target 。

向数组中的每个整数前添加'+' 或 '-' ，然后串联起所有整数，可以构造一个表达式 ：
例如，nums = [2, 1] ，可以在2之前添加'+' ，在1之前添加'-' ，然后串联起来得到表达式"+2-1" 。
返回可以通过上述方法构造的、运算结果等于target的不同表达式的数目。

示例 1：
输入：nums = [1,1,1,1,1], target = 3
输出：5
解释：一共有 5 种方法让最终目标和为 3 。
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3

示例 2：
输入：nums = [1], target = 1
输出：1

提示：
1 <= nums.length <= 20
0 <= nums[i] <= 1000
0 <= sum(nums[i]) <= 1000
-1000 <= target <= 1000
*/

/*
思路:题目的目标是如何使得数组所有元素组成的表达式运算结果为target,假如我们将数组元素和记为sum,
将全部带有+表达式和记为left,全都带有-表达式和记为right,则有left-right=target, left+right=sum.
则可推导出left = (sum+target)/2。而sum和target是固定的，left自然不难求出，此时问题转化为在nums中找出
和为left的组合数

动态规划五部曲

1 确定dp数组以及下标含义
dp[j]表示要装满容量为j的背包，一共有dp[j]种方法

2 确定递推公式
那么如何推导出dp[j]呢？我们知道，要装满容量为j-nums[i]的背包，一共有dp[j-nums[i]]种方法,那么只要找到nums[i]，自然就能填满
容量为j的背包，也就是说此时有dp[j-nums[i]]种方法可以填满容量为j的背包；以此类推，将多个dp[j-nums[i]]累加起来就得到dp[j]
所以递推公式就是dp[j] += dp[j-nums[i]]

3 初始化dp数组
首先，dp[0] = 0，这个很好理解，装满容量为0的背包，有1种方法，就是装0件物品。dp数组长度即为left+1

4 确定遍历顺序
一维数组遍历顺序，是先遍历物品，再遍历背包，且遍历背包时必须是倒序

5 举例推导dp数组
nums: [1, 1, 1, 1, 1], target:3
参见目标和.pn
*/

// FindTargetSumWays 时间复杂度O(n * capacity)，空间复杂度：O(capacity)， n为nums数组长度，capacity为背包容量，
func FindTargetSumWays(nums []int, target int) int {
	sum := Utils.SumOfArray(nums)
	if Utils.Abs(target) > sum {
		return 0
	}
	if (sum+target)%2 == 1 {
		return 0
	}
	capacity := (sum + target) / 2
	dp := make([]int, capacity+1)
	dp[0] = 1
	for i := 0; i < len(nums); i++ {
		for j := capacity; j >= nums[i]; j-- {
			dp[j] += dp[j-nums[i]]
		}
	}
	return dp[capacity]
}