package medium

/*
medium contains middle level problems
*/

import (
	"algorithm-practise/utils"
	"container/heap"
	"math"
	"sort"
	"strings"
)

/*
leetcode 96
1.1  不同的二叉搜索树
给你一个整数n，求由n个节点组成且节点值从1到n互不相同的二叉搜索树有多少种？返回满足题意的二叉搜索树的种数。
*/

/*
解题思路:由于1,2...n这个数列是递增的，所以我们从任意一个位置“提起”这棵树，都满足二叉搜索
树的这个条件：左子树小于根节点，右子树大于根节点.从1,2,...n数列构建搜索树，实际上只是一个不断细分的过程
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

// numOfBST 时间复杂度O(n^2)，空间复杂度O(n)
func numOfBST(n int) int {
	if n <= 1 {
		return 1
	}
	dp := make([]int, n+1)
	dp[0], dp[1] = 1, 1
	// dp[0]和dp[1]都已经初始化好了，所以外层循环遍历从2开始
	for i := 2; i <= n; i++ {
		for j := 0; j < i; j++ {
			// 两个下标和为i-1
			dp[i] += dp[j] * dp[i-1-j]
		}
	}
	return dp[n]
}

/*
1.2 01背包理论
有N件物品和一个最多能背重量为W的背包。第i件物品的重量是weight[i]，得到的价值是value[i]。每件物品只能用一次，
求解将哪些物品装入背包里物品价值总和最大。

示例: weight:[1,3,4];value:[15,20,30],w=4
输出35

1 确定dp数组及其下标含义
dp[i][j]表示从下标为[0, i]的物品里任意取物品，放进容量为j的背包，所得到的最大价值总和。

2 确定递推公式
根据dp数组定义，我们可以从两个方向推导出dp[i][j]
不放物品i:由dp[i-1][j]推出，即背包容量为j，里面不放物品i的最大价值，此时dp[i][j]就是dp[i-1][j]。(此时物品i无法放进背包中，
因为背包容量已经是j,再放物品背包就装不下了，weight[i]+j>j, 所以背包内的价值依然和前面相同。)
放物品i,由dp[i-1][j-weight[i]]推出, dp[i-1][j-weight[i]]为背包容量为j-weight[i]时不放物品i的最大价值,那么
dp[i-1][j-weight[i]]+value[i](物品i的价值),就是容量为j(此时容量为j-weight[i]+weight[i]=j)的背包放入物品i后的最大价值

所以递推公式为: dp[i][j] = max(dp[i-1][j],dp[i-1][j-weight[i]]+value[i])

3 dp数组初始化
从dp数组定义出发，如果背包容量j为0的话，即dp[i][0]，无论选取哪些物品，背包价值总和一定为0，因为背包无法容纳任何物品。
再看其他情况:
从状态转移方程 dp[i][j] = max(dp[i-1][j],dp[i-1][j-weight[i]]+value[i]) 可以看出i是由i-1推导出来，那么i为0时一定要
初始化。dp[0][j]即只选取下标为0的物品时，容量为j的背包所能得到的最大物品价值。
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

// bagProblem 时间复杂度O(m*n)，空间复杂度O(m*n),m为weight数组长度，n为capacity
func bagProblem(weight, value []int, capacity int) int {
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
				dp[i][j] = utils.Max(dp[i-1][j], dp[i-1][j-weight[i]]+value[i])
			}
		}
	}
	return dp[n-1][capacity]
}

// bagProblemSimple 用一维数组空间复杂度可以优化空间复杂度，时间复杂度O(m*n)，空间复杂度O(n),m为weight数组长度，n为capacity
func bagProblemSimple(weight, value []int, capacity int) int {
	n := len(weight)
	dp := make([]int, capacity+1)
	for i := 0; i < n; i++ {
		// 必须逆序遍历背包, 确保元素不会被重复放入
		for j := capacity; j >= weight[i]; j-- {
			// dp一维数组递推公式
			dp[j] = utils.Max(dp[j], dp[j-weight[i]]+value[i])
		}
	}
	return dp[capacity]
}

/*
leetcode 416
1.3 分割等和子集
给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
注意: 每个数组中的元素不会超过100，数组的大小不会超过200

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
体积也就是sum/2不会超过10000，故dp数组的长度可定为10001; 当然，最精确的做法是遍历nums数组，
累加数组元素得到数组元素和sum,长度就等于sum/2+1(整除是向下取整，所以要+1)

2 确定递推公式
dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])

3 dp数组如何初始化
dp[0] = 0 这一点是可以确定的。至于其他下标根据题目条件都初始化为0就可以了，因为题目说集合中的元素都是正整数，那么
初始化为0就好了，这样可已经足以确保dp数组在递归过程中所取的最大值不会被初始值覆盖掉。如果题目给定的元素有负数，那么
其他下标就需要初始化为负无穷大，以确保递归得到的最大值不会被初始值覆盖。

4 确定遍历顺序
一维数组遍历顺序，是先遍历物品，再遍历背包，且遍历背包时必须是倒序

5 举例推导dp数组
下标j  0  1  2  3  4  5  6  7  8  9  10  11
dp[j] 0  1  1  1  1  5  6  6  6  6  10  11
target := sum(array) / 2 = 11
dp[target] = target, 返回true
*/

// canPartition 时间复杂度O(n^2)，空间复杂度O(n)
func canPartition(nums []int) bool {
	sum := utils.SumOfArray(nums)
	// 如果数组nums元素之和sum为奇数则不可能平分为两个子集
	if sum%2 == 1 {
		return false
	}
	target := sum / 2
	dp := make([]int, target+1)
	n := len(nums)
	for i := 0; i < n; i++ {
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
leetcode 1046
1.4 最后一块石头的重量I
有一堆石头，每块石头的重量都是正整数。
每一回合，从中选出两块最重的石头，然后将它们一起粉碎。假设石头的重量分别为x和y，且x <= y。那么粉碎的可能
结果如下：
如果x == y，那么两块石头都会被完全粉碎；
如果x != y，那么重量为x的石头将会完全粉碎，而重量为y的石头新重量为y-x。
最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 0。

示例：
输入：[2,7,4,1,8,1]
输出：1
解释：
先选出 7 和 8，得到 1，所以数组转换为 [2,4,1,1,1]，
再选出 2 和 4，得到 2，所以数组转换为 [2,1,1,1]，
接着是 2 和 1，得到 1，所以数组转换为 [1,1,1]，
最后选出 1 和 1，得到 0，最终数组转换为 [1]，这就是最后剩下那块石头的重量。

提示：
1 <= stones.length <= 30
1 <= stones[i] <= 1000
*/

/*
思路:将所有石头的重量放入最大堆中。每次依次从队列中取出最重的两块石头a和b，必有a≥b。如果 a>b，则将
新石头a-b放回到最大堆中；如果a=b，两块石头完全被粉碎，因此不会产生新的石头。重复上述操作，直到剩下的
石头少于2块。
最终可能剩下1块石头，该石头的重量即为最大堆中剩下的元素，返回该元素；也可能没有石头剩下，此时最大堆为空，
返回0。
*/

type hp struct {
	sort.IntSlice
}

func (h hp) Less(i, j int) bool {
	return h.IntSlice[i] > h.IntSlice[j]
}
func (h *hp) Push(v interface{}) {
	h.IntSlice = append(h.IntSlice, v.(int))
}
func (h *hp) Pop() interface{} {
	a := h.IntSlice
	v := a[len(a)-1]
	h.IntSlice = a[:len(a)-1]
	return v
}
func (h *hp) push(v int) {
	heap.Push(h, v)
}
func (h *hp) pop() int {
	return heap.Pop(h).(int)
}

func LastStoneWeight(stones []int) int {
	q := &hp{stones}
	heap.Init(q)
	for q.Len() > 1 {
		x, y := q.pop(), q.pop()
		if x > y {
			q.push(x - y)
		}
	}
	if q.Len() > 0 {
		return q.IntSlice[0]
	}
	return 0
}

func lastStoneWeightSimple(stones []int) int {
	n := len(stones)
	mh := utils.NewMaxHeap(n)
	for _, v := range stones {
		mh.Add(v)
	}
	for mh.Length() > 1 {
		x, y := mh.Extract(), mh.Extract()
		if x > y {
			mh.Add(x - y)
		}
	}
	if mh.Length() > 0 {
		return mh.Extract()
	}
	return 0
}

/*
leetcode 1049
1.5 最后一块石头的重量II
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

// lastStoneWeight 时间复杂度O(sum/2 * n), 空间复杂度为O(n), n为stones数组长度，sum/2为stones数组之和的一半
func lastStoneWeight(stones []int) int {
	sum := utils.SumOfArray(stones)
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
leetcode 494
1.6 目标和
给你一个整数数组nums和一个整数target 。
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
将全部带有+表达式和记为left,全都带有-表达式和记为right(正数和),则有left-right=target, left+right=sum.
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
首先，dp[0] = 1，这个很好理解，装满容量为0的背包，有1种方法，就是装0件物品。dp数组长度即为left+1

4 确定遍历顺序
一维数组遍历顺序，是先遍历物品，再遍历背包，且遍历背包时必须是倒序

5 举例推导dp数组
nums: [1, 1, 1, 1, 1], target:3
参见目标和.pn
*/

// findTargetSumWays 时间复杂度O(n * capacity)，空间复杂度：O(capacity)， n为nums数组长度，capacity为背包容量，
func findTargetSumWays(nums []int, target int) int {
	sum := utils.SumOfArray(nums)
	// target的绝对值比数组和还大，是不可能有结果的
	if utils.Abs(target) > sum {
		return 0
	}
	// 因为本题转化为在nums数组中找和为left的组合数，也就是找(sum + target) / 2 的组合数
	// 那么sum + target就不能为奇数
	if (sum+target)%2 == 1 {
		return 0
	}
	capacity := (sum + target) / 2
	dp := make([]int, capacity+1)
	dp[0] = 1
	for i, n := 0, len(nums); i < n; i++ {
		for j := capacity; j >= nums[i]; j-- {
			dp[j] += dp[j-nums[i]]
		}
	}
	return dp[capacity]
}

/*
leetcode 474
1.7 一和零
给你一个二进制字符串数组strs和两个整数m和n 。
请你找出并返回strs的最大子集的长度，该子集中最多有m个0和n个1 。

示例 1：
输入：strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3
输出：4
解释：最多有5个0和3个1的最大子集是 {"10","0001","1","0"} ，因此答案是 4 。
其他满足题意但较小的子集包括 {"0001","1"} 和 {"10","1","0"} 。{"111001"} 不满足题意，因为它含4个1 ，大于n的值3 。

示例 2：
输入：strs = ["10", "0", "1"], m = 1, n = 1
输出：2
解释：最大的子集是 {"0", "1"} ，所以答案是2 。
*/

/*
思路:
1 确定dp数组以及下标的含义
dp[i][j]：最多有i个0和j个1的strs的最大子集的大小为dp[i][j]。

2 确定递推公式
dp[i][j]可以由前一个strs里的字符串推导出来，该字符串有zeroNum个0，oneNum个1。
dp[i][j]就可以是dp[i-zeroNum][j-oneNum]+1(加1就是子集长度要加上当前字符串这个元素)。
然后我们在遍历的过程中，取dp[i][j]的最大值。
所以递推公式：dp[i][j] = max(dp[i][j], dp[i-zeroNum][j-oneNum]+1);

此时回想一下01背包的递推公式：dp[j]=max(dp[j],dp[j-weight[i]]+value[i]);
对比一下就会发现，字符串的zeroNum和oneNum相当于物品的重量（weight[i]），字符串本身的个数相当于物品的价值（value[i]）。
这就是一个典型的01背包！只不过物品的重量有了两个维度而已。

3 dp数组如何初始化
01背包的dp数组初始化为0就可以。
因为物品价值不会是负数，初始为0，保证递推的时候dp[i][j]不会被初始值覆盖。

4 确定遍历顺序
01背包一定是外层for循环遍历物品，内层for循环遍历背包容量且从后向前遍历！
那么本题也是，物品就是strs里的字符串，背包容量就是题目描述中的m和n。
有同学可能想，那个遍历背包容量的两层for循环先后循序有没有什么讲究？
没讲究，都是物品重量的一个维度，先遍历哪个都行！

举例推导dp数组
以输入：["10","0001","111001","1","0"]，m = 3，n = 3为
最后dp数组的状态参见1和0.png

复杂度分析
时间复杂度：O(lmn+L)，其中l是数组strs的长度，m和n分别是0和1的容量，L是数组strs中的所有字符串的长度之和。
动态规划需要计算的状态总数是O(lmn)，每个状态的值需要O(1)的时间计算。
对于数组strs中的每个字符串，都要遍历字符串得到其中的0和1的数量，因此需要O(L)的时间遍历所有的字符串。
总时间复杂度是O(lmn+L)。
空间复杂度：O(mn)，其中m和n分别是0和1的容量。使用空间优化的实现，需要创建m+1行n+1列的二维数组dp。
*/

// findMaxForm 本题较难，注意理解m和n都是背包容量，导致题目有多个背包维度
func findMaxForm(strs []string, m, n int) int {
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	for _, str := range strs {
		zeroNum := strings.Count(str, "0")
		oneNum := strings.Count(str, "1")
		for i := m; i >= zeroNum; i-- {
			for j := n; j >= oneNum; j-- {
				dp[i][j] = utils.Max(dp[i][j], dp[i-zeroNum][j-oneNum]+1)
			}
		}
	}
	return dp[m][n]
}

/*
1.8 完全背包问题
有N件物品和一个最多能背重量为W的背包。第i件物品的重量是weight[i]，得到的价值是value[i] 。每件物品都有无限个（也就是可以放入背包多次），
求解将哪些物品装入背包里物品价值总和最大。
完全背包和01背包问题唯一不同的地方就是，每种物品可以重复放入，次数不限。

完全背包与01背包主要不同在于遍历顺序，注意完全背包的物品是可以添加多次的，所以内层遍历背包要从小到大去遍历(正序遍历)
*/

// completeBagProblem 时间复杂度O(M*N)，空间复杂度O(N)
func completeBagProblem(weight, value []int, capacity int) int {
	dp := make([]int, capacity+1)
	for i := 0; i < len(weight); i++ {
		for j := weight[i]; j <= capacity; j++ {
			dp[j] = utils.Max(dp[j], dp[j-weight[i]]+value[i])
		}
	}
	return dp[capacity]
}

/*
leetcode 518
1.9 零钱兑换II
给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。

示例1:
输入: amount = 5, coins = [1, 2, 5] 输出:4 解释: 有四种方式可以凑成总金额: 5=5 5=2+2+1 5=2+1+1+1 5=1+1+1+1+1

示例2:
输入: amount = 3, coins = [2] 输出: 0 解释: 只用面额2的硬币不能凑成总金额3。

示例3:
输入: amount = 10, coins = [10] 输出: 1

注意，你可以假设：
0 <= amount (总金额) <= 5000
1 <= coin (硬币面额) <= 5000
硬币种类不超过500种
结果符合32位符号整数
*/

/*
思路:
因为每一种面额的硬币有无限个，所以这是一个求组合数的完全背包问题
1 确定dp数组以及下标含义
dp[j]表示凑足钱币总额为j的组合个数为dp[j]

2 确定递推公式
dp[j]可以由dp[j-coins[i]]推出，凑足钱币总额为j-coins[i]的组合个数为dp[j-coins[i]]，此时再找到coins[i]的钱币，便能
得出凑足钱币总额为j的组合，也就是有dp[j-coins[i]]个组合可以凑出钱币总额为j，明显这里就是所有的dp[j-coins[i]]累加
所以，递推公式为dp[j] += dp[j - coins[i]]

3 dp数组初始化
dp[0] = 1
从dp[i]的含义上来讲就是，凑成总金额0的货币组合数为1。

4 确定遍历顺序
本题只能是先遍历物品，后遍历背包，因为题目求的是组合数，不讲究元素的顺序，如果先遍历背包，后遍历物品计算的是排列数，会有重复
特别注意:
如果求组合数就是外层for循环遍历物品，内层for遍历背包。
如果求排列数就是外层for遍历背包，内层for循环遍历物品。

5 举例推导dp数组
输入: amount = 5, coins = [1, 2, 5] ，dp状态图如下：
零钱兑换.png
*/

// change 时间复杂度O(amount*len(coins))，空间复杂度O(amount)
func change(amount int, coins []int) int {
	dp := make([]int, amount+1)
	dp[0] = 1
	for i := 0; i < len(coins); i++ {
		for j := coins[i]; j <= amount; j++ {
			dp[j] += dp[j-coins[i]]
		}
	}
	return dp[amount]
}

/*
leetcode 377
1.10 组合总和
给定一个由正整数组成且不存在重复数字的数组，找出和为给定目标正整数的组合的个数。

示例:
nums = [1, 2, 3] target = 4
所有可能的组合为： (1, 1, 1, 1) (1, 1, 2) (1, 2, 1) (1, 3) (2, 1, 1) (2, 2) (3, 1)
请注意，顺序不同的序列被视作不同的组合。
因此输出为7。
*/

/*
本题与1.8 零钱兑换类似，唯一不同的是本题顺序不同的序列被视作不同的组合，所以求的是排列数，而后者求的是组合数，因此
在遍历顺序上必须先遍历背包，再遍历物品
*/

func combinationSum(nums []int, target int) int {
	dp := make([]int, target+1)
	dp[0] = 1
	// 在遍历顺序上必须先遍历背包，再遍历物品
	for j := 0; j <= target; j++ {
		for i := 0; i < len(nums); i++ {
			if j >= nums[i] {
				dp[j] += dp[j-nums[i]]
			}
		}
	}
	return dp[target]
}

/*
1.11 爬楼梯进阶版
假设你正在爬楼梯。需要n阶你才能到达楼顶。
每次你可以爬一个台阶，两个台阶，三个台阶，.......，直到m个台阶。你有多少种不同的方法可以爬到楼顶呢？
本题其实是一个背包问题，楼顶n阶是背包，每次爬的一个台阶，两个台阶，三个台阶就是物品，问爬到楼顶有几种方法其实就是问装满背包有几种方法。
由于爬的台阶数可以重复，所以具体而言这是一个完全背包问题，并且先爬一个台阶，再爬两个台阶到三层；与先爬两个个台阶，再爬一个台阶到三层是不同
的两种爬楼梯方法，所以本题是完全背包的求排列问题，因此遍历顺序就必须是先遍历背包，再遍历物品，且遍历背包时必须是正序遍历
*/

// climbStairsComplex m表示一次最多可以爬m个台阶
func climbStairsComplex(m, n int) int {
	dp := make([]int, n+1)
	dp[0] = 1
	for j := 1; j <= n; j++ {
		for i := 1; i <= m; i++ {
			if j >= i {
				dp[j] += dp[j-i]
			}
		}
	}
	return dp[n]
}

/*
leetcode 322
1.12 零钱兑换I
给定不同面额的硬币 coins 和一个总金额amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成
总金额，返回 -1。

你可以认为每种硬币的数量是无限的。

示例1：
输入：coins = [1, 2, 5], amount = 11 输出：3 解释：11 = 5 + 5 + 1

示例2：
输入：coins = [2], amount = 3 输出：-1

示例3：
输入：coins = [1], amount = 0 输出：0

示例4：
输入：coins = [1], amount = 1 输出：1

示例5：
输入：coins = [1], amount = 2 输出：2

提示：
1 <= coins.length <= 12
1 <= coins[i] <= 2^31 - 1
0 <= amount <= 10^4
*/

/*
思路: 由于每种硬币的数量是无限的，所以这是一个完全背包问题
1 确定dp数组以及下标含义
dp[j]表示凑足总金额为j所需的最少硬币数为dp[j]

2 确定递推公式
dp[j]明显可以由dp[j-coins[i]]推出，凑足金额为j-coins[i]的最少硬币数为[j-coins[i]]，那么只需要加上一个硬币coins[i]就可以
凑足j，也就是dp[j] = dp[j-coins[i]] + 1
所以dp[j]要取所有dp[j-coins[i]] + 1中最小的。
故递推公式为dp[j] = min(dp[j], dp[j-coins[i]] + 1) 不断迭代dp[j]，得到最小值

3 dp数组初始化
首先dp[0] = 0 这个很好理解，凑足金额为0的最少硬币数为0，由于是求最小值，所以其他下标对应元素一律初始化为最大整数，
这样可以确保遍历中对dp[j]赋值时不会被初始值覆盖掉。

4 确定遍历顺序
由于题目求最少硬币个数，并非求组合数或排列数，所以遍历顺序无所谓，可以先遍历物品，再遍历背包，也可以先遍历背包，再遍历物品。
但由于是完全背包，所以遍历背包时，必须是正序

5 举例推导dp数组
以输入：coins = [1, 2, 5], amount = 5为例
下标   0  1  2  3  4   5
dp[j] 0  1  1  2   2  1
*/

func leastCoinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	dp[0] = 0
	for i := 1; i <= amount; i++ {
		dp[i] = math.MaxInt32
	}
	for i, n := 0, len(coins); i < n; i++ {
		for j := coins[i]; j <= amount; j++ {
			if dp[j-coins[i]] != math.MaxInt32 {
				dp[j] = utils.Min(dp[j], dp[j-coins[i]]+1)
			}
		}
	}
	// dp[amount] == math.MaxInt32，说明dp[amount]的值还是初始值，并未被更新，此时返回-1
	if dp[amount] == math.MaxInt32 {
		return -1
	}
	return dp[amount]
}

/*
leetcode 279
1.13 完全平方数
给定正整数n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于n。你需要让组成和的完全平方数的个数最少。

给你一个整数n ，返回和为n的完全平方数的最少数量 。

完全平方数是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而3和11不是。

示例1：
输入：n = 12 输出：3 解释：12 = 4 + 4 + 4

示例2：
输入：n = 13 输出：2 解释：13 = 4 + 9

提示：
1 <= n <= 10^4
*/

/*
初看此问题感觉无从下手，其实这就是一个完全背包问题，物品就是完全平方数，背包就是目标值n, 翻译过来就是最少需要多少个物品(完全平方数)能够凑够
背包的容量n，由于你可以重复使用完全平方数(物品),所以这实际上是一个完全背包问题。

1 确定dp数组及其下标含义
dp[j]表示和为j所需要的最少完全平方数个数为dp[j]

2 确定递推公式
dp[j]明显只能由dp[j-i^2]推出，要凑够和为j-i*i所需的最少完全平方数为dp[j-i^2],那么只要再来一个完全平方数i^2,就能得到j,所以
dp[j] = dp[j-i^2] + 1，由于是求最小值，所以dp[j] = min(dp[j], dp[j-i^2]+1)

3 初始化dp数组
dp[0] = 0, 这个完全是为了递推公式，dp[1]=1=dp[1-1*1]+1=dp[0]+1=1, 所以dp[0]=0
非0下标的值应该初始化为最大整数，以便遍历时可以被最小值迭代替换掉。

4 确定遍历顺序，完全背包，且本题是求最小数，不是求排列，也不是求组合，所以可以外层遍历物品，内层遍历背包，或者反过来都行，但是完全背包
遍历背包必须是正序遍历
*/

func numSquares(n int) int {
	dp := make([]int, n+1)
	for i := 1; i <= n; i++ {
		dp[i] = math.MaxInt32
	}
	for i := 1; i*i <= n; i++ {
		for j := i * i; j <= n; j++ {
			dp[j] = utils.Min(dp[j], dp[j-i*i]+1)
		}
	}
	return dp[n]
}

/*
leetcode 139
1.14 单词拆分
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
初看此题感觉无从下手，其实本题是完全背包问题。
首先将本题转化为背包问题，单词就是物品，字符串s就是背包，单词列表wordDict中的单词就是物品，单词能否组成字符串s，
就是问物品能不能把背包装满。由于可以重复使用字典中的单词，说明这是一个完全背包问题。

动规五部曲分析如下：

1 确定dp数组以及下标的含义
dp[i]表示字符串s长度为i的话，dp[i]为true，表示可以拆分为一个或多个在字典中出现的单词。

2 确定递推公式
如果确定dp[j]是true，且 [j, i] 这个区间的子串出现在字典里，那么dp[i]一定是true。（j < i ）。
所以递推公式是如果([j, i]这个区间的子串出现在wordDict中并且dp[j]是true，那么dp[i] = true。

3 dp数组如何初始化
从递归公式中可以看出，dp[i] 的状态依靠 dp[j]是否为true，那么dp[0]就是递归的根基，dp[0]一定要为true，
否则递归下去后面都是false了。那么dp[0]初始为true完全就是为了推导公式，不必深究。

下标非0的dp[i]初始化为false，只要没有被覆盖说明都是不可拆分为一个或多个在字典中出现的单词。

4 确定遍历顺序
题目中说是拆分为一个或多个在字典中出现的单词，所以这是完全背包。
但本题有特殊性，因为是要求子串，最好是遍历背包放在外循环，将遍历物品放在内循环。

如果要是外层for循环遍历物品，内层for遍历背包，就需要把所有的子串都预先放在一个容器里。（如果不理解的话，可以自己尝试这么写一写就理解了）
所以最终的遍历顺序为：遍历背包放在外循环，将遍历物品放在内循环。因为是完全背包，所以内循环从前到后。

5 举例推导dp数组
略
*/

// wordBreak 时间复杂度O(N*N),空间复杂度O(N),N为字符串s的长度
func wordBreak(s string, wordDict []string) bool {
	n := len(s)
	// 记录wordDict中出现的单词
	seen := make(map[string]bool)
	for _, word := range wordDict {
		seen[word] = true
	}
	dp := make([]bool, n+1)
	dp[0] = true
	for i := 1; i <= n; i++ {
		for j := 0; j < i; j++ {
			if dp[j] && seen[s[j:i]] {
				dp[i] = true
				break
			}
		}
	}
	return dp[n]
}

/*
1.15 多重背包问题
有N种物品和一个容量为V的背包。第i种物品最多有Mi件可用，每件耗费的空间是Ci，价值是Wi。求解将哪些物品装入背包可使这些物品的耗费的空间
总和不超过背包容量，且价值总和最大。

多重背包和01背包是非常像的，为什么和01背包像呢？
每件物品最多有Mi件可用，把Mi件摊开，其实就是一个01背包问题了。
时间复杂度：O(m * n * k) m：物品种类个数，n背包容量，k单类物品数量
例如：
背包最大重量为10。

物品为：
       重量  价值  数量
物品0    1  15  2
物品1    3  20  3
物品2    4  30  2
问背包能背的物品最大价值是多少？
*/

func multiBagProblem(weight, value, nums []int, capacity int) int {
	dp := make([]int, capacity+1)
	n := len(nums)
	for i := 0; i < n; i++ {
		for nums[i] > 1 {
			weight = append(weight, weight[i])
			value = append(value, value[i])
			nums[i]--
		}
	}
	for i := 0; i < n; i++ {
		for j := capacity; j >= weight[i]; j-- {
			dp[j] = utils.Max(dp[j], dp[j-weight[i]]+value[i])
		}
	}
	return dp[capacity]
}

/*
leetcode 931. 下降路径最小和
1.16 给你一个n x n 的方形整数数组matrix ，请你找出并返回通过matrix的下降路径的最小和。
下降路径可以从第一行中的任何元素开始，并从每一行中选择一个元素。在下一行选择的元素和当前行所选元素最多相隔
一列（即位于正下方或者沿对角线向左或者向右的第一个元素）。具体来说，位置 (row, col) 的下一个元素应当是
(row + 1, col - 1)、(row + 1, col) 或者 (row + 1, col + 1) 。
输入：matrix = [[2,1,3],[6,5,4],[7,8,9]]
输出：13
*/

/*
动规五部曲分析如下：

1 确定dp数组以及下标的含义
dp[i][j]表示从位置matrix[i][j]开始的下降路径最小和。

2 确定递推公式
根据题意，从位置(i,j)可以下降到(i+1,j-1),(i+1,j),(i+1,j+1)三个位置。但要注意考虑特殊情况，如果
j==0, 也就是最左列，此时就只能下降到(i+1,j),(i+1,j+1)；同理，如果j==n-1,也就是最右边一列，那么就只能下降到
(i+1,j-1),(i+1,j)两个位置，所以递推公式是:
j=0时，dp[i][j]=Min(dp[i+1][j], dp[i+1][j+1]) + matrix[i][j]
1<=j<n-1时，dp[i][j]=Min(dp[i+1][j-1], Min(dp[i+1][j], dp[i+1][j+1])) + matrix[i][j]
j=n-1时，dp[i][j]=Min(dp[i+1][j-1], dp[i+1][j]) + matrix[i][j]

3 dp数组如何初始化
从递归公式中可以看出，dp[i]的值是从下一行(i+1行)推导而来，而最后一行，即n-1行的下降路径最小和显然就等于
matrix[n-1],于是有dp[n-1][j] = matrix[n-1][j]

4 确定遍历顺序
从递推公式可知，dp[i]的值是从下一行(i+1行)推导而来, 所以遍历顺序应该是从下往上。

5 举例推导dp数组
略
由于下降路径可以从第一行中的任何元素开始，因此最终的答案肯定是dp[0]这一行中的最小值。
*/

func minFallingPathSum(matrix [][]int) int {
	n := len(matrix)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
	}
	for j := 0; j < n; j++ {
		dp[n-1][j] = matrix[n-1][j]
	}
	for i := n - 2; i >= 0; i-- {
		for j := 0; j < n; j++ {
			if j == 0 {
				dp[i][j] = utils.Min(dp[i+1][j], dp[i+1][j+1]) + matrix[i][j]
			} else if j == n-1 {
				dp[i][j] = utils.Min(dp[i+1][j], dp[i+1][j-1]) + matrix[i][j]
			} else {
				dp[i][j] = utils.Min(dp[i+1][j], utils.Min(dp[i+1][j-1], dp[i+1][j+1])) + matrix[i][j]
			}
		}
	}
	return utils.MinValueOfArray(dp[0])
}

/*
leetcode 120. 三角形最小路径和
1.17 给定一个三角形triangle，找出自顶向下的最小路径和。
每一步只能移动到下一行中相邻的结点上。相邻的结点在这里指的是下标与上一层结点下标相同或者等于上一层结点下标+1
的两个结点。也就是说，如果正位于当前行的下标i，那么下一步可以移动到下一行的下标i或i+1 。

示例1：
输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
输出：11
解释：如下面简图所示：
   2
  3 4
 6 5 7
4 1 8 3
自顶向下的最小路径和为11（即，2+3+5+1= 11）。
*/

/*
本题与leetcode 931. 下降路径最小和逻辑基本一致，所以很容易以相同的思路解决。
*/

func minimumTotal(triangle [][]int) int {
	n := len(triangle)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, i+1)
	}
	for j := 0; j < n; j++ {
		dp[n-1][j] = triangle[n-1][j]
	}
	for i := n - 2; i >= 0; i-- {
		for j := 0; j < i+1; j++ {
			dp[i][j] = utils.Min(dp[i+1][j], dp[i+1][j+1]) + triangle[i][j]
		}
	}
	return dp[0][0]
}

/*
leetcode 1277. 统计全为1的正方形子矩阵
1.18 给你一个m * n的矩阵，矩阵中的元素不是 0 就是 1，请你统计并返回其中完全由1组成的正方形子矩阵的个数。

示例1：
输入：matrix =
[
[0,1,1,1],
[1,1,1,1],
[0,1,1,1]
]
输出：15
解释：
边长为 1 的正方形有 10 个。
边长为 2 的正方形有 4 个。
边长为 3 的正方形有 1 个。
正方形的总数 = 10 + 4 + 1 = 15.

提示：
1 <= arr.length <= 300
1 <= arr[0].length <= 300
0 <= arr[i][j] <= 1
*/

/*
思路:动态规划
我们用dp[i][j]表示以matrix[i][j]为右下角的正方形最大边长。dp[i][j] = x 也表示以 (i, j)
为右下角的正方形的数目为x（即边长为 1, 2, ..., x 的正方形各一个）。
在计算出所有的dp[i][j]后，我们将它们累加，即可得到矩阵matrix中正方形的数目。
我们尝试挖掘dp[i][j]与相邻位置的关系来计算出dp[i][j]的值。
若dp[i][j]=4，那么我们可以看到其左侧位置(i,j-1),上方位置(i-1,j),左上方位置(i-1,j-1)均可作为一个边长为3
的正方形的右下角，也就是说，这些位置的dp值(边长)至少为3，即:
如图矩阵中的正方形.png所示:
dp[i,j-1] >= dp[i][j]-1
dp[i-1,j] >= dp[i][j]-1
dp[i-1,j-1] >= dp[i][j]-1
将以上不等式联立，可得:
min(dp[i,j-1], dp[i-1,j], dp[i-1,j-1]) >= dp[i][j]-1
简单移项，可得:
dp[i][j] <= min(dp[i,j-1], dp[i-1,j], dp[i-1,j-1]) + 1

这是我们通过固定dp[i][j] 的值，判断其相邻位置与之的关系得到的不等式。同理，我们也可以固定dp[i][j]相邻位置
的值，得到另外的限制条件。

若dp[i,j-1], dp[i-1,j], dp[i-1,j-1]中最小值为3, 也就是说(i,j-1),(i-1,j),(i-1,j-1)均可作为一个边长
至少为3的正方形的右下角，那么如果位置(i,j)的元素为1(即matrix[i][j]=1)，那么位置(i,j)可以作为一个边长为4的
正方形的右下角，即dp[i][j]>=4, 即:
dp[i][j] >= min(dp[i,j-1], dp[i-1,j], dp[i-1,j-1]) + 1
将其与上一个不等式联立，可得:
dp[i][j] = min(dp[i,j-1], dp[i-1,j], dp[i-1,j-1]) + 1

这样我们就得到了dp[i][j]的递推公式，此外还要考虑一些边界条件和特殊情况，即可得到完整的递推公式
1 if i==0 || j==0, dp[i][j] = matrix[i][j](因为dp[i][j]表示以matrix[i][j]为右下角的正方形最大边长)
2 else if matrix[i][j] == 0, dp[i][j] = 0
3 else dp[i][j] = min(dp[i,j-1], dp[i-1,j], dp[i-1,j-1]) + 1
*/

func countSquares(matrix [][]int) int {
	m, n := len(matrix), len(matrix[0])
	count := 0
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
		for j := 0; j < n; j++ {
			if i == 0 || j == 0 {
				dp[i][j] = matrix[i][j]
			} else if matrix[i][j] == 0 {
				dp[i][j] = 0
			} else {
				dp[i][j] = utils.Min(dp[i-1][j], utils.Min(dp[i][j-1], dp[i-1][j-1])) + 1
			}
			count += dp[i][j]
		}
	}
	return count
}

/*
leetcode 221. 最大正方形
1.19 在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。

示例1:
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],
["1","0","0","1","0"]]
输出：4

提示：
m == matrix.length
n == matrix[i].length
1 <= m, n <= 300
matrix[i][j] 为 '0' 或 '1'
*/

func maximalSquare(matrix [][]byte) int {
	maxSide := 0
	dp := make([][]int, len(matrix))
	for i := 0; i < len(matrix); i++ {
		dp[i] = make([]int, len(matrix[i]))
		for j := 0; j < len(matrix[i]); j++ {
			dp[i][j] = int(matrix[i][j] - '0')
			if dp[i][j] == 1 {
				maxSide = 1
			}
		}
	}
	for i := 1; i < len(matrix); i++ {
		for j := 1; j < len(matrix[i]); j++ {
			if dp[i][j] == 1 {
				dp[i][j] = utils.Min(dp[i-1][j], utils.Min(dp[i][j-1], dp[i-1][j-1])) + 1
				if maxSide < dp[i][j] {
					maxSide = dp[i][j]
				}
			}
		}
	}
	return maxSide * maxSide
}

/*
圆环上有10个点，编号为0~9。从0点出发，每次可以逆时针和顺时针走一步，问走n步回到0点共有多少种走法。

输入: 2
输出: 2
解释：有2种方案。分别是0->1->0和0->9->0
*/

/*
思路:动态规划
走n步到0的方案数=走n-1步到1的方案数+走n-1步到9的方案数。
类似的，走i步到j点的方案数等于走i-1步到j点的相邻两个点，也就是j-1和j+1两个点的方案数之和
因此，若设dp[i][j]为从0点出发走i步到j点的方案数，则递推式为：
dp[i][j] = dp[i-1][(j-1+length)%length] + dp[i-1][(j+1)%length]
公式之所以取余是因为j-1或j+1可能会超过圆环0~9的范围
*/

func backToOrigin(n int) int {
	length := 10
	dp := make([][]int, n+1)
	for i := 0; i <= n; i++ {
		dp[i] = make([]int, length)
	}
	dp[0][0] = 1
	for i := 1; i <= n; i++ {
		for j := 0; j < length; j++ {
			dp[i][j] = dp[i-1][(j-1+length)%length] + dp[i-1][(j+1)%length]
		}
	}
	return dp[n][0]
}

func backToOriginSimple(n int) int {
	if n == 0 {
		return 1
	}
	length := 10
	// 初始化两个数组，分别表示上一步和当前步
	prev, cur := make([]int, length), make([]int, length)
	// 初始状态，第0步在点0
	prev[0] = 1
	for i := 1; i <= n; i++ {
		for j := 0; j < length; j++ {
			// 计算当前步的方案数
			cur[j] = prev[(j-1+length)%length] + prev[(j+1)%length]
		}
		// 更新上一步为当前步
		prev, cur = cur, prev
	}
	// 返回走n步回到点0的方案数
	return prev[0]
}

/*
leetcode 97 交错字符串
给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的。
两个字符串 s 和 t 交错 的定义与过程如下，其中每个字符串都会被分割成若干非空
子字符串：
s = s1 + s2 + ... + sn
t = t1 + t2 + ... + tm
|n - m| <= 1
交错 是 s1 + t1 + s2 + t2 + s3 + t3 + ... 或者 t1 + s1 + t2 + s2 + t3 + s3 + ...
注意：a + b 意味着字符串 a 和 b 连接。

输入：s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
输出：true

提示：

0 <= s1.length, s2.length <= 100
0 <= s3.length <= 200
s1、s2、和 s3 都由小写英文字母组成

进阶：您能否仅使用 O(s2.length) 额外的内存空间来解决它?
*/

/*
思路:动态规划
定义状态：

dp[i][j] 表示 s1 的前 i 个字符 和 s2 的前 j 个字符 是否能够交错组成 s3 的前 i+j 个字符。
状态转移方程：

如果 s1[i-1] == s3[i+j-1] 且 dp[i-1][j] 为真，则 dp[i][j] = true。
如果 s2[j-1] == s3[i+j-1] 且 dp[i][j-1] 为真，则 dp[i][j] = true。
即：
dp[i][j] = (s1[i-1] == s3[i+j-1] && dp[i-1][j]) || (s2[j-1] == s3[i+j-1] && dp[i][j-1])

推导:
dp[i][j] = true 的条件为：
当前字符由 s1 提供，且之前的状态满足：
如果 s1[i-1] == s3[i+j-1] 且 dp[i-1][j] = true：
意味着 s3 的第 i+j-1 个字符来源于 s1 的第 i-1 个字符；
同时，剩下的 s1 的前 i-1 和 s2 的前 j 已经可以交错组成 s3 的前 i+j-1 个字符。

当前字符由 s2 提供，且之前的状态满足：
如果 s2[j-1] == s3[i+j-1] 且 dp[i][j-1] = true：
意味着 s3 的第 i+j-1 个字符来源于 s2 的第 j-1 个字符；
同时，剩下的 s1 的前 i 和 s2 的前 j-1 已经可以交错组成 s3 的前 i+j-1 个字符

边界条件：
dp[0][0] = true，表示空的 s1 和空的 s2 可以组成空的 s3。

当 i=0 时，dp[0][j] 仅依赖 s2：
dp[0][j] = dp[0][j-1] && (s2[j-1] == s3[j-1])

当 j=0 时，dp[i][0] 仅依赖 s1：
dp[i][0] = dp[i-1][0] && (s1[i-1] == s3[i-1])

最终结果：
返回 dp[len(s1)][len(s2)]。
*/

func isInterleave(s1 string, s2 string, s3 string) bool {
	m, n, k := len(s1), len(s2), len(s3)
	// 如果长度不匹配，直接返回 false
	if m+n != k {
		return false
	}
	// 初始化二维 DP 数组
	dp := make([][]bool, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]bool, n+1)
	}
	// 边界条件
	// 空的 s1 和空的 s2 可以组成空的 s3
	dp[0][0] = true
	// 填充第一列 (只使用 s1)
	for i := 1; i <= m; i++ {
		dp[i][0] = dp[i-1][0] && s1[i-1] == s3[i-1]
	}
	// 填充第一行 (只使用 s2)
	for j := 1; j <= n; j++ {
		dp[0][j] = dp[0][j-1] && s2[j-1] == s3[j-1]
	}
	// 填充整个dp数组
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			// 状态转移方程
			dp[i][j] = (dp[i-1][j] && s1[i-1] == s3[i+j-1]) || (dp[i][j-1] && s2[j-1] == s3[i+j-1])
		}
	}
	// 返回最终结果
	return dp[m][n]
}

// isInterleaveSimple 使用一维数组降低空间复杂度
func isInterleaveSimple(s1 string, s2 string, s3 string) bool {
	n, m, k := len(s1), len(s2), len(s3)
	// 长度不匹配，直接返回 false
	if n+m != k {
		return false
	}
	// dp[j] 表示 s1 的前 i 个字符和 s2 的前 j 个字符是否能组成 s3 的前 i+j 个字符
	dp := make([]bool, m+1)
	// 初始化 dp 数组
	for j := 0; j <= m; j++ {
		dp[j] = j == 0 || (dp[j-1] && s2[j-1] == s3[j-1])
	}
	// 动态规划填表
	for i := 1; i <= n; i++ {
		dp[0] = dp[0] && s1[i-1] == s3[i-1] // 更新第一列
		for j := 1; j <= m; j++ {
			dp[j] = (dp[j] && s1[i-1] == s3[i+j-1]) || (dp[j-1] && s2[j-1] == s3[i+j-1])
		}
	}
	return dp[m]
}
