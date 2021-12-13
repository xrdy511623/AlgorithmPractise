package Greedy

import (
	"AlgorithmPractise/Utils"
	"sort"
)

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

/*
1.2 摆动序列
如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为摆动序列。第一个差（如果存在的话）可能是正数或负数。少于两个元素的序列也是摆动序列。

例如， [1,7,4,9,2,5] 是一个摆动序列，因为差值 (6,-3,5,-7,3) 是正负交替出现的。相反, [1,4,7,2,5] 和 [1,7,4,5,5] 不是摆动序列，第一个序列是
因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。

给定一个整数序列，返回作为摆动序列的最长子序列的长度。 通过从原始序列中删除一些（也可以不删除）元素来获得子序列，剩下的元素保持其原始顺序。

示例1:
输入: [1,7,4,9,2,5]
输出: 6
解释: 整个序列均为摆动序列。

示例2:
输入: [1,17,5,10,13,15,10,5,16,8]
输出: 7
解释: 这个序列包含几个长度为 7 摆动序列，其中一个可为[1,17,10,13,10,16,8]。

示例3:
输入: [1,2,3,4,5,6,7,8,9]
输出: 2
*/

/*
贪心的本质是选择每一阶段的局部最优,从而达到全局最优, 本题符合这个套路，而且举不出反例，所以可以用贪心解决
本题实质其实是让序列有尽可能多的局部峰值。
局部最优：删除单调坡度上的节点（不包括单调坡度两端的节点），那么这个坡度就可以有两个局部峰值。
整体最优：整个序列有最多的局部峰值，从而达到最长摆动序列。
*/

// WiggleMaxLength 时间复杂度O(N),空间复杂度O(1)
func WiggleMaxLength(nums []int) int {
	n := len(nums)
	if n <= 1 {
		return n
	}
	preDiff, curDiff := 0, 0
	// 单个元素也是摆动序列，故maxLength初始化为1
	maxLength := 1
	for i := 1; i < n; i++ {
		curDiff = nums[i] - nums[i-1]
		// 只要满足正负数交替出现，则累加最大摆动子序列长度，并更新preDiff为curDiff
		if (curDiff > 0 && preDiff <= 0) || (curDiff < 0 && preDiff >= 0) {
			maxLength++
			preDiff = curDiff
		}
	}
	return maxLength
}

/*
1.3 最大子序和
给定一个整数数组nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
示例:
输入: [-2,1,-3,4,-1,2,1,-5,4] 输出: 6 解释: 连续子数组 [4,-1,2,1]的和最大，为 6。
*/

/*
贪心的思路为局部最优：当前“连续和”为负数的时候立刻放弃，从下一个元素重新计算“连续和”，因为负数加上下一个元素 “连续和”只会越来越小。
从而推出全局最优：选取最大“连续和”
*/

// MaxSubArray 时间复杂度O(N),空间复杂度O(1)
func MaxSubArray(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return nums[0]
	}
	max := nums[0]
	for i := 1; i < n; i++ {
		if nums[i-1] > 0 {
			// 前一个数大于0，才有累加的价值,否则重新开始计算子序列和
			nums[i] += nums[i-1]
		}
		if max < nums[i] {
			max = nums[i]
		}
	}
	return max
}

/*
1.4 买卖股票的最佳时机II
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

示例 1:

输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4。随后，在第 4 天（股票价格 = 3）
的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。

示例 2:
输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。注意你不能在第 1 天和第 2 天
接连购买股票，之后再将它们卖出。因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。

示例 3:
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为0。

提示：
1 <= prices.length <= 3 * 10 ^ 4
0 <= prices[i] <= 10 ^ 4
*/

// MaxProfit 时间复杂度O(N),空间复杂度O(1)
func MaxProfit(prices []int) int {
	maxProfit := 0
	for i := 1; i < len(prices); i++ {
		if profit := prices[i] - prices[i-1]; profit > 0 {
			maxProfit += profit
		}
	}
	return maxProfit
}

/*
1.5 跳跃游戏
给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个位置。

示例 1:
输入: [2,3,1,1,4]
输出: true
解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。

示例 2:
输入: [3,2,1,0,4]
输出: false
解释: 无论怎样，你总会到达索引为3的位置。但该位置的最大跳跃长度是0， 所以你永远不可能到达最后一个位置。
*/

// CanJump 时间复杂度O(N),空间复杂度O(1)
func CanJump(nums []int) bool {
	// 最远可到达位置rightMost，初始值为0
	n, rightMost := len(nums), 0
	for i := 0; i < n; i++ {
		// i <= rightMost证明i这个位置可达，然后更新rightMost
		if i <= rightMost {
			rightMost = Utils.Max(i+nums[i], rightMost)
			// 如果rightMost能延伸到数组末尾位置，证明可以跳到末尾
			if rightMost >= n-1 {
				return true
			}
		}
	}
	return false
}

/*
1.6 跳跃游戏II
给定一个非负整数数组，你最初位于数组的第一个位置。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
你的目标是使用最少的跳跃次数到达数组的最后一个位置。

示例:
输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是2。从下标为0跳到下标为1的位置，跳1步，然后跳3步到达数组的最后一个位置。
说明: 假设你总是可以到达数组的最后一个位置。
*/

/*
思路:反向查找出发位置
我们的目标是到达数组的最后一个位置，因此我们可以考虑最后一步跳跃前所在的位置，该位置通过跳跃能够到达最后一个位置。
如果有多个位置通过跳跃都能够到达最后一个位置，那么我们应该如何进行选择呢？直观上来看，我们可以「贪心」地选择距离最后一个位置最远的那个位置，
也就是对应下标最小的那个位置。因此，我们可以从左到右遍历数组，选择第一个满足要求的位置。
找到最后一步跳跃前所在的位置之后，我们继续贪心地寻找倒数第二步跳跃前所在的位置，以此类推，直到找到数组的开始位置。
*/

// Jump 时间复杂度O(N*N),空间复杂度O(1)
func Jump(nums []int) int {
	position := len(nums) - 1
	steps := 0
	for position > 0 {
		for i := 0; i < position; i++ {
			if i+nums[i] >= position {
				position = i
				steps++
				break
			}
		}
	}
	return steps
}

/*
思路:正向查找可到达的最大位置
我们维护当前能够到达的最大下标位置，记为边界。我们从左到右遍历数组，到达边界时，更新边界并将跳跃次数增加1。
*/

// JumpSimple 时间复杂度O(N),空间复杂度O(1)
func JumpSimple(nums []int) int {
	n := len(nums)
	// 初始化右边界end，当前所能到达的最大位置maxPosition，以及跳跃次数count为0
	end, maxPosition, count := 0, 0, 0
	for i := 0; i < n-1; i++ {
		// 更新当前所能到达的最大位置maxPosition
		maxPosition = Utils.Max(i+nums[i], maxPosition)
		// 当下标移动到右边界时，跳跃次数需要加1，同时更新右边界为maxPosition
		if i == end {
			end = maxPosition
			count++
		}
	}
	return count
}

/*
1.7 K次取反后最大化的数组和
给定一个整数数组 A，我们只能用以下方法修改该数组：我们选择某个索引i并将A[i]替换为-A[i]，然后总共重复这个过程K次。（我们可以多次选择同一个索引i。）
以这种方式修改数组后，返回数组可能的最大和。

示例1：
输入：A = [4,2,3], K = 1
输出：5
解释：选择索引 (1,) ，然后 A 变为 [4,-2,3]。

示例 2：
输入：A = [3,-1,0,2], K = 3
输出：6
解释：选择索引 (1, 2, 2) ，然后 A 变为 [3,1,0,2]。

示例3：
输入：A = [2,-3,-1,5,-4], K = 2
输出：13
解释：选择索引 (1, 4) ，然后 A 变为 [2,3,-1,5,4]。
提示：

1 <= A.length <= 10000
1 <= K <= 10000
-100 <= A[i] <= 100
*/

/*
思路:
由于我们必须反转k次，那么有不止k个负数的话，我们要反转里面最小的k个，这样数组和sum最大。有不到k个负数的话（数组会变为全部为正），
剩下的次数就反复反转所有数里面绝对值最小的那个(如果剩下偶数次负负得正所以sum不变，奇数次相当于只反转一次最小的那个）
*/

// 时间复杂度O(N),空间复杂度O(1)
func LargestSumAfterKNegations(nums []int, k int) int {
	sum := 0
	minAbs := 101
	sort.Ints(nums)
	for _, num := range nums {
		minAbs = Utils.MinAbs(minAbs, num)
		if num < 0 && k > 0 {
			sum -= num
			k--
		} else {
			sum += num
		}
	}
	if k > 0 && k%2 == 1 {
		return sum - 2*minAbs
	}
	return sum
}

/*
1.8 加油站
在一条环路上有N个加油站，其中第i个加油站有汽油gas[i]升。

你有一辆油箱容量无限的的汽车，从第i个加油站开往第i+1个加油站需要消耗汽油cost[i] 升。你从其中的一个加油站出发，
开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

说明:

如果题目有解，该答案即为唯一答案。
输入数组均为非空数组，且长度相同。
输入数组中的元素均为非负数。

示例 1:
输入:
gas = [1,2,3,4,5]
cost = [3,4,5,1,2]
输出: 3 解释:

从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
因此，3 可为起始索引。

示例2:
输入:
gas = [2,3,4]

cost = [3,4,3]

输出: -1

解释: 你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。我们从 2 号加油站出发，
可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油。开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油。
开往 1 号加油站，此时油箱有 3 - 3 + 3 = 3 升汽油。你无法返回 2 号加油站，因为返程需要消耗 4 升汽油，
但是你的油箱只有 3 升汽油。因此，无论怎样，你都不可能绕环路行驶一周。
*/

/*
思路:
根据题意，我们可以得出两个结论:
1 将每个加油站的剩余油量累加给left,即left+=gas[i]-cost[i],如果left<0,那么从出发站到i都不是起点；
2 如果sum(gas)>=cost(gas)，那么问题一定有解。

为什么？

先说第一条，从起点站a出发，必须gas[a]−cost[a]>=0才能到b站。假设去到c站加完油发现去不了d站，只是说明a站不是合格起点吗？
到不了d站，就说明：(gas[a]−cost[a])+(gas[b]−cost[b])+(gas[c]−cost[c])<0
即：left(a)+left(b)+left(c)<0

已知：left(a)>=0
所以：left(b)+left(c)<0

所以，b站不能作为起点，因为去不了d。

能来到c站，肯定到过b站，所以有:left(a)+left(b)>=0

又因为:left(a)+left(b)+left(c)<0

有:left(c)<0

c站也不能作为起点，因为去不了d！我们归纳出下面结论1。

再说第二条
我们从起点0开始，累加每个站点的gas[i]−cost[i]，即left(i)
当站i累加完left(i)后，如果小于0，则站0到站i都不是起点，[0,i]段的sum(left)<0
我们将i+1作为新的起点，重新累加每个站点的left(i)
当站j累加完left(j)，如果小于0，则站i+1到站j都不是起点。[i+1,j]段sum(left)<0
继续考察新起点……但是，不可能一直sum(left)<0下去
因为sum(gas)>=sum(cost)是前提，对于整个数组有sum(left)>=0
因此必然有一段sum(left)>0，假设此时起点更新为k，以k为起点的这一段能加到足够的油，足以填补其他段欠缺的量(<0)。

sum(0,i)+sum(i+1,j)+sum(j+1, k-1)+sum(k,n-1)>=0
如果sum(0,i)+sum(i+1,j)+sum(j+1, k-1)<0, 则sum(k,n-1)>0,且(k,n-1)这一段正数足以覆盖前面这一段负数才会有
总和大于等于0
*/

// CanCompleteCircuit 时间复杂度O(N),空间复杂度O(1)
func CanCompleteCircuit(gas []int, cost []int) int {
	start, totalSum, curSum := 0, 0, 0
	for i := 0; i < len(gas); i++ {
		totalSum += gas[i] - cost[i]
		curSum += gas[i] - cost[i]
		if curSum < 0 {
			start = i + 1
			curSum = 0
		}
	}
	if totalSum < 0 {
		return -1
	}
	return start
}

/*
1.9 分发糖果
老师想给孩子们分发糖果，有N个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。

你需要按照以下要求，帮助老师给这些孩子分发糖果：

每个孩子至少分配到 1 个糖果。
相邻的孩子中，评分高的孩子必须获得更多的糖果。
那么这样下来，老师至少需要准备多少颗糖果呢？

示例 1:

输入: [1,0,2]
输出: 5
解释: 你可以分别给这三个孩子分发 2、1、2 颗糖果。
示例 2:

输入: [1,2,2]
输出: 4
解释: 你可以分别给这三个孩子分发 1、2、1 颗糖果。第三个孩子只得到 1 颗糖果，这已满足上述两个条件。
*/

/*
这道题目一定是要确定一边之后，再确定另一边，例如比较每一个孩子的左边，然后再比较右边，如果两边一起考虑一定会顾此失彼。
先确定右边评分大于左边的情况（也就是从前向后遍历）

此时局部最优：只要右边评分比左边大，右边的孩子就多一个糖果，全局最优：相邻的孩子中，评分高的右孩子获得比左边孩子更多
的糖果

局部最优可以推出全局最优。

如果ratings[i]>ratings[i-1] 那么[i]的糖 一定要比[i-1]的糖多一个，所以贪心：candyVec[i]=candyVec[i-1]+1

再确定左孩子大于右孩子的情况（从后向前遍历）

为什么不能从前向后遍历呢？

因为如果从前向后遍历，根据ratings[i+1]来确定ratings[i]对应的糖果，那么每次都不能利用上前一次的比较结果了。

所以确定左孩子大于右孩子的情况一定要从后向前遍历！

如果ratings[i]>ratings[i+1]，此时candyVec[i]（第i个小孩的糖果数量）就有两个选择了，一个是candyVec[i+1]+1，
一个是candyVec[i]（之前比较右孩子大于左孩子得到的糖果数量）。

那么又要贪心了，局部最优：取candyVec[i+1]+1 和 candyVec[i]最大的糖果数量，保证第i个小孩的糖果数量即大于左边的
也大于右边的。全局最优：相邻的孩子中，评分高的孩子获得更多的糖果。

局部最优可以推出全局最优。

所以就取candyVec[i+1]+1和candyVec[i] 最大的糖果数量，candyVec[i]只有取最大的才能既保持对左边
candyVec[i-1]的糖果多，也比右边candyVec[i+1]的糖果多。
*/

// DistributeCandy 时间复杂度O(N),空间复杂度O(N)
func DistributeCandy(ratings []int) int {
	n := len(ratings)
	candies := make([]int, n)
	// 先考虑右边评分比左边高的情况
	for i := 0; i < n; i++ {
		if i > 0 && ratings[i] > ratings[i-1] {
			candies[i] = candies[i-1] + 1
		} else {
			// 因为每个孩子最少分到一个糖果，所以初始值定为1
			candies[i] = 1
		}
	}
	sum := candies[n-1]
	// 从后向前遍历，考虑左边评分高于右边的情况
	for i := n - 2; i >= 0; i-- {
		if ratings[i] > ratings[i+1] {
			candies[i] = Utils.Max(candies[i], candies[i+1]+1)
		}
		sum += candies[i]
	}
	return sum
}

/*
1.10 柠檬水找零
在柠檬水摊上，每一杯柠檬水的售价为 5 美元。

顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。

每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客
向你支付 5 美元。

注意，一开始你手头没有任何零钱。

如果你能给每位顾客正确找零，返回 true ，否则返回 false 。

示例 1：

输入：[5,5,5,10,20]
输出：true
解释：
前 3 位顾客那里，我们按顺序收取 3 张 5 美元的钞票。
第 4 位顾客那里，我们收取一张 10 美元的钞票，并返还 5 美元。
第 5 位顾客那里，我们找还一张 10 美元的钞票和一张 5 美元的钞票。
由于所有客户都得到了正确的找零，所以我们输出 true。
示例 2：

输入：[5,5,10]
输出：true
示例 3：

输入：[10,10]
输出：false
示例 4：

输入：[5,5,10,10,20]
输出：false
解释：
前 2 位顾客那里，我们按顺序收取 2 张 5 美元的钞票。
对于接下来的 2 位顾客，我们收取一张 10 美元的钞票，然后返还 5 美元。
对于最后一位顾客，我们无法退回 15 美元，因为我们现在只有两张 10 美元的钞票。
由于不是每位顾客都得到了正确的找零，所以答案是 false。
提示：

0 <= bills.length <= 10000
bills[i] 不是 5 就是 10 或是 20
*/

/*
只需要维护两种金额的数量，5，10。

有如下三种情况：
情况一：账单是5，直接收下。
情况二：账单是10，消耗一个5，增加一个10
情况三：账单是20，优先消耗一个10和一个5，如果不够，再消耗三个5
此时大家就发现 情况一，情况二，都是固定策略，都不用我们来做分析了，而唯一不确定的其实在情况三。

而情况三逻辑也不复杂甚至感觉纯模拟就可以了，其实情况三这里是有贪心的。

账单是20的情况，为什么要优先消耗一个10和一个5呢？

因为美元10只能给账单20找零，而美元5可以给账单10和账单20找零，美元5更万能！

所以局部最优：遇到账单20，优先消耗美元10，完成本次找零。全局最优：完成全部账单的找零。

局部最优可以推出全局最优，并找不出反例，那么就试试贪心算法！
*/

// LemonadeChange 时间复杂度O(N),空间复杂度O(1)
func LemonadeChange(bills []int) bool {
	five, ten := 0, 0
	for _, bill := range bills {
		if bill == 5 {
			five++
		} else if bill == 10 {
			if five <= 0 {
				return false
			}
			five--
			ten++
		} else {
			if ten > 0 && five > 0 {
				ten--
				five--
			} else if five >= 3 {
				five -= 3
			} else {
				return false
			}
		}
	}
	return true
}