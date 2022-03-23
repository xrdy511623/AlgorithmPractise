package Greedy

import (
	"AlgorithmPractise/BinaryTree/Entity"
	"AlgorithmPractise/Utils"
	"math"
	"sort"
	"strconv"
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
	maxP := 0
	for i := 1; i < len(prices); i++ {
		if profit := prices[i] - prices[i-1]; profit > 0 {
			maxP += profit
		}
	}
	return maxP
}

/*
1.5 跳跃游戏
给定一个非负整数数组，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断
你是否能够到达最后一个位置。

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
你的目标是使用最少地跳跃次数到达数组的最后一个位置。

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
	end, rightMost, steps := 0, 0, 0
	for i := 0; i < n-1; i++ {
		// 更新当前所能到达的最大位置rightMost
		rightMost = Utils.Max(i+nums[i], rightMost)
		// 当下标移动到右边界时，跳跃次数需要加1，同时更新右边界为maxPosition
		if i == end {
			end = rightMost
			steps++
		}
	}
	return steps
}

/*
1.7 跳跃游戏
这里有一个非负整数数组arr，你最开始位于该数组的起始下标start处。当你位于下标i处时，你可以跳到i + arr[i]
或者 i - arr[i]。

请你判断自己是否能够跳到对应元素值为0的任一下标处。
注意，不管是什么情况下，你都无法跳到数组之外。

示例 1：
输入：arr = [4,2,3,0,3,1,2], start = 5
输出：true
解释：
到达值为 0 的下标 3 有以下可能方案：
下标 5 -> 下标 4 -> 下标 1 -> 下标 3
下标 5 -> 下标 6 -> 下标 4 -> 下标 1 -> 下标 3

示例2：
输入：arr = [4,2,3,0,3,1,2], start = 0
输出：true
解释：
到达值为 0 的下标 3 有以下可能方案：
下标 0 -> 下标 4 -> 下标 1 -> 下标 3

提示：
1 <= arr.length <= 5 * 10^4
0 <= arr[i] <arr.length
0 <= start < arr.length
*/

// CanReach BFS解决，时间复杂度O(N),空间复杂度O(N)
func CanReach(arr []int, start int) bool {
	// 优先处理起始位置就满足的情况
	if arr[start] == 0 {
		return true
	}
	// 哈希表used记录已经使用过的位置
	used := make(map[int]bool)
	used[start] = true
	queue := []int{start}
	for len(queue) != 0 {
		pos := queue[0]
		queue = queue[1:]
		// 记录下一个可能的位置(有两个，pos+arr[pos], pos-arr[pos])
		nextPos := []int{pos + arr[pos], pos - arr[pos]}
		for _, v := range nextPos {
			// 这个位置不能越界，而且不能是已经使用过的位置，否则会陷入死循环
			if v >= 0 && v < len(arr) && !used[v] {
				// 如果这个位置对应的值为0，返回true
				if arr[v] == 0 {
					return true
				}
				// 如果没有找到值为0的位置，将该位置添加到队列末尾，以便之后从该位置计算下一个可能的位置
				queue = append(queue, v)
				// 同时标记该位置已经使用过了
				used[v] = true
			}
		}
	}
	return false
}

/*
1.8 K次取反后最大化的数组和
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
[-8,-5,-5,-3,-2,3]
*/

// LargestSumAfterKNegations 时间复杂度O(N),空间复杂度O(1)
func LargestSumAfterKNegations(nums []int, k int) int {
	sum := 0
	// 根据题意，数组中的最大绝对值不会超过100
	minAbs := 101
	// 对原数组进行排序，得到升序排列数组
	sort.Ints(nums)
	for _, num := range nums {
		// 迭代数组中的最小绝对值
		minAbs = Utils.MinAbs(minAbs, num)
		if num < 0 && k > 0 {
			sum += -1 * num
			k--
		} else {
			sum += num
		}
	}
	// 如果k为正数，而且是偶数，那么此时任选一个正数j取反k次得到的值还是j本身, 之前遍历数组时累加的值
	// 也是j，所以可以直接返回sum
	if k > 0 && k%2 == 1 {
		return sum - 2*minAbs
	}
	return sum
}

/*
1.9 加油站
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
思路:贪心
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

c站也不能作为起点，因为去不了d！我们归纳出结论1。

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
1.10 分发糖果
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
1.11 柠檬水找零
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
	// 维护5元和10元钞票的数量
	five, ten := 0, 0
	for _, bill := range bills {
		// 如果是收到5元，直接收下，不用找零了
		if bill == 5 {
			five++
		} else if bill == 10 {
			// 收到10元，消耗一张5元钞票找零，如果没有5元钞票可以消耗，返回false
			if five <= 0 {
				return false
			}
			// 5元钞票数-1，10元超票数+1
			five--
			ten++
		} else {
			// 此时收到的是20，优先消耗一张10元和5元的钞票找零
			if ten > 0 && five > 0 {
				ten--
				five--
				// 实在不行，也可以消耗三张5元找零
			} else if five >= 3 {
				five -= 3
			} else {
				// 都不满足，返回false
				return false
			}
		}
	}
	return true
}

/*
1.12 根据身高重建队列
假设有打乱顺序的一群人站成一个队列，数组people表示队列中一些人的属性（不一定按顺序）。每个people[i]=[hi, ki]
表示第i个人的身高为hi ，前面正好有ki个身高大于或等于hi的人。

请你重新构造并返回输入数组people所表示的队列。返回的队列应该格式化为数组queue ，其中queue[j] = [hj, kj]
是队列中第j个人的属性（queue[0] 是排在队列前面的人）。

示例1：
输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
解释：
编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面。
编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面。
编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人。
编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人。
编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
因此 [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] 是重新构造后的队列。

示例 2：
输入：people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
输出：[[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]

提示：
1 <= people.length <= 2000
0 <= hi <= 10^6
0 <= ki < people.length
题目数据确保队列可以被重建
*/

// ReconstructQueue 时间复杂度O(N^2),空间复杂度O(logN)
func ReconstructQueue(people [][]int) [][]int {
	sort.Slice(people, func(i, j int) bool {
		a, b := people[i], people[j]
		return a[0] < b[0] || a[0] == b[0] && a[1] > b[1]
	})
	queue := make([][]int, len(people))
	for _, person := range people {
		position := person[1] + 1
		for i := range queue {
			if queue[i] == nil {
				position--
				if position == 0 {
					queue[i] = person
					break
				}
			}
		}
	}
	return queue
}

/*
1.13 用最少数量的箭引爆气球
在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，
所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。

一支弓箭可以沿着x轴从不同点完全垂直地射出。在坐标x处射出一支箭，若有一个气球的直径的开始和结束坐标为
xstart，xend， 且满足xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。弓箭一旦被射出
之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。

给你一个数组points ，其中points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。

示例1：
输入：points = [[10,16],[2,8],[1,6],[7,12]]
输出：2
解释：对于该样例，x = 6 可以射爆 [2,8],[1,6] 两个气球，以及 x = 11 射爆另外两个气球

示例2：
输入：points = [[1,2],[3,4],[5,6],[7,8]]
输出：4

示例3：
输入：points = [[1,2],[2,3],[3,4],[4,5]]
输出：2

示例4：
输入：points = [[1,2]]
输出：1

示例5：
输入：points = [[2,3],[2,3]]
输出：1

提示：
0 <= points.length <= 10^4
points[i].length == 2
-2^31 <= xstart < xend <= 2^31 - 1
*/

// FindMinArrowShots 时间复杂度O(NlogN),空间复杂度O(logN)
func FindMinArrowShots(points [][]int) int {
	n := len(points)
	if n <= 1 {
		return n
	}
	sort.Slice(points, func(i, j int) bool {
		return points[i][0] < points[j][0]
	})
	num := 1
	for i := 1; i < n; i++ {
		// 两个气球不重叠，则需要的弓箭数+1
		if points[i][0] > points[i-1][1] {
			num++
		} else {
			// 两个气球重叠，则更新右边气球的最小右边界
			points[i][1] = Utils.Min(points[i-1][1], points[i][1])
		}
	}
	return num
}

/*
1.14 给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。
注意: 可以认为区间的终点总是大于它的起点。 区间[1,2]和[2,3] 的边界相互“接触”，但没有相互重叠。

示例1:
输入: [ [1,2], [2,3], [3,4], [1,3] ] 输出: 1 解释: 移除 [1,3] 后，剩下的区间没有重叠。

示例2:
输入: [ [1,2], [1,2], [1,2] ] 输出: 2 解释: 你需要移除两个 [1,2] 来使剩下的区间没有重叠。

示例3:
输入: [ [1,2], [2,3] ] 输出: 0 解释: 你不需要移除任何区间，因为它们已经是无重叠的了。
*/

/*
思路一:动态规划
题目的要求等价于选出最多数量的区间，使得它们互不重叠。最后intervals的长度减去最大的不重叠区间数，即为
重叠的区间数，也就是需要移除的最小区间数，注意题目只要求出这个需要移除的最小区间数就可以了，没说一定要
去实际的移除这些重叠区间，千万不要陷在这个思维误区里出不来哈。

首先我们需要对数组intervals按照左端进行升序排序，然后对有序的intervals进行动态规划。

1 确定dp数组及其下标含义
dp[i]表示intervals[0:i](注意是左闭右闭区间)区间范围内的最大的不重叠区间数。

2 确定递推公式
对于j:=0;j<i;j++, 由于我们已经按照左端点进行升序排序了，所以只要满足intervals[j][1] <= intervals[i][0]，
那就意味着第j个区间一定不与第i个区间重叠，我们又找到了一个不重叠区间，所以在所有满足要求的dp[j]中，选择最大的
那一个进行状态转移，递推公式就是dp[i] = Max(dp[i], dp[j]+1)

3 初始化dp数组
显然，在intervals[0:i]区间范围内的不重叠区间数至少都是1，极限情况下所有的区间都是一样的，重叠区间为n-1,
那不重叠区间至少也有1个

4 确定遍历顺序
从递推公式可知，dp[i]依赖于位置比它靠前的dp[j]，所以肯定是从前往后遍历

5 举例推导dp数组
以排序后的intervals数组为例:
     [ [1,2], [1,3], [2,3], [3,4] ]
下标   0       1      2      3
值     1       1     2       3
所以最后返回len(intervals)-maxValueOfArray(dp) = 4 - 3 = 1
*/

// EraseOverlapIntervalsComplex 时间复杂度O(N^2+NlogN), 空间复杂度O(N)
func EraseOverlapIntervalsComplex(intervals [][]int) int {
	n := len(intervals)
	if n <= 1 {
		return 0
	}
	dp := make([]int, n)
	for i := 0; i < n; i++ {
		dp[i] = 1
	}
	// 升序排序，时间复杂度O(NlogN)
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	maxNonOverlap := 1
	// 动态规划，为动归数组dp进行状态转移，时间复杂度O(N^2)
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			if intervals[j][1] <= intervals[i][0] {
				dp[i] = Utils.Max(dp[i], dp[j]+1)
			}
		}
		// 迭代最大的不重叠区间数
		if maxNonOverlap < dp[i] {
			maxNonOverlap = dp[i]
		}
	}
	// len(intervals)-maxNonOverlap即为所求要移除的最小重叠区间数
	return n - maxNonOverlap
}

/*
思路二:贪心
仔细分析，本题其实与1.12 用最少数量的箭引爆气球非常类似，唯一的差别只是判定重叠的边界条件略有不同。
所以只需将该题的代码拿过来略加修改便可AC了,是不是很爽啊？
*/

// EraseOverlapIntervals 时间复杂度O(NlogN+N), 空间复杂度O(1)
func EraseOverlapIntervals(intervals [][]int) int {
	n := len(intervals)
	if n <= 1 {
		return 0
	}
	// 升序排序，时间复杂度O(NlogN)
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	minOverlap := 0
	// 本段代码时间复杂度O(N)
	for i := 1; i < n; i++ {
		// 因为已经按照左端点进行升序排序，所以只要左边的右端点越过了右边的左端点，便出现了重叠区间
		if intervals[i-1][1] > intervals[i][0] {
			// 此时需要移除的最小重叠区间数累加1
			minOverlap++
			// 要移除的重叠区间数最小，自然是要右边的右端点尽可能小，才能尽可能的不与后继区间(i+1区间)重叠
			intervals[i][1] = Utils.Min(intervals[i-1][1], intervals[i][1])
		}
	}
	return minOverlap
}

/*
1.15 划分字母区间
字符串 S 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示
每个字符串片段的长度的列表。

示例： 输入：S = "ababcbacadefegdehijhklij" 输出：[9,7,8] 解释： 划分结果为 "ababcbaca", "defegde",
"hijhklij"。 每个字母最多出现在一个片段中。 像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分
的片段数较少。

提示：
S的长度在[1, 500]之间。
S只包含小写字母'a' 到 'z' 。
*/

/*
思路:
在遍历的过程中相当于是要找每一个字母的边界，如果找到之前遍历过的所有字母的最远边界，说明这个边界就是分割点了。
此时前面出现过所有字母，最远也就到这个边界了。

可以分为如下两步：
统计每一个字符最后出现的位置
从头遍历字符，并更新字符的最远出现下标，如果找到字符最远出现位置下标和当前下标相等了，则找到了分割点。
参看分割字符串.png
*/

// PartitionLabels 时间复杂度O(N), 空间复杂度O(1)
func PartitionLabels(s string) []int {
	var res []int
	positionMap := make(map[byte]int)
	left, right, size := 0, 0, len(s)
	for i := 0; i < size; i++ {
		positionMap[s[i]] = i
	}
	for i := 0; i < size; i++ {
		right = Utils.Max(right, positionMap[s[i]])
		if i == right {
			res = append(res, right-left+1)
			left = i + 1
		}
	}
	return res
}

// 用数组替代map来存储每个字符出现的最远位置，更节省空间一些，可以写成下面这样
func PartitionLabelsTwo(s string) []int {
	var res []int
	marks := make([]int, 26)
	size := len(s)
	for i := 0; i < size; i++ {
		marks[s[i]-'a'] = i
	}
	left, right := 0, 0
	for i := 0; i < size; i++ {
		right = Utils.Max(right, marks[s[i]-'a'])
		if i == right {
			res = append(res, right-left+1)
			left = right + 1
		}
	}
	return res
}

/*
1.16 合并区间
给出一个区间的集合，请合并所有重叠的区间。

示例1:
输入: intervals = [[1,3],[2,6],[8,10],[15,18]] 输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

示例2:
输入: intervals = [[1,4],[4,5]] 输出: [[1,5]]
解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。

提示：
intervals[i][0] <= intervals[i][1]
*/

/*
思路:首先，我们将列表中的区间按照左端点升序排序。然后我们将第一个区间加入merged数组中，并按顺序依次考虑之后
的每个区间：
如果当前区间的左端点在数组merged中最后一个区间的右端点之后，那么它们不会重合，我们可以直接将这个区间加入数组
merged的末尾；
否则，它们重合，我们需要用当前区间的右端点更新数组merged中最后一个区间的右端点，将其置为二者的较大值。
*/

// Merge 时间复杂度O(NlogN), 空间复杂度O(logN)
func Merge(intervals [][]int) [][]int {
	n := len(intervals)
	if n <= 1 {
		return intervals
	}
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	var merged [][]int
	for i := 0; i < n; i++ {
		size := len(merged)
		// 若merged为空或merged数组末尾元素不与intervals[i]重叠，则将intervals[i]添加到merged中
		if size == 0 || merged[size-1][1] < intervals[i][0] {
			merged = append(merged, intervals[i])
		} else {
			// 否则将merged数组末尾元素与intervals[i]合并
			merged[size-1][1] = Utils.Max(merged[size-1][1], intervals[i][1])
		}
	}
	return merged
}

/*
1.17 删除被覆盖区间
给你一个区间列表，请你删除列表中被其他区间所覆盖的区间。
只有c <= a且b <= d时，我们才认为区间[a,b)被区间[c,d) 覆盖。

在完成所有删除操作后，请你返回列表中剩余区间的数目。


示例：
输入：intervals = [[1,4],[3,6],[2,8]]
输出：2
解释：区间 [3,6] 被区间 [2,8] 覆盖，所以它被删除了。


提示：
1 <= intervals.length <= 1000
0 <= intervals[i][0] < intervals[i][1] <= 10^5
对于所有的i != j：intervals[i] != intervals[j]
*/

/*
思路:贪心
首先对intervals数组进行排序，排序规则是按照左端点升序排序，左端点相同的情况下，按照右端点降序排序，
右边界rMax为排序后数组第一个元素的右端点，那么从前往后顺序遍历排序数组，一旦intervals[i]的右端点小于等于
右边界rMax，则证明该元素(区间)被前一个元素(区间)覆盖，因为按照排序规则，此时有:
intervals[i-1][0]<=intervals[i][0]<=intervals[i][1]<=intervals[i-1][1]。
注意题目只要求解删除被覆盖区间后的区间(元素)数，所以并不需要实际的去删除被覆盖区间，只需要总区间数(元素数)减去
1就可以了。如果intervals[i]的右端点大于右边界rMax，则证明该元素(区间)没有被前一个元素(区间)覆盖，此时将
右边界rMax向右推进，更新为max(rMax, intervals[i][1])，以备下一次判断，如此循环遍历结束后返回的区间数
即为所求答案。
*/

// RemoveCoveredIntervals 时间复杂度O(NlogN), 空间复杂度O(logN)
func RemoveCoveredIntervals(intervals [][]int) int {
	// 按照左端点升序排序，左端点相同的情况下，按照右端点降序排序
	sort.Slice(intervals, func(i, j int) bool {
		a, b := intervals[i], intervals[j]
		return a[0] < b[0] || a[0] == b[0] && a[1] > b[1]
	})
	n := len(intervals)
	// 总区间数，初始值为len(intervals)
	count := n
	// 右边界rMax初始值为排序数组第一个元素的右端点
	rMax := intervals[0][1]
	for i := 1; i < n; i++ {
		// 此时区间intervals[i]被覆盖，总区间数count--
		if intervals[i][1] <= rMax {
			count--
		} else {
			// 此时更新右边界rMax的值
			rMax = Utils.Max(rMax, intervals[i][1])
		}
	}
	// 最后返回的count即为所求答案
	return count
}

// 或者也可以这么写
func removeCoveredIntervals(intervals [][]int) int {
	sort.Slice(intervals, func(i, j int) bool {
		a, b := intervals[i], intervals[j]
		return a[0] < b[0] || a[0] == b[0] && a[1] > b[1]
	})
	n := len(intervals)
	count := n
	for i := 1; i < n; i++ {
		if intervals[i][1] <= intervals[i-1][1] {
			count--
			intervals[i][1] = intervals[i-1][1]
		}
	}
	return count
}

/*
1.18 单调递增的数字
给定一个非负整数N，找出小于或等于N的最大的整数，同时这个整数需要满足其各个位数上的数字是单调递增。
（当且仅当每个相邻位数上的数字 x 和 y 满足 x <= y 时，我们称这个整数是单调递增的。）

示例1:
输入: N = 10 输出: 9

示例2:
输入: N = 1234 输出: 1234

示例3:
输入: N = 332 输出: 299

说明: N是在 [0, 10^9] 范围内的一个整数。
*/

/*
思路:
题目要求小于等于N的最大单调递增的整数，那么拿一个两位的数字来举例。
例如：98，一旦出现strNum[i - 1] > strNum[i]的情况（非单调递增），首先想让strNum[i - 1]--，然后
strNum[i]给为9，这样这个整数就是89，即小于98的最大的单调递增整数。

这一点如果想清楚了，这道题就好办了。

局部最优：遇到strNum[i - 1] > strNum[i]的情况，让strNum[i - 1]--，然后strNum[i]给为9，可以保证这两位
变成最大单调递增整数。

全局最优：得到小于等于N的最大单调递增的整数。

但这里局部最优推出全局最优，还需要其他条件，即遍历顺序和标记从哪一位开始统一改成9。
此时是从前向后遍历还是从后向前遍历呢？

从前向后遍历的话，遇到strNum[i - 1] > strNum[i]的情况，让strNum[i - 1]减一，但此时如果strNum[i - 1]
减一了，可能又小于strNum[i - 2]。
这么说有点抽象，举个例子，数字：332，从前向后遍历的话，那么就把变成了329，此时2又小于了第一位的3了，真正的结果
应该是299。

所以从前后向遍历会改变已经遍历过的结果！

那么从后向前遍历，就可以重复利用上次比较得出的结果了，从后向前遍历332的数值变化为：332 -> 329 -> 299
确定了遍历顺序之后，那么此时局部最优就可以推出全局，找不出反例，试试贪心。
*/

// MonotoneIncreasingDigits 时间复杂度O(N), 空间复杂度O(N)
func MonotoneIncreasingDigits(n int) int {
	strN := strconv.Itoa(n)
	ss := []byte(strN)
	size := len(ss)
	if size <= 1 {
		return n
	}
	for i := size - 1; i > 0; i-- {
		if ss[i-1] > ss[i] {
			ss[i-1]--
			for j := i; j < size; j++ {
				ss[j] = '9'
			}
		}
	}
	res, _ := strconv.Atoi(string(ss))
	return res
}

/*
1.19 买卖股票的最佳时机含手续费
给定一个整数数组prices，其中第i个元素代表了第i天的股票价格 ；非负整数fee代表了交易股票的手续费用。
你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续
购买股票了。

返回获得利润的最大值。

注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

示例1:
输入: prices = [1, 3, 2, 8, 4, 9], fee = 2 输出: 8
解释: 能够达到的最大利润: 在此处买入 prices[0] = 1 在此处卖出 prices[3] = 8 在此处买入 prices[4] = 4
在此处卖出 prices[5] = 9 总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8.

注意:
0 < prices.length <= 50000.
0 < prices[i] < 50000.
0 <= fee < 50000.
*/

/*
思路:贪心
本题不同于1.4 买卖股票的最佳时机II在于每次交易股票有手续费的成本，所以不能按照原来累加每天的正利润思路来做。
因为这样可能会有多笔手续费的成本，而实际最低点买进最高点卖出只有一笔手续费。

我们在做收获利润操作的时候其实有三种情况：
情况一：收获利润的这一天并不是收获利润区间里的最后一天（不是真正的卖出，相当于持有股票），所以后面要继续收获
利润。
情况二：前一天是收获利润区间里的最后一天（相当于真正的卖出了），今天要重新记录最小价格了。
情况三：不作操作，保持原有状态（买入，卖出，不买不卖）
*/

// MaxProfitIncludeFee 时间复杂度O(N), 空间复杂度O(1)
func MaxProfitIncludeFee(prices []int, fee int) int {
	minPrice := prices[0]
	maxProfit := 0
	for i := 1; i < len(prices); i++ {
		// 不能卖出,重新记录最低股票价格
		if prices[i] < minPrice {
			minPrice = prices[i]
		}
		// 卖掉股票的收益不足以覆盖买入和手续费的成本，也不能卖出
		if prices[i] >= minPrice && prices[i]-minPrice-fee <= 0 {
			continue
		}
		// 卖掉股票的收益足以覆盖买入和手续费的成本，才能卖出
		if profit := prices[i] - minPrice - fee; profit > 0 {
			// 累加交易的正利润
			maxProfit += profit
			// 可能有多次计算利润，最后一次计算利润才是真正意义的卖出
			// 所以为了防止重复的手续费成本,minPrice要更新为prices[i]-fee
			minPrice = prices[i] - fee
		}
	}
	return maxProfit
}

/*
1.20 监控二叉树
给定一个二叉树，我们在树的节点上安装摄像头。
节点上的每个摄影头都可以监视其父对象、自身及其直接子对象。
计算监控树的所有节点所需的最小摄像头数量。

示例1：
输入：[0,0,null,0,0] 输出:1 解释：如图所示，一台摄像头足以监控所有节点。

示例2：
输入：[0,0,null,0,null,0,null,null,0] 输出:2 解释：需要至少两个摄像头来监视树的所有节点。

提示：
给定树的节点数的范围是 [1, 1000]。
每个节点的值都是0。
*/

/*
思路:贪心算法解决
如果一个节点有孩子节点且孩子节点没有被摄像机覆盖,则我们需要放置一个摄像机在该节点.此外,
如果一个节点没有父节点且自己没有被覆盖，则必须放置一个摄像机在该节点.
还有一点很关键，就是遍历树的过程中，会遇到空节点，那么问题来了，空节点究竟是哪一种状态呢？空节点表示无覆盖？
表示有摄像头？还是有覆盖呢？
回归本质，为了让摄像头数量最少，我们要尽量让叶子节点的父节点安装摄像头，这样才能摄像头的数量最少。
那么空节点不能是无覆盖的状态，这样叶子节点就要放摄像头了，空节点也不能是有摄像头的状态，这样叶子节点的父节点
就没有必要放摄像头了，而是可以把摄像头放在叶子节点的爷爷节点上。
所以空节点的状态只能是有覆盖，这样就可以在叶子节点的父节点放摄像头了。
*/

// MinCameraCover 时间复杂度O(N), 空间复杂度O(2N)
func MinCameraCover(root *Entity.TreeNode) int {
	// 定义哈希表covered记录节点是否被监视
	covered := make(map[*Entity.TreeNode]bool)
	// 空节点一律记录为已被监控
	covered[nil] = true
	// 定义哈希表parentMap记录节点的父亲节点
	parentMap := make(map[*Entity.TreeNode]*Entity.TreeNode)
	// 需要的摄像头数量，初始值为0
	camera := 0
	var dfs func(*Entity.TreeNode, *Entity.TreeNode)
	dfs = func(node, parent *Entity.TreeNode) {
		if node != nil {
			// 记录当前节点的父节点
			parentMap[node] = parent
			// 递归遍历当前节点的左子树
			dfs(node.Left, node)
			// 递归遍历当前节点的右子树
			dfs(node.Right, node)
			// 如果是必须要放置摄像头的情况，那么所有相关节点都要记录为已被监控
			if (parentMap[node] == nil && !covered[node]) || !covered[node.Left] || !covered[node.Right] {
				camera++
				covered[node] = true
				covered[parentMap[node]] = true
				covered[node.Left] = true
				covered[node.Right] = true
			}
		}
	}
	dfs(root, nil)
	return camera
}

/*
leetcode 1014. 最佳观光组合
1.7 给你一个正整数数组values，其中values[i]表示第i个观光景点的评分，并且两个景点i和j之间的距离为j - i。
一对景点（i < j）组成的观光组合的得分为values[i] + values[j] + i - j ，也就是景点的评分之和减去它们两者
之间的距离。

返回一对观光景点能取得的最高分。

示例1：
输入：values = [8,1,5,2,6]
输出：11
解释：i = 0, j = 2, values[i] + values[j] + i - j = 8 + 5 + 0 - 2 = 11

示例2：
输入：values = [1,2]
输出：2

提示：
2 <= values.length <= 5 * 104
1 <= values[i] <= 1000
*/

/*
对于两个景点i和j而言(i<j)，它们的最高得分是values[i]+values[j]+i-j。那么我们可以对每一个景点j,遍历它之前的
景点i(i的区间为[0,j-1])，对于景点j而言，values[j]-j值是固定的，所以要values[i]+values[j]+i-j最大，就是要
values[i]+i最大，也就是需要在j之前的所有景点中找到一个最大的values[i]+[i]，我们记为preMax.
所以我们的思路就呼之欲出了。
我们从前向后遍历values数组(从下标1开始)，动态的维护景点j之前的preMax, 不断迭代最高得分highestScore即可。
于是有highestScore = Max(highestScore, preMax+values[j]-j)，计算景点j的最高得分，并迭代highestScore。
preMax = Max(preMax, values[j]+j)，更新preMax为下一个景点j+1所用
*/

func MaxScoreSightseeingPair(values []int) int {
	highestScore := math.MinInt32
	preMax := values[0] + 0
	for j := 1; j < len(values); j++ {
		// 计算景点j的最高得分，并迭代highestScore
		highestScore = Utils.Max(highestScore, preMax+values[j]-j)
		// 更新preMax为下一个景点j+1所用
		preMax = Utils.Max(preMax, values[j]+j)
	}
	return highestScore
}
