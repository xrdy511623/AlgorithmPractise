package Medium

import (
	"AlgorithmPractise/BinaryTree/Entity"
	"AlgorithmPractise/Utils"
)

/*
1.1 打家劫舍I
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的
房屋在同一晚上被小偷闯入，系统会自动报警。
给定一个代表每个房屋存放金额的非负整数数组，计算你不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

示例1：
输入：[1,2,3,1] 输出：4 解释：偷窃1号房屋 (金额 = 1) ，然后偷窃3号房屋 (金额 = 3)。 偷窃到的最高金额 = 1 + 3 = 4 。

示例2：
输入：[2,7,9,3,1] 输出：12 解释：偷窃1号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃5号房屋 (金额 = 1)。
偷窃到的最高金额 = 2 + 9 + 1 = 12 。

提示：
0 <= nums.length <= 100
0 <= nums[i] <= 400
*/

/*
思路:
1 确定dp数组及其下标含义
dp[i]表示经过[0,i]这些房屋，最多可以偷窃的金额为dp[i]

2 确定递推公式
决定dp[i]的因素是第i间房偷还是不偷，如果偷第i间房，那么由于相邻房屋不能偷，否则会触发报警，dp[i]便只能由于dp[i-2]推出，此时
dp[i] = dp[i-2] + nums[i];如果不偷第i间房，那么dp[i]=dp[i-1],即考虑i-1房
所以dp[i]=max(dp[i-2] + nums[i], dp[i-1])

3 dp数组初始化
dp[0]=nums[0], dp[1]=max(nums[0], nums[1])

4 确定遍历顺序
由递推公式可知，dp[i]是由dp[i-2]以及dp[i-1]推导而来，所以从左到右遍历nums数组即可

5 举例推导dp数组
nums=[2,7,9,3,1]
dp下标  0   1   2   3   4
值      2  7   11  11  12
*/

// Rob 时间复杂度O(N),空间复杂度O(N)
func Rob(nums []int) int {
	maxValue := 0
	n := len(nums)
	switch n {
	case 0:
		maxValue = 0
	case 1:
		maxValue = nums[0]
	case 2:
		maxValue = Utils.Max(nums[0], nums[1])
	default:
		dp := make([]int, n)
		dp[0], dp[1] = nums[0], Utils.Max(nums[0], nums[1])
		for i := 2; i < n; i++ {
			dp[i] = Utils.Max(dp[i-1], dp[i-2]+nums[i])
		}
		maxValue = dp[n-1]
	}
	return maxValue
}

/*
实际上我们只需要维护两个状态值，所以可以写成下面这样，大大降低算法的空间复杂度
 */

// RobSimple 时间复杂度O(N),空间复杂度O(1)
func RobSimple(nums []int) int {
	maxValue := 0
	n := len(nums)
	switch n {
	case 0:
		maxValue = 0
	case 1:
		maxValue = nums[0]
	case 2:
		maxValue = Utils.Max(nums[0], nums[1])
	default:
		dp := make([]int, 2)
		dp[0], dp[1] = nums[0], Utils.Max(nums[0], nums[1])
		for i := 2; i < n; i++ {
			newMax := Utils.Max(dp[0]+nums[i], dp[1])
			dp[0] = dp[1]
			dp[1] = newMax
		}
		maxValue = dp[1]
	}
	return maxValue
}


/*
1.2 打家劫舍II
你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。
同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。

给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下 ，能够偷窃到的最高金额。

示例1：
输入：nums = [2,3,2] 输出：3 解释：你不能先偷窃1号房屋（金额 = 2），然后偷窃3号房屋（金额 = 2）, 因为他们是相邻的。

示例2：
输入：nums = [1,2,3,1] 输出：4 解释：你可以先偷窃1号房屋（金额 = 1），然后偷窃3号房屋（金额 = 3）。偷窃到的最高金额 = 1 + 3 = 4。

示例3：
输入：nums = [0] 输出：0

提示：
1 <= nums.length <= 100
0 <= nums[i] <= 1000
*/

/*
思路:
本题是1.1 打家劫舍的变形题，基本思路是一样的，唯一的区别是成环了
对于一个数组，成环的话主要有以下三种情况:
a 不考虑首尾元素
b 不考虑尾元素
c 不考虑首元素
而b,c其实都包含了a这种情况，所以只需要考虑b,c这两种情况，最后比较两种情况下的最大值，取较大值即可
*/

// RobRing 时间复杂度O(N),空间复杂度O(N)
func RobRing(nums []int) int {
	maxValue := 0
	n := len(nums)
	switch n {
	case 0:
		maxValue = 0
	case 1:
		maxValue = nums[0]
	case 2:
		// 我觉得按照题意，此时应该返回0才对，但是这样操作AC不了
		maxValue = Utils.Max(nums[0], nums[1])
	default:
		c1 := RobRange(nums, 0, n-2)
		c2 := RobRange(nums, 1, n-1)
		maxValue = Utils.Max(c1, c2)
	}
	return maxValue
}

func RobRange(nums []int, start, end int) int {
	if start == end {
		return nums[start]
	}
	dp := make([]int, len(nums))
	dp[start] = nums[start]
	dp[start+1] = Utils.Max(nums[start], nums[start+1])
	for i := start + 2; i <= end; i++ {
		dp[i] = Utils.Max(dp[i-2]+nums[i], dp[i-1])
	}
	return dp[end]
}

/*
实际上我们只需要维护两个状态值，所以可以写成下面这样，将算法的空间复杂度降低为O(1)
 */

// RobRangeSimple 时间复杂度O(N),空间复杂度O(1)
func RobRangeSimple(nums []int, start, end int) int {
	if start == end {
		return nums[start]
	}
	dp := make([]int, 2)
	dp[0] = nums[start]
	dp[1] = Utils.Max(nums[start], nums[start+1])
	for i:=start+2;i<=end;i++{
		newMax := Utils.Max(dp[0]+nums[i], dp[1])
		dp[0] = dp[1]
		dp[1] = newMax
	}
	return dp[1]
}


/*
1.3 打家劫舍III
在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

示例 1:
输入: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \
     3   1

输出: 7
解释:小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.

示例2:
输入: [3,4,5,1,3,null,1]

     3
    / \
   4   5
  / \   \
 1   3   1

输出: 9
解释:小偷一晚能够盗取的最高金额 = 4 + 5 = 9.
*/

/*
1 确定递归函数的参数和返回值
这里我们要求一个节点 偷与不偷的两个状态所得到的金钱，那么返回值就是一个长度为2的数组。
其实这里的返回数组就是dp数组。
所以dp数组（dp table）以及下标的含义：下标为0记录不偷该节点所得到的的最大金钱，下标为1记录偷该节点所得到的的最大金钱。

2 确定终止条件
在遍历的过程中，如果遇到空节点的话，很明显，无论偷还是不偷都是0，所以就返回

3 确定遍历顺序
首先明确的是使用后序遍历。 因为通过递归函数的返回值来做下一步计算。
通过递归左节点，得到左节点偷与不偷的金钱。
通过递归右节点，得到右节点偷与不偷的金钱。

4 确定单层递归的逻辑
如果是偷当前节点，那么左右孩子就不能偷，val1 = cur->val + left[0] + right[0]; （如果对下标含义不理解就在回顾一下dp数组的含义）
如果不偷当前节点，那么左右孩子就可以偷，至于到底偷不偷一定是选一个最大的，所以:
val2 = max(left[0], left[1]) + max(right[0], right[1]);
最后当前节点的状态就是{val2, val1}; 即：{不偷当前节点得到的最大金钱，偷当前节点得到的最大金钱}

5 举例推导dp数组
以示例1为例，dp数组状态如下：（注意用后序遍历的方式推导）
略

最后头结点就是取下标0和下标1的最大值就是偷得的最大金钱。
*/

// RobBinaryTree 时间复杂度：O(n) 每个节点只遍历了一次; 空间复杂度：O(logn) 算上递推系统栈的空间
func RobBinaryTree(root *Entity.TreeNode) int {
	dp := RobTree(root)
	return Utils.Max(dp[0], dp[1])
}

func RobTree(node *Entity.TreeNode) []int {
	if node == nil {
		return []int{0, 0}
	}
	left := RobTree(node.Left)
	right := RobTree(node.Right)
	notRobCur := Utils.Max(left[0], left[1]) + Utils.Max(right[0], right[1])
	robCur := node.Val + left[0] + right[0]
	return []int{notRobCur, robCur}
}

/*
1.4 按摩师预约
一个有名的按摩师会收到源源不断的预约请求，每个预约都可以选择接或不接。在每次预约服务之间要有休息时间，因此她
不能接受相邻的预约。给定一个预约请求序列，替按摩师找到最优的预约集合（总预约时间最长），返回总的分钟数。

注意：本题相对原题稍作改动

示例1：
输入： [1,2,3,1]
输出： 4
解释： 选择1号预约和3号预约，总时长 = 1 + 3 = 4。

示例2：
输入： [2,7,9,3,1]
输出： 12
解释： 选择1号预约、 3号预约和5号预约，总时长 = 2 + 9 + 1 = 12。

示例3：
输入： [2,1,4,5,3,1,1,3]
输出： 12
解释： 选择1号预约、 3号预约、 5号预约和8号预约，总时长 = 2 + 4 + 3 + 3 = 12。
*/

/*
稍加分析便可发现本题跟1.1 打家劫舍I几乎是雷同，所以代码不用改就可以AC了，爽歪歪。
*/

// MassageArrange 时间复杂度O(N),空间复杂度O(N)
func MassageArrange(nums []int) int {
	maxValue := 0
	n := len(nums)
	switch n {
	case 0:
		maxValue = 0
	case 1:
		maxValue = nums[0]
	case 2:
		maxValue = Utils.Max(nums[0], nums[1])
	default:
		dp := make([]int, n)
		dp[0], dp[1] = nums[0], Utils.Max(nums[0], nums[1])
		for i := 2; i < n; i++ {
			dp[i] = Utils.Max(dp[i-1], dp[i-2]+nums[i])
		}
		maxValue = dp[n-1]
	}
	return maxValue
}

/*
或者也可以区分两种状态下的递推公式
dp[i][0]表示第i个预约不接的最长预约时间，dp[i][1]表示第i个预约接的最长预约时间，那么递推公式很明显应该是
dp[i][0] = max(dp[i-1][0], dp[i-1][1])
dp[i][1] = dp[i-1][0]+nums[i]
最后返回max(dp[n-1][0], dp[n-1][1])即可
*/

// MassageArrangement 时间复杂度O(2N),空间复杂度O(2N)
func MassageArrangement(nums []int) int {
	maxValue := 0
	n := len(nums)
	switch n {
	case 0:
		maxValue = 0
	case 1:
		maxValue = nums[0]
	case 2:
		maxValue = Utils.Max(nums[0], nums[1])
	default:
		dp := make([][]int, n)
		for i := 0; i < n; i++ {
			dp[i] = make([]int, 2)
		}
		dp[0][0] = 0
		dp[0][1] = nums[0]
		for i := 1; i < n; i++ {
			dp[i][0] = Utils.Max(dp[i-1][0], dp[i-1][1])
			dp[i][1] = dp[i-1][0] + nums[i]
		}
		maxValue = Utils.Max(dp[n-1][0], dp[n-1][1])
	}
	return maxValue
}