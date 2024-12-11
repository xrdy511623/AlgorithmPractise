package hard

/*
package hard contains complex dynamicProgramming problems
*/

import "algorithm-practise/utils"

/*
1.1 leetcode 42 接雨水
给定n个非负整数表示每个宽度为1的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
*/

/*
思路:暴力
按题意，要接到雨水，前提是当前柱子的左右两边必须都比它高，根据木桶效应，接到的雨水量为左右两边最矮的那边柱子
的高度减去当前柱子的高度。简单粗暴的解法是遍历下标1~n-2(0和n-1只有一边，没有左右两边)，累加符合积水条件时的
积水量
*/

// TrapBrutal 暴力 时间复杂度O(N^2),空间复杂度O(N)
func TrapBrutal(height []int) int {
	sum := 0
	leftMax, rightMax, n := 0, 0, len(height)
	for i := 1; i < n-1; i++ {
		leftMax = utils.MaxValueOfArray(height[:i])
		rightMax = utils.MaxValueOfArray(height[i+1:])
		if leftMax > height[i] && rightMax > height[i] {
			sum += utils.Min(leftMax, rightMax) - height[i]
		}
	}
	return sum
}

/*
思路:动态规划
暴力解法时间复杂度较高是因为需要对每个下标位置都向两边扫描。如果已经知道每个位置两边的最大高度，则可以在O(n)
的时间内得到能接的雨水总量。使用动态规划的方法，可以在O(n)的时间内预处理得到每个位置两边的最大高度。

创建两个长度为n的数组leftMax和rightMax。对于0≤i<n，leftMax[i] 表示下标i左边的位置中，height的最大高度，
rightMax[i]表示下标i右边的位置中，height的最大高度。

显然，leftMax[1]=height[0]，rightMax[n−2]=height[n−1]。两个数组的其余元素的计算如下：

当2≤i≤n−2时，leftMax[i]=max(leftMax[i−1],height[i-1])；

当1≤i≤n−3时，rightMax[i]=max(rightMax[i+1],height[i+1])。

因此可以正向遍历数组height得到数组leftMax的每个元素值，反向遍历数组height得到数组rightMax的每个元素值。
在得到数组leftMax和rightMax的每个元素值之后，对于1≤i<=n-2，下标i处能接的雨水量等于
min(leftMax[i], rightMax[i])−height[i]。遍历每个下标位置即可得到能接的雨水总量。
*/

func trapUseDp(height []int) int {
	sum := 0
	n := len(height)
	// 至少需要3根柱子才能积水
	if n <= 2 {
		return 0
	}
	leftMax, rightMax := make([]int, n), make([]int, n)
	leftMax[1], rightMax[n-2] = height[0], height[n-1]
	for i := 2; i < n-1; i++ {
		leftMax[i] = utils.Max(height[i-1], leftMax[i-1])
	}
	for j := n - 3; j >= 1; j-- {
		rightMax[j] = utils.Max(height[j+1], rightMax[j+1])
	}
	for k := 1; k <= n-2; k++ {
		if value := utils.Min(leftMax[k], rightMax[k]) - height[k]; value > 0 {
			sum += value
		}
	}
	return sum
}

/*
思路:单调递减栈
维护一个单调递减栈，如果当前柱子高度h大于栈顶元素对应的高度(s[n-1])，由于是单调递减栈，此时在栈顶元素对应
的柱子便形成了低洼处，即s[n-1]<s[n-2]且s[n-1]<h，积水量便等于积水区域的宽度*高度。
width = i - left - 1  i为当前柱子高度h所对应的下标, left为s[n-2]对应的下标。
height = min(s[n-2], h)
*/

func trapUseStack(height []int) int {
	var stack []int
	sum := 0
	for i, h := range height {
		for len(stack) > 0 && h > height[stack[len(stack)-1]] {
			// 显然会在stack[len(stack)-1]处形成积水，积水处柱子高度为height[stack[len(stack)-1]]
			low := height[stack[len(stack)-1]]
			stack = stack[:len(stack)-1]
			// 只有右侧高，是无法形成积水的
			if len(stack) == 0 {
				break
			}
			// 积水处左侧位置为stack[len(stack)-1]
			left := stack[len(stack)-1]
			// 积水区域宽度为积水处左侧柱子位置与右侧柱子位置之差-1
			curWidth := i - left - 1
			// 积水区域高度便等于积水处左侧高度与右侧高度的最小值-低洼处高度
			curHeight := utils.Min(height[left], h) - low
			// 此处积水区域的面积即为此处收集到的雨水量
			sum += curWidth * curHeight
		}
		stack = append(stack, i)
	}
	return sum
}

/*
思路:双指针
动态规划的做法中，需要维护两个数组leftMax和rightMax，因此空间复杂度是O(n)。是否可以将空间复杂度降到O(1)？
注意到下标i处能接的雨水量由leftMax[i]和rightMax[i]中的最小值决定。由于数组leftMax是从左往右计算，数组
rightMax是从右往左计算，因此可以使用双指针和两个变量代替两个数组。

维护两个指针left和right，以及两个变量leftMax和rightMax，初始时left=0,right=n−1,leftMax=0,rightMax=0。
指针left只会向右移动，指针right只会向左移动，在移动指针的过程中维护两个变量leftMax和rightMax的值。

当两个指针没有相遇时，进行如下操作：
使用height[left]和height[right]的值更新leftMax和rightMax的值；
如果height[left]<height[right]，则必有leftMax<rightMax，下标left处能接的雨水量等于leftMax−height[left]，
将下标left处能接的雨水量加到能接的雨水总量，然后将left加1（即向右移动一位）；
如果height[left]≥height[right]，则必有leftMax≥rightMax，下标right处能接的雨水量等于rightMax−height[right]，
将下标right处能接的雨水量加到能接的雨水总量，然后将right减1（即向左移动一位）。

当两个指针相遇时，即可得到能接的雨水总量。
*/

func trapSimple(height []int) int {
	sum := 0
	n := len(height)
	if n < 3 {
		return sum
	}
	leftMax, rightMax := height[0], height[n-1]
	left, right := 1, n-2
	for i := 1; i < n-1; i++ {
		if height[left-1] < height[right+1] {
			leftMax = utils.Max(leftMax, height[left-1])
			if leftMax > height[left] {
				sum += leftMax - height[left]
			}
			left++
		} else {
			rightMax = utils.Max(rightMax, height[right+1])
			if rightMax > height[right] {
				sum += rightMax - height[right]
			}
			right--
		}
	}
	return sum
}

/*
leetcode 10  正则表达式匹配
给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素
所谓匹配，是要涵盖 整个 字符串 s 的，而不是部分字符串。


示例 1：
输入：s = "aa", p = "a"
输出：false
解释："a" 无法匹配 "aa" 整个字符串。

示例 2:
输入：s = "aa", p = "a*"
输出：true
解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。

示例 3：
输入：s = "ab", p = ".*"
输出：true
解释：".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。

提示：
1 <= s.length <= 20
1 <= p.length <= 20
s 只包含从 a-z 的小写字母。
p 只包含从 a-z 的小写字母，以及字符 . 和 *。
保证每次出现字符 * 时，前面都匹配到有效的字符
*/

/*
思路:动态规划
这个问题可以通过动态规划（Dynamic Programming, DP）来高效解决。我们需要模拟正则表达式的匹配过程，并检查字符串 s 是否能够完全匹配模式 p。

正则表达式中有两个特殊字符：
. 匹配任意单个字符。
* 匹配零个或多个前面字符。

定义一个二维的 DP 数组 dp[i][j]，表示 s[0..i-1] 和 p[0..j-1] 是否匹配。具体来说，dp[i][j] 为 true 表示 s 的前 i 个字符与
p 的前 j 个字符匹配，false 表示不匹配。

状态转移
初始化：dp[0][0] = true，因为空字符串和空模式总是匹配的。

处理 * 的情况：
如果当前字符 p[j-1] 是 *，它可以有两种情况：
* 代表零次出现，跳过 p[j-2] 和 *，即 dp[i][j] = dp[i][j-2]。
* 代表一次或多次出现，当前字符 s[i-1] 必须与 p[j-2] 匹配，或者 p[j-2] 是 . 即 dp[i][j] = dp[i-1][j]。

处理 . 的情况：
如果当前字符 p[j-1] 是 . 则它总是与任意字符匹配。即 dp[i][j] = dp[i-1][j-1]。

处理普通字符：
如果当前字符 s[i-1] 和 p[j-1] 相同，或者 p[j-1] 是 . 则 dp[i][j] = dp[i-1][j-1]。
最终的答案是 dp[len(s)][len(p)]，即 s 和 p 完全匹配的结果。
*/

func isMatch(s string, p string) bool {
	m, n := len(s), len(p)
	// 创建二维 dp 数组，dp[i][j] 表示 s 的前 i 个字符和 p 的前 j 个字符是否匹配
	dp := make([][]bool, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]bool, n+1)
	}
	// 初始条件：空字符串和空模式匹配
	dp[0][0] = true
	// 处理模式 p 中以 '*' 开头的情况
	// p[j-1] == '*' 时，dp[0][j] 代表的是 s 为空字符串与模式 p 的前 j 个字符是否匹配
	for j := 1; j <= n; j++ {
		// '*' 可以匹配零个字符
		if p[j-1] == '*' {
			dp[0][j] = dp[0][j-2]
		}
	}
	// 遍历字符串 s 和模式 p，更新 dp 数组
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if p[j-1] == s[i-1] || p[j-1] == '.' {
				dp[i][j] = dp[i-1][j-1] // 字符匹配，继承上一行的状态
			} else if p[j-1] == '*' {
				// '*' 匹配多个字符
				if p[j-2] == s[i-1] || p[j-2] == '.' {
					dp[i][j] = dp[i][j] || dp[i-1][j] // '*' 匹配多个字符
				}
				// '*' 匹配零个字符
				dp[i][j] = dp[i][j] || dp[i][j-2]
			}
		}
	}
	// 返回最终结果，s 和 p 是否完全匹配
	return dp[m][n]
}

/*
leetcode 329 矩阵中的最长递增路径
给定一个 m x n 整数矩阵 matrix ，找出其中 最长递增路径 的长度。
对于每个单元格，你可以往上，下，左，右四个方向移动。 你 不能 在 对角线 方向上移动或移动到 边界外（即不允许环绕）。

输入：matrix = [[3,4,5],[3,2,6],[2,2,1]]
输出：4
解释：最长递增路径是 [3, 4, 5, 6]。注意不允许在对角线方向上移动。

提示：
m == matrix.length
n == matrix[i].length
1 <= m, n <= 200
0 <= matrix[i][j] <= 231 - 1
*/

/*
思路:动态规划+dfs搜索
这是一道典型的动态规划和深度优先搜索（DFS）结合的题目，适合用 记忆化搜索 来优化性能。具体步骤如下：

定义问题：
我们需要找到矩阵中最长的递增路径。递增路径的定义是当前位置值严格小于下一位置值。
每个位置可以向上下左右四个方向移动，但不能越界或重复访问。
解法：

使用深度优先搜索（DFS）遍历矩阵，从每个位置开始尝试寻找最长路径。
使用一个二维数组 cache 来保存从每个位置开始的最长递增路径长度。如果某个位置已经计算过，直接返回缓存值，避免重复计算。

优化：
DFS + 记忆化搜索将时间复杂度从指数级降到线性级别
O(m×n)，因为每个单元格最多只会被计算一次。

边界条件：
空矩阵或单元格长度为 1 时直接返回结果。
*/

func check(r, c, rows, cols int) bool {
	return r >= 0 && r < rows && c >= 0 && c < cols
}

func longestIncreasingPath(matrix [][]int) int {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return 0
	}
	m, n := len(matrix), len(matrix[0])
	// 缓存数组，cache[i][j] 表示从位置 (i, j) 开始的最长递增路径长度
	cache := make([][]int, m)
	for i := 0; i < m; i++ {
		cache[i] = make([]int, n)
	}
	maxLength := 0
	var dfs func(int, int) int
	dfs = func(r, c int) int {
		// 如果当前位置已经计算过，直接返回缓存值，避免重复计算
		if cache[r][c] != 0 {
			return cache[r][c]
		}
		length := 1
		// 遍历上下左右邻近位置，寻找最长递增路径
		// 判断边界条件及递增条件
		if check(r-1, c, m, n) && matrix[r][c] < matrix[r-1][c] {
			length = utils.Max(length, 1+dfs(r-1, c))
		}
		if check(r+1, c, m, n) && matrix[r][c] < matrix[r+1][c] {
			length = utils.Max(length, 1+dfs(r+1, c))
		}
		if check(r, c-1, m, n) && matrix[r][c] < matrix[r][c-1] {
			length = utils.Max(length, 1+dfs(r, c-1))
		}
		if check(r, c+1, m, n) && matrix[r][c] < matrix[r][c+1] {
			length = utils.Max(length, 1+dfs(r, c+1))
		}
		// 缓存结果并返回
		cache[r][c] = length
		return length
	}
	// 遍历每个位置，找到最长路径
	for r := 0; r < m; r++ {
		for c := 0; c < n; c++ {
			// 迭代最长路径长度 maxLength
			maxLength = utils.Max(maxLength, dfs(r, c))
		}
	}
	return maxLength
}
