package hard

/*
package hard contains complex dynamicProgramming problems
*/

import "AlgorithmPractise/Utils"

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
		leftMax = Utils.MaxValueOfArray(height[:i])
		rightMax = Utils.MaxValueOfArray(height[i+1:])
		if leftMax > height[i] && rightMax > height[i] {
			sum += Utils.Min(leftMax, rightMax) - height[i]
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
		leftMax[i] = Utils.Max(height[i-1], leftMax[i-1])
	}
	for j := n - 3; j >= 1; j-- {
		rightMax[j] = Utils.Max(height[j+1], rightMax[j+1])
	}
	for k := 1; k <= n-2; k++ {
		if value := Utils.Min(leftMax[k], rightMax[k]) - height[k]; value > 0 {
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
			curHeight := Utils.Min(height[left], h) - low
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
			leftMax = Utils.Max(leftMax, height[left-1])
			if leftMax > height[left] {
				sum += leftMax - height[left]
			}
			left++
		} else {
			rightMax = Utils.Max(rightMax, height[right+1])
			if rightMax > height[right] {
				sum += rightMax - height[right]
			}
			right--
		}
	}
	return sum
}
