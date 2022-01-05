package Hard

import "AlgorithmPractise/Utils"

/*
1.1 接雨水
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
	leftMax, rightMax := 0, 0
	for i := 1; i < len(height)-1; i++ {
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

创建两个长度为n的数组leftMax和rightMax。对于0≤i<n，leftMax[i] 表示下标i及其左边的位置中，height的最大高度，
rightMax[i]表示下标i及其右边的位置中，height的最大高度。

显然，leftMax[0]=height[0]，rightMax[n−1]=height[n−1]。两个数组的其余元素的计算如下：

当1≤i≤n−1时，leftMax[i]=max(leftMax[i−1],height[i])；

当0≤i≤n−2时，rightMax[i]=max(rightMax[i+1],height[i])。

因此可以正向遍历数组height得到数组leftMax的每个元素值，反向遍历数组height得到数组rightMax的每个元素值。
在得到数组leftMax和rightMax的每个元素值之后，对于0≤i<n，下标i处能接的雨水量等于
min(leftMax[i],rightMax[i])−height[i]。遍历每个下标位置即可得到能接的雨水总量。
*/

func TrapUseDp(height []int) int {
	sum := 0
	n := len(height)
	leftMax, rightMax := make([]int, n), make([]int, n)
	leftMax[0], rightMax[n-1] = height[0], height[n-1]
	for i := 1; i < n; i++ {
		leftMax[i] = Utils.Max(leftMax[i-1], height[i])
	}
	for i := n - 2; i >= 0; i-- {
		rightMax[i] = Utils.Max(rightMax[i+1], height[i])
	}
	for i := 1; i < n; i++ {
		if value := Utils.Min(leftMax[i], rightMax[i]) - height[i]; value > 0 {
			sum += value
		}
	}
	return sum
}

/*
思路:双指针
动态规划的做法中，需要维护两个数组leftMax和rightMax，因此空间复杂度是O(n)。是否可以将空间复杂度降到O(1)？
注意到下标i处能接的雨水量由leftMax[i]和rightMax[i]中的最小值决定。由于数组leftMax是从左往右计算，数组
rightMax是从右往左计算，因此可以使用双指针和两个变量代替两个数组。

维护两个指针left和right，以及两个变量leftMax和rightMax，初始时left=0,right=n−1,leftMax=0,rightMax=0。
指针left只会向右移动，指针right只会向左移动，在移动指针的过程中维护两个变量 leftMax和rightMax的值。

当两个指针没有相遇时，进行如下操作：
使用height[left]和height[right]的值更新leftMax和rightMax的值；
如果height[left]<height[right]，则必有leftMax<rightMax，下标left处能接的雨水量等于leftMax−height[left]，
将下标left处能接的雨水量加到能接的雨水总量，然后将left加1（即向右移动一位）；
如果height[left]≥height[right]，则必有leftMax≥rightMax，下标right处能接的雨水量等于rightMax−height[right]，
将下标right处能接的雨水量加到能接的雨水总量，然后将right减1（即向左移动一位）。

当两个指针相遇时，即可得到能接的雨水总量。
*/

func TrapSimple(height []int) int {
	sum := 0
	left, right := 0, len(height)-1
	leftMax, rightMax := 0, 0
	for left < right {
		leftMax = Utils.Max(leftMax, height[left])
		rightMax = Utils.Max(rightMax, height[right])
		if height[left] < height[right] {
			sum += leftMax - height[left]
			left++
		} else {
			sum += rightMax - height[right]
			right--
		}
	}
	return sum
}