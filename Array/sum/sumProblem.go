package sum

import (
	"AlgorithmPractise/Utils"
	"math"
	"sort"
	"strconv"
)

/*
1.1 连续子数组的最大和
输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。
要求时间复杂度为O(n)。
输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1]的和最大，为6。
*/

func maxSubArray(nums []int) int {
	for i := 1; i < len(nums); i++ {
		nums[i] += Utils.Max(nums[i-1], 0)
	}
	return Utils.MaxValueOfArray(nums)
}

/*
1.2 字符串相加
给定两个字符串形式的非负整数num1和num2，计算它们的和并同样以字符串形式返回。
你不能使用任何內建的用于处理大整数的库，也不能直接将输入的字符串转换为整数形式。

示例 1：

输入：num1 = "11", num2 = "123"
输出："134"
示例 2：

输入：num1 = "456", num2 = "77"
输出："533"
示例 3：

输入：num1 = "0", num2 = "0"
输出："0"

*/

/*
思路与算法:
本题我们只需要对两个大整数模拟「竖式加法」的过程。竖式加法就是我们平常学习生活中常用的对两个整数相加的方法，回想一下我们在纸上对两个整数
相加的操作，是不是如下图将相同数位对齐，从低到高逐位相加，如果当前位和超过10，则向高位进一位？因此我们只要将这个过程用代码写出来即可。
图:字符串相加.png
具体实现也不复杂，我们定义两个指针i和j分别指向num1和num2的末尾，即最低位，同时定义一个变量add维护当前是否有进位，然后从末尾到开头逐位
相加即可。你可能会想两个数字位数不同怎么处理，这里我们统一在指针当前下标处于负数的时候返回0，等价于对位数较短的数字进行了补零操作，
这样就可以除去两个数字位数不同情况的处理，具体可以看下面的代码。

复杂度分析
时间复杂度：O(max(len1,len2)),其中len1=num1.length, len2=num2.length,竖式加法的次数取决于较大数的位数。
空间复杂度：O(1)。除答案外我们只需要常数空间存放若干变量.
*/

func AddStrings(num1, num2 string) string {
	// 进位值初始值为0
	add := 0
	result := ""
	// 从低位开始相加，每循环一次，索引向左移动一位
	for i, j := len(num1)-1, len(num2)-1; i >= 0 || j >= 0 || add != 0; i, j = i-1, j-1 {
		var x, y int
		if i >= 0 {
			x = int(num1[i] - '0')
		}
		if j >= 0 {
			y = int(num2[j] - '0')
		}
		sum := x + y + add
		result = strconv.Itoa(sum%10) + result
		add = sum / 10
	}
	return result
}

/*
1.3 字符串转换整数
请你来实现一个myAtoi(string s)函数，使其能将字符串转换成一个32位有符号整数（类似 C/C++ 中的atoi函数）。

函数myAtoi(string s) 的算法如下：

读入字符串并丢弃无用的前导空格
检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤2开始）。
如果整数数超过32位有符号整数范围 [−2^31,2^31−1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于−2^31 的整数应该被固定为
−2^31 ，大于 2^31-1 的整数应该被固定为2^31−1 。
返回整数作为最终结果。
注意：

本题中的空白字符只包括空格字符 ' ' 。
除前导空格或数字后的其余字符串外，请勿忽略任何其他字符。

输入：s = "42"
输出：42

输入：s = "   -42"
输出：-42
*/

func MyAtoi(s string) int {
	sum, sign, i, n := 0, 1, 0, len(s)
	// 丢弃无用的前导空格
	for i < n && s[i] == ' ' {
		i++
	}
	// 判定正负
	if i < n {
		if s[i] == '-' {
			sign = -1
			i++
		} else if s[i] == '+' {
			sign = 1
			i++
		}
	}
	// 从左到右依次累加
	for i < n && s[i] >= '0' && s[i] <= '9' {
		sum = 10*sum + int(s[i]-'0')
		// 整数超过32位有符号整数范围,特殊处理
		if sign*sum < math.MinInt32 {
			return math.MinInt32
		} else if sign*sum > math.MaxInt32 {
			return math.MaxInt32
		}
		i++
	}
	return sign * sum
}

/*
1.4 四数之和
给你一个由n个整数组成的数组nums ，和一个目标值target 。请你找出并返回满足下述全部条件且不重复的四元组
[nums[a], nums[b], nums[c], nums[d]]（若两个四元组元素一一对应，则认为两个四元组重复）：

0 <= a, b, c, d< n
a、b、c 和 d 互不相同
nums[a] + nums[b] + nums[c] + nums[d] == target
你可以按任意顺序返回答案 。

示例1：
输入：nums = [1,0,-1,0,-2,2], target = 0
输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]

示例2：
输入：nums = [2,2,2,2,2], target = 8
输出：[[2,2,2,2]]

提示：
1 <= nums.length <= 200
-109 <= nums[i] <= 109
-109 <= target <= 109
*/

func FourSum(nums []int, target int) [][]int {
	var res [][]int
	n := len(nums)
	if n < 4 {
		return res
	}
	// 首先对数组进行排序
	sort.Ints(nums)
	for i := 0; i < n-3; i++ {
		// 防止重复数组进入res
		if i >= 1 && nums[i] == nums[i-1] {
			continue
		}
		// 最小的四个元素和都大于target,那么后续的数字都不用遍历了
		if nums[i]+nums[i+1]+nums[i+2]+nums[i+3] > target {
			break
		}
		// nums[i]加上最大的三个元素和都小于target，说明i需要向后移动
		if nums[i]+nums[n-3]+nums[n-2]+nums[n-1] < target {
			continue
		}
		// 遍历第二个元素
		for j := i + 1; j < n-2; j++ {
			// 同理，防止重复数组进入res
			if j-i > 1 && nums[j] == nums[j-1] {
				continue
			}
			// 同理，最小的四个元素和都大于target,那么后续的数字都不用遍历了
			if nums[i]+nums[j]+nums[j+1]+nums[j+2] > target {
				break
			}
			// nums[i]+nums[j]加上最大的两个元素和都小于target，说明j需要向后移动
			if nums[i]+nums[j]+nums[n-2]+nums[n-1] < target {
				continue
			}
			// 双指针
			l, r := j+1, n-1
			for l < r {
				sum := nums[i] + nums[j] + nums[l] + nums[r]
				if sum == target {
					res = append(res, []int{nums[i], nums[j], nums[l], nums[r]})
					// nums[l]去重
					for l < r && nums[l] == nums[l+1] {
						l++
					}
					// nums[r]去重
					for l < r && nums[r] == nums[r-1] {
						r--
					}
					l++
					r--
				} else if sum > target {
					r--
				} else {
					l++
				}
			}
		}
	}
	return res
}