package sum

import (
	"math"
	"sort"
	"strconv"

	"algorithmpractise/utils"
)

/*
leetcode 415. 字符串相加
1.1 给定两个字符串形式的非负整数num1和num2，计算它们的和并同样以字符串形式返回。
你不能使用任何內建的用于处理大整数的库，也不能直接将输入的字符串转换为整数形式。

示例1：
输入：num1 = "11", num2 = "123"
输出："134"

示例2：
输入：num1 = "456", num2 = "77"
输出："533"

示例3：
输入：num1 = "0", num2 = "0"
输出："0"

*/

/*
思路与算法:
本题我们只需要对两个大整数模拟「竖式加法」的过程。竖式加法就是我们平常学习生活中常用的对两个整数相加的方法，
回想一下我们在纸上对两个整数相加的操作，是不是如下图将相同数位对齐，从低到高逐位相加，如果当前位和超过10，
则向高位进一位？因此我们只要将这个过程用代码写出来即可。
图:字符串相加.png
具体实现也不复杂，我们定义两个指针i和j分别指向num1和num2的末尾，即最低位，同时定义一个变量add维护当前是否
有进位，然后从末尾到开头逐位相加即可。你可能会想两个数字位数不同怎么处理，这里我们统一在指针当前下标处于
负数的时候返回0，等价于对位数较短的数字进行了补零操作，
这样就可以除去两个数字位数不同情况的处理，具体可以看下面的代码。

复杂度分析
时间复杂度：O(max(len1,len2)),其中len1=num1.length, len2=num2.length,竖式加法的次数取决于较大数的
位数。
空间复杂度：O(1)。除答案外我们只需要常数空间存放若干变量.
*/

func AddStrings(num1, num2 string) string {
	// 进位值初始值为0
	add := 0
	res := ""
	// 从低位开始相加，每循环一次，索引向左移动一位
	for i, j := len(num1)-1, len(num2)-1; i >= 0 || j >= 0 || add != 0; i, j = i-1, j-1 {
		sum := 0
		if i >= 0 {
			sum += int(num1[i] - '0')
		}
		if j >= 0 {
			sum += int(num2[j] - '0')
		}
		sum += add
		res = strconv.Itoa(sum%10) + res
		add = sum / 10
	}
	return res
}

/*
leetcode 8. 字符串转换整数 (atoi)
1.2 请你来实现一个myAtoi(string s)函数，使其能将字符串转换成一个32位有符号整数（类似C/C++中的atoi函数）。
函数myAtoi(string s) 的算法如下：

读入字符串并丢弃无用的前导空格
检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤2开始）。
如果整数数超过32位有符号整数范围 [−2^31,2^31−1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于−2^31 的整数应该被固定为
−2^31 ，大于 2^31-1 的整数应该被固定为2^31−1 。
返回整数作为最终结果。
注意：

本题中的空白字符只包括空格字符' ' 。
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
leetcode 18. 四数之和
1.3 给你一个由n个整数组成的数组nums，和一个目标值target。请你找出并返回满足下述全部条件且不重复的四元组
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
			if j > i+1 && nums[j] == nums[j-1] {
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

/*
leetcode 410 分割数组的最大值
给定一个非负整数数组 nums 和一个整数 k ，你需要将这个数组分成 k 个非空的连续子数组，使得这 k 个子数组各自和的最大值
最小。

返回分割后最小的和的最大值。
子数组 是数组中连续的部份。

示例 1：
输入：nums = [7,2,5,10,8], k = 2
输出：18
解释：
一共有四种方法将 nums 分割为 2 个子数组。
其中最好的方式是将其分为 [7,2,5] 和 [10,8] 。
因为此时这两个子数组各自的和的最大值为18，在所有情况中最小。

示例 2：
输入：nums = [1,2,3,4,5], k = 2
输出：9

示例 3：
输入：nums = [1,4,4], k = 3
输出：4

提示：
1 <= nums.length <= 1000
0 <= nums[i] <= 106
1 <= k <= min(50, nums.length)
*/

/*
思路:二分查找+动态规划
这个问题是一个经典的动态规划 + 二分查找 的结合问题。

1. 二分查找的关键思想
我们可以通过二分查找来寻找最小的 "最大子数组和"。即我们要找到一个最小的最大和，使得可以将数组分成 k 个子数组，每个子
数组的和都不超过这个最大值。

最小值：max(nums)，因为每个子数组的和必须至少为一个元素的值。
最大值：sum(nums)，这是没有任何分割时数组的总和。
然后我们通过二分查找从 max(nums) 到 sum(nums) 的区间来找到最小的最大和。

2. 动态规划的辅助判断
对于每一个 "最大和" 值，我们需要判断是否可以将数组分成不超过 k 个子数组，使得每个子数组的和都小于等于当前的最大和。
我们可以通过一个贪心算法来判断。
从头开始遍历数组，当当前子数组的和超过最大和时，我们就开始分割，开始新的一组子数组。

步骤总结：
初始化二分查找的边界：l = max(nums)，r = sum(nums)。
在二分查找中，判断中间值 mid 是否能将数组分成 k 个子数组，若能则尝试减小最大和，否则增大。
返回最终的二分查找结果。
*/

func splitArray(nums []int, k int) int {
	// 二分查找的最小和最大值
	l := utils.MaxValueOfArray(nums)
	r := utils.SumOfArray(nums)
	for l < r {
		mid := l + (r-l)/2
		// 如果可以分割，说明可以尝试更小的最大和
		if canSplit(nums, mid, k) {
			r = mid
		} else {
			// 否则说明 maxSum 需要更大
			l = mid + 1
		}
	}
	return l
}

// canSplit 判断是否能将数组 nums 分割成 k 个子数组，使得每个子数组的和不超过 maxSum
func canSplit(nums []int, maxSum, k int) bool {
	// curSum当前子数组和初始化为0，count初始化为1，表示至少需要一个子数组
	curSum, count := 0, 1
	for _, num := range nums {
		// 如果当前子数组的和加上 num 超过了 maxSum，就需要分割
		if curSum+num > maxSum {
			count++
			curSum = num
			// 如果分割的子数组超过了 k 个，则返回 false
			if count > k {
				return false
			}
		} else {
			curSum = curSum + num
		}
	}
	return true
}
