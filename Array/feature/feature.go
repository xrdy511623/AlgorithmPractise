package feature

import (
	"algorithm-practise/utils"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"
)

/*
1.1 反转数组
*/

// ReverseArraySimple 可以创建新数组
func ReverseArraySimple(nums []int) []int {
	n := len(nums)
	var res []int
	for i := n - 1; i >= 0; i-- {
		res = append(res, nums[i])
	}
	return res
}

// ReverseArray 原地反转
func ReverseArray(nums []int) []int {
	for i, n := 0, len(nums); i < n/2; i++ {
		nums[i], nums[n-1-i] = nums[n-1-i], nums[i]
	}
	return nums
}

/*
出现次数专题
*/

/*
leetcode 169. 多数元素
1.2 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
你可以假设数组是非空的，并且给定的数组总是存在多数元素。
*/

/*
第一种思路，最简单的办法就是遍历数组，用哈希表记录数组中每个元素出现的次数，如果哪个元素的出现次数大于数组长度的一半，
那么这个元素就是众数，这样做时间复杂度O(n)，空间复杂度O(n/2),显然空间复杂度较高，不是最优解。

第二种思路:如果我们把众数的值记为+1，把其他数记为−1，将它们全部加起来，显然和大于0，从结果本身我们可以看出众数比
其他数多。我们维护一个候选众数candidate和它出现的次数count。初始时candidate可以为任意值，count为0；
我们遍历数组nums中的所有元素，对于每个元素x，在判断x之前，如果count的值为0，我们先将x的值赋予candidate，随后我们判断x：
如果x与candidate相等，那么计数器count的值增加1；
如果x与candidate不等，那么计数器count的值减少1。
在遍历完成后，candidate即为整个数组的众数。为什么？因为非众数在遍历过程中一定会遇到出现次数比它多的众数，这样count值会被减到0，
从而引发candidate的重新赋值，同理，只有众数赋值给candidate后，count才不会减到0，因为所有非众数出现的次数加起来都没有它多，
这样candidate的值就会一直是众数，直到遍历数组结束。

时间复杂度O(n)，空间复杂度O(1)
*/

func MajorityElement(nums []int) int {
	candidate, count := 0, 0
	for _, v := range nums {
		if count == 0 {
			candidate = v
		}
		if v == candidate {
			count++
		} else {
			count--
		}
	}
	return candidate
}

/*
leetcode 229. 求众数II
1.3 给定一个大小为n的整数数组，找出其中所有出现超过n/3次的元素。

示例1：
输入：[3,2,3]
输出：[3]

示例2：
输入：nums = [1]
输出：[1]

示例3：
输入：[1,1,1,3,3,2,2,2]
输出：[1,2]

提示：
1 <= nums.length <= 5 * 104
-109 <= nums[i] <= 109

进阶：尝试设计时间复杂度为 O(n)、空间复杂度为 O(1)的算法解决此问题。
*/

/*
思路:摩尔投票法。数组长度为n, 那么数组中最多只能有两个出现次数超过n/3次的元素。我们可以用反证法证明，如果有两个以上
的元素出现次数超过n/3次(最有可能的情况是3个元素，如果3个都不行，其他就更不可能了)，3个元素的出现次数之和肯定就
超过了n(x>n/3,3x>n/3*3,于是3x>n)，也就是超出数组长度了，显然是不可能的。所以数组中如果有出现次数超过n/3次的元素
顶多两个。

*/

func majorityElement(nums []int) []int {
	res := []int{}
	n := len(nums)
	if n == 0 {
		return res
	}
	// 初始化两个候选人，以及他们的得票数
	cand1, cand2 := nums[0], nums[0]
	count1, count2 := 0, 0
	// 配对和抵消阶段
	for _, num := range nums {
		// 投票
		if num == cand1 {
			count1++
			continue
		}
		if num == cand2 {
			count2++
			continue
		}
		if count1 == 0 {
			cand1 = num
			count1++
			continue
		}
		if count2 == 0 {
			cand2 = num
			count2++
			continue
		}
		// 如果遍历的元素num既不是cand1,cand2，那么两个候选人的得票数都减一
		count1--
		count2--
	}
	// 上面得到的候选人cand1和cand2不一定就是出现次数超过n/3次的元素
	// 只有数组中确实有两个元素出现次数超过n/3次时，他俩才是。
	// 否则需要计票确认。显然，如果数组中元素很分散(最极端的情况是各不相同)，cand1和cand2就会是数组中
	// 倒数第二个和倒数第三个元素，如果数组中只有一个元素出现次数超过n/3次，cand1会是我们要的候选人， cand2绝对不是
	// 计票阶段，确定两个候选人得票数是否超过n/3次
	count1, count2 = 0, 0
	for _, num := range nums {
		// 如果数组中只有一个元素或者数组中有多个元素，但这些元素值相等，那么经过第一轮投票后,c1和c2会相等
		// 譬如数组[1]和数组[3,3,3]，所以此时第二轮计票时，既不能使用两个独立的if判断，否则会出现结果重复的情况
		if num == cand1 {
			count1++
		} else if num == cand2 {
			count2++
		}
	}
	// 将得票数超过n/3的候选人添加到结果集中
	if count1 > n/3 {
		res = append(res, cand1)
	}
	if count2 > n/3 {
		res = append(res, cand2)
	}
	return res
}

/*
leetcode 26. 删除有序数组中的重复项
1.7 给你一个有序数组nums，请你原地删除重复出现的元素，使每个元素只出现一次 ，返回删除后数组的新长度。
不要使用额外的数组空间，你必须在原地修改输入数组并在使用O(1) 额外空间的条件下完成。
*/

/*
数组是有序的，那么重复的元素一定会相邻。在同一个数组里面操作，也就是将不重复的元素移到数组的左侧，
最后取左侧的数组的值。
*/

func RemoveDuplicates(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	slow := 0
	for fast := 1; fast < n; fast++ {
		if nums[fast] != nums[fast-1] {
			slow++
			nums[slow] = nums[fast]
		}
	}
	return slow + 1
}

/*
leetcode 27. 移除元素
1.8 给你一个数组nums和一个值val，你需要原地移除所有数值等于val的元素，并返回移除后数组的新长度。
不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并原地修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]

输入：nums = [0,1,2,2,3,0,4,2], val = 2
输出：5, nums = [0,1,4,0,3]
*/

func RemoveElement(nums []int, val int) int {
	for i := 0; i < len(nums); i++ {
		if nums[i] == val {
			nums = append(nums[:i], nums[i+1:]...)
			i--
		}
	}
	return len(nums)
}

/*
双指针法，快慢指针fast,slow初始位置均为0，fast指向当前要处理的元素，slow指向下一个将要赋值的位置，
如果fast指向的元素不等于val，那它一定是新数组的一个元素，我们将它赋值给slow指向的位置，同时快慢
指针同时向右移动，若fast指向的元素等于val，那它不是新数组想要的元素，此时slow指针不动，fast向后移动一位。
*/

func RemoveElementsTwo(nums []int, val int) int {
	fast, slow := 0, 0
	for fast < len(nums) {
		if nums[fast] != val {
			nums[slow] = nums[fast]
			slow++
		}
		fast++
	}
	nums = nums[:slow]
	return slow
}

/*
leetcode 283. 移动零
1.9 给定一个数组nums，编写一个函数将所有0移动到数组的末尾，同时保持非零元素的相对顺序。

示例:
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]

说明:
必须在原数组上操作，不能拷贝额外的数组。
尽量减少操作次数。
*/

/*
思路:先处理非0元素，最后处理0
index指针指向非0元素，初始值为0.在for循环遍历过程中，如果遇到非0元素，则将其赋值给nums[index]，同时index指针
向右移动一位。遍历结束后，index即指向所有非0元素最后一位的右边,这就意味着nums[index:]范围内的元素都应该是0。
*/

func MoveZeroes(nums []int) {
	index, n := 0, len(nums)
	for i := 0; i < n; i++ {
		if nums[i] != 0 {
			nums[index] = nums[i]
			index++
		}
	}
	for index < n {
		nums[index] = 0
		index++
	}
}

/*
leetcode 14. 最长公共前缀
1.10 编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串""。

示例 1：
输入：strs = ["flower","flow","flight"]
输出："fl"

示例 2：
输入：strs = ["dog","racecar","car"]
输出：""
解释：输入不存在公共前缀。

1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] 仅由小写英文字母组成
*/

/*
思路:化繁为简，多个字符串的最长公共前缀不好求，两个字符串的最长公共前缀是很容易求得的，那么我们可以先求出
字符串数组中前两个字符串的最长公共前缀prefix, 然后在遍历字符串数组strs时，迭代这个prefix就好了，也就是
求prefix和下一个字符串strs[i]的最长公共前缀。特别的，如果循环中，prefix长度为0，说明strs[0:i]范围内的
所有字符串最长公共前缀为空串，后续的遍历也就没有意义了，直接break退出循环。当然，还需要考虑特殊情况，如果
字符串数组的长度为0，直接返回空串。
时间复杂度：O(mn)，其中m是字符串数组中的字符串的平均长度，n是字符串的数量。最坏情况下，字符串数组中的每个
字符串的每个字符都会被比较一次。
空间复杂度：O(1)。使用的额外空间复杂度为常数。
*/

func LongestCommonPrefix(strs []string) string {
	n := len(strs)
	if n == 0 {
		return ""
	}
	prefix := strs[0]
	for i := 1; i < n; i++ {
		prefix = lcp(prefix, strs[i])
		if len(prefix) == 0 {
			break
		}
	}
	return prefix
}

func lcp(str1, str2 string) string {
	length := utils.Min(len(str1), len(str2))
	index := 0
	for index < length && str1[index] == str2[index] {
		index++
	}
	return str1[:index]
}

/*
leetcode 674. 最长连续递增序列
1.11 给定一个未经排序的整数数组，找到最长且连续递增的子序列，并返回该序列的长度。
连续递增的子序列，可以由两个下标l和r（l < r）确定，如果对于每个l <= i < r，都有nums[i] < nums[i + 1] ，
那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。

输入：nums = [1,3,5,4,7]
输出：3
解释：最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。

输入：nums = [2,2,2,2,2]
输出：1
解释：最长连续递增序列是 [2], 长度为1。
*/

func FindLengthOfLCIS(nums []int) int {
	maxLength, start := 0, 0
	for i, v := range nums {
		if i > 0 && v <= nums[i-1] {
			start = i
		}
		maxLength = utils.Max(maxLength, i-start+1)
	}

	return maxLength
}

/*
leetcode 53. 最大子数组和
1.12 给定一个整数数组nums，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

示例 1：
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组[4,-1,2,1] 的和最大，为6 。

示例 2：
输入：nums = [1]
输出：1
示例 3：

输入：nums = [0]
输出：0
示例 4：

输入：nums = [-1]
输出：-1
*/

func MaxSubArray(nums []int) int {
	maxSub, n := nums[0], len(nums)
	for i := 1; i < n; i++ {
		if nums[i-1] > 0 {
			nums[i] += nums[i-1]
		}
		if nums[i] > maxSub {
			maxSub = nums[i]
		}
	}
	return maxSub
}

/*
leetcode 209. 长度最小的子数组
剑指Offer II 008. 和大于等于target的最短子数组
1.13 给定一个含有n个正整数的数组和一个正整数s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组，并返回其长度。
如果不存在符合条件的子数组，返回0。

示例：
输入：s = 7, nums = [2,3,1,2,4,3] 输出：2 解释：子数组 [4,3] 是该条件下的长度最小的子数组。
*/

/*
思路:滑动窗口
所谓滑动窗口，就是不断的调节子序列的起始位置和终止位置，从而得出我们要想的结果。
实现滑动窗口，主要确定以下三点：
窗口内是什么？
如何移动窗口的起始位置？
如何移动窗口的结束位置？
窗口就是满足其和 ≥ target 的长度最小的连续子数组。
窗口的起始位置如何移动：如果当前窗口的值大于target了，窗口起始位置就要向右移动了（也就是该缩小窗口了）。
窗口的结束位置如何移动：窗口的结束位置就是遍历数组的指针，窗口结束位置初始值设置为数组的起始位置就可以了。
滑动窗口的精妙之处在于根据当前子序列和大小的情况，不断调节子序列的起始位置。
子数组长度 length := end-start+1(窗口结束位置-窗口起始位置+1)
*/

// MinSubArrayLen 时间复杂度O(2N),空间复杂度O(1)
func MinSubArrayLen(target int, nums []int) int {
	// 和大于target的连续子数组的最小长度，初始值设为最大的int，便于后续迭代
	minLength := math.MaxInt32
	// 连续子数组之和，初始值为0
	sum, n := 0, len(nums)
	// 滑动窗口的起始位置,初始值0
	start := 0
	for end := 0; end < n; end++ {
		sum += nums[end]
		for sum >= target {
			// 计算窗口大小，即满足和大于等于target的子数组长度
			length := end - start + 1
			minLength = utils.Min(minLength, length)
			// 缩小窗口，也就是将滑动窗口的起始位置向右移动，看看是否仍满足和大于target
			sum -= nums[start]
			start++
		}
	}
	// 如果minLength的值没有被更新，说明不存在和大于等于target的子数组，返回0
	if minLength == math.MaxInt32 {
		return 0
	}
	return minLength
}

/*
最多购买宝石数目
橱窗里有一排宝石，不同的宝石对应不同的价格，宝石的价格标记为 gems[i],0<=i<n, n = gems.length
宝石可同时出售0个或多个，如果同时出售多个，则要求出售的宝石编号连续；
例如客户最大购买宝石个数为m，购买的宝石编号必须为gems[i],gems[i+1]...gems[i+m-1](0<=i<n,m<=n)
假设你当前拥有总面值为value的钱，请问最多能购买到多少个宝石,如无法购买宝石，则返回 0。

输入描述
第一行输入n，参数类型为int，取值范围：[0,10^6]，表示橱窗中宝石的总数量。
之后n行分别表示从第0个到第n-1个宝石的价格，即gems[0]到gems[n-1]的价格，类型为int，取值范围：(0,1000]。
之后一行输入v，类型为int，取值范围：[0,10^9]表示你拥有的钱。

输出描述
输出int类型的返回值，表示最大可购买的宝石数量。

示例1
输入：
7
8
4
6
3
1
6
7
10

输出：
3
示例2
输入：
0
1

输出：
0

说明：
因为没有宝石，所以返回 0
示例3
输入：
9
6
1
3
1
8
9
3
2
4
15

输出：
4
*/

/*
思路:滑动窗口
*/

func mostGems(gems []int, sum int) int {
	if sum == 0 {
		return 0
	}
	// 初始化窗口最大长度，窗口左边界，窗口宝石价值总和为0
	maxLength, l, curSum, n := 0, 0, 0, len(gems)
	for r := 0; r < n; r++ {
		curSum += gems[r]
		// 当窗口总和超过 value 时，收缩左边界
		for curSum > sum && l <= r {
			curSum -= gems[l]
			l++
		}
		// 更新最大长度（当前窗口长度为 right - left + 1
		maxLength = utils.Max(maxLength, r-l+1)
	}
	return maxLength
}

/*
leetcode 1423 可获得的最大点数
几张卡牌 排成一行，每张卡牌都有一个对应的点数。点数由整数数组 cardPoints 给出。

每次行动，你可以从行的开头或者末尾拿一张卡牌，最终你必须正好拿 k 张卡牌。

你的点数就是你拿到手中的所有卡牌的点数之和。

给你一个整数数组 cardPoints 和整数 k，请你返回可以获得的最大点数。



示例 1：

输入：cardPoints = [1,2,3,4,5,6,1], k = 3
输出：12
解释：第一次行动，不管拿哪张牌，你的点数总是 1 。但是，先拿最右边的卡牌将会最大化你的可获得点数。最优策略是拿右边的三张牌，
最终点数为 1 + 6 + 5 = 12 。
示例 2：

输入：cardPoints = [2,2,2], k = 2
输出：4
解释：无论你拿起哪两张卡牌，可获得的点数总是 4 。
示例 3：

输入：cardPoints = [9,7,7,9,7,7,9], k = 7
输出：55
解释：你必须拿起所有卡牌，可以获得的点数为所有卡牌的点数之和。
示例 4：

输入：cardPoints = [1,1000,1], k = 1
输出：1
解释：你无法拿到中间那张卡牌，所以可以获得的最大点数为 1 。
示例 5：

输入：cardPoints = [1,79,80,1,1,1,200,1], k = 3
输出：202


提示：

1 <= cardPoints.length <= 10^5
1 <= cardPoints[i] <= 10^4
1 <= k <= cardPoints.length
*/

/*
思路:滑动窗口
这个问题要求从数组 cardPoints 的开头或末尾拿正好 k 张卡牌，求最大点数和。直接枚举所有可能的开头和末尾组合会非常复杂，
时间复杂度为 O(2^k)，对于 (k) 较大时不可行。因此，我们需要一个更高效的算法。

思路分析
问题本质：从数组两端拿 k 个元素，等价于从数组中选择一个长度为 n−k
的连续子数组（剩余部分），使得这个子数组的和最小，剩余的 (k) 个元素（从两端取）的和最大。

滑动窗口：总和是固定的（数组所有元素之和），最大化 (k) 个元素的和等价于最小化剩余 n−k
个元素的和。我们可以用滑动窗口来计算长度为 n−k 的子数组的最小和。

步骤：
计算数组总和 (total)。
用滑动窗口找到长度为 n −k的子数组的最小和 (minWindowSum)。

最大点数 = total−minWindow

时间复杂度
计算总和：(O(n))。
滑动窗口遍历：(O(n))，窗口大小固定为 n−k，只需遍历一次。
总时间复杂度：(O(n))。

空间复杂度
只需几个变量存储总和和窗口和，空间复杂度为 (O(1))。

特殊情况
当 k=n 时，直接返回数组总和。
当 k=0 时，返回 0（题目保证 k≥1)
)。
*/

func maxScore(cardPoints []int, k int) int {
	// 计算数组cardPoints总和
	n := len(cardPoints)
	total := 0
	for _, point := range cardPoints {
		total += point
	}
	// 特殊处理，如果k=n，直接返回数组cardPoints的所有点数之和
	if k == n {
		return total
	}
	// 使用滑动窗口找到长度为 n-k 的子数组的最小和
	// 窗口大小恒定为n-k
	windowSize := n - k
	// 窗口和初初始化为0
	windowSum := 0
	// 初始化第一个窗口的和
	for i := 0; i < windowSize; i++ {
		windowSum += cardPoints[i]
	}
	// 记录最小窗口和，初始为第一个窗口
	minWindowSum := windowSum
	// 滑动窗口，从左到右移动，更新最小窗口和
	for i := windowSize; i < n; i++ {
		// 移除窗口左端元素，加入右端新元素
		windowSum = windowSum - cardPoints[i-windowSize] + cardPoints[i]
		if windowSum < minWindowSum {
			minWindowSum = windowSum
		}
	}
	// 最大点数 = 总和 - 最小窗口和
	return total - minWindowSum
}

/*
leetcode 1004 最大连续1的个数 III
给定一个二进制数组 nums 和一个整数 k，假设最多可以翻转 k 个 0 ，则返回执行操作后 数组中连续 1 的最大个数 。

示例 1：
输入：nums = [1,1,1,0,0,0,1,1,1,1,0], K = 2
输出：6
解释：[1,1,1,0,0,1,1,1,1,1,1]
粗体数字从 0 翻转到 1，最长的子数组长度为 6。

示例 2：
输入：nums = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], K = 3
输出：10
解释：[0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
粗体数字从 0 翻转到 1，最长的子数组长度为 10。

提示：
1 <= nums.length <= 105
nums[i] 不是 0 就是 1
0 <= k <= nums.length
*/

/*
思路: 滑动窗口
这个问题要求在二进制数组 nums 中最多翻转 k 个 0，返回操作后连续 1 的最大个数。直接枚举所有可能的翻转组合复杂度太高
O(2^n)，因此需要一个高效的算法。

思路分析
问题本质：找到一个最长的子数组，使得其中 0 的个数不超过 k（因为我们可以将这些 0 翻转为 1）。
滑动窗口：使用滑动窗口来解决这个问题：
窗口内的 0 的个数不能超过 k。
当窗口内的 0 超过 k 时，缩小窗口（移动左边界），直到 0 的个数不超过 k。
窗口长度即为当前连续 1 的个数（假设所有 0 都被翻转为 1）。

步骤：
用两个指针（left 和 right）维护一个滑动窗口。
扩展右边界（right），统计窗口内的 0 的个数。
当 0 的个数超过 k 时，移动左边界（left），减少 0 的计数。
记录窗口的最大长度。

时间复杂度
每个元素最多被访问两次（右指针扩展一次，左指针收缩一次）。
总时间复杂度：(O(n))，其中 (n) 是数组长度。

空间复杂度
只需几个变量记录窗口状态，空间复杂度为 (O(1))。

特殊情况
当 k=0 时，返回数组中最长的连续 1 子数组。
当 k≥n时，可以翻转所有 0，返回整个数组长度。
*/

func longestOnes(nums []int, k int) int {
	n := len(nums)
	// 特殊情况：如果 k >= n，可以翻转所有 0，返回整个数组长度
	if k >= n {
		return n
	}
	// 初始化最大窗口长度，窗口内0计数，窗口左边界为0
	maxLength, zeroCount, l := 0, 0, 0
	// 遍历数组，扩展右边界
	for r, num := range nums {
		// 如果当前元素是 0，增加 zeroCount计数
		if num == 0 {
			zeroCount++
		}
		// 当窗口内 0 的个数超过 k 时，收缩左边界
		for zeroCount > k {
			// 将左边界移出窗口，减少 0 的计数
			if nums[l] == 0 {
				zeroCount--
			}
			// 右移左边界
			l++
		}
		// 更新最大窗口长度
		maxLength = utils.Max(maxLength, r-l+1)
	}
	return maxLength
}

/*
leetcode 557. 反转字符串中的单词III
1.14 给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。

示例：
输入："Let's take LeetCode contest"
输出："s'teL ekat edoCteeL tsetnoc"
*/

/*
思路:以空格分割字符串s得到单词(字符串)集合，将每个单词反转后添加到结果集合中(字符串集合)
最后返回strings.Join(res, " ")即可。
*/

func ReverseWords(s string) string {
	array := strings.Split(s, " ")
	var res []string
	for _, str := range array {
		bytes := []byte(str)
		for i, n := 0, len(bytes); i < n/2; i++ {
			bytes[i], bytes[n-1-i] = bytes[n-1-i], bytes[i]
		}
		res = append(res, string(bytes))
	}
	return strings.Join(res, " ")
}

/*
声明一个新字符串。然后从头到尾遍历原字符串，直到找到空格为止，此时找到了一个单词，并能得到单词的起止位置。
随后，根据单词的起止位置，可以将该单词逆序放到新字符串当中。如此循环多次，直到遍历完原字符串，就能得到翻转
后的结果。
*/

func Reverse(s []byte) {
	for i, n := 0, len(s); i < n/2; i++ {
		s[i], s[n-1-i] = s[n-1-i], s[i]
	}
}

func ReverseWordsTwo(s string) string {
	n := len(s)
	var res []byte
	for i := 0; i < n; {
		start := i
		// 找到一个单词的末尾位置
		for i < n && s[i] != ' ' {
			i++
		}
		// 将s[start,i]区间内的字符逆序放入到res中
		for p := i - 1; p >= start; p-- {
			res = append(res, s[p])
		}
		// 空格保持原样，添加到res中
		for i < n && s[i] == ' ' {
			res = append(res, ' ')
			i++
		}
	}
	return string(res)
}

/*
leetcode 541. 反转字符串II
1.15 给定一个字符串s和一个整数k，从字符串开头算起，每计数至2k个字符，就反转这2k字符中的前k个字符。
如果剩余字符少于k个，则将剩余字符全部反转。
如果剩余字符小于2k但大于或等于k个，则反转前k个字符，其余字符保持原样。

输入：s = "abcdefg", k = 2
输出："bacdfeg"
*/

func ReverseStr(s string, k int) string {
	bytes := []byte(s)
	n := len(s)
	for i := 0; i < n; i += 2 * k {
		// 每2k个字符对前k个字符进行反转
		// 剩余字符小于2k但大于或等于k个，则反转前k个字符
		if i+k <= n {
			reverse(bytes[i : i+k])
		} else {
			// 剩余字符少于k个，则将剩余字符全部反转。
			reverse(bytes[i:n])
		}
	}
	return string(bytes)
}

func reverse(s []byte) {
	for i, n := 0, len(s); i < n/2; i++ {
		s[i], s[n-1-i] = s[n-1-i], s[i]
	}
}

/*
剑指Offer 05. 替换空格
1.16 请实现一个函数，把字符串s中的每个空格替换成"%20"。

示例1：
输入：s = "We are happy."
输出："We%20are%20happy."
*/

// ReplaceSpace 遍历添加 时间复杂度O(N),空间复杂度O(N)
func ReplaceSpace(s string) string {
	var res []byte
	for i := 0; i < len(s); i++ {
		if s[i] != ' ' {
			res = append(res, s[i])
		} else {
			res = append(res, []byte("%20")...)
		}
	}
	return string(res)
}

// ReplaceSpaceSimple 原地修改 时间复杂度O(N),空间复杂度O(1)
func ReplaceSpaceSimple(s string) string {
	b := []byte(s)
	length := len(b)
	spaceCount := 0
	// 计算空格数量
	for _, v := range b {
		if v == ' ' {
			spaceCount++
		}
	}
	// 扩展原有切片
	tmp := make([]byte, spaceCount*2)
	b = append(b, tmp...)
	i := length - 1
	j := len(b) - 1
	// 从后向前填充
	for i >= 0 {
		if b[i] != ' ' {
			b[j] = b[i]
			i--
			j--
		} else {
			b[j] = '0'
			b[j-1] = '2'
			b[j-2] = '%'
			i--
			j = j - 3
		}
	}
	return string(b)
}

/*
leetcode 151. 翻转字符串里的单词
1.17 给你一个字符串s，逐个翻转字符串中的所有单词 。
单词是由非空格字符组成的字符串。s中使用至少一个空格将字符串中的单词分隔开。
请你返回一个翻转s中单词顺序并用单个空格相连的字符串。

说明：
输入字符串s可以在前面、后面或者单词间包含多余的空格。
翻转后单词间应当仅用一个空格分隔。
翻转后的字符串中不应包含额外的空格。

示例1：
输入：s = "the sky is blue"
输出："blue is sky the"

示例2：
输入：s = " hello world "
输出："world hello"
解释：输入字符串可以在前面或者后面包含多余的空格，但是翻转后的字符不能包括。

提示：
1 <= s.length <= 104
s 包含英文大小写字母、数字和空格 ' '
s 中至少存在一个单词
*/

/*
思路:先去掉冗余的空格然后反转整个字符串，然后再依次反转字符串中的每个单词。
想一下，如果我们将整个字符串都反转过来，那么单词的顺序指定是倒序了，只不过单词本身也倒叙了，那么再把单词
反转一下，单词不就正过来了。

所以解题思路如下：
移除多余空格
将整个字符串反转
将每个单词反转
举个例子，源字符串为：" the sky is blue  "
移除多余空格 : "the sky is blue"
字符串反转："eulb si yks eht"
单词反转："blue is sky the"
这样我们就完成了翻转字符串里的单词。
*/

func reverseWords(s string) string {
	ss := []byte(s)
	length := len(ss)
	slow, fast := 0, 0
	// 去掉字符串最左边的冗余空格
	for fast < length && ss[fast] == ' ' {
		fast++
	}
	// 去掉单词之间的冗余空格
	for ; fast < length; fast++ {
		if fast > 1 && ss[fast] == ss[fast-1] && ss[fast] == ' ' {
			continue
		}
		ss[slow] = ss[fast]
		slow++
	}
	// 去掉字符串最右边的冗余空格, 也就是如果字符串s最后一位是空格要去掉
	if slow > 1 && ss[slow-1] == ' ' {
		ss = ss[:slow-1]
	} else {
		ss = ss[:slow]
	}
	// 反转整个字符串
	reverse(ss)
	i, n := 0, len(ss)
	for i < n {
		// 反转单个单词
		j := i
		// 找到单词的结束位置
		for j < n && ss[j] != ' ' {
			j++
		}
		// 反转
		reverse(ss[i:j])
		// 更新下一个单词的起始位置，+1是要跳过单词间的空格
		i = j + 1
	}
	return string(ss)
}

/*
剑指Offer 58 - II. 左旋转字符串
1.18 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。
比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

示例1：
输入: s = "abcdefg", k = 2
输出:"cdefgab"

示例2：
输入: s = "lrloseumgh", k = 6
输出:"umghlrlose"

限制：
1 <= k < s.length <= 10000
*/

/*
思路一:在ss[k:length]集合后边依次添加ss中前k个元素即可。
*/

// ReverseLeftWords 时间复杂度O(N),空间复杂度O(k)
func ReverseLeftWords(s string, k int) string {
	var res strings.Builder
	res.Grow(len(s))
	res.WriteString(s[k:])
	res.WriteString(s[:k])
	return res.String()
}

// ReverseLeftWordsSimple 时间复杂度O(N),空间复杂度O(1)
func ReverseLeftWordsSimple(s string, n int) string {
	ss := []byte(s)
	reverse(ss[:n])
	reverse(ss[n:])
	reverse(ss)
	return string(ss)
}

/*
leetcode 28. 实现 strStr()
1.19 给你两个字符串haystack和needle ，请你在haystack字符串中找出needle字符串出现的第一个位置（下标从0开始）。
如果不存在，则返回 -1 。

示例1：
输入：haystack = "hello", needle = "ll"
输出：2

示例2：
输入：haystack = "aaaaa", needle = "bba"
输出：-1

示例3：
输入：haystack = "", needle = ""
输出：0

提示：

0 <= haystack.length, needle.length <= 5 * 104
haystack和needle仅由小写英文字符组成
*/

// strStr 时间复杂度O(N*M),空间复杂度O(1)
func strStr(haystack string, needle string) int {
	n, m := len(haystack), len(needle)
	if m == 0 {
		return 0
	}
	for i := 0; i+m <= n; i++ {
		if needle == haystack[i:i+m] {
			return i
		}
	}
	return -1
}

/*
KMP算法的核心为前缀函数，记作π(i)，其定义如下：

对于长度为m的字符串s，其前缀函数 π(i)(0≤i<m)表示s的子串s[0:i]的最长的相等的真前缀与真后缀的长度。特别地，
如果不存在符合条件的前后缀，那么π(i)=0。其中真前缀与真后缀的定义为不等于自身的的前缀与后缀。

字符串的前缀是指不包含最后一个字符的所有以第一个字符开头的连续子串；后缀是指不包含第一个字符的所有以
最后一个字符结尾的连续子串。

我们举个例子说明：字符串aabaaab的前缀函数值依次为 0,1,0,1,2,2,3

π(0)=0，因为a没有真前缀和真后缀，根据规定为0（可以发现对于任意字符串π(0)=0必定成立）；

π(1)=1，因为aa最长的一对相等的真前后缀为a，长度为1；

π(2)=0，因为aab没有对应的真前缀和真后缀，根据规定为0；

π(3)=1，因为aaba最长的一对相等的真前后缀为a，长度为1；

π(4)=2，因为aabaa最长的一对相等的真前后缀为aa，长度为2；

π(5)=2，因为aabaaa最长的一对相等的真前后缀为aa，长度为2；

π(6)=3，因为aabaaab 最长的一对相等的真前后缀为aab，长度为3。

而有了前缀函数，我们就可以快速地计算出模式串在主串中的每一次出现。

如何求解前缀函数

长度为m的字符串s的所有前缀函数的求解算法的总时间复杂度是严格O(m)的，且该求解算法是增量算法，即我们可以一边
读入字符串，一边求解当前读入位的前缀函数。

为了叙述方便，我们接下来将说明几个前缀函数的性质：

π(i)≤π(i−1)+1
依据 π(i)定义得：s[0:π(i)−1]=s[i−π(i)+1:i]
将两区间的右端点同时左移，可得：s[0:π(i)−2]=s[i−π(i)+1:i−1]
依据 π(i−1)定义得：π(i−1)≥π(i)−1 即 π(i)≤π(i−1)+1

如果 s[i]=s[π(i−1)]，那么 π(i)=π(i−1)+1
依据 π(i−1)定义得：s[0:π(i−1)−1]=s[i−π(i−1):i−1]
因为 s[π(i−1)]=s[i]，可得 s[0:π(i−1)]=s[i−π(i−1):i]
依据 π(i) 定义得：π(i)≥π(i−1)+1，结合第一个性质可得 π(i)=π(i−1)+1。
这样我们可以依据这两个性质提出求解 π(i)的方案：找到最大的j，满足 s[0:j−1]=s[i−j:i−1]，且 s[i]=s[j]
（这样就有 s[0:j]=s[i−j:i]，即 π(i)=j+1）。

注意这里提出了两个要求：

j要求尽可能大，且满足 s[0:j−1]=s[i−j:i−1]
j要求满足 s[i]=s[j]
由 π(i−1)定义可知：

(1) s[0:π(i−1)−1]=s[i−π(i−1):i−1]
那么 j=π(i−1)符合第一个要求。如果 s[i]=s[π(i−1)]，我们就可以确定 π(i)。

否则如果 s[i]≠s[π(i−1)]，那么 π(i)≤π(i−1)，因为j=π(i)−1，所以j<π(i−1)，于是可以取(1)式两子串的长度
为j的后缀，它们依然是相等的：s[π(i−1)−j:π(i−1)−1]=s[i−j:i−1]。

当s[i]≠s[π(i−1)]时，我们可以修改我们的方案为：找到最大的j，满足 s[0:j−1]=s[π(i−1)−j:π(i−1)−1]，
且 s[i]=s[π(i−1)]（这样就有s[0:j]=s[π(i−1)−j:π(i−1)]，即 π(i)=π(i−1)+1。

注意这里提出了两个要求：

j要求尽可能大，且满足 s[0:j−1]=s[π(i−1)−j:π(i−1)−1];
j要求满足 s[i]=s[j]。
由 π(π(i−1)−1)定义可知 j=π(π(i−1)−1)符合第一个要求。如果 s[i]=s[π(π(i−1)−1)]，我们就可以确定π(i)。

此时，我们可以发现j的取值总是被描述为 π(π(π(…)−1)−1)的结构（初始为 π(i−1)）。于是我们可以描述我们的算法：
设定 π(i)=j+1，j的初始值为 π(i−1)。我们只需要不断迭代j（令j 变为 π(j−1)）直到 s[i]=s[j] 或 j=0 即可，如果
最终匹配成功（找到了j使得 s[i]=s[j]，那么 π(i)=j+1，否则 π(i)=0。
*/

// strStrSimple KMP算法 时间复杂度O(N+M),空间复杂度O(M)
func strStrSimple(haystack string, needle string) int {
	n, m := len(haystack), len(needle)
	if m == 0 || m > n {
		return -1
	}
	next := make([]int, m)
	GetNext(next, needle)
	// 因为next数组里记录的起始位置为0
	j := 0
	// i从0开始匹配
	for i := 0; i < n; i++ {
		// 如果不匹配，就寻找之前匹配的位置
		for j > 0 && haystack[i] != needle[j] {
			j = next[j-1]
		}
		// 如果匹配，i和j同时向后移动
		if haystack[i] == needle[j] {
			j++
		}
		// 如果j从0移动到m的位置，意味着模式串needle与文本串haystack匹配成功
		if j == m {
			return i - m + 1
		}
	}
	return -1
}

/*
为什么使用前缀表
因为找到了最长相等的前缀和后缀，匹配失败的位置是后缀子串的后面，那么我们找到与其相同的前缀的后面重新
匹配就可以了。
字符串的前缀是指不包含最后一个字符的所有以第一个字符开头的连续子串；后缀是指不包含第一个字符的所有以
最后一个字符结尾的连续子串。
为什么要找不匹配字符的前一个字符的前缀表数值？
找到不匹配的位置，此时我们要看它的前一个字符的前缀表的数值是多少。
为什么要前一个字符的前缀表的数值呢，因为要找前面字符串的最长相同的前缀和后缀。
所以要看前一位的前缀表的数值。前一个字符的前缀表的数值是x，所有把下标移动到下标x的位置继续匹配。
*/

func GetNext(next []int, s string) {
	// next[j]就是记录着j（包括j）之前的子串的相同前后缀的长度。
	j := 0
	next[0] = 0
	// j指向前缀起始位置，i指向后缀起始位置
	for i := 1; i < len(s); i++ {
		// 如果前后缀不相同，那么j就要向前回退
		for j > 0 && s[i] != s[j] {
			j = next[j-1]
		}
		// 说明找到了相同的前后缀, j++，同时记录next[i]
		if s[i] == s[j] {
			j++
		}
		next[i] = j
	}
}

/*
leetcode 459. 重复的子字符串
1.20 重复的子字符串
给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。给定的字符串只含有小写英文字母，并且长度
不超过10000。

示例1:
输入: "abab"
输出: True
解释: 可由子字符串"ab"重复两次构成。

示例2:
输入: "aba"
输出: False

示例3:
输入: "abcabcabcabc"
输出: True
解释: 可由子字符串 "abc" 重复四次构成。 (或者子字符串 "abcabc" 重复两次构成。)
*/

func RepeatedSubstringPattern(s string) bool {
	length := len(s)
	if length == 0 {
		return false
	}
	next := make([]int, length)
	GetNext(next, s)
	if next[length-1] != 0 && length%(length-next[length-1]) == 0 {
		return true
	}
	return false
}

/*
为了判断一个字符串是否可以通过其某个子串重复多次构成，可以使用以下方法：

拼接字符串法：
将字符串 s 拼接成 s+s。
从拼接后的字符串中移除头尾字符，即截取 s[1:-1]。
检查原字符串 s 是否在截取后的字符串中存在：
如果存在，则说明 s 是由其某个子串重复构成的。
如果不存在，则不是。
原理：
如果 s 是由某个子串重复构成的，那么拼接后的字符串 s+s 中，去掉头尾后一定会包含原字符串 s。
*/

func RepeatedSubstringPatternSimple(s string) bool {
	// 将两个原字符串拼接起来
	doubled := s + s
	// 移除首尾字符
	left := doubled[1 : len(doubled)-1]
	// 检查原字符串是否存在于截取后的字符串中
	return strings.Contains(left, s)
}

/*
剑指Offer 21. 调整数组顺序使奇数位于偶数前面
1.21 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数在数组的前半部分，所有偶数在数组的后半部分。

示例：
输入：nums = [1,2,3,4]
输出：[1,3,2,4]
注：[3,1,2,4] 也是正确的答案之一。
*/

func Exchange(nums []int) []int {
	var odd, even []int
	for _, v := range nums {
		if v%2 == 1 {
			odd = append(odd, v)
		} else {
			even = append(even, v)
		}
	}
	odd = append(odd, even...)
	return odd
}

func ExchangeSimple(nums []int) []int {
	l, r := 0, len(nums)-1
	for l < r {
		for l < r && nums[l]%2 == 1 {
			l++
		}
		for l < r && nums[r]%2 == 0 {
			r--
		}
		nums[l], nums[r] = nums[r], nums[l]
	}
	return nums
}

/*
leetcode 263. 丑数I
1.22 给你一个整数n ，请你判断n是否为丑数 。如果是，返回true ；否则，返回false 。
丑数就是只包含质因数2、3 或5的正整数。

示例1：
输入：n = 6
输出：true
解释：6 = 2 × 3

示例2：
输入：n = 8
输出：true
解释：8 = 2 × 2 × 2

示例3：
输入：n = 14
输出：false
解释：14 不是丑数，因为它包含了另外一个质因数7 。

示例4：
输入：n = 1
输出：true
解释：1 通常被视为丑数。

提示：
-2^31 <= n <= 2^31 - 1
*/

/*
根据丑数的定义，0和负整数一定不是丑数。
当n>0 时，若n是丑数，则n可以写成 n = 2^a * 3^b * 5^c的形式，其中 a,b,c都是非负整数。特别地，当a,b,c
都是0时，n=1。
为判断n是否满足上述形式，可以对n反复除以2,3,5直到n不再包含质因数2,3,5。若剩下的数等于1，则说明n不包含其
他质因数，是丑数；否则，说明n包含其他质因数，不是丑数。
*/

func IsUgly(n int) bool {
	if n <= 0 {
		return false
	}
	factors := []int{2, 3, 5}
	for _, f := range factors {
		for n%f == 0 {
			n /= f
		}
	}
	return n == 1
}

/*
leetcode 1201. 丑数III
1.23 给你四个整数：n 、a 、b 、c ，请你设计一个算法来找出第n个丑数。
丑数是可以被a或b或c整除的正整数 。
*/

// NthUglyNumber 二分法+容斥原理
func NthUglyNumber(n int, a int, b int, c int) int {
	ab, ac, bc := lcm(a, b), lcm(a, c), lcm(b, c)
	abc := lcm(ab, c)
	left, right := 0, n*utils.Min(a, utils.Min(b, c))
	for left <= right {
		mid := (left + right) / 2
		count := mid/a + mid/b + mid/c - mid/ab - mid/ac - mid/bc + mid/abc
		if count == n {
			if mid%a == 0 && mid%b == 0 && mid%c == 0 {
				return mid
			} else {
				right = mid - 1
			}
		} else if count < n {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return left
}

// lcm 求两个正整数的最小公倍数
func lcm(x, y int) int {
	return x * y / gcd(x, y)
}

// gcd 求两个正整数的最大公约数
func gcd(x, y int) int {
	temp := x % y
	if temp > 0 {
		return gcd(y, temp)
	}
	return y
}

/*
leetcode 878. 第N个神奇数字
1.24 如果正整数可以被A或B整除，那么它是神奇的。
返回第N个神奇数字。由于答案可能非常大，返回它模 10^9 + 7 的结果。
*/

func NthMagicalNumber(n int, a int, b int) int {
	mod := int(1e9) + 7
	ab := lcm(a, b)
	left, right := 0, n*utils.Min(a, b)
	for left <= right {
		mid := (left + right) / 2
		count := mid/a + mid/b - mid/ab
		if count == n {
			if mid%a == 0 && mid%b == 0 {
				return mid % mod
			} else {
				right = mid - 1
			}
		} else if count > n {
			left = mid - 1
		} else {
			right = mid + 1
		}
	}
	return left % mod
}

/*
leetcode 54. 螺旋矩阵
1.25 给你一个m行n列的矩阵matrix ，请按照顺时针螺旋顺序，返回矩阵中的所有元素。
示例1:
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]

示例2:
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]

提示：
m == matrix.length
n == matrix[i].length
1 <= m, n <= 10
-100 <= matrix[i][j] <= 100
*/

/*
思路：按层模拟
可以将矩阵看成若干层，首先输出最外层的元素，其次输出次外层的元素，直到输出最内层的元素。
定义矩阵的第k层是到最近边界距离为k的所有顶点。例如，下图矩阵最外层元素都是第1层，次外层元素都是第2层，剩下
的元素都是第3层。

[[1, 1, 1, 1, 1, 1, 1],
 [1, 2, 2, 2, 2, 2, 1],
 [1, 2, 3, 3, 3, 2, 1],
 [1, 2, 2, 2, 2, 2, 1],
 [1, 1, 1, 1, 1, 1, 1]]

对于每层，从左上方开始以顺时针的顺序遍历所有元素。假设当前层的左上角位于(top,left)，右下角位于(bottom,right)，
按照如下顺序遍历当前层的元素。
从左到右遍历上侧元素，依次为(top,left) 到 (top,right)。
从上到下遍历右侧元素，依次为(top+1,right) 到(bottom,right)。
如果left<right 且 top<bottom，则从右到左遍历下侧元素，依次为(bottom,right−1) 到 (bottom,left+1)，
以及从下到上遍历左侧元素，依次为(bottom,left) 到 (top+1,left)。
遍历完当前层的元素之后，将left和top分别增加1，将right和bottom分别减少1，进入下一层继续遍历，直到遍历完
所有元素为止。
*/

// spiralOrder 时间复杂度O(MN), 空间复杂度O(1)
func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return []int{}
	}
	rows, columns := len(matrix), len(matrix[0])
	order := make([]int, rows*columns)
	index := 0
	top, bottom, left, right := 0, rows-1, 0, columns-1
	for top <= bottom && left <= right {
		for column := left; column <= right; column++ {
			order[index] = matrix[top][column]
			index++
		}
		for row := top + 1; row <= bottom; row++ {
			order[index] = matrix[row][right]
			index++
		}
		if top < bottom && left < right {
			for column := right - 1; column > left; column-- {
				order[index] = matrix[bottom][column]
				index++
			}
			for row := bottom; row > top; row-- {
				order[index] = matrix[row][left]
				index++
			}
		}
		top++
		bottom--
		left++
		right--
	}
	return order
}

/*
变形题:逆时针螺旋矩阵
给你一个m行n列的矩阵matrix，请按照逆时针螺旋顺序，返回矩阵中的所有元素。
示例1:
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1, 4, 7, 8, 9, 6, 3, 2, 5]
示例2:
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1, 5, 9, 10, 11, 12, 8, 4, 3, 2, 6, 7]
*/

func antiSpiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return []int{}
	}
	rows, columns := len(matrix), len(matrix[0])
	top, bottom, left, right := 0, rows-1, 0, columns-1
	order := make([]int, rows*columns)
	index := 0
	for top <= bottom && left <= right {
		for row := top; row <= bottom; row++ {
			order[index] = matrix[row][left]
			index++
		}
		for column := left + 1; column <= right; column++ {
			order[index] = matrix[bottom][column]
			index++
		}
		if top < bottom && left < right {
			for row := bottom - 1; row > top; row-- {
				order[index] = matrix[row][right]
				index++
			}
			for column := right; column > left; column-- {
				order[index] = matrix[top][column]
				index++
			}
		}
		top++
		bottom--
		left++
		right--
	}
	return order
}

/*
leetcode 59 螺旋矩阵II
给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。

示例1:
输入：n = 3
输出：[[1,2,3],[8,9,4],[7,6,5]]

提示：
1 <= n <= 20
*/

func generateMatrix(n int) [][]int {
	matrix := make([][]int, n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]int, n)
	}
	order := 1
	top, bottom, left, right := 0, n-1, 0, n-1
	for top <= bottom && left <= right {
		for col := left; col <= right; col++ {
			matrix[top][col] = order
			order++
		}
		for row := top + 1; row <= bottom; row++ {
			matrix[row][right] = order
			order++
		}
		if top < bottom && left < right {
			for col := right - 1; col >= left; col-- {
				matrix[bottom][col] = order
				order++
			}
			for row := bottom - 1; row > top; row-- {
				matrix[row][left] = order
				order++
			}
		}
		top++
		bottom--
		left++
		right--
	}
	return matrix
}

/*
1.26 随机组队
一群朋友组队玩游戏，至少有5组人，一组至少2人
要求:
1 每2个人组一队或者3人组一队，每个人只能加到一个队伍里，不能落单
2 2人队和3人队各自的队伍数均不得少于1，队伍中的人不能来自相同组
3 随机组队，重复执行程序得到的结果不一样，总队伍数也不能一样
4 必须有注释
注:要同时满足条件1-4
*/

var GroupList = [][]string{
	{"少华", "少平", "少军", "少安", "少康"},
	{"福军", "福堂", "福民", "福平", "福心"},
	{"小明", "小红", "小花", "小丽", "小强"},
	{"大壮", "大力", "大1", "大2", "大3"},
	{"阿花", "阿朵", "阿蓝", "阿紫", "阿红"},
	{"A", "B", "C", "D", "E"},
	{"一", "二", "三", "四", "五"},
}

// Combination 结构体标记2人队和3人队队伍数的组合
type Combination struct {
	Two   int
	Three int
}

var r = rand.New(rand.NewSource(time.Now().UnixNano()))

// 计算出两人队和三人队的所有可能解，a*2+b*3=sum，即a和b的组合
func calCombination(sum int) Combination {
	result := make([]Combination, 0)
	// 2人一队或3人一队，不能落单
	// 2人队、3人队各自的队伍数均不得少于1
	// tow标记2人队队伍数量
	for two := 1; two*2 <= sum-3*1; two++ {
		num := sum - 2*two
		// 减去两人队剩余人数后，如果剩下的人还能组成至少一组3人队，则我们找到一个组队方案
		if num%3 == 0 && num/3 >= 1 {
			result = append(result, Combination{two, num / 3})
		}
	}
	// 没有符合题意的组队方式
	if len(result) == 0 {
		return Combination{}
	}
	n := len(result)
	// 随机组队，重复执行程序得到的结果不一样，总队伍数也不能一样
	return result[r.Intn(n)%n]
}

func constructGroup() [][]string {
	// sum为GroupList总人数
	sum := len(GroupList) * len(GroupList[0])
	// 根据总人数，得到一个随机组队方案cr
	cr := calCombination(sum)
	fmt.Println(cr)
	// 2人队队伍数不能少于1
	if cr.Two == 0 {
		fmt.Println("没有满足条件的组合")
		return [][]string{}
	}
	// 将GroupList每一组成员随机打乱，避免后续从二维数组左上角开始，从上到下，从左到右固定取数据的影响
	for _, group := range GroupList {
		r.Shuffle(len(group), func(i, j int) {
			group[i], group[j] = group[j], group[i]
		})
	}
	//  res为最后的组队结果集，其长度为随机组队方案cr的2人队和3人队队伍数之和
	res := make([][]string, cr.Two+cr.Three)
	// index为组队结果集res的下标索引，初始值为0
	index := 0
	groups := len(GroupList)
	for i := 0; i < sum; i++ {
		// 要求队伍中的人不能来自相同组，所以每次填充队伍都选不同的组
		res[index] = append(res[index], GroupList[i%groups][i/groups])
		// 先尝试排2人队队伍，然后再尝试排3人队队伍
		if i < cr.Two*2 && len(res[index]) == 2 {
			index++
		} else if i >= cr.Two*2 && len(res[index]) == 3 {
			index++
		}
	}
	return res
}

/*
剑指 Offer 57. 和为s的两个数字
1.27 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，
则输出任意一对即可。

示例 1：
输入：nums = [2,7,11,15], target = 9
输出：[2,7] 或者 [7,2]

示例 2：
输入：nums = [10,26,30,31,47,60], target = 40
输出：[10,30] 或者 [30,10]
*/

// twoSum 因为是升序数组，采用双指针的思路时间复杂度O(N)，空间复杂度O(1)
func twoSum(nums []int, target int) []int {
	i, j := 0, len(nums)-1
	for i < j {
		sum := nums[i] + nums[j]
		if sum > target {
			j--
		} else if sum < target {
			i++
		} else {
			return []int{nums[i], nums[j]}
		}
	}
	return []int{}
}

/*
leetcode 498 对角线遍历
1.29 给你一个大小为 m x n 的矩阵 mat ，请以对角线遍历的顺序，用一个数组返回这个矩阵中的所有元素。

示例1:
输入：mat = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,4,7,5,3,6,8,9]

示例2:
输入：mat = [[1,2],[3,4]]
输出：[1,2,3,4]

示：

m == mat.length
n == mat[i].length
1 <= m, n <= 104
1 <= m * n <= 104
-105 <= mat[i][j] <= 105
*/

/*
方法一：直接模拟
思路与算法

根据题目要求，矩阵按照对角线进行遍历。设矩阵的行数为m, 矩阵的列数为n, 我们仔细观察对角线遍历的规律可以得到如下信息:
一共有 m+n−1条对角线，相邻的对角线的遍历方向不同，当前遍历方向为从左下到右上，则紧挨着的下一条对角线遍历方向为从右上
到左下；

对角线上的行列索引i+j=k(k指对角线编号)

设对角线从上到下的编号为 i∈[0,m+n−2]：
当i为偶数时，则第i条对角线的走向是从下往上遍历；
当i为奇数时，则第i条对角线的走向是从上往下遍历；
当第i条对角线从下往上遍历时，每次行索引减1，列索引加1，直到矩阵的边缘为止：
当 i<m 时，则此时对角线遍历的起点位置为 (i,0)；
当 i≥m 时，则此时对角线遍历的起点位置为 (m−1,i−m+1)；

当第i条对角线从上往下遍历时，每次行索引加1，列索引减1，直到矩阵的边缘为止：

当 i<n 时，则此时对角线遍历的起点位置为 (0,i)；
当 i≥n 时，则此时对角线遍历的起点位置为 (i−n+1,n−1)；
根据以上观察得出的结论，我们直接模拟遍历所有的对角线即可。
*/

func findDiagonalOrder(mat [][]int) []int {
	m, n := len(mat), len(mat[0])
	res := make([]int, 0, m*n)
	for i := 0; i < m+n-1; i++ {
		if i%2 == 0 {
			// 偶数对角线从下到上遍历
			// 确定对角线的起始位置
			row, col := 0, 0
			if i < m {
				row, col = i, 0
			} else {
				row, col = m-1, i-m+1
			}
			for row >= 0 && col < n {
				res = append(res, mat[row][col])
				row--
				col++
			}
		} else {
			// 奇数对角线从上到下遍历
			// 确定对角线的起始位置
			row, col := 0, 0
			if i < n {
				row, col = 0, i
			} else {
				row, col = i-n+1, n-1
			}
			for row < m && col >= 0 {
				res = append(res, mat[row][col])
				row++
				col--
			}
		}
	}
	return res
}

/*
leetcode 41 缺失的第一个整数
给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。


示例 1：

输入：nums = [1,2,0]
输出：3
解释：范围 [1,2] 中的数字都在数组中。
示例 2：

输入：nums = [3,4,-1,1]
输出：2
解释：1 在数组中，但 2 没有。
示例 3：

输入：nums = [7,8,9,11,12]
输出：1
解释：最小的正数 1 没有出现。


提示：

1 <= nums.length <= 105
-231 <= nums[i] <= 231 - 1
*/

/*
实际上，对于一个长度为 N 的数组，其中没有出现的最小正整数只能在 [1,N+1] 中。这是因为如果 [1,N] 都出现了，那么答案是 N+1，
否则答案是 [1,N] 中没有出现的最小正整数。这样一来，我们将所有在 [1,N] 范围内的数放入哈希表，也可以得到最终的答案。而给定的
数组恰好长度为 N，这让我们有了一种将数组设计成哈希表的思路.
要解决这个问题并满足 时间复杂度为𝑂(n)和 常数级别额外空间 的约束，我们可以使用“原地哈希”的方法。基本思路是将数字尽量
放到对应的索引位置上，使数组元素形成一种索引-值的映射关系。然后通过一次遍历找到第一个不满足这种关系的数字。

算法步骤
原地哈希调整：遍历数组，将每个数字 nums[i] 放到它的正确位置（即 nums[i] - 1），直到数组中的数字不符合条件或已经
在正确位置为止。
条件为 nums[i] > 0 && nums[i] <= n && nums[nums[i]-1] != nums[i]。
交换 nums[i] 和 nums[nums[i]-1]。
查找缺失的正整数： 再次遍历数组，找到第一个不满足 nums[i] == i + 1 的位置，返回 i + 1。
如果数组中所有位置都满足： 返回 n + 1，其中 n 是数组长度。
*/

func firstMissingPositive(nums []int) int {
	n := len(nums)
	for i := 0; i < n; i++ {
		x := nums[i]
		// 对于一个长度为 N 的数组，其中没有出现的最小正整数只能在 [1,N+1] 中，所以数组中的负数和0对结果没影响
		if x > 0 && x <= n && nums[x-1] != x {
			// 将数组中的元素x(nums[i])放到其对应的位置上(nums[i]-1,即x-1位置)
			x, nums[x-1] = nums[x-1], x
		}
	}
	// 经过第一轮for循环原地哈希后，符合条件的数组中的元素x都被放到了正确的位置x-1上，即x=x-1+1
	for i := 0; i < n; i++ {
		// 不满足x=x-1+1的即为缺失的第一个正整数
		if nums[i] != i+1 {
			return i + 1
		}
	}
	// 如果满足x=x-1+1，说明1-n的正整数都在数组nums中出现了，返回n+1
	return n + 1
}

/*
旋转图像专题
leetcode 48 旋转图像
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

参见images目录下的 matrix_one.jpg
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]

参见images目录下的 matrix_two.jpg
输入：matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
输出：[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

提示：
n == matrix.length == matrix[i].length
1 <= n <= 20
-1000 <= matrix[i][j] <= 1000
*/

func rotate(matrix [][]int) {
	n := len(matrix)
	// 先水平翻转
	for i := 0; i < n/2; i++ {
		matrix[i], matrix[n-1-i] = matrix[n-1-i], matrix[i]
	}
	// 再做主对角线翻转
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
}

// 顺时针旋转135度
func rotate135(matrix [][]int) {
	n := len(matrix)
	// 上下翻转
	for i := 0; i < n/2; i++ {
		matrix[i], matrix[n-1-i] = matrix[n-1-i], matrix[i]
	}
	// 反对角线翻转
	for i := 0; i < n; i++ {
		for j := 0; j < n-i-1; j++ {
			matrix[i][j], matrix[n-1-j][n-1-i] = matrix[n-1-j][n-1-i], matrix[i][j]
		}
	}
}

// 顺(逆)时针旋转180度
func rotate180(matrix [][]int) {
	n := len(matrix)
	// 上下翻转
	for i := 0; i < n/2; i++ {
		matrix[i], matrix[n-1-i] = matrix[n-1-i], matrix[i]
	}
	// 左右翻转
	for i := 0; i < n; i++ {
		for j := 0; j < n/2; j++ {
			matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]
		}
	}
}

// 顺时针旋转270度
func rotate270(matrix [][]int) {
	n := len(matrix)
	// 左右翻转
	for i := 0; i < n; i++ {
		for j := 0; j < n/2; j++ {
			matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]
		}
	}
	// 主对角线翻转
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
}

// 逆时针旋转90度
func rotateCounter90(matrix [][]int) {
	n := len(matrix)
	// 左右翻转
	for i := 0; i < n; i++ {
		for j := 0; j < n/2; j++ {
			matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]
		}
	}
	// 反对角线翻转
	for i := 0; i < n; i++ {
		for j := 0; j < n-i-1; j++ {
			matrix[i][j], matrix[n-1-j][n-1-i] = matrix[n-1-j][n-1-i], matrix[i][j]
		}
	}
}

// 逆时针旋转135度
func rotateCounter135(matrix [][]int) {
	n := len(matrix)
	// 左右翻转
	for i := 0; i < n; i++ {
		for j := 0; j < n/2; j++ {
			matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]
		}
	}
	// 主对角线翻转
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
}

/*
逆时针旋转 180°
等价于顺时针旋转 180°，直接复用 rotate180(matrix)。
逆时针旋转 270°
等价于顺时针旋转 90°，直接复用 rotate(matrix)。
*/

/*
leetcode 470 用Rand7()实现Rand10()
给定方法 rand7 可生成 [1,7] 范围内的均匀随机整数，试写一个方法 rand10 生成 [1,10] 范围内的均匀随机整数。

你只能调用 rand7() 且不能调用其他方法。请不要使用系统的 Math.random() 方法。

每个测试用例将有一个内部参数 n，即你实现的函数 rand10() 在测试时将被调用的次数。请注意，这不是传递给 rand10() 的参数。


示例 1:
输入: 1
输出: [2]

示例 2:
输入: 2
输出: [2,8]

示例 3:
输入: 3
输出: [3,8,10]


提示:
1 <= n <= 105

进阶:
rand7()调用次数的 期望值 是多少 ?
你能否尽量少调用 rand7() ?
*/

func rand7() int {
	return rand.Intn(7) + 1
}

func rand10() int {
	for {
		/*
			这里 rand7() 生成一个均匀分布的随机整数 [1, 7]。
			我们分别生成两个随机数 row 和 col，可以看成一个二维网格的行和列编号，构成一个 7x7 的矩阵。
		*/
		row := rand7()
		col := rand7()
		/*
			这个公式将二维坐标 (row, col) 转换成一个一维编号 idx，范围是 [1, 49]，表示 7x7 矩阵中每个位置的唯一索引。
			row*col 的结果不是均匀分布。例如，rand7() 的结果为 [1, 7]，乘积可能会集中在较小的数字范围（如 1、4），而较大的数字（如 49）更难出现，导致随机性不均匀。
		*/
		idx := (row-1)*7 + col
		if idx <= 40 {
			/*
					[1, 40] 是可以被均匀分成 10 个区间的范围（即每个区间大小是 4），满足我们需要的目标范围 [1, 10]。
					超出 40 的值（41-49）会被舍弃，重新尝试生成一个随机值。这种拒绝采样 确保了结果的均匀性。
				    41-49 的数字分布无法均匀分成 [1, 10] 的范围，因此被舍弃。
			*/
			return 1 + (idx-1)%10
		}
		/*
			减少 rand7() 调用次数
			在最坏情况下，上述算法会多次生成超出范围的随机数（41-49），导致效率下降。我们可以尝试优化拒绝采样的范围。
			例如：
			进一步利用舍弃的区间重新生成所需数字。
			将超出范围的部分映射回需要的范围。
		*/
		// 映射剩余的 41-49 到新的随机范围
		idx = (idx-41)*7 + rand7()
		if idx <= 60 {
			return 1 + (idx-1)%10
		}
		// 最后再处理 61-63
		idx = (idx-61)*7 + rand7()
		if idx <= 20 {
			return 1 + (idx-1)%10
		}
	}
}

/*
leetcode 162 寻找峰值
峰值元素是指其值严格大于左右相邻值的元素。

给你一个整数数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。

你可以假设 nums[-1] = nums[n] = -∞ 。

你必须实现时间复杂度为 O(log n) 的算法来解决此问题。



示例 1：
输入：nums = [1,2,3,1]
输出：2
解释：3 是峰值元素，你的函数应该返回其索引 2。

示例 2：
输入：nums = [1,2,1,3,5,6,4]
输出：1 或 5
解释：你的函数可以返回索引 1，其峰值元素为 2；
     或者返回索引 5， 其峰值元素为 6。


提示：
1 <= nums.length <= 1000
-231 <= nums[i] <= 231 - 1
对于所有有效的 i 都有 nums[i] != nums[i + 1]
*/

func findPeakElement(nums []int) int {
	n := len(nums)
	var get func(int) int
	get = func(i int) int {
		if i == -1 || i == n {
			return math.MinInt64
		}
		return nums[i]
	}
	l, r := 0, n-1
	for {
		mid := (l + r) / 2
		if get(mid-1) < get(mid) && get(mid) > get(mid+1) {
			return mid
		}
		if get(mid) < get(mid+1) {
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
}

/*
leetcode 11 盛最多水的容器
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
返回容器可以储存的最大水量。
说明：你不能倾斜容器。

输入：[1,8,6,2,5,4,8,3,7]
输出：49

提示：
n == height.length
2 <= n <= 105
0 <= height[i] <= 104
*/

/*
容器的水量计算方法：
容器的水量由两条线的宽度和高度共同决定。
宽度由两条线的横坐标差决定。
高度由这两条线中的较短的那条决定。
因此，容器的容量可以表示为：
容量 = min(height[left], height[right]) * (right - left)

其中：
left 和 right 是容器的左右边界的索引。
height[left] 和 height[right] 是这两条垂直线的高度。

暴力解法：
遍历所有可能的线对 (i, j)，计算它们构成的容器的水量。时间复杂度为 O(n^2)，对于较大的 n，效率较低，不适合题目中的规模。
双指针法（优化方案）：
我们可以利用双指针的方法来优化解决方案。
初始化两个指针，一个指向数组的开始位置 (l) 和一个指向数组的结束位置 (r)。
计算这两个指针之间的容器的水量，并记录最大值。
然后，移动较短的那条线的指针，目的是寻找可能更高的线，以增加容器的高度。
如果 height[l] 小于 height[r]，则将 l 指针向右移动，否则将 r 指针向左移动。
这样做的理由是，移动较短的线有可能找到一个更高的线，从而增加容器的容量。
这个过程持续直到两个指针相遇，时间复杂度为 O(n)，空间复杂度为 O(1)。
*/

func maxArea(height []int) int {
	maxWater := 0
	l, r := 0, len(height)-1
	for l < r {
		width := r - l
		minHeight := utils.Min(height[l], height[r])
		maxWater = utils.Max(maxWater, width*minHeight)
		if height[l] < height[r] {
			l++
		} else {
			r--
		}
	}
	return maxWater
}

func reversePairs(arr []int) int {
	n := len(arr)
	if n <= 1 {
		return 0
	}
	temp := make([]int, n)
	return mergeAndCount(0, n-1, arr, temp)
}

func mergeAndCount(l, r int, arr, temp []int) int {
	if l >= r {
		return 0
	}
	mid := (l + r) / 2
	leftCount := mergeAndCount(l, mid, arr, temp)
	rightCount := mergeAndCount(mid+1, r, arr, temp)
	count := mergeComplex(l, r, mid, arr, temp)
	return leftCount + rightCount + count
}

func mergeComplex(l, r, mid int, arr, temp []int) int {
	i, j, k, count := l, mid+1, 0, 0
	for i <= mid && j <= r {
		if arr[i] <= arr[j] {
			temp[k] = arr[i]
			i++
		} else {
			temp[k] = arr[j]
			count += mid - i + 1
			j++
		}
		k++
	}
	for ; i <= mid; i++ {
		temp[k] = arr[i]
		k++
	}
	for ; j <= r; j++ {
		temp[k] = arr[j]
		k++
	}
	for m := l; m <= r; m++ {
		arr[m] = temp[m]
	}
	return count
}

/*
leetcode 75 颜色分类
给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地 对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色
顺序排列。

我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
必须在不使用库内置的 sort 函数的情况下解决这个问题。

示例 1：
输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]

示例 2：
输入：nums = [2,0,1]
输出：[0,1,2]

提示：
n == nums.length
1 <= n <= 300
nums[i] 为 0、1 或 2

进阶：
你能想出一个仅使用常数空间的一趟扫描算法吗？
*/

/*
遍历一遍数组，统计红白蓝三种颜色元素出现的次数
然后按照它们的出现次数，从下标0开始赋值即可。
*/

func sortColors(nums []int) {
	red, white, blue := 0, 0, 0
	for i, n := 0, len(nums); i < n; i++ {
		if nums[i] == 0 {
			red++
		} else if nums[i] == 1 {
			white++
		} else {
			blue++
		}
	}
	index := 0
	for i := 0; i < red; i++ {
		nums[index] = 0
		index++
	}
	for i := 0; i < white; i++ {
		nums[index] = 1
		index++
	}
	for i := 0; i < blue; i++ {
		nums[index] = 2
		index++
	}
}

/*
剑指offer 03 数组中重复的数字
在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，
也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。要求时间复杂度为O(N), 空间复杂度为O(1).
*/

/*
题目要求在一个长度为 n 的数组 nums 中找出任意一个重复的数字，所有数字都在 0 到 n-1 范围内，且要求时间复杂度为 O(n)，
空间复杂度为 O(1)。由于数组长度为 n，而数字范围是 0 到 n-1，根据鸽笼原理，必然存在至少一个重复的数字。
直接使用哈希表可以轻松解决，但空间复杂度会是 O(n)，不符合要求。因此，我们需要利用数组本身的特点。观察到数字范围是
0 到 n-1，正好与数组索引对应，可以采用原地交换的方法，将每个数字放到其对应的索引位置上。例如，数字 i 应该放在索引 i 处。
如果在交换过程中发现某个位置已有相同的数字，则该数字是重复的。
*/

func findRepeatNum(nums []int) int {
	n := len(nums)
	for i := 0; i < n; {
		// 当前数字
		current := nums[i]
		// 如果当前数字已在正确位置（即 nums[i] == i），继续下一个
		if current == i {
			i++
			continue
		}
		// 检查目标位置是否已有相同数字
		targetIdx := current
		if nums[targetIdx] == current {
			// 发现重复数字，直接返回
			return current
		}
		// 否则，将当前数字交换到其对应索引位置
		nums[i], nums[targetIdx] = nums[targetIdx], nums[i]
	}
	// 理论上不会到达这里，因为题目保证存在重复数字
	return -1
}

/*
leetcode 442 数组中重复的数据
给你一个长度为 n 的整数数组 nums ，其中 nums 的所有整数都在范围 [1, n] 内，且每个整数出现 最多两次。请你找出所有出现
两次的整数，并以数组形式返回。

你必须设计并实现一个时间复杂度为 O(n) 且仅使用常量额外空间（不包括存储输出所需的空间）的算法解决此问题。

示例 1：
输入：nums = [4,3,2,7,8,2,3,1]
输出：[2,3]

示例 2：
输入：nums = [1,1,2]
输出：[1]

示例 3：
输入：nums = [1]
输出：[]

提示：
n == nums.length
1 <= n <= 105
1 <= nums[i] <= n
nums 中的每个元素出现 一次 或 两次
*/

/*
我们可以给 nums[i] 加上「负号」表示数 i+1 已经出现过一次。具体地，我们首先对数组进行一次遍历。当遍历到位置 i 时，
我们考虑 nums[nums[i]−1] 的正负性：

如果 nums[nums[i]−1] 是正数，说明 nums[i] 还没有出现过，我们将 nums[nums[i]−1] 加上负号；
如果 nums[nums[i]−1] 是负数，说明 nums[i] 已经出现过一次，我们将 nums[i] 放入答案。

细节
由于 nums[i] 本身可能已经为负数，因此在将 nums[i] 作为下标或者放入答案时，需要取绝对值
*/

func findDuplicates(nums []int) []int {
	n := len(nums)
	res := []int{}
	for i := 0; i < n; i++ {
		idx := utils.Abs(nums[i]) - 1
		if nums[idx] < 0 {
			res = append(res, utils.Abs(nums[i]))
		} else {
			nums[idx] = -nums[idx]
		}
	}
	return res
}

/*
leetcode 384 打乱数组
给你一个整数数组 nums，设计算法来打乱一个没有重复元素的数组。打乱后，数组的所有排列应该是等可能的。

实现 Solution class:
Solution(int[] nums) 使用整数数组 nums 初始化对象
int[] reset() 重设数组到它的初始状态并返回
int[] shuffle() 返回数组随机打乱后的结果

示例 1：
输入
["Solution", "shuffle", "reset", "shuffle"]
[[[1, 2, 3]], [], [], []]
输出
[null, [3, 1, 2], [1, 2, 3], [1, 3, 2]]

解释
Solution solution = new Solution([1, 2, 3]);
solution.shuffle();    // 打乱数组 [1,2,3] 并返回结果。任何 [1,2,3]的排列返回的概率应该相同。例如，返回 [3, 1, 2]
solution.reset();      // 重设数组到它的初始状态 [1, 2, 3] 。返回 [1, 2, 3]
solution.shuffle();    // 随机返回数组 [1, 2, 3] 打乱后的结果。例如，返回 [1, 3, 2]

提示：
1 <= nums.length <= 50
-106 <= nums[i] <= 106
nums 中的所有元素都是 唯一的
最多可以调用 104 次 reset 和 shuffle
*/

/*
问题分析：
reset 方法需要返回数组的初始状态，直接返回存储的原始数组即可。
shuffle 方法需要打乱数组，且所有排列都应具有相同的概率。

随机打乱方法：
使用 Fisher-Yates 洗牌算法。该算法在O(n) 时间复杂度内完成洗牌，且确保所有排列具有相同概率：
从数组末尾开始，随机选择一个索引并交换当前元素和随机索引处的元素。
重复上述过程直到数组的起始位置。

数据结构设计：
nums：存储初始数组，供 reset 方法使用。
shuffled：在 shuffle 方法中操作的数组。
*/

type Solution struct {
	original []int
	shuffled []int
}

func Constructor(nums []int) Solution {
	original := make([]int, len(nums))
	copy(original, nums)
	return Solution{
		original: original,
		shuffled: nums,
	}
}

func (s *Solution) Reset() []int {
	s.shuffled = make([]int, len(s.original))
	copy(s.shuffled, s.original)
	return s.shuffled
}

func (s *Solution) Shuffle() []int {
	rand.Seed(time.Now().UnixNano())
	n := len(s.shuffled)
	for i := n - 1; i >= 0; i-- {
		j := rand.Intn(i + 1)
		s.shuffled[i], s.shuffled[j] = s.shuffled[j], s.shuffled[i]
	}
	return s.shuffled
}

/*
leetcode 287 寻找重复的数
给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。
假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。
你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。

示例 1：
输入：nums = [1,3,4,2,2]
输出：2

示例 2：
输入：nums = [3,1,3,4,2]
输出：3

示例 3 :
输入：nums = [3,3,3,3,3]
输出：3

提示：
1 <= n <= 105
nums.length == n + 1
1 <= nums[i] <= n
nums 中 只有一个整数 出现 两次或多次 ，其余整数均只出现 一次
*/

/*
基本思路
这道题的关键是把数组看作一个链表。

数组的值 nums[i] 可以看作链表节点的“指向下一个节点的索引”。例如，如果 nums[i] = 3，那么节点 i 就指向节点 3。
因为题目保证数组有 重复数字，所以在这个“链表”中会出现环。环的起点就是那个重复的数字。
为什么数组有重复数字，链表就一定有环？

假设数组是 nums = [1, 3, 4, 2, 2]：
从索引 0 出发，根据 nums[i] 的值找到下一个节点：
0 -> 1 -> 3 -> 2 -> 4 -> 2
注意，最后指向了 2，而 2 已经出现过。这就是环的存在。
这里的环结构是因为数组中某个数字重复，所以这个数字在链表中会被多次访问，从而形成环。

快慢指针法寻找重复数字
快慢指针算法（Floyd 判圈算法）主要解决两个问题：
是否有环？
用快慢指针，一个每次移动 1 步（慢指针 slow），一个每次移动 2 步（快指针 fast）。如果两者最终相遇，则说明有环。
环的入口在哪？
相遇后，将其中一个指针重置到链表头，两个指针每次移动 1 步。再次相遇时的位置就是环的入口，也就是重复数字。
*/

func findDuplicate(nums []int) int {
	fast, slow := nums[0], nums[0]
	for {
		slow = nums[slow]
		fast = nums[nums[fast]]
		if fast == slow {
			break
		}
	}
	fast = nums[0]
	for fast != slow {
		fast = nums[fast]
		slow = nums[slow]
	}
	return fast
}

/*
文物朝代判断
展览馆展出来自 13 个朝代的文物，每排展柜展出 5 个文物。某排文物的摆放情况记录于数组 places，其中 places[i]
表示处于第 i 位文物的所属朝代编号。其中，编号为 0 的朝代表示未知朝代。请判断并返回这排文物的所属朝代编号是否
能够视为连续的五个朝代（如遇未知朝代可算作连续情况）。

示例 1：
输入: places = [0, 6, 9, 0, 7]
输出: True

示例 2：
输入: places = [7, 8, 9, 10, 11]
输出: True

提示：
places.length = 5
0 <= places[i] <= 13
*/

func checkDynasty(places []int) bool {
	// 初始化最大值和最小值
	max, min := 0, 14
	// 记录非零朝代编号出现次数
	seen := make(map[int]int)
	for _, num := range places {
		// 遇到未知朝代直接跳过
		if num == 0 {
			continue
		}
		seen[num]++
		// 如果非零朝代编号重复，返回失败
		if seen[num] > 1 {
			return false
		}
		// 迭代数组中的最大值和最小值
		if max < num {
			max = num
		}
		if min > num {
			min = num
		}
	}
	// 检查最大值和最小值的差距是否可被零填充
	return max-min <= 4
}

/*
leetcode 191 位1的个数
给定一个正整数 n，编写一个函数，获取一个正整数的二进制形式并返回其二进制表达式中设置位的个数（也被称为汉明重量）。

示例 1：
输入：n = 11
输出：3
解释：输入的二进制串 1011 中，共有 3 个设置位。

示例 2：
输入：n = 128
输出：1
解释：输入的二进制串 10000000 中，共有 1 个设置位。

示例 3：
输入：n = 2147483645
输出：30
解释：输入的二进制串 1111111111111111111111111111101 中，共有 30 个设置位。

提示：
1 <= n <= 231 - 1

进阶：
如果多次调用这个函数，你将如何优化你的算法？
*/

/*
位操作法（高效）：
我们可以利用 n & (n - 1) 的技巧，通过每次消去二进制表示中的最低位 1 来高效计算设置位的数量。
位操作法的解释：
n & (n - 1) 会把 n 中最右边的 1 设置为 0。
例如，假设 n = 12，二进制为 1100，n - 1 = 1011，那么 n & (n - 1) = 1000，把最右边的 1 位清除掉。
这样我们每调用一次 n & (n - 1)，n 的二进制表示中就少一个 1，直到所有的 1 都被消除。

进阶：
如果多次调用这个函数，我们可以使用缓存或者预计算，例如，对于多个数字频繁查询，我们可以预先计算每个数字的汉明重量
并存储下来，直接返回结果。
但对于单个调用，使用 n & (n - 1) 这种方法已经是最优化的，不需要额外的空间，时间复杂度是 O(k)，其中 k 是二进制表示中
1 的个数，最坏情况下 k = O(log n)。
*/

func hammingWeight(n int) int {
	count := 0
	for n != 0 {
		// 每次清除最低位的 1
		n = n & (n - 1)
		count++
	}
	return count
}

/*
leetcode 485
给定一个二进制数组 nums， 计算其中最大连续 1 的个数。

示例 1：
输入：nums = [1,1,0,1,1,1]
输出：3
解释：开头的两位和最后的三位都是连续 1 ，所以最大连续 1 的个数是 3.

示例 2:
输入：nums = [1,0,1,1,0,1]
输出：2

提示：
1 <= nums.length <= 105
nums[i] 不是 0 就是 1.
*/

func findMaxConsecutiveOnes(nums []int) int {
	maxOneCount := 0
	oneCount := 0
	for _, num := range nums {
		if num == 1 {
			oneCount++
		} else {
			maxOneCount = utils.Max(maxOneCount, oneCount)
			oneCount = 0
		}
	}
	return utils.Max(maxOneCount, oneCount)
}

/*
leetcode 1438 绝对差不超过限制的最长连续子数组
给你一个整数数组 nums ，和一个表示限制的整数 limit，请你返回最长连续子数组的长度，该子数组中的任意两个元素之间的绝对差
必须小于或者等于 limit 。

如果不存在满足条件的子数组，则返回 0 。

示例 1：
输入：nums = [8,2,4,7], limit = 4
输出：2
解释：所有子数组如下：
[8] 最大绝对差 |8-8| = 0 <= 4.
[8,2] 最大绝对差 |8-2| = 6 > 4.
[8,2,4] 最大绝对差 |8-2| = 6 > 4.
[8,2,4,7] 最大绝对差 |8-2| = 6 > 4.
[2] 最大绝对差 |2-2| = 0 <= 4.
[2,4] 最大绝对差 |2-4| = 2 <= 4.
[2,4,7] 最大绝对差 |2-7| = 5 > 4.
[4] 最大绝对差 |4-4| = 0 <= 4.
[4,7] 最大绝对差 |4-7| = 3 <= 4.
[7] 最大绝对差 |7-7| = 0 <= 4.
因此，满足题意的最长子数组的长度为 2 。

示例 2：
输入：nums = [10,1,2,4,7,2], limit = 5
输出：4
解释：满足题意的最长子数组是 [2,4,7,2]，其最大绝对差 |2-7| = 5 <= 5 。

示例 3：
输入：nums = [4,2,2,2,4,4,2,2], limit = 0
输出：3

提示：
1 <= nums.length <= 10^5
1 <= nums[i] <= 10^9
0 <= limit <= 10^9
*/

/*
思路: 滑动窗口 + 单调队列
这个问题要求找到一个最长的连续子数组，使得子数组中任意两个元素之间的绝对差不超过给定的 limit。暴力枚举所有可能的子数组
复杂度为 O(n^2)，对于 n≤ 10^5 不可行。因此，我们需要一个更高效的算法。

思路分析
问题本质：对于一个连续子数组，其最大绝对差由子数组的最大值和最小值决定，即 max−min≤limit。我们需要找到最长的子数组，
满足这个条件。

滑动窗口 + 单调队列：
使用滑动窗口维护一个连续子数组。
在窗口内，需要快速获取最大值和最小值，以判断是否满足 max−min≤limit。
使用两个单调队列（一个递增队列记录最小值，一个递减队列记录最大值）来高效维护窗口内的极值。

步骤：
扩展右边界（right），将新元素加入窗口。
更新最大值队列和最小值队列。
检查 max−min≤limit
如果不满足，收缩左边界（left），移除左端元素并更新队列。
如果满足，更新最大窗口长度。
重复直到遍历整个数组。

时间复杂度
每个元素最多入队和出队一次，单调队列操作复杂度为 (O(1))（均摊）。
窗口左右指针遍历数组，总时间复杂度为 (O(n))。

空间复杂度
两个单调队列的空间与窗口大小相关，最坏情况下为 (O(n))。

特殊情况
当 limit=0时，要求子数组元素完全相等。
当 n=1 时，返回 1（单个元素总是满足条件）。



*/

func longestSubarray(nums []int, limit int) int {
	n := len(nums)
	// 单个元素总是满足条件
	if n == 1 {
		return n
	}
	// 单调递减队列，维护窗口内的最大值
	maxQueue := make([]int, 0)
	// 单调递增队列，维护窗口内的最小值
	minQueue := make([]int, 0)
	// 初始化窗口左边界和最大窗口长度为0
	l, maxLength := 0, 0
	// 遍历数组，扩展右边界
	for r := 0; r < n; r++ {
		// 更新最大值队列（保持单调递减）
		for len(maxQueue) > 0 && nums[maxQueue[len(maxQueue)-1]] < nums[r] {
			// 移除小于当前值的元素
			maxQueue = maxQueue[:len(maxQueue)-1]
		}
		maxQueue = append(maxQueue, r)
		// 更新最小值队列（保持单调递增）
		for len(minQueue) > 0 && nums[minQueue[len(minQueue)-1]] > nums[r] {
			// 移除大于当前值的元素
			minQueue = minQueue[:len(minQueue)-1]
		}
		minQueue = append(minQueue, r)
		// 检查窗口是否满足条件
		for len(maxQueue) > 0 && len(minQueue) > 0 && nums[maxQueue[0]]-nums[minQueue[0]] > limit {
			// 如果最大值-最小值超过 limit，收缩左边界
			if maxQueue[0] < minQueue[0] {
				l = maxQueue[0] + 1
				maxQueue = maxQueue[1:]
			} else {
				l = minQueue[0] + 1
				minQueue = minQueue[1:]
			}
		}
		// 更新最大窗口长度
		maxLength = utils.Max(maxLength, r-l+1)
	}
	return maxLength
}

/*
简易内存池
题目描述：
请实现一个简易内存池,根据请求命令完成内存分配和释放。内存池支持两种操作命令，REQUEST和RELEASE，其格式为：

REQUEST=请求的内存大小，表示请求分配指定大小内存.

如果分配成功，返回分配到的内存首地址；
如果内存不足，或指定的大小为0，则输出error；

RELEASE=释放的内存首地址 表示释放掉之前分配的内存，
释放成功无需输出，如果释放不存在的首地址则输出error；

注意：
1. 内存池总大小为100字节；
2. 内存池地址分配必须是连续内存，并优先从低地址分配；
3. 内存释放后可被再次分配，已释放的内存在空闲时不能被二次释放；
4. 不会释放已申请的内存块的中间地址；
5. 释放操作只是针对首地址所对应的单个内存块进行操作，不会影响其它内存块；


输入描述：
首行为整数 N , 表示操作命令的个数，取值范围：0 < N <= 100;
接下来的N行, 每行将给出一个操作命令，操作命令和参数之间用 “=”分割;

输出描述：

1. 请求分配指定大小内存时，如果分配成功，返回分配到的内存首地址；如果内存不足，或指定的大小为0，则输出error；
2. 释放掉之前分配的内存时，释放成功无需输出，如果释放不存在的首地址则输出error。

示例1：
输入
2
REQUEST=10
REQUEST=20

输出
0
10

示例2：
输入
5
REQUEST=10
REQUEST=20
RELEASE=0
REQUEST=20
REQUEST=10

输出
0
10
30
0

示例说明：
第一条指令，申请地址0~9的10个字节内存，返回首地址0；
第二条指令，申请地址10~29的20个字节内存，返回首地址10；
第三条指令，释放首地址为0的内存申请，0~9的地址内存被释放，变为空闲，释放成功，无需输出；
第四条指令，申请20字节内存，0~9的地址内存连续空间不足20字节，往后查找到30~49地址，返回首地址30；
第五条指令，申请10字节内存，0~9的地址内存连续空间足够，返回首地址0；
*/

// MemoryBlock 表示一个内存块，包含首地址和大小
type MemoryBlock struct {
	// 内存块的首地址
	Addr int
	// 内存块的大小
	Size int
}

// MemoryPool 管理内存池
type MemoryPool struct {
	// 已分配的内存块列表
	Allocated []MemoryBlock
	// 空闲内存块列表，按首地址升序排序
	Free []MemoryBlock
	// 内存池总大小
	TotalSize int
}

// NewMemoryPool 初始化一个新的内存池
func NewMemoryPool(totalSize int) *MemoryPool {
	return &MemoryPool{
		Allocated: []MemoryBlock{},
		// 初始时整个内存为空闲
		Free:      []MemoryBlock{{Addr: 0, Size: totalSize}},
		TotalSize: totalSize,
	}
}

// Request 分配指定大小的内存
func (mp *MemoryPool) Request(size int) (int, string) {
	// 请求大小无效
	if size <= 0 {
		return -1, "error"
	}
	for i, block := range mp.Free {
		if block.Size >= size {
			// 分配的首地址
			addr := block.Addr
			if block.Size == size {
				// 完全使用该空闲内存块，移除它
				mp.Free = append(mp.Free[:i], mp.Free[i+1:]...)
			} else {
				// 部分使用，更新该空闲内存块的首地址和大小
				mp.Free[i].Addr += size
				mp.Free[i].Size -= size
			}
			// 将分配的内存块记录到已分配内存块列表
			mp.Allocated = append(mp.Allocated, MemoryBlock{Addr: addr, Size: size})
			// 保持已分配内存块按首地址升序排序
			sort.Slice(mp.Allocated, func(i, j int) bool {
				return mp.Allocated[i].Addr < mp.Allocated[j].Addr
			})
			// 返回分配内存块的首地址
			return addr, ""
		}
	}
	// 无足够空闲块
	return -1, "error"
}

// Release 释放指定首地址的内存块
func (mp *MemoryPool) Release(addr int) string {
	// 在已分配内存块中查找
	for i, block := range mp.Allocated {
		if block.Addr == addr {
			// 找到，移除该内存块
			released := block
			mp.Allocated = append(mp.Allocated[:i], mp.Allocated[i+1:]...)
			// 加入空闲内存块列表
			mp.Free = append(mp.Free, released)
			// 合并相邻空闲块
			mp.MergeFreeBlocks()
			return ""
		}
	}
	// 未找到该内存地址
	return "error"
}

func (mp *MemoryPool) MergeFreeBlocks() {
	// 按首地址排序空闲内存块
	sort.Slice(mp.Free, func(i, j int) bool {
		return mp.Free[i].Addr < mp.Free[j].Addr
	})
	// 从头遍历，合并相邻空闲内存块
	i, n := 0, len(mp.Free)-1
	for i < n {
		// 相邻，合并
		if mp.Free[i].Addr+mp.Free[i].Size == mp.Free[i+1].Addr {
			mp.Free[i].Size += mp.Free[i+1].Size
			mp.Free = append(mp.Free[:i+1], mp.Free[i+2:]...)
		} else {
			i++
		}
	}
}

/*
数组去重和排序
给定一个乱序的数组，删除所有的重复元素，使得每个元素只出现一次，并且按照出现的次数从高到低进行排序，相同出现次数按照
第一次出现顺序进行先后排序。

输入描述
一个数组，数组大小不超过100 数组元素值大小不超过100

输出描述
去重排序后的数组

示例1
输入：
1,3,3,2,4,4,4,5

输出：
4,3,1,2,5
*/

type Item struct {
	Val  int
	Freq int
	Pos  int
}

func removeDuplicatesAndSort(nums []int) []int {
	n := len(nums)
	mark := make(map[int]Item, n)
	for i, num := range nums {
		if item, ok := mark[num]; ok {
			item.Freq++
			mark[num] = item
		} else {
			item := Item{
				Val:  num,
				Freq: 1,
				Pos:  i,
			}
			mark[num] = item
		}
	}
	items := make([]Item, 0, len(mark))
	for _, item := range mark {
		items = append(items, item)
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].Freq == items[j].Freq {
			return items[i].Pos < items[j].Pos
		}
		return items[i].Freq > items[j].Freq
	})
	res := make([]int, len(items))
	for i, item := range items {
		res[i] = item.Val
	}
	return res
}

/*
请设计一个机械累加器，计算从 1、2... 一直累加到目标数值 target 的总和。注意这是一个只能进行加法操作的程序，
不具备乘除、if-else、switch-case、for 循环、while 循环，及条件判断语句等高级功能。

示例 1：
输入: target = 5
输出: 15

示例 2：
输入: target = 7
输出: 28

提示：
1 <= target <= 10000
*/

/*
递归累加：每次调用sum时，将当前值加到结果变量ans上，并递归处理下一个较小的值。
终止条件：利用逻辑运算符&&的短路特性，在不使用if-else的情况下控制递归的结束。
副作用：函数虽然返回一个布尔值，但我们真正关心的是它对全局变量ans的修改。


工作原理
函数sum(target)会将当前的target加到ans上。
然后通过表达式target > 0 && sum(target-1)决定是否继续递归：
如果target > 0为true，则执行sum(target-1)，继续递归。
如果target > 0为false（即target <= 0），&&后面的sum(target-1)不会执行，递归停止。
最终，ans中保存了从1到target的累加和。

为何有效
短路特性：在Go语言中，&&是短路求值的。如果左侧条件为false，则右侧表达式不会执行。这巧妙地替代了if-else。
递归展开：从target开始，逐步递减到0，每次递归累加一个值。
*/

func mechanicalAccumulator(target int) int {
	ans := 0
	var sum func(int) bool
	sum = func(target int) bool {
		ans += target
		return target > 0 && sum(target-1)
	}
	sum(target)
	return ans
}

/*
剑指offer 66 构建乘积数组
给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B[i] 的值是数组 A 中除了下标 i 以外的元素的积,
即 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。
*/

/*
解题思路:两次遍历再合并
给定一个数组 A[0,1,…,n-1]，我们需要构建一个新数组 B[0,1,…,n-1]，其中 B[i] 是数组 A 中除了下标 i 以外所有元素的乘积，
即 B[i] = A[0] × A[1] × … × A[i-1] × A[i+1] × … × A[n-1]。题目明确要求不能使用除法，因此我们不能通过计算数组 A
的总乘积再除以 A[i] 的方式来得到 B[i]。

一个朴素的思路是，对于每个 B[i]，遍历数组 A，跳过下标 i，将其他所有元素相乘。然而，这种方法的时间复杂度为 O(n^2)，
因为对于每个 i，都需要 O(n) 时间来计算乘积。当数组较大时，这种方法效率太低。
为了优化，我们可以观察到 B[i] 是 A 中除了 A[i] 以外元素的乘积，可以将其分解为两部分：
左边乘积：A[0] × A[1] × … × A[i-1]（从开头到 i-1 的累积乘积）。
右边乘积：A[i+1] × A[i+2] × … × A[n-1]（从 i+1 到末尾的累积乘积）。

于是，B[i] = 左边乘积 × 右边乘积。基于这个分解，我们可以通过以下步骤高效计算：
从左到右计算左边乘积：
用一个数组（可以直接用 B）存储从左到右的累积乘积。
例如，B[i] 初始时存储 A[0] × A[1] × … × A[i-1]。

从右到左计算右边乘积并合并：
用一个变量维护从右到左的累积乘积，并将其与 B[i] 的值相乘。
这样，B[i] 最终成为左边乘积和右边乘积的乘积。

这种方法只需两次线性遍历：
第一次从左到右，计算每个位置左边的乘积。
第二次从右到左，将右边的乘积合并到结果中。

时间复杂度：O(n)，因为我们只进行了两次长度为 n 的遍历。
空间复杂度：O(1)（不计输出数组 B），因为除了必要的输出数组外，只使用了常数级别的额外变量。
*/

func constructMultiArray(A []int) []int {
	n := len(A)
	if n == 0 {
		return []int{}
	}
	B := make([]int, n)
	temp := 1
	for i := 0; i < n; i++ {
		B[i] = temp
		temp *= A[i]
	}
	temp = 1
	for i := n - 1; i >= 0; i-- {
		B[i] *= temp
		temp *= A[i]
	}
	return B
}

/*
leetcode 233 数字1的个数
给定一个整数 n，计算所有小于等于 n 的非负整数中数字 1 出现的个数。

示例 1：
输入：n = 13
输出：6

示例 2：
输入：n = 0
输出：0

提示：
0 <= n <= 109
*/

/*
我们需要计算从 0 到 n 中所有数字中出现数字 1 的总次数。直接遍历所有数字显然不够高效，尤其当 n 较大时。因此我们使用
按位分析法来统计每一位上出现数字 1 的次数。

对于数字 n，我们可以逐位（个位、十位、百位……）考虑：
设当前考察的位对应的权重为 factor（例如个位 factor=1，十位 factor=10，百位 factor=100）。
将 n 分解为三部分：

high = n / (factor * 10)：当前位左侧的高位数字；

cur = (n / factor) % 10：当前位数字；

low = n % factor：当前位右侧的低位数字。

根据当前位数字 cur 的值，有如下三种情况：

cur == 0：此时当前位上为 1 的情况只取决于高位数字，因此出现次数为：
high * factor

cur == 1：当前位为 1 时，除了高位影响外，低位也会决定出现次数，具体为：
high * factor + (low + 1)

cur > 1：当当前位大于 1 时，无论低位如何，总会多出一段 1 出现的情况，计算公式为：
(high + 1) * factor

对所有位累加得到的结果就是从 0 到 n 中数字 1 出现的总次数。

我们考虑 n 的每一位（个位、十位、百位……），在每一位上分析“1”出现的情况。为此，我们定义一个变量 factor，表示当前考察位所处的位置权重。
例如：

个位的 factor = 1

十位的 factor = 10

百位的 factor = 100

……

当我们考虑某一位时，我们可以将 n 分成三部分：

高位（high）：当前位左边所有的数字

当前位（cur）：当前正在考察的数字

低位（low）：当前位右边所有的数字

具体计算方法为：

high = n / (factor * 10)
例如：如果 factor = 10（表示当前考察的是十位），则 factor * 10 = 100，n 除以 100 就去掉了最后两位数字，
结果就是十位左边的所有数字。

cur = (n / factor) % 10
用 n 除以当前位的权重，再取余 10，就可以获得当前位上的数字。

low = n % factor
n 除以 factor 的余数就是当前位右边的所有数字。

不同情况的推导基于如下事实：

当前位的数字决定了在这一位上出现“1”的次数。

分解数字
对于任意数字 n，我们可以写成如下形式（假设 n 大于等于 100）：
n = high * 100 + cur*10 + low

其中：

high 表示 n 的百位及以上的部分

cur 表示十位数字

low 表示个位数字

完整循环的概念
观察十位上的数字，它的变化周期为 100（即 0～99，100～199，……）。
每 100 个连续数字（比如 0 到 99）中，十位数字会完整遍历 0～9。
在这些 100 个数字中，当十位为 1 的时候，正好有 10 个数字（10 到 19），也就是 factor = 10 个数字。


1 当 cur 为 0 时
假设当前考察的十位数字 cur 为 0，此时 n 的十位部分为 0。
那么，数字中所有完全的 100 个数字的循环就由 high 决定：
例如，若 high = 5，则说明从 0 到 n 中，已经有 5 个完整的 0～99、100～199、…… 的循环。
每个完整的循环中，十位出现数字 1 的次数都是固定的，即 10 次（对应于 10～19）。
因此，总的次数就是：
次数 = high * factor   （此处 factor=10）

2 当 cur == 1
当当前位为 1 时，有两部分贡献：
高位部分：与上面类似，高位决定了之前有多少个完整的循环，每个循环中当前位都有 factor 个数字出现“1”。这部分贡献为：
high * factor
低位部分：当前位为 1 时，还要考虑低位的情况，因为低位可以从 0 到 low（包括 low 自身）。因此还要加上：
low + 1
次数 = high * factor + (low + 1)

3. 当前位 cur > 1
举个例子，对于 n = 12123541，当 factor = 100 时，我们把 n 拆分成三部分：

high = n / (factor * 10)
= 12123541 / 1000
= 12123

cur = (n / factor) % 10
= (12123541 / 100) % 10
= 121235 % 10
= 5

low = n % factor
= 12123541 % 100
= 41

此时 cur = 5，大于 1。

考虑当前我们正在统计的是百位（即 factor = 100）上“1”出现的次数。

完整循环的概念
百位数字每变化一个周期，其周期长度为 factor * 10 = 1000。
在每个完整的 1000 个数字中，百位恰好为 1 的那一段数字是连续的，从 x100 到 x199，这段共有 100 个数字（即 factor = 100 个数字）。

分为两部分：
1 高位决定的完整循环

high = 12123 表示在 n 的最高位部分，有 12123 个完整的 1000 数字的区间。

每个完整的区间中，百位为 1 的数字有 100 个。
因此这部分贡献为：
12123 * 100

2 当前周期中额外的完整块
此时当前位 cur = 5，大于 1，说明当前这个不完全的 1000 数字区间中，百位已经经历了从 0 到 5。
关键点在于：对于当前这个区间，当百位数字本应从 0 变化到 9 时，出现“1”的那一段（即从 x100 到 x199）已经完整出现了。
所以当前为贡献的百位为 1 的数字已经有 100 个， 从100到199。

不论 low（这里为 41）如何，只要 cur > 1，就证明当前区间中那段数字已经完整存在。
因此，我们需要再加上一个完整的 100 数字的贡献。
换句话说，就是把完整的区间数从 high 扩展到 high+1，乘上 factor（即 100）。

合在一起，百位上出现“1”的次数就是
次数 = (high + 1) * factor = (12123 + 1) * 100 = 12124 * 100 = 1212400
*/

func countDigitOne(n int) int {
	// 结果变量，保存总的出现次数
	count := 0
	// factor 表示当前位的权重（个位、十位、百位……）
	factor := 1

	// 当 n/factor > 0 时，说明还存在当前位
	for n/factor > 0 {
		// high 表示当前位左侧的数字
		high := n / (factor * 10)
		// cur 表示当前位数字
		cur := (n / factor) % 10
		// low 表示当前位右侧的数字
		low := n % factor

		// 根据当前位数字的情况来计算贡献
		if cur == 0 {
			// 当前位数字为 0 时，只能由高位决定
			count += high * factor
		} else if cur == 1 {
			// 当前位数字为 1 时，高位决定部分贡献，低位决定额外贡献
			count += high*factor + low + 1
		} else {
			// 当前位数字大于 1 时，多出一段完整的 factor 个数字
			count += (high + 1) * factor
		}

		// 处理下一位
		factor *= 10
	}

	return count
}

/*
剑指offer 57 和为S的连续正数序列
输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
*/

/*
思路:滑动窗口
假设我们要求的连续正整数序列为 [a, a+1, a+2, ..., b]，其和为：

sum=(start+end)×(end−start+1) / 2

我们希望sum=target，并且序列至少包含两个数字。
基于连续序列的特点，我们可以使用双指针（或滑动窗口）的方式来寻找满足条件的序列：
初始状态：设左指针 start = 1，右指针 end = 2（因为至少有两个数字）。

计算当前窗口的和：利用等差数列求和公式，可以快速计算从 start 到 end 的和：
sum=(start+end)×(end−start+1) / 2

判断 sum 与 target 的关系：

如果 sum 等于 target，则将当前序列记录下来，同时将窗口右边界右移（end++），寻找其他可能的序列。
如果 sum 小于 target，则右边界 end 右移（扩展窗口），使 sum 增大。
如果 sum 大于 target，则左边界 start 右移（缩小窗口），使 sum 变小。

终止条件：由于序列至少包含两个数字，当 start 到达 target/2（严格来说：start < target/2+1）时，
窗口中已经无法找到满足条件的连续序列，可以结束搜索。

这种方法的时间复杂度为 O(target)，在 target 较大时依然高效。
*/

func fileCombination(target int) [][]int {
	res := make([][]int, 0, target)
	start, end := 1, 2
	for start < (target+1)/2 {
		sum := (start + end) * (end - start + 1) / 2
		if sum == target {
			seq := make([]int, end-start+1)
			for i := start; i <= end; i++ {
				seq[i-start] = i
			}
			res = append(res, seq)
			end++
		} else if sum < target {
			end++
		} else {
			start++
		}
	}
	return res
}

/*
剑指Offer 46：孩子们的游戏（圆圈中最后剩下的数）
题目
每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。
其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的
那个小朋友要出列唱首歌, 然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....
这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。
请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)
*/

func lastRemaining(n, m int) int {
	res := 0
	for i := 2; i <= n; i++ {
		res = (res + m) % i
	}
	return res
}
