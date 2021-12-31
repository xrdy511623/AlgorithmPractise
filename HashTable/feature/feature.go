package feature

import (
	"AlgorithmPractise/Utils"
	"math"
)

/*
1.0  有效的字母异位词
给定两个字符串s和t ，编写一个函数来判断t是否是s的字母异位词。
注意：若s和t中每个字符出现的次数都相同，则称s和t互为字母异位词。

输入: s = "anagram", t = "nagaram"
输出: true

输入: s = "rat", t = "car"
输出: false

提示:
1 <= s.length, t.length <= 5 * 104
s 和 t 仅包含小写字母
*/

// IsAnagram 时间复杂度O(S+T)，空间复杂度O(S)
func IsAnagram(s, t string) bool {
	if len(s) != len(t) {
		return false
	}
	freqS := make(map[byte]int)
	for i := 0; i < len(s); i++ {
		freqS[s[i]]++
	}
	for i := 0; i < len(t); i++ {
		freqS[t[i]]--
		if freqS[t[i]] < 0 {
			return false
		}
	}
	return true
}

func IsIsAnagramSimple(s, t string) bool {
	var c1, c2 [26]int
	for _, ch := range s {
		c1[ch-'a']++
	}
	for _, ch := range t {
		c2[ch-'a']++
	}
	return c1 == c2
}

/*
1.1 两个数组的交集
给定两个数组，编写一个函数来计算它们的交集。

示例1：
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2]

示例2：
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[9,4]
*/

// FindIntersection 时间复杂度O(M+N)，空间复杂度O(M)
func FindIntersection(nums1 []int, nums2 []int) []int {
	var res []int
	hashTable := make(map[int]bool)
	for i := 0; i < len(nums1); i++ {
		hashTable[nums1[i]] = true
	}
	for i := 0; i < len(nums2); i++ {
		if hashTable[nums2[i]] {
			res = append(res, nums2[i])
		}
		// 因为交集中相同的元素只保留一个，所以需要这么操作
		delete(hashTable, nums2[i])
	}
	return res
}

/*
1.2 快乐数
编写一个算法来判断一个数n是不是快乐数。
「快乐数」定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数
变为 1，也可能是无限循环但始终变不到1。如果可以变为1，那么这个数就是快乐数。
如果n是快乐数就返回True；不是，则返回False 。

示例：
输入：19
输出：true
解释：
1^2 + 9^2 = 82
8^2 + 2^2 = 68
6^2 + 8^2 = 100
1^2 + 0^2 + 0^2 = 1
*/

/*
思路:利用哈希表检查循环
算法分为两部分，我们需要设计和编写代码。
给一个数字n，它的下一个数字是什么？
按照一系列的数字来判断我们是否进入了一个循环。
第1部分我们按照题目的要求做数位分离，求平方和。
第2部分可以使用哈希集合完成。每次生成链中的下一个数字时，我们都会检查它是否已经在哈希集合中。
如果它不在哈希集合中，我们应该添加它。
如果它在哈希集合中，这意味着我们处于一个死循环中，因此应该返回 false。
*/

// IsHappy 时间复杂度O(logN)，空间复杂度O(logN)
func IsHappy(n int) bool {
	occurred := make(map[int]bool)
	for {
		sum := GetSquareSum(n)
		if sum == 1 {
			return true
		}
		// 如果这个平方和重复出现，就证明陷入死循环，直接返回false
		if occurred[sum] {
			return false
		} else {
			// 否则，记录这个平方和出现过
			occurred[sum] = true
		}
		// 重置n的值为前一个平方和
		n = sum
	}
}

// GetSquareSum 求正整数每个位置上的数字的平方和
func GetSquareSum(n int) int {
	sum := 0
	for n > 0 {
		sum += (n % 10) * (n % 10)
		n = n / 10
	}
	return sum
}

/*
思路二:快慢双指针法
通过反复调用getNext(n) 得到的链是一个隐式的链表。隐式意味着我们没有实际的链表节点和指针，但数据仍然形成
链表结构。起始数字是链表的头“节点”，链中的所有其他数字都是节点。next 指针是通过调用 getNext(n) 函数获得。
意识到我们实际有个链表，那么这个问题就可以转换为检测一个链表是否有环。快慢指针法就派上了用场，如果链表有环，
也就是平方和重复出现，那就意味着快慢指针一定会相遇，此时返回false,否则不会相遇，那么只需要判断fast是否
等于1就可以了。
*/

// IsHappyNumber 时间复杂度O(logN)，空间复杂度O(1)
func IsHappyNumber(n int) bool {
	slow, fast := n, n
	var step func(int) int
	step = func(n int) int {
		sum := 0
		for n > 0 {
			sum += (n % 10) * (n % 10)
			n = n / 10
		}
		return sum
	}
	for fast != 1 {
		slow = step(slow)
		fast = step(step(fast))
		if slow == fast && slow != 1 {
			return false
		}
	}
	return fast == 1
}

/*
1.3 无重复字符的最长子串
给定一个字符串s ，请你找出其中不含有重复字符的最长子串的长度。
输入: s = "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是"wke"，所以其长度为 3。
请注意，你的答案必须是子串的长度，"pwke"是一个子序列，不是子串。

输入: s = ""
输出: 0
*/

func LengthOfLongestSubString(s string) int {
	// 哈希表，记录字符串中的字符是否出现过
	occurred := make(map[byte]int, 0)
	// 最长子串长度初始值为0，最长子串最右边界(右指针)初始值为-1，表示此时右指针尚未移动
	maxLength, rp := 0, -1
	n := len(s)
	// i从0开始递增，它就是子串的左边界，左指针
	for i := 0; i < n; i++ {
		// 左指针每向右移动一位，则在哈希表中将其前面的字符删除
		if i != 0 {
			delete(occurred, s[i-1])
		}
		// 只要右指针不越界(超出字符串的长度-1)且右指针对应的字符没有在哈希表中出现过，则哈希表中
		// 记录该字符，同时右指针持续向右移动一位
		for rp+1 < n && occurred[s[rp+1]] == 0 {
			occurred[s[rp+1]]++
			rp++
		}
		// i每迭代一次，则迭代maxLength的值，最大长度等于右指针-左指针+1
		maxLength = Utils.Max(maxLength, rp-i+1)
	}
	return maxLength
}

/*
1.4 最长连续序列
给定一个未排序的整数数组nums，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
请你设计并实现时间复杂度为O(n)的算法解决此问题。

示例 1：
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。

示例 2：
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9
*/

/*
思路:哈希表
我们考虑枚举数组中的每个数x，考虑以其为起点，不断尝试匹配x+1,x+2,⋯ 是否存在，假设最长匹配到了x+y，那么以x为起点的最长连续序列即为x,x+1,x+2,⋯,
x+y，其长度为y+1，我们不断枚举并更新答案即可。
对于匹配的过程，暴力的方法是O(n) 遍历数组去看是否存在这个数，但其实更高效的方法是用一个哈希表存储数组中的数，这样查看一个数是否存在即能优化至O(1)
的时间复杂度。
仅仅是这样我们的算法时间复杂度最坏情况下还是会达到 O(n^2)（即外层需要枚举O(n)个数，内层需要暴力匹配O(n)次），无法满足题目的要求。但仔细分析这个过程，
我们会发现其中执行了很多不必要的枚举，如果已知有一个x,x+1,x+2,⋯,x+y 的连续序列，而我们却重新从x+1，x+2或者是x+y处开始尝试匹配，那么得到的结果肯定
不会优于枚举x为起点的答案，因此我们在外层循环的时候碰到这种情况跳过即可。

那么怎么判断是否跳过呢？由于我们要枚举的数x一定是在数组中不存在前驱数x-1的，不然按照上面的分析我们会从x−1开始尝试匹配，因此我们每次在哈希表中检查是
否存在x−1即能判断是否需要跳过了。

增加了判断跳过的逻辑之后，时间复杂度是多少呢？外层循环需要O(n) 的时间复杂度，只有当一个数是连续序列的第一个数的情况下才会进入内层循环，然后在内层循环
中匹配连续序列中的数，因此数组中的每个数只会进入内层循环一次。根据上述分析可知，总时间复杂度为O(n)，符合题目要求。
*/

func LongestConsecutive(nums []int) int {
	hashTable := make(map[int]bool)
	// 哈希表去重
	for _, num := range nums {
		hashTable[num] = true
	}
	maxLength := 0
	for num := range hashTable {
		// 跳过num-1，否则会重复做无用功
		if !hashTable[num-1] {
			cur := num
			length := 1
			// 不断枚举cur+1,cur+2 ...,判断是否存在于哈希表中
			for hashTable[cur+1] {
				cur++
				// 如果每次cur累加1都满足，则以cur为起点的连续递增序列长度累加1
				length++
			}
			// 迭代maxLength
			if maxLength < length {
				maxLength = length
			}
		}
	}
	return maxLength
}

/*
1.5 最小覆盖子串
给你一个字符串s、一个字符串t。返回s中涵盖t所有字符的最小子串。如果s中不存在涵盖t所有字符的子串，则返回空字符串""。

注意：
对于t中重复字符，我们寻找的子字符串中该字符数量必须不少于t中该字符数量。
如果s中存在这样的子串，我们保证它是唯一的答案。

示例1：
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"

示例2：
输入：s = "a", t = "a"
输出："a"

示例3:
输入: s = "a", t = "aa"
输出: ""
*/

func MinWindow(s, t string) string {
	ori, window := make(map[byte]int), make(map[byte]int)
	// 用哈希表ori记录字符串t中字符出现的频次
	for i := 0; i < len(t); i++ {
		ori[t[i]]++
	}
	sLen := len(s)
	wL, wR := -1, -1
	length := math.MaxInt32
	check := func() bool {
		for k, v := range ori {
			if window[k] < v {
				return false
			}
		}
		return true
	}
	for l, r := 0, 0; r < sLen; r++ {
		if r < sLen && ori[s[r]] > 0 {
			window[s[r]]++
		}
		for check() && l <= r {
			// 只要window内仍包含ori存储的所有字符，且字符数量不低于ori的字符个数
			// 就可以继续更新窗口长度和窗口首尾下标
			if r-l+1 < length {
				length = r - l + 1
				wL, wR = l, l+length
			}
			if _, ok := ori[s[l]]; ok {
				window[s[l]]--
			}
			// 左指针向右移动，试着减小窗口长度
			l++
		}
	}
	if wL == -1 {
		return ""
	}
	return s[wL:wR]
}

/*
1.6 和为k的连续子数组
给你一个整数数组nums和一个整数k ，请你统计并返回该数组中和为k的连续子数组的个数。

示例1：
输入：nums = [1,1,1], k = 2
输出：2

示例2：
输入：nums = [1,2,3], k = 3
输出：2
*/

/*
思路一:枚举法
考虑以i结尾和为k的连续子数组个数，我们需要统计符合条件的下标j的个数，其中0≤j≤i 且[j..i]这个子数组的和恰好为k 。
我们可以枚举 [0..i]里所有的下标j来判断是否符合条件。
*/

// SubarraySumSimple 时间复杂度O(N^2)，空间复杂度O(1)
func SubarraySumSimple(nums []int, k int) int {
	count := 0
	for i := 0; i < len(nums); i++ {
		sum := 0
		for end := i; end >= 0; end-- {
			sum += nums[end]
			if sum == k {
				count++
			}
		}
	}
	return count
}

/*
思路二:前缀和+哈希表优化
我们可以基于方法一利用数据结构进行进一步的优化，我们知道方法一的瓶颈在于对每个i，我们需要枚举所有的j来判断是否符合
条件，这一步是否可以优化呢？答案是可以的。

我们定义pre[i]为[0..i]里所有数的和，则pre[i]可以由pre[i−1]递推而来，即：
pre[i]=pre[i−1]+nums[i]

那么[j..i]这个子数组和为k这个条件我们可以转化为
pre[i]−pre[j−1]=k

简单移项可得符合条件的下标j需要满足
pre[j−1]==pre[i]−k

所以我们考虑以i结尾的和为k的连续子数组个数时只要统计有多少个前缀和为pre[i]−k的pre[j]即可。我们建立哈希表
mp，以和为键，出现次数为对应的值，记录pre[i]出现的次数，从左往右边更新mp边计算答案，那么以i结尾的答案
mp[pre[i]−k]即可在O(1) 时间内得到。最后的答案即为所有下标结尾的和为k的子数组个数之和。

需要注意的是，从左往右边更新边计算的时候已经保证了mp[pre[i]−k]里记录的pre[j] 的下标范围是0≤j≤i 。同时，
由于pre[i]的计算只与前一项的答案有关，因此我们可以不用建立pre数组，直接用pre变量来记录pre[i-1]的答案即可。
*/

// SubarraySum 时间复杂度O(N)，空间复杂度O(N)
func SubarraySum(nums []int, k int) int {
	pre, count := 0, 0
	hashTable := make(map[int]int)
	hashTable[0] = 1
	for i := 0; i < len(nums); i++ {
		pre += nums[i]
		if v, ok := hashTable[pre-k]; ok {
			count += v
		}
		hashTable[pre]++
	}
	return count
}

/*
1.7 前k个高频元素
给你一个整数数组nums和一个整数k ，请你返回其中出现频率前k高的元素。你可以按任意顺序返回答案。

示例1:
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]

示例2:
输入: nums = [1], k = 1
输出: [1]

提示：
1 <= nums.length <= 105
k 的取值范围是 [1, 数组中不相同的元素的个数]
题目数据保证答案唯一，换句话说，数组中前k个高频元素的集合是唯一的
进阶：你所设计算法的时间复杂度必须优于O(NlogN) ，其中n是数组大小。
*/

func TopKFrequent(nums []int, k int) []int {
	freqMap := make(map[int]int)
	maxFreq := math.MinInt32
	for _, v := range nums {
		if _, ok := freqMap[v]; ok {
			freqMap[v]++
		} else {
			freqMap[v] = 1
		}
		if freqMap[v] > maxFreq {
			maxFreq = freqMap[v]
		}
	}
	hashTop := make([][]int, maxFreq+1)
	for key, val := range freqMap {
		hashTop[val] = append(hashTop[val], key)
	}
	res := make([]int, 0)
	for i := maxFreq; i >= 0; i-- {
		res = append(res, hashTop[i]...)
		k -= len(hashTop[i])
		if k == 0 {
			break
		}
	}
	return res
}