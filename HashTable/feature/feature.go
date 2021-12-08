package feature

import (
	"AlgorithmPractise/Utils"
	"math"
)

/*
1.1 无重复字符的最长子串
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
1.2 最长连续序列
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
1.3 最小覆盖子串
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