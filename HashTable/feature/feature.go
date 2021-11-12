package feature

import "AlgorithmPractise/Utils"

/*
1.0 无重复字符的最长子串
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

func LengthOfLongestSubString(s string)int{
	// 哈希表，记录字符串中的字符是否出现过
	occurred := make(map[byte]int, 0)
	// 最长子串长度初始值为0，最长子串最右边界(右指针)初始值为-1，表示此时右指针尚未移动
	maxLength, rp := 0, -1
	n := len(s)
	// i从0开始递增，它就是子串的左边界，左指针
	for i:=0;i<n;i++{
		// 左指针每向右移动一位，则在哈希表中将其前面的字符删除
		if i != 0{
			delete(occurred, s[i-1])
		}
		// 只要右指针不越界(超出字符串的长度-1)且右指针对应的字符没有在哈希表中出现过，则哈希表中
		// 记录该字符，同时右指针持续向右移动一位
		for rp + 1 < n && occurred[s[rp+1]] == 0{
			occurred[s[rp+1]]++
			rp++
		}
		// i每迭代一次，则迭代maxLength的值，最大长度等于右指针-左指针+1
		maxLength = Utils.Max(maxLength, rp-i+1)
	}
	return maxLength
}