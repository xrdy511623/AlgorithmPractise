package feature

import "AlgorithmPractise/Utils"

/*
字符串变位词(异位词)专题
*/

/*
leetcode 242. 有效的字母异位词
1.1 给定两个字符串s和t ，编写一个函数来判断t是否是s的字母异位词。
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
剑指 Offer II 014. 字符串中的变位词
leetcode 567. 字符串的排列
1.2 给定两个字符串s1和s2，写一个函数来判断s2是否包含s1的某个变位词。
换句话说，第一个字符串的排列之一是第二个字符串的子串。

示例1：
输入: s1 = "ab" s2 = "eidbaooo"
输出: True
解释: s2 包含 s1 的排列之一 ("ba").

示例2：
输入: s1= "ab" s2 = "eidboaoo"
输出: False

提示：
1 <= s1.length, s2.length <= 104
s1 和 s2 仅包含小写字母
*/

/*
思路:滑动窗口
由于变位词不会改变字符串中每个字符的个数，所以只有当两个字符串每个字符的个数均相等时，一个字符串才是另一个
字符串的变位词。根据这一性质，记s1的长度为n，我们可以遍历s2中的每个长度为n的子串，判断子串和s1中每个字符的
个数是否相等，若相等则说明该子串是s1的一个变位词。
使用两个数组nums1和nums2分别统计s1中各个字符的个数，当前遍历的s2的子串中各个字符的个数。由于需要遍历的子串
长度均为n，我们可以使用一个固定长度为n的滑动窗口来维护nums2。滑动窗口每向右滑动一次，就多统计一次进入窗口的
字符，少统计一次离开窗口的字符。然后，判断nums1与nums2是否相等，若相等则意味着s1的变位词之一是s2的子串。
*/

// CheckInclusion 比较s1中每个字符的个数与s2中连续长度为len(s1)的子串中字符的个数是否相等。
func CheckInclusion(s1 string, s2 string) bool {
	n, m := len(s1), len(s2)
	if n > m {
		return false
	}
	var nums1, nums2 [26]int
	for i, v := range s1 {
		nums1[v-'a']++
		nums2[s2[i]-'a']++
	}
	if nums1 == nums2 {
		return true
	}
	for i := n; i < m; i++ {
		nums2[s2[i]-'a']++
		nums2[s2[i-n]-'a']--
		if nums1 == nums2 {
			return true
		}
	}
	return false
}

/*
优化
注意到每次窗口滑动时，只统计了一进一出两个字符，却比较了整个nums1和nums2数组。从这个角度出发，我们可以用一个
变量diff来记录两个数组nums1和nums2中不同值的个数，这样判断nums1与nums2是否相等就转换成了判断diff是否为0.
每次窗口滑动，记一进一出两个字符为x和y.
若x=y则对nums2无影响，可以直接跳过。
若x!=y，对于字符x，在修改nums2之前若有nums2[x]=0,则将diff加一；在修改nums2[x]之后若有nums2[x]=0
则将diff减一。字符y同理。
此外，为简化上述逻辑，我们可以只用一个数组nums，其中nums[x]=nums2[x]-nums1[x],这样就将nums1和nums2的比较
替换成nums[x]与0的比较，即统计nums中0的个数是否大于0，也就是diff是否等于0。
*/

// CheckInclusionSimple
func CheckInclusionSimple(s1 string, s2 string) bool {
	n, m := len(s1), len(s2)
	if n > m {
		return false
	}
	var nums [26]int
	for i, v := range s1 {
		nums[v-'a']--
		nums[s2[i]-'a']++
	}
	diff := 0
	for _, num := range nums {
		if num != 0 {
			diff++
		}
	}
	if diff == 0 {
		return true
	}
	for i := n; i < m; i++ {
		x, y := s2[i]-'a', s2[i-n]-'a'
		if x == y {
			continue
		}
		if nums[x] == 0 {
			diff++
		}
		nums[x]++
		if nums[x] == 0 {
			diff--
		}
		if nums[y] == 0 {
			diff++
		}
		nums[y]--
		if nums[y] == 0 {
			diff--
		}
		if diff == 0 {
			return true
		}
	}
	return false
}

/*
剑指Offer II 015. 字符串中的所有变位词
leetcode 438. 找到字符串中所有字母异位词
1.3 给定两个字符串s和p，找到s中所有p的变位词的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
变位词指字母相同，但排列不同的字符串。

示例1:
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的变位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的变位词。

示例 2:
输入: s = "abab", p = "ab"
输出: [0,1,2]
解释:
起始索引等于 0 的子串是 "ab", 它是 "ab" 的变位词。
起始索引等于 1 的子串是 "ba", 它是 "ab" 的变位词。
起始索引等于 2 的子串是 "ab", 它是 "ab" 的变位词。

提示:
1 <= s.length, p.length <= 3 * 104
s和p仅包含小写字母
*/

func FindAnagrams(s string, p string) []int {
	var res []int
	m, n := len(s), len(p)
	if n > m {
		return res
	}
	var nums [26]int
	for i, v := range p {
		nums[v-'a']--
		nums[s[i]-'a']++
	}
	diff := 0
	for _, num := range nums {
		if num != 0 {
			diff++
		}
	}
	if diff == 0 {
		// 此时s的起始索引当然是0
		res = append(res, 0)
	}
	for i := n; i < m; i++ {
		// x, y分别为进入滑动窗口和离开滑动窗口的元素
		// i其实就是滑动窗口的末尾位置
		x, y := s[i]-'a', s[i-n]-'a'
		if nums[x] == 0 {
			diff++
		}
		nums[x]++
		if nums[x] == 0 {
			diff--
		}
		if nums[y] == 0 {
			diff++
		}
		nums[y]--
		if nums[y] == 0 {
			diff--
		}
		if diff == 0 {
			// 此时的i为符合条件的子串的末尾位置
			// 所以子串的起始位置就是i-len(p)+1,也就是i-n+1
			res = append(res, i-n+1)
		}
	}
	return res
}

/*
leetcode 49. 字母异位词分组
1.4 给你一个字符串数组，请你将字母异位词组合在一起。可以按任意顺序返回结果列表。
字母异位词是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。

示例1:
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]

示例2:
输入: strs = [""]
输出: [[""]]

示例3:
输入: strs = ["a"]
输出: [["a"]]

提示：
1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i]仅包含小写字母
*/

func GroupAnagrams(strs []string) [][]string {
	hashTable := make(map[[26]int][]string)
	for _, str := range strs {
		nums := [26]int{}
		for _, v := range str {
			nums[v-'a']++
		}
		// 利用哈希表将字符出现次数相同的字符串放到同一个数组中
		hashTable[nums] = append(hashTable[nums], str)
	}
	res := make([][]string, 0, len(hashTable))
	for _, v := range hashTable {
		res = append(res, v)
	}
	return res
}

/*
剑指Offer II 016. 不含重复字符的最长子字符串
leetcode 3. 无重复字符的最长子串
1.5 给定一个字符串s ，请你找出其中不含有重复字符的最长连续子字符串的长度。

示例1:
输入: s = "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子字符串是"abc"，所以其长度为3。

示例2:
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子字符串是"b"，所以其长度为1。

示例3:
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。

示例4:
输入: s = ""
输出: 0

提示：
0 <= s.length <= 5 * 104
s由英文字母、数字、符号和空格组成
*/

/*
思路:哈希表+滑动窗口
这道题需要通过滑动窗口来解决，只不过这次的边界获取要通过哈希表来实现。
首先我们创建一个哈希表visited，并且初始化滑动窗口左边界start=0，字符串s中最长不含重复字符的连续子串长度
maxLength=0
下来我们从下标0开始遍历字符串:
每当遍历到字符串中的一个字符时，首先需要判断该字符在哈希表visited中是否出现过(判重)
如果该字符串没有在哈希表中，表示该字符不重复，无需移动左边界start，将该字符串及对应下标加入哈希表中。
如果该字符在哈希表中出现过，表示找到了重复的元素，此时我们需要移动左边界start:
若start小于哈希表中该字符之前出现的位置pos，则移动至pos+1（因为在位置pos处已经重复了，需要跳过）
若start大于哈希表中该字符之前出现的位置pos，表示重复的字符在左边界以外，忽略即可。
将哈希表visited中当前字符的出现位置pos更新为当前位置i。
每次遍历后，迭代最大长度maxLength的值。
最终返回maxLength即可。
*/

// LengthOfLongestSubstring
func LengthOfLongestSubstring(s string) int {
	start, maxLength := 0, 0
	visited := make(map[byte]int)
	// i其实就是窗口右边界
	for i := 0; i < len(s); i++ {
		if pos, ok := visited[s[i]]; ok {
			// 如果当前字符在哈希表中出现过，更新窗口左边界start
			start = Utils.Max(start, pos+1)
		}
		visited[s[i]] = i
		maxLength = Utils.Max(maxLength, i-start+1)
	}
	return maxLength
}