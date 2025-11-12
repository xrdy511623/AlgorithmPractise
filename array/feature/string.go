package feature

import (
	"math"
	"strconv"
	"strings"
	"unicode"

	"algorithmpractise/utils"
)

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
	if len(s) != len(t) {
		return false
	}
	need := [26]int{}
	for _, v := range s {
		need[v-'a']++
	}
	for _, v := range t {
		need[v-'a']--
		if need[v-'a'] < 0 {
			return false
		}
	}
	return true
}

/*
剑指Offer II 014. 字符串中的变位词
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

// CheckInclusionSimple 简化写法
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
思路:滑动窗口
异位词的定义：两个字符串的字符及其出现次数相同。
滑动窗口：
使用一个长度为 p 的窗口在字符串 s 上滑动。
对每个窗口内的字符串进行字符计数，判断其是否与 p 的字符计数匹配。
优化：
使用一个固定大小为 26 的数组（针对小写字母），分别记录窗口内字符和目标字符串 p 的字符计数。
每次滑动时，仅更新窗口内新增和移除的字符的计数，而不是重新计算整个窗口的字符计数。
*/

func FindAnagramsTwo(s string, p string) []int {
	res := []int{}
	m, n := len(s), len(p)
	// 特殊情况：s 的长度小于 p，直接返回
	if n > m {
		return res
	}
	// 初始化两个计数数组
	var pCount, sCount [26]int
	for i, v := range p {
		pCount[v-'a']++
		sCount[s[i]-'a']++
	}
	// 滑动窗口开始
	l, r := 0, len(p)-1
	for r < m {
		// 判断当前窗口是否是异位词
		if pCount == sCount {
			res = append(res, l)
		}
		// 移动窗口右边界
		r++
		if r < m {
			// 新增窗口右边字符
			sCount[s[r]-'a']++
		}
		// 移除窗口左边字符
		sCount[s[l]-'a']--
		l++
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
1.5 给定一个字符串s，请你找出其中不含有重复字符的最长连续子字符串的长度。

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
解释: 因为无重复字符的最长子串是"wke"，所以其长度为 3。
请注意，你的答案必须是子串的长度，"pwke"是一个子序列，不是子串。

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
接下来我们从下标0开始遍历字符串:
"abcabcbb"
每当遍历到字符串中的一个字符时，首先需要判断该字符在哈希表visited中是否出现过(判重)
如果该字符串没有在哈希表中，表示该字符不重复，无需移动左边界start，将该字符及对应下标加入哈希表中。
如果该字符在哈希表中出现过，表示找到了重复的元素，此时我们需要移动左边界start:
若start小于哈希表中该字符之前出现的位置pos，则移动至pos+1（因为在位置pos处已经重复了，需要跳过）
若start大于哈希表中该字符之前出现的位置pos，表示重复的字符在左边界以外，忽略即可。
将哈希表visited中当前字符的出现位置pos更新为当前位置i。
每次遍历后，迭代最大长度maxLength的值。
最终返回maxLength即可。
*/

// LengthOfLongestSubstring 哈希表+滑动窗口解决
func LengthOfLongestSubstring(s string) int {
	start, maxLength := 0, 0
	visited := make(map[byte]int)
	// i其实就是窗口右边界
	for i, length := 0, len(s); i < length; i++ {
		if pos, ok := visited[s[i]]; ok {
			// 如果当前字符在哈希表中出现过，更新窗口左边界start
			start = utils.Max(start, pos+1)
		}
		visited[s[i]] = i
		maxLength = utils.Max(maxLength, i-start+1)
	}
	return maxLength
}

/*
leetcode 43. 字符串相乘
给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
注意：不能使用任何内置的 BigInteger 库或直接将输入转换为整数。

示例 1:
输入: num1 = "2", num2 = "3"
输出: "6"

示例2:
输入: num1 = "123", num2 = "456"
输出: "56088"


提示：
1 <= num1.length, num2.length <= 200
num1 和 num2 只能由数字组成。
num1 和 num2 都不包含任何前导零，除了数字0本身。
*/

func multiply(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	m, n := len(num1), len(num2)
	arr := make([]int, m+n)
	for i := m - 1; i >= 0; i-- {
		n1 := int(num1[i] - '0')
		for j := n - 1; j >= 0; j-- {
			n2 := int(num2[j] - '0')
			sum := n1*n2 + arr[i+j+1]
			arr[i+j+1] = sum % 10
			arr[i+j] += sum / 10
		}
	}
	res := ""
	for i, v := range arr {
		if i == 0 && v == 0 {
			continue
		}
		res += strconv.Itoa(v)
	}
	return res
}

/*
leetcode 165. 比较版本号
给你两个版本号字符串 version1 和 version2 ，请你比较它们。版本号由被点 '.' 分开的修订号组成。修订号的值是它转换
为整数并忽略前导零。

比较版本号时，请按从左到右的顺序依次比较它们的修订号。如果其中一个版本字符串的修订号较少，则将缺失的修订号视为0。

返回规则如下：
如果 version1 < version2 返回 -1，
如果 version1 > version2 返回 1，
除此之外返回 0。

示例 1：

输入：version1 = "1.2", version2 = "1.10"

输出：-1

解释：

version1 的第二个修订号为 "2"，version2 的第二个修订号为 "10"：2 < 10，所以 version1 < version2。

示例 2：

输入：version1 = "1.01", version2 = "1.001"

输出：0

解释：

忽略前导零，"01" 和 "001" 都代表相同的整数 "1"。

示例 3：

输入：version1 = "1.0", version2 = "1.0.0.0"

输出：0

解释：
version1 有更少的修订号，每个缺失的修订号按 "0" 处理。

提示：
1 <= version1.length, version2.length <= 500
version1 和 version2 仅包含数字和 '.'
version1 和 version2 都是 有效版本号
version1 和 version2 的所有修订号都可以存储在 32 位整数 中
*/

/*
思路：字符串分割
我们可以将版本号按照点号分割成修订号，然后从左到右比较两个版本号的相同下标的修订号。在比较修订号时，需要将字符串转换
成整数进行比较。注意根据题目要求，如果版本号不存在某个下标处的修订号，则该修订号视为0。
*/

func compareVersion(version1 string, version2 string) int {
	v1, v2 := strings.Split(version1, "."), strings.Split(version2, ".")
	for i := 0; i < len(v1) || i < len(v2); i++ {
		x, y := 0, 0
		if i < len(v1) {
			x, _ = strconv.Atoi(v1[i])
		}
		if i < len(v2) {
			y, _ = strconv.Atoi(v2[i])
		}
		if x > y {
			return 1
		}
		if x < y {
			return -1
		}
	}
	return 0
}

/*
leetcode 394 字符串解码
给定一个经过编码的字符串，返回它解码后的字符串。
编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。

示例 1：
输入：s = "3[a]2[bc]"
输出："aaabcbc"

示例 2：
输入：s = "3[a2[c]]"
输出："accaccacc"

示例 3：
输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"

示例 4：
输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"


提示：
1 <= s.length <= 30
s 由小写英文字母、数字和方括号 '[]' 组成
s 保证是一个 有效 的输入。
s 中所有整数的取值范围为 [1, 300]
*/

func decodeString(s string) string {
	// 保存重复次数的栈
	countStack := []int{}
	// 保存当前字符串的栈
	stringStack := []string{}
	// 当前构建的字符串
	currentString := ""
	// 当前解析的数字
	currentNum := 0
	for i := range s {
		// 遇到数字,累计数字，多位数处理
		if s[i] >= '0' && s[i] <= '9' {
			currentNum = currentNum*10 + int(s[i]-'0')
		} else if s[i] == '[' {
			// 遇到左括号时，保存当前构建的字符串和解析的数字到对应栈中并重置它们
			countStack = append(countStack, currentNum)
			stringStack = append(stringStack, currentString)
			currentNum = 0
			currentString = ""
		} else if s[i] == ']' {
			// 遇到右括号,处理当前重复字符串并拼接之前构建的字符串
			count := countStack[len(countStack)-1]
			countStack = countStack[:len(countStack)-1]
			previousString := stringStack[len(stringStack)-1]
			stringStack = stringStack[:len(stringStack)-1]
			currentString = previousString + repeat(currentString, count)
		} else {
			// 遇到字母, 加入到当前构建的字符串中
			currentString += string(s[i])
		}
	}
	return currentString
}

func repeat(s string, count int) string {
	res := strings.Builder{}
	for i := 0; i < count; i++ {
		res.WriteString(s)
	}
	return res.String()
}

/*
leetcode 468 验证IP地址
给定一个字符串 queryIP。如果是有效的 IPv4 地址，返回 "IPv4" ；如果是有效的 IPv6 地址，返回 "IPv6" ；
如果不是上述类型的 IP 地址，返回 "Neither" 。

有效的IPv4地址 是 “x1.x2.x3.x4” 形式的IP地址。 其中 0 <= xi <= 255 且 xi 不能包含 前导零。
例如: “192.168.1.1” 、 “192.168.1.0” 为有效IPv4地址， “192.168.01.1” 为无效IPv4地址; “192.168.1.00” 、
“192.168@1.1” 为无效IPv4地址。

一个有效的IPv6地址 是一个格式为“x1:x2:x3:x4:x5:x6:x7:x8” 的IP地址，其中:
1 <= xi.length <= 4
xi 是一个 十六进制字符串，可以包含数字、小写英文字母( 'a' 到 'f' )和大写英文字母( 'A' 到 'F' )。
在 xi 中允许前导零。
例如 "2001:0db8:85a3:0000:0000:8a2e:0370:7334" 和 "2001:db8:85a3:0:0:8A2E:0370:7334" 是有效的 IPv6 地址，
而 "2001:0db8:85a3::8A2E:037j:7334" 和 "02001:0db8:85a3:0000:0000:8a2e:0370:7334" 是无效的 IPv6 地址。

示例 1：
输入：queryIP = "172.16.254.1"
输出："IPv4"
解释：有效的 IPv4 地址，返回 "IPv4"

示例 2：
输入：queryIP = "2001:0db8:85a3:0:0:8A2E:0370:7334"
输出："IPv6"
解释：有效的 IPv6 地址，返回 "IPv6"

示例 3：
输入：queryIP = "256.256.256.256"
输出："Neither"
解释：既不是 IPv4 地址，又不是 IPv6 地址

提示：
queryIP 仅由英文字母，数字，字符 '.' 和 ':' 组成。
*/

func validIPAddress(queryIP string) string {
	if checkIpV4(queryIP) {
		return "IPv4"
	}
	if checkIpV6(queryIP) {
		return "IPv6"
	}
	return "Neither"
}

func checkIpV4(queryIP string) bool {
	ipParts := strings.Split(queryIP, ".")
	// 必须是“x1.x2.x3.x4” 形式的IP地址,也就是被.分隔为4部分
	if len(ipParts) != 4 {
		return false
	}
	for _, part := range ipParts {
		// 每部分长度在1到3之间
		if len(part) < 1 || len(part) > 3 {
			return false
		}
		// 每部分不能含前导零
		if len(part) > 1 && part[0] == '0' {
			return false
		}
		// 每部分都是数字，且值在0~255之间
		num := 0
		for _, ch := range part {
			// 判断是否为合法的数字
			if !(ch >= '0' && ch <= '9') {
				return false
			}
			num = num*10 + int(ch-'0')
		}
		if num < 0 || num > 255 {
			return false
		}
	}
	return true
}

func checkIpV6(queryIP string) bool {
	ipParts := strings.Split(queryIP, ":")
	// 一个有效的IPv6地址 是一个格式为“x1:x2:x3:x4:x5:x6:x7:x8” 的IP地址,也就是被:分隔为8部分
	if len(ipParts) != 8 {
		return false
	}
	for _, part := range ipParts {
		// 每部分长度在1~4之间
		if len(part) < 1 || len(part) > 4 {
			return false
		}
		// 每个部分包含数字、小写英文字母( 'a' 到 'f' )和大写英文字母( 'A' 到 'F' )
		for _, ch := range part {
			if !(ch >= '0' && ch <= '9') && !(ch >= 'a' && ch <= 'f') && !(ch >= 'A' && ch <= 'F') {
				return false
			}
		}
	}
	return true
}

/*
leetcode 7 整数反转
给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。
如果反转后整数超过 32 位的有符号整数的范围 [−231,  231 − 1] ，就返回 0。
假设环境不允许存储 64 位整数（有符号或无符号）。

示例 1：
输入：x = 123
输出：321

示例 2：
输入：x = -123
输出：-321

示例 3：
输入：x = 120
输出：21

示例 4：
输入：x = 0
输出：0

提示：
-231 <= x <= 231 - 1
*/

func reverseNum(x int) int {
	numStr := strconv.Itoa(x)
	sign := 1
	ss := []byte(numStr)
	if ss[0] == '-' {
		sign = -1
		ss = ss[1:]
	}
	reversedNumStr := utils.ReverseString(ss)
	num, _ := strconv.Atoi(reversedNumStr)
	res := num * sign
	if res <= math.MinInt32 || res >= math.MaxInt32 {
		return 0
	}
	return res
}

/*
leetcode 125 验证回文串
如果在将所有大写字符转换为小写字符、并移除所有非字母数字字符之后，短语正着读和反着读都一样。则可以认为该短语是一个 回文串 。

字母和数字都属于字母数字字符。
给你一个字符串 s，如果它是 回文串 ，返回 true ；否则，返回 false 。

示例 1：
输入: s = "A man, a plan, a canal: Panama"
输出：true
解释："amanaplanacanalpanama" 是回文串。

示例 2：
输入：s = "race a car"
输出：false
解释："raceacar" 不是回文串。

示例 3：
输入：s = " "
输出：true
解释：在移除非字母数字字符之后，s 是一个空字符串 "" 。
由于空字符串正着反着读都一样，所以是回文串。

提示：
1 <= s.length <= 2 * 105
s 仅由可打印的 ASCII 字符组成
*/

func isPalindrome(s string) bool {
	if s == " " || len(s) == 1 {
		return true
	}
	ss := []byte(s)
	n := len(ss)
	ssNew := []byte{}
	for i := 0; i < n; i++ {
		if !utils.CheckAlphaNumeric(s[i]) {
			continue
		}
		ssNew = append(ssNew, ss[i])
	}
	res := strings.ToLower(string(ssNew))
	for i, n := 0, len(res); i < n/2; i++ {
		if res[i] != res[n-1-i] {
			return false
		}
	}
	return true
}

/*
双指针，性能更优，主要是空间复杂度更低
*/
func isPalindromeSimple(s string) bool {
	l, r := 0, len(s)-1
	for l < r {
		for l < r && !utils.CheckAlphaNumeric(s[l]) {
			l++
		}
		for l < r && !utils.CheckAlphaNumeric(s[r]) {
			r--
		}
		if strings.ToLower(string([]byte{s[l]})) != strings.ToLower(string([]byte{s[r]})) {
			return false
		}
		l++
		r--
	}
	return true
}

/*
leetcode 906 超级回文数
如果一个正整数自身是回文数，而且它也是一个回文数的平方，那么我们称这个数为 超级回文数 。
现在，给你两个以字符串形式表示的正整数 left 和 right，统计并返回区间 [left, right] 中的 超级回文数 的数目。

示例 1：
输入：left = "4", right = "1000"
输出：4
解释：4、9、121 和 484 都是超级回文数。
注意 676 不是超级回文数：26 * 26 = 676 ，但是 26 不是回文数。

示例 2：
输入：left = "1", right = "2"
输出：1

提示：
1 <= left.length, right.length <= 18
left 和 right 仅由数字（0 - 9）组成。
left 和 right 不含前导零。
left 和 right 表示的整数在区间 [1, 1018 - 1] 内。
left 小于等于 right 。
*/

func superPalindromesInRange(left string, right string) int {
	low, _ := strconv.ParseInt(left, 10, 64)
	high, _ := strconv.ParseInt(right, 10, 64)
	count := 0
	// 为了保证 y^2 <= 10^18，y 最大值约为 10^9。我们枚举回文数根 y。
	// 采用构造回文数的方式：枚举前半部分 i
	// 注意：由于 i 的值决定了回文数的长度，对于 i 较大的情况生成的回文数已经会超过 10^9，
	// 因此当 y*y 超出 high 时可以提前结束循环。
	// 枚举上界取 100000 是一个经验值（因为 10^5 构造出的回文数一般可覆盖到 10^9 的范围）
	for i := 1; i < 100000; i++ {
		s := strconv.Itoa(i)
		// 构造奇数长度的回文数：将 s 与 s[0:len(s)-1] 的反转拼接
		oddStr := s + utils.ReverseStr(s[:len(s)-1])
		oddPalind, _ := strconv.ParseInt(oddStr, 10, 64)
		// 检查回文奇数的平方数是否在low,high区间内并且是否也是回文字符串
		oddSquare := oddPalind * oddPalind
		if oddSquare >= low && oddSquare < high && utils.CheckPalindrome(strconv.FormatInt(oddSquare, 10)) {
			count++
		}
		// 构造偶数长度的回文数：将 s 与 reverse(s) 拼接
		evenStr := s + utils.ReverseStr(s)
		// 检查回文偶数的平方数是否在low,high区间内并且是否也是回文字符串
		evenPalind, _ := strconv.ParseInt(evenStr, 10, 64)
		evenSquare := evenPalind * evenPalind
		if evenSquare >= low && evenSquare < high && utils.CheckPalindrome(strconv.FormatInt(evenSquare, 10)) {
			// 注意：当 s 长度大于1时，偶数和奇数构造可能生成相同的回文数
			// 为避免重复计数，我们只在不重复的情况下计数
			if len(s) == 1 || evenStr != oddStr {
				count++
			}
		}
	}
	return count
}

/*
补充题
36进制由0-9，a-z，共36个字符表示。
要求按照加法规则计算出任意两个36进制正整数的和，如1b + 2x = 48  （解释：47+105=152）
要求：不允许使用先将36进制数字整体转为10进制，相加后再转回为36进制的做法
*/

/*
36进制字符和数字的映射：

0-9 表示数字 0-9。
a-z 表示数字 10-35。
建立两个映射：一个用于将字符转为数值（例如 map['a'] = 10），另一个用于将数值转为字符（例如 map[10] = 'a'）。

逐位相加：
从两个字符串的最低位开始（即从右到左），逐位相加。
如果相加的结果超过36，记录进位值，进入下一轮计算。

处理进位：
如果最终仍有进位，需在结果前追加一位。

返回结果：
因为是从最低位开始计算，最后得到的结果需要反转。
*/

// charToValue: 将36进制字符映射为对应的数值
func charToValue(c byte) int {
	if c >= '0' && c <= '9' {
		return int(c - '0')
	}
	return int(c-'a') + 10
}

// valueToChar: 将数值映射为对应的36进制字符
func valueToChar(v int) byte {
	if v >= 0 && v <= 9 {
		return byte(v - '0')
	}
	return byte(v-10) + 'a'
}

func AddBase36(num1, num2 string) string {
	l1, l2 := len(num1)-1, len(num2)-1
	carry := 0
	res := strings.Builder{}
	for l1 >= 0 || l2 >= 0 || carry > 0 {
		sum := 0
		if l1 >= 0 {
			sum += charToValue(num1[l1])
			l1--
		}
		if l2 > 0 {
			sum += charToValue(num2[l2])
			l2--
		}
		// 计算当前位的和及进位
		sum += carry
		carry = sum / 36
		// 存储当前位的结果
		res.WriteByte(valueToChar(sum % 36))
	}
	// 由于结果是从低位开始存储的，需要反转
	result := []byte(res.String())
	for i, n := 0, len(result); i < n/2; i++ {
		result[i], result[n-1-i] = result[n-1-i], result[i]
	}
	return string(result)
}

/*
leetcode 395 至少有K个重复字符的最长子串
给你一个字符串 s 和一个整数 k ，请你找出 s 中的最长子串， 要求该子串中的每一字符出现次数都不少于 k 。返回这一子串的长度。
如果不存在这样的子字符串，则返回 0。

示例 1：
输入：s = "aaabb", k = 3
输出：3
解释：最长子串为 "aaa" ，其中 'a' 重复了 3 次。

示例 2：
输入：s = "ababbc", k = 2
输出：5
解释：最长子串为 "ababb" ，其中 'a' 重复了 2 次， 'b' 重复了 3 次。

提示：
1 <= s.length <= 104
s 仅由小写英文字母组成
1 <= k <= 105
*/

/*
思路: 递归+分治
我们可以用递归的方法来解决这个问题，通过分治策略有效地优化性能：

统计字符频率：
首先遍历字符串，统计每个字符的出现次数。

分治递归：
如果某个字符的出现次数小于k，这个字符无法出现在满足条件的子串中。因此，以该字符为分隔符，将字符串分成多个子串，分别递归处理每个子串。
如果所有字符的出现次数都大于等于k，则整个字符串即为满足条件的子串。

终止条件：
如果字符串长度小于k，直接返回 0，因为不可能有符合条件的子串。
这种方法的时间复杂度接近O(n⋅26)，其中n 是字符串长度，26 是英文字母的种类。
*/

func longestSubstring(s string, k int) int {
	n := len(s)
	// 边界条件，如果字符串长度为0或字符串长度小于k，返回0
	if n == 0 || n < k {
		return 0
	}
	// 特殊处理，如果k等于1，直接返回1
	if k == 1 {
		return n
	}
	// 统计字符频率
	freqMap := make(map[byte]int)
	for i := range s {
		freqMap[s[i]]++
	}
	// 遍历字符串，找到出现次数小于 k 的字符作为分隔符
	for i := range s {
		if freqMap[s[i]] < k {
			// 分隔字符串并递归处理每段子串
			l := longestSubstring(s[:i], k)
			r := longestSubstring(s[i+1:], k)
			// 返回左右子串中较大的结果
			return utils.Max(l, r)
		}
	}
	// 如果没有找到任何字符的频率小于 k，说明整个字符串满足条件
	return n
}

/*
leetcode 168 Excel表列名称
给你一个整数 columnNumber ，返回它在 Excel 表中相对应的列名称。

例如：
A -> 1
B -> 2
C -> 3
...
Z -> 26
AA -> 27
AB -> 28
...

示例 1：
输入：columnNumber = 1
输出："A"

示例 2：
输入：columnNumber = 28
输出："AB"

示例 3：
输入：columnNumber = 701
输出："ZY"

示例 4：
输入：columnNumber = 2147483647
输出："FXSHRXW"

提示：
1 <= columnNumber <= 231 - 1
*/

/*
Excel 的列编号与字母表的关系类似于 26 进制数：
每一列可以看作是基于 26 的数字，其中字母 A 表示 1，B 表示 2，...，Z 表示 26。
与传统 26 进制不同的是，Excel 的列编号是从 1 开始的，而不是从 0 开始。这需要特别处理：
例如：第 26 列是 Z，第 27 列是 AA，这意味着每次减去 1，使得编号对齐到从 0 开始的 26 进制表示。

转换步骤：
对 columnNumber 不断取模 26，得到当前位的字母。
将 columnNumber 减去当前位的值，然后除以 26，继续处理更高位。
直到 columnNumber 为 0，停止。
*/

func convertToTitle(columnNumber int) string {
	// 定义结果字符串
	res := ""
	for columnNumber > 0 {
		// Excel 列号是从 1 开始的，调整到 0 开始
		columnNumber--
		// 当前位的字符：'A' + 余数
		char := 'A' + (columnNumber % 26)
		// 字符加到结果前面
		res = string(rune(char)) + res
		// 减去当前位的值，并继续处理更高位
		columnNumber /= 26
	}
	return res
}

/*
leetcode 567 字符串的排列
给你两个字符串 s1 和 s2 ，写一个函数来判断 s2 是否包含 s1 的排列。如果是，返回 true ；否则，返回 false 。
换句话说，s1 的排列之一是 s2 的 子串 。

示例 1：
输入：s1 = "ab" s2 = "eidbaooo"
输出：true
解释：s2 包含 s1 的排列之一 ("ba").

示例 2：
输入：s1= "ab" s2 = "eidboaoo"
输出：false

提示：
1 <= s1.length, s2.length <= 104
s1 和 s2 仅包含小写字母
*/

/*
思路:滑动窗口
排列的特性：
s1 的排列意味着其字符的频次与 s2 中一个子串的字符频次完全相同。
也就是说，s1 的每一个字符都必须在 s2 的某个子串中出现相同的次数。

滑动窗口：
我们可以用滑动窗口的方式，遍历 s2 的所有子串，判断这些子串是否是 s1 的排列。
具体来说，我们在 s2 上维护一个窗口，其大小为 s1 的长度，然后将该窗口内字符的频次与 s1 的字符频次进行比较。
如果在某一时刻，窗口内的字符频次和 s1 的字符频次相同，就说明 s2 中包含了 s1 的一个排列。

具体步骤：
使用两个频率数组：一个记录 s1 中字符的频次，另一个记录当前滑动窗口内字符的频次。
初始时，比较 s1 和 s2 前 len(s1) 个字符的频次。
然后，通过滑动窗口的方式更新窗口内的字符频次，并与 s1 的字符频次进行比较。
如果在某个时刻，两者相等，则说明找到了一种排列，返回 true。
*/

func checkInclusion(s1 string, s2 string) bool {
	// 获取 s1 和 s2 的长度
	m, n := len(s1), len(s2)
	// 如果 s1 的长度大于 s2，肯定不可能包含排列
	if m > n {
		return false
	}
	// 频率数组
	need1, need2 := [26]int{}, [26]int{}
	// 初始化 s1 和 s2 的前 m 个字符的频率
	for i := 0; i < m; i++ {
		need1[s1[i]-'a']++
		need2[s2[i]-'a']++
	}
	// 比较频率数组
	if need1 == need2 {
		return true
	}
	// 滑动窗口，从 s2 的第 m 个字符开始滑动
	for i := m; i < n; i++ {
		// 当前窗口右移
		need2[s2[i]-'a']++
		// 移除当前窗口左边界字符
		need2[s2[i-m]-'a']--
		// 比较当前窗口的频率与 s1 的频率
		if need1 == need2 {
			return true
		}
	}
	return false
}

/*
对于给定的十六进制数，输出其对应的十进制表示。
在本题中，十六进制数的格式为：0x 开头，后跟若干个十六进制数字0-9 和A-F。其中，
A-F 依次代表10−15 。
*/

func hexToDecimal(hex string) int {
	hex = strings.ToUpper(hex[2:])
	n := len(hex)
	res := 0
	for i := 0; i < n; i++ {
		value := 0
		char := hex[n-1-i]
		if char >= '0' && char <= '9' {
			value = int(char - '0')
		} else {
			value = int(char-'A') + 10
		}
		res += value * int(math.Pow(16, float64(i)))
	}
	return res
}

/*
数据单元的变量替换
将一个csv格式的数据文件中包含有单元格引用的内容替换为对应单元格内容的实际值。
csv格式的数据文件使用逗号作为分隔符将各单位的内容进行分隔。

输入描述

1. 输入只有一行数据，用逗号分隔每个单元格，行尾没有逗号。最多26个单元格，对应编号A-Z。

2. 每个单元格的内容包含字母和数字，以及使用<>分隔的单元格引用，例如：<A>表示引用第一个单元的值。

3. 每个单元格的内容，在替换前和替换后均不超过100个字符。

5. 引用单元格的位置不受限制，运行排在后面的单元格被排在前面的单元格引用。

6. 不存在循环引用的情况

7. 不存在多重<>的情况，一个单元格只能引用一个其他单元格。

输出描述：

输出所有单元格展开的内容，单元格之间用逗号分隔。处理过程中出现错误时，输出字符串“-1”表示出错。

示例1

输入1, 2<A>00

输出1，2100

说明

第二个单元中有对A单元的引用，A单元格的值为1，替换时，将A单元的内容替代<A>的位置，并和其他内容合并。

示例2

输入
<B>12,1

输出

112，1

说明

第一个单元中有对B单元的引用，B单元格的值为1，替换时，将第二个数据单元的内容替代<B>的位置，并和其他内容合并。

示例3

输入

<B<12,1

输出

-1

说明
第一个单元中有错误的单元格引用方式，输出-1
*/

func ReplaceVariables(input string) string {
	cells := strings.Split(input, ",")
	// 遍历每个单元格
	for i, cell := range cells {
		// 查找单元格中是否存在引用标记"<"和">"
		l := strings.Index(cell, "<")
		r := strings.Index(cell, ">")
		// 如果找到 "<" 或 ">" 中只有一个，或者 "<" 在 ">" 后面，格式错误
		if (l == -1 && r != -1) || (r == -1 && l != -1) || l > r {
			return "-1"
		}
		// 如果存在引用，例如 "<A>"
		if l != -1 && r != -1 {
			// 截取引用的内容
			ref := cell[l+1 : r]
			// 引用必须为单个大写字母
			if len(ref) != 1 || !unicode.IsUpper(rune(ref[0])) {
				return "-1"
			}
			// 计算被引用单元格的索引，A 对应 0，B 对应 1，…… Z 对应 25
			refIndex := int(ref[0] - 'A')
			// 如果引用索引超出数组范围，说明引用不存在
			if refIndex < 0 || refIndex >= len(cells) {
				return "-1"
			}
			// 替换 "<X>" 为被引用单元格的内容
			// 例如单元格B: "2<A>00"，被替换为 "2" + cells[0] + "00"
			cells[i] = cell[:l] + cells[refIndex] + cell[r+1:]
		}
	}
	return strings.Join(cells, ",")
}

/*
字符串中所有整数的最小和
输入字符串s，输出s中包含所有整数的最小和。说明:字符串s，只包含a-z A-Z +-;合法的整数包括1正整数一个或者多个0-9组成，
如0230021022 负整数负号-开头，数字部分由一个或者多个0-9组成，如-0 -012 -23 -00023
输入描述包含数字的字符串输出描述所有整数的最小和
输入bb1234aa，输出10；
输入bb12-34aa，输出-31
*/

func minSumOfString(s string) int {
	n := len(s)
	minSum, i := 0, 0
	for i < n {
		// 跳过非数字和非符号字符
		// 如果当前字符不是 '+'、'-'、数字，则跳过
		if !unicode.IsDigit(rune(s[i])) && s[i] != '+' && s[i] != '-' {
			i++
			continue
		}
		// 判断是否遇到正负号
		sign := 1
		// 如果遇到 '-' 且后面是数字，则这是负数块
		if s[i] == '-' && i+1 < n && unicode.IsDigit(rune(s[i+1])) {
			sign = -1
			// 跳过符号
			i++
		} else if s[i] == '+' && i+1 < n && unicode.IsDigit(rune(s[i+1])) {
			sign = 1
			// 跳过符号
			i++
		}
		// 此时，s[i] 应该为数字，否则跳过（防止输入错误）
		if i >= n || !unicode.IsDigit(rune(s[i])) {
			i++
			continue
		}
		// 记录数字块的开始位置
		start := i
		// 取连续的数字
		for i < n && unicode.IsDigit(rune(s[i])) {
			i++
		}
		// 记录这个数字块
		block := s[start:i]
		if sign == 1 {
			// 正数块：为了最小和，将每个数字拆分成单个的数字
			for _, ch := range block {
				num, _ := strconv.Atoi(string(ch))
				minSum += num
			}
		} else {
			// 负数块：整体作为一个数字
			num, _ := strconv.Atoi(block)
			minSum -= num
		}
	}
	return minSum
}
