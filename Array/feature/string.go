package feature

import (
	"AlgorithmPractise/Utils"
	"strconv"
	"strings"
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
			start = Utils.Max(start, pos+1)
		}
		visited[s[i]] = i
		maxLength = Utils.Max(maxLength, i-start+1)
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
