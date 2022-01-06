package feature

import (
	"AlgorithmPractise/Utils"
	"math"
	"strings"
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
func ReverseArray(nums []int){
	n := len(nums)
	for i := 0; i < n/2; i++ {
		temp := nums[n-1-i]
		nums[n-1-i] = nums[i]
		nums[i] = temp
	}
}

/*
1.2 找众数
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
你可以假设数组是非空的，并且给定的数组总是存在多数元素。
*/

/*
第一种思路，最简单的办法就是遍历数组，用哈希表记录数组中每个元素出现的次数，如果哪个元素的出现次数大于数组长度的一半，
那么这个元素就是众数，这样做时间复杂度O(n)，空间复杂度O(n/2),显然空间复杂度较高，不是最优解。
第二种思路:如果我们把众数的值记为+1，把其他数记为−1，将它们全部加起来，显然和大于0，从结果本身我们可以看出众数比其他数多。
我们维护一个候选众数candidate和它出现的次数count。初始时candidate可以为任意值，count为0；
我们遍历数组nums中的所有元素，对于每个元素x，在判断x之前，如果count的值为0，我们先将x的值赋予candidate，随后我们判断x：
如果x与candidate相等，那么计数器count的值增加1；
如果x与candidate不等，那么计数器count的值减少1。
在遍历完成后，candidate 即为整个数组的众数。为什么？因为非众数在遍历过程中一定会遇到出现次数比它多的众数，这样count值会被减到0，
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
1.3 删除有序数组中的重复项
给你一个有序数组nums，请你原地删除重复出现的元素，使每个元素只出现一次 ，返回删除后数组的新长度。
不要使用额外的数组空间，你必须在原地修改输入数组 并在使用O(1) 额外空间的条件下完成。
*/

/*
数组是有序的，那么重复的元素一定会相邻。在同一个数组里面操作，也就是不重复的元素移到数组的左侧，
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
1.4 进阶 移除元素
给你一个数组 nums和一个值val，你需要原地移除所有数值等于val的元素，并返回移除后数组的新长度。
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
	return slow
}

// 双指针法可以写得更简单一些
func RemoveElementSimple(nums []int, val int) int {
	index := 0
	// 此时i就是快指针fast
	for i := 0; i < len(nums); i++ {
		if nums[i] != val {
			nums[index] = nums[i]
			index++
		}
	}
	return index
}

/*
1.5 移动零
给定一个数组nums，编写一个函数将所有0移动到数组的末尾，同时保持非零元素的相对顺序。

示例:
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]

说明:
必须在原数组上操作，不能拷贝额外的数组。
尽量减少操作次数。
*/

/*
思路:先处理非0元素，最后处理0
index指针指向非0元素，count统计数组中零的个数，index的位置和count的数量初始值均为0，
在for循环遍历过程中，如果遇到非0元素，则将其赋值给nums[index]，同时index指针向右移动一位。
若遇到0，则count累加1。遍历结束后，index即指向所有非0元素最后一位的右边。意味着
nums[index:index+count]区间内的元素都应该是0，那就循环count次，将0赋值给nums[index]就可以了，
当然，每次赋值后index还得向后移动一位。
*/

func MoveZeroes(nums []int) {
	index, count := 0, 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[index] = nums[i]
			index++
		} else {
			count++
		}
	}
	for i := 0; i < count; i++ {
		nums[index] = 0
		index++
	}
}

/*
1.5 最长公共前缀
编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串""。

示例 1：
输入：strs = ["flower","flow","flight"]
输出："fl"

示例 2：
输入：strs = ["dog","racecar","car"]
输出：""
解释：输入不存在公共前缀。
*/

/*
依次遍历字符串数组中的每个字符串，对于每个遍历到的字符串，更新最长公共前缀，当遍历完所有的字符串以后，即可得到字符串数组中的最长公共前缀。
如果在尚未遍历完所有的字符串时，最长公共前缀已经是空串，则最长公共前缀一定是空串，因此不需要继续遍历剩下的字符串，直接返回空串即可。
复杂度分析
时间复杂度：O(mn)，其中m是字符串数组中的字符串的平均长度，n是字符串的数量。最坏情况下，字符串数组中的每个字符串的每个字符都会被比较一次。
空间复杂度：O(1)。使用的额外空间复杂度为常数。
*/

func LongestCommonPrefix(strs []string) string {
	count := len(strs)
	if count == 0 {
		return ""
	}
	prefix := strs[0]
	for i := 1; i < count; i++ {
		prefix = lcp(prefix, strs[i])
		if len(prefix) == 0 {
			break
		}
	}
	return prefix
}

func lcp(str1, str2 string) string {
	length := Utils.Min(len(str1), len(str2))
	index := 0
	for index < length && str1[index] == str2[index] {
		index++
	}
	return str1[:index]
}

/*
1.6 最长连续递增子序列
给定一个未经排序的整数数组，找到最长且连续递增的子序列，并返回该序列的长度。
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
		maxLength = Utils.Max(maxLength, i-start+1)
	}

	return maxLength
}

/*
1.7 最大子序列和
给定一个整数数组 nums，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

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
	max := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i-1] > 0 {
			nums[i] += nums[i-1]
		}
		if nums[i] > max {
			max = nums[i]
		}
	}
	return max
}

/*
1.8 长度最小的子数组
给定一个含有n个正整数的数组和一个正整数s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组，并返回其长度。如果不存在符合条件的子数组，返回 0。

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
窗口的结束位置如何移动：窗口的结束位置就是遍历数组的指针，窗口的起始位置设置为数组的起始位置就可以了。
动窗口的精妙之处在于根据当前子序列和大小的情况，不断调节子序列的起始位置。
子数组长度 length := end-start+1(窗口结束位置-窗口起始位置+1)
*/

// MinSubArrayLen 时间复杂度O(2N),空间复杂度O(1)
func MinSubArrayLen(target int, nums []int) int {
	// 和大于target的连续子数组的最小长度，初始值设为最大的int，便于后续迭代
	minLength := math.MaxInt32
	// 连续子数组之和，初始值为0
	sum := 0
	// 滑动窗口的起始位置,初始值0
	start := 0
	for end := 0; end < len(nums); end++ {
		sum += nums[end]
		for sum >= target {
			// 计算窗口大小，即满足和大于等于target的子数组长度
			length := end - start + 1
			minLength = Utils.Min(minLength, length)
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
1.9 反转字符串中的单词III
给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。

示例：
输入："Let's take LeetCode contest"
输出："s'teL ekat edoCteeL tsetnoc"
*/

/*
思路:以空格分割字符串s得到单词(字符串)集合，将每个单词反转后添加到结果集合中(字符串集合)
最后返回strings.Join(res, " ")即可
*/

func ReverseWords(s string) string {
	array := strings.Split(s, " ")
	var res []string
	for _, str := range array {
		bytes := []byte(str)
		n := len(bytes)
		for i := 0; i < n/2; i++ {
			temp := bytes[n-1-i]
			bytes[n-1-i] = bytes[i]
			bytes[i] = temp
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
		for i < n && s[i] == ' ' {
			i++
			res = append(res, ' ')
		}
	}
	return string(res)
}

/*
1.10 反转字符串II
给定一个字符串s和一个整数k，从字符串开头算起，每计数至2k个字符，就反转这2k字符中的前k个字符。
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
			reverse(bytes[i:i+k])
		} else {
			// 剩余字符少于k个，则将剩余字符全部反转。
			reverse(bytes[i:n])
		}
	}
	return string(bytes)
}

func reverse(s []byte) {
	n := len(s)
	for i := 0; i < n/2; i++ {
		temp := s[n-1-i]
		s[n-1-i] = s[i]
		s[i] = temp
	}
}

/*
1.11 替换空格
请实现一个函数，把字符串s中的每个空格替换成"%20"。

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
1.12 反转字符串里的单词
给你一个字符串s，逐个翻转字符串中的所有单词 。
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
	for length > 0 && fast < length && ss[fast] == ' ' {
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
	// 去掉字符串最右边的冗余空格
	if slow > 1 && ss[slow-1] == ' ' {
		ss = ss[:slow-1]
	} else {
		ss = ss[:slow]
	}
	// 反转整个字符串
	reverse(ss)
	i := 0
	for i < len(ss) {
		// 反转单个单词
		j := i
		// 找到单词的结束位置
		for ; j < len(ss) && ss[j] != ' '; j++ {
			continue
		}
		// 反转
		reverse(ss[i:j])
		// 更新下一个单词的起始位置，+1是要跳过单词间的空格
		i = j + 1
	}
	return string(ss)
}

/*
1.13 左旋转字符串
字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。
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
思路一:将前n个元素添加到tmp集合中，然后在ss[n:length]集合后边依次添加tmp集合中的元素即可。
*/

//  ReverseLeftWords 时间复杂度O(N),空间复杂度O(k)
func ReverseLeftWords(s string, k int) string {
	ss := []byte(s)
	length := len(ss)
	var tmp []byte
	for i := 0; i < k; i++ {
		tmp = append(tmp, ss[i])
	}
	ss = append(ss[k:length], tmp...)
	return string(ss)
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
1.14  实现strStr()
给你两个字符串haystack和needle ，请你在haystack字符串中找出needle字符串出现的第一个位置（下标从0开始）。
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

// strStrSimple KMP算法 时间复杂度O(N+M),空间复杂度O(M)
func strStrSimple(haystack string, needle string) int {
	n, m := len(haystack), len(needle)
	if m == 0 {
		return 0
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
		// 如果j从0移动到m的位置，意味着模式串与文本串匹配成功
		if j == m {
			return i - m + 1
		}
	}
	return -1
}

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
1.15 重复的子字符串
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