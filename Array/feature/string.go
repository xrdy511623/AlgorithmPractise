package feature

/*
剑指 Offer II 014. 字符串中的变位词
leetcode 567. 字符串的排列
1.1 给定两个字符串s1和s2，写一个函数来判断s2是否包含s1的某个变位词。
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