package split

import "strconv"

/*
leetcode 131. 分割回文串
1.1 给定一个字符串s，将s分割成一些子串，使每个子串都是回文串。
返回s所有可能的分割方案。

示例:
输入: "aab" 输出: [ ["aa","b"], ["a","a","b"] ]
*/

/*
本题难点
知道切割问题可以抽象为组合问题，这个画图就一目了然了，这其实是一棵N叉树。
知道如何模拟那些切割线，起始下标start一直向右移动。
知道切割问题中递归如何终止，当start移动到字符串末尾时。
知道在递归循环中如何截取子串，其实就是s[start:i+1]。
知道如何判断回文,这个用双指针法解决。
剪枝:如果s[start:i+1]不是回文子串，就可以跳过本次循环了。
*/

func PartitionSubString(s string) [][]string {
	var res [][]string
	var path []string
	size := len(s)
	var backTrack func(int)
	backTrack = func(start int) {
		// 递归终止条件
		// 如果切割线，也就是start移动到了字符串s末尾之后，说明我们找到了一个符合要求的切割方案
		if start == size {
			temp := make([]string, len(path))
			copy(temp, path)
			res = append(res, temp)
			return
		}
		for i := start; i < size; i++ {
			// 如果s[start:i+1]不是回文子串，就可以跳过本次循环了
			if IsPalindrome(s, start, i) {
				path = append(path, s[start:i+1])
			} else {
				continue
			}
			// 递归
			backTrack(i + 1)
			// 回溯
			path = path[:len(path)-1]
		}
	}
	backTrack(0)
	return res
}

// IsPalindrome 双指针法判断s[start:end+1]是否是回文字符串
func IsPalindrome(s string, start, end int) bool {
	for start < end {
		if s[start] != s[end] {
			return false
		}
		start++
		end--
	}
	return true
}

/*
leetcode 680 验证回文串II
给你一个字符串 s，最多 可以从中删除一个字符。
请你判断 s 是否能成为回文字符串：如果能，返回 true ；否则，返回 false 。

示例 1：
输入：s = "aba"
输出：true

示例 2：
输入：s = "abca"
输出：true
解释：你可以删除字符 'c' 。

示例 3：
输入：s = "abc"
输出：false

提示：
1 <= s.length <= 105
s 由小写英文字母组成
*/

/*
思路:双指针
回文的定义：从前向后和从后向前读都是一样的字符串。
双指针法：
使用两个指针 left 和 right，分别指向字符串的头和尾。
比较 s[left] 和 s[right]：
如果相等，两个指针继续向中间移动；
如果不相等，可以选择删除一个字符：
删除左边字符：验证子字符串 s[left+1:right+1] 是否为回文；
删除右边字符：验证子字符串 s[left:right] 是否为回文。
如果上述两种情况都不满足，返回 false。
子函数验证回文：
编写一个辅助函数判断字符串是否为严格回文，继续使用双指针。
*/

func validPalindrome(s string) bool {
	l, r := 0, len(s)-1
	for l < r {
		if !IsPalindrome(s, l, r) {
			return IsPalindrome(s, l+1, r) || IsPalindrome(s, l, r-1)
		}
		l++
		r--
	}
	return true
}

/*
leetcode 93. 复原IP地址
1.2 给定一个只包含数字的字符串，复原它并返回所有可能的IP地址格式。
有效的IP地址正好由四个整数（每个整数位于0到255之间组成，且不能含有前导0），整数之间用'.'分隔。
例如："0.1.2.201" 和 "192.168.1.1" 是有效的IP地址，但是"0.011.255.245"、"192.168.1.312"
和 "192.168@1.1" 是 无效的IP地址。

示例1：
输入：s = "25525511135"
输出：["255.255.11.135","255.255.111.35"]

示例2：
输入：s = "0000"
输出：["0.0.0.0"]

示例3：
输入：s = "1111"
输出：["1.1.1.1"]

示例4：
输入：s = "010010"
输出：["0.10.0.10","0.100.1.0"]

示例5：
输入：s = "101023"
输出：["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]

提示：
0 <= s.length <= 3000
s 仅由数字组成
*/

func RestoreIpAddresses(s string) []string {
	var res []string
	var path []string
	size := len(s)
	if size < 4 {
		return res
	}
	var backTrack func(int)
	backTrack = func(start int) {
		// 剪枝，若path长度为4段但start未移动到s的末尾位置或者path长度大于4
		if (len(path) == 4 && start < size) || len(path) > 4 {
			return
		}
		// 递归终止条件
		if len(path) == 4 {
			ip := path[0] + "." + path[1] + "." + path[2] + "." + path[3]
			res = append(res, ip)
			return
		}
		for i := start; i < size; i++ {
			// 剪枝，path中的每一个子串长度都不能超过3，而且当前path的长度必须小于4
			// 同时添加到path中的子串必须是合法的IP字符串
			if i-start+1 <= 3 && len(path) < 4 && IsValidIP(s[start:i+1]) {
				path = append(path, s[start:i+1])
			} else {
				// 否则直接结束运行,不需要继续深入递归了。
				return
			}
			// 递归
			backTrack(i + 1)
			// 回溯
			path = path[:len(path)-1]
		}
	}
	backTrack(0)
	return res
}

// IsValidIP 检查字符串s是否是合法IP
func IsValidIP(s string) bool {
	val, _ := strconv.Atoi(s)
	if len(s) > 1 && s[0] == '0' {
		return false
	}
	if val > 255 {
		return false
	}
	return true
}
