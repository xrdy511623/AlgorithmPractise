package SplitProblem

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
