package Medium

import "AlgorithmPractise/Utils"

/*
1.1 最长递增子序列
给你一个整数数组nums ，找到其中最长严格递增子序列的长度。
子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7]
的子序列。

示例1：
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为4 。

示例2：
输入：nums = [0,1,0,3,2,3]
输出：4

示例3：
输入：nums = [7,7,7,7,7,7,7]
输出：1
*/

/*
动态规划解法:

1 dp数组及其下标含义
dp[i]表示在数组nums[0,i]范围内的最长递增子序列的长度

2 确定递推公式
位置i的最长递增子序列长度等于j从0到i-1各个位置的最长递增子序列+1的最大值
也就是，如果nums[i] > nums[j] && j:=0;j<i;j++ 则有dp[i] = max(dp[i], dp[j] + 1);

3 初始化dp数组
每一个i，对应的dp[i]（即最长递增子序列）起始大小至少都是是1.

4 确定遍历顺序
dp[i]是由0到i-1各个位置的最长递增子序列推导而来，那么遍历i一定是从前向后遍历。
j其实就是0到i-1，遍历i的循环在外层，遍历j则在内层，

5 举例推导dp数组
参见最长连续递增子序列.png
*/

// LengthOfLTS 时间复杂度O(N^2),空间复杂度O(N)
func LengthOfLTS(nums []int) int {
	n := len(nums)
	if n <= 1 {
		return n
	}
	dp := make([]int, n)
	// 初始化dp数组
	for i := 0; i < n; i++ {
		dp[i] = 1
	}
	maxLength := 0
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			if nums[i] > nums[j] {
				dp[i] = Utils.Max(dp[i], dp[j]+1)
			}
		}
		if dp[i] > maxLength {
			maxLength = dp[i]
		}
	}
	return maxLength
}

/*
思路:贪心+二分查找
考虑一个简单的贪心，如果我们要使上升子序列尽可能的长，则我们需要让序列上升得尽可能慢，因此我们希望每次在上升子序列最后加上的那个数尽可能的小。

基于上面的贪心思路，我们维护一个数组d[i] ，表示长度为i的最长上升子序列的末尾元素的最小值，用len记录目前最长上升子序列的长度，起始时len为1，
d[1] =nums[0]。
同时我们可以注意到d[i]是关于i单调递增的。
根据d数组的单调性，我们可以使用二分查找寻找下标i，优化时间复杂度。

最后整个算法流程为：
设当前已求出的最长上升子序列的长度为len（初始时为1），从前往后遍历数组nums，在遍历到nums[i]时：
如果nums[i]>d[len]，则直接加入到d数组末尾，并更新len=len+1；
否则，在d数组中二分查找，找到第一个比nums[i]小的数d[k]，并更新d[k+1]=nums[i]。

以输入序列 [0, 8, 4, 12, 2][0,8,4,12,2] 为例：
第一步插入0，d=[0]；
第二步插入8，d=[0,8]；
第三步插入4，d=[0,4]；
第四步插入12，d=[0,4,12]；
第五步插入2，d=[0,2,12]。
最终得到最大递增子序列长度为len(d)=3。
*/

// LengthOfLTSSimple  时间复杂度O(NlogN),空间复杂度O(N)
func LengthOfLTSSimple(nums []int) int {
	n := len(nums)
	if n <= 1 {
		return n
	}
	var res []int
	for _, num := range nums {
		if len(res) == 0 || num > res[len(res)-1] {
			res = append(res, num)
		} else {
			pos := FindFirstLessNum(res, num)
			if pos == -1 {
				// 如果找不到res数组中所有元素都比num大，此时更新res[0] = num
				res[0] = num
			} else {
				// 否则更新res[pos+1]=num
				res[pos+1] = num
			}
		}
	}
	return len(res)
}

// FindFirstLessNum 在连续递增的有序数组中寻找第一个小于target的元素的位置
func FindFirstLessNum(nums []int, target int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	if target < nums[0] {
		return -1
	}
	if target > nums[n-1] {
		return n - 1
	}
	l, r := 0, n-1
	for l <= r {
		mid := (l + r) / 2
		if nums[mid] > target {
			r = mid - 1
			// 由于nums是连续递增的有序数组，所以此时mid-1就是第一个小于target的元素
		} else if nums[mid] == target {
			return mid - 1
		} else {
			// 此时nums[mid]<target<=nums[mid+1],所以mid就是我们要找的位置
			if nums[mid+1] >= target {
				return mid
				// 此时nums[mid+1]<target,说明mid不是第一个小于target的元素，得在[mid+1, r]范围继续寻找
			} else {
				l = mid + 1
			}
		}
	}
	return -1
}

/*
1.2 最长连续递增序列
给定一个未经排序的整数数组，找到最长且连续递增的子序列，并返回该序列的长度。
连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个l <= i < r，都有nums[i] < nums[i + 1] ，那么子序列
[nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。

示例1：
输入：nums = [1,3,5,4,7] 输出：3 解释：最长连续递增序列是 [1,3,5], 长度为3。 尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，
因为 5 和 7 在原数组里被 4 隔开。

示例2：
输入：nums = [2,2,2,2,2] 输出：1 解释：最长连续递增序列是 [2], 长度为1。

提示：
0 <= nums.length <= 10^4
-10^9 <= nums[i] <= 10^9
*/

/*
思路:动态规划

1 确定dp数组（dp table）以及下标的含义
dp[i]：以下标i为结尾的数组的连续递增的子序列长度为dp[i]。
注意这里的定义，一定是以下标i为结尾，并不是说一定以下标0为起始位置。

2 确定递推公式
如果 nums[i] > nums[i-1]，那么以i为结尾的数组的连续递增的子序列长度 一定等于以i-1为结尾的数组的连续递增的子序列长度+1 。
即：dp[i] = dp[i-1]+1;

3 dp数组如何初始化
以下标i为结尾的数组的连续递增的子序列长度最少也应该是1，即就是nums[i]这一个元素。
所以dp[i]应该初始1;

4 确定遍历顺序
从递推公式上可以看出， dp[i]依赖dp[i-1]，所以一定是从前向后遍历。

5 举例推导dp数组
以输入nums = [1,3,5,4,7]为例，dp数组状态如下：
下标     0 1 2 3 4
length  1 2 3 1 2
所以返回3
*/

// FindLengthOfLCIS 动态规划解决，时间复杂度O(N),空间复杂度O(N)
func FindLengthOfLCIS(nums []int) int {
	n := len(nums)
	if n <= 1 {
		return n
	}
	dp := make([]int, n)
	for i := 0; i < n; i++ {
		dp[i] = 1
	}
	maxLength := 0
	for i := 1; i < n; i++ {
		if nums[i] > nums[i-1] {
			dp[i] = dp[i-1] + 1
		}
		if maxLength < dp[i] {
			maxLength = dp[i]
		}
	}
	return maxLength
}

/*
思路:
这道题目也可以用贪心来做，也就是遇到nums[i] > nums[i-1]的情况，count就++，否则count为1，记录count的最大值就可以了。
*/

// FindLengthOfLCISSimple 贪心解决，时间复杂度O(N),空间复杂度O(1)
func FindLengthOfLCISSimple(nums []int) int {
	n := len(nums)
	if n <= 1 {
		return n
	}
	maxLength, count := 1, 1
	for i := 1; i < n; i++ {
		if nums[i] > nums[i-1] {
			count++
		} else {
			count = 1
		}
		if maxLength < count {
			maxLength = count
		}
	}
	return maxLength
}

/*
1.3 最长重复子数组
给两个整数数组A和B ，返回两个数组中公共的、长度最长的子数组的长度。

示例：
输入： A: [1,2,3,2,1] B: [3,2,1,4,7]
输出：3 解释：长度最长的公共子数组是 [3, 2, 1] 。

提示：
1 <= len(A), len(B) <= 1000
0 <= A[i], B[i] < 100
*/

/*
思路:动态规划

1 确定dp数组及其下标含义
dp[i][j]表示数组A[:i]和数组B[:j]的最长重复子数组的长度

2 确定递推公式
对于数组A[:i]和数组B[:j]，如果有A[i-1]==B[j-1],那么数组A[:i]和数组B[:j]的最长重复子数组的长度就等于数组A[:i-1]和数组B[:j-1]
的最长重复子数组的长度+1，也就是dp[i][j]=dp[i-1][j-1]+1
所以，递推公式为dp[i][j]=dp[i-1][j-1]+1

3 初始化dp数组
为了方便递推公式dp[i][j] = dp[i-1][j-1] + 1;
所以dp[i][0]和dp[0][j]初始化为0。

4 确定遍历顺序
由递推公式可知，dp[i][j]根据dp[i-1][j-1]得来，所以i和j都是正序遍历

5 举例推导dp数组
略
*/

// FindLongestLengthOfCSS 时间复杂度O(M*N)，空间复杂度O(M*N)
func FindLongestLengthOfCSS(nums1, nums2 []int) int {
	m, n := len(nums1), len(nums2)
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	maxLength := 0
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if nums1[i-1] == nums2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			}
			if maxLength < dp[i][j] {
				maxLength = dp[i][j]
			}
		}
	}
	return maxLength
}

/*
1.4 最长公共子序列
给定两个字符串text1和text2，返回这两个字符串的最长公共子序列的长度。
一个字符串的子序列是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。

若这两个字符串没有公共子序列，则返回0。

示例1:
输入：text1 = "abcde", text2 = "ace"
输出：3
解释：最长公共子序列是 "ace"，它的长度为 3。

示例2:
输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc"，它的长度为 3。

示例 3:
输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0。

提示:
1 <= text1.length <= 1000
1 <= text2.length <= 1000 输入的字符串只含有小写英文字符。
*/

/*
思路:动态规划
本题与1.3 最长重复子数组最大的不同是公共子序列不要求是连续的了。

1 确定dp数组及其下标含义
dp[i][j]表示字符串A[:i]字符串B[:j]的最长公共子序列的长度

2 确定递推公式
对于字符串A[:i]和字符串B[:j]，如果A[i-1]==B[j-1],那么字符串A[:i]和字符串B[:j]的最长公共子序列的长度就等于字符串A[:i-1]和字符串B[:j-1]
的最长公共子序列的长度+1，也就是dp[i][j]=dp[i-1][j-1]+1
如果A[i-1]!=B[:j-1], 那就看字符串A[:i-1]和字符串B[:j]的最长公共子序列的长度dp[i-1][j]以及字符串A[:i]和字符串B[:j-1]的最长公共子序列的长度
dp[i][j-1]哪个更大，取较大值。
所以，递推公式为
当A[i-1]==B[j-1]时，
dp[i][j]=dp[i-1][j-1]+1
否则:
dp[i][j]=max(dp[i-1][j],dp[i][j-1])

3 初始化dp数组
dp[i][0]是0，因为字符串A[:i]和空字符串的最长公共子序列的长度为0，同理，dp[0][j]也是0

4 确定遍历顺序
由递推公式可知，dp[i][j]根据dp[i-1][j-1]或dp[i-1][j],dp[i][j-1]得来，所以i和j都是正序遍历

5 举例推导dp数组
略
*/

// LongestCommonSubSequence 时间复杂度O(M*N)，空间复杂度O(M*N)
func LongestCommonSubSequence(text1, text2 string) int {
	m, n := len(text1), len(text2)
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = Utils.Max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[m][n]
}

/*
1.5 不相交的线
leetcode 1035
我们在两条独立的水平线上按给定的顺序写下A和B中的整数。
现在，我们可以绘制一些连接两个数字A[i]和B[j]的直线，只要A[i] == B[j]，且我们绘制的直线不与任何其他连线（非水平线）相交。
以这种方法绘制线条，并返回我们可以绘制的最大连线数。

示例:
输入：nums1 = [1,4,2], nums2 = [1,2,4]
输出：2
解释：可以画出两条不交叉的线，如图不相交线路图.png所示。
但无法画出第三条不相交的直线，因为从 nums1[1]=4 到 nums2[2]=4 的直线将与从 nums1[2]=2 到 nums2[1]=2 的直线相交。
*/

/*
思路:本题与1.4 最长公共子序列实质上是一模一样的。
绘制一些连接两个数字 A[i] 和 B[j] 的直线，只要 A[i] == B[j]，且直线不能相交！
直线不能相交，这就是说明在字符串A中 找到一个与字符串B相同的子序列，且这个子序列不能改变相对顺序，只要相对顺序不改变，链接相同数字的直线就不会相交。

其实也就是说A和B的最长公共子序列是[1,4]，长度为2。 这个公共子序列指的是相对顺序不变（即数字4在字符串A中数字1的后面，那么数字4也应该在字符串B数字1
的后面）

这么分析完之后，可以发现本题说是求绘制的最大连线数，其实就是求两个字符串的最长公共子序列的长度！
那么本题就和1.4 最长公共子序列实质上是一模一样的了。
*/

// MaxUncrossLines 时间复杂度O(M*N)，空间复杂度O(M*N)
func MaxUncrossLines(nums1, nums2 []int) int {
	m, n := len(nums1), len(nums2)
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if nums1[i-1] == nums2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = Utils.Max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[m][n]
}

/*
1.6 最大子序和
给定一个整数数组nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
示例:
输入: [-2,1,-3,4,-1,2,1,-5,4] 输出: 6 解释: 连续子数组 [4,-1,2,1]的和最大，为6。
*/

/*
思路:动态规划
1 确定dp数组及其下标含义
dp[i]表示数组nums[:i]的最大子数组和

2 确定递推公式
显然，dp[i]的值取决于dp[i-1]和nums[i]。
当nums[i]加入当前连续子序列时，dp[i]=dp[i-1]+nums[i]
从头开始，即从nums[i]开始计算当前连续子序列，dp[i]=nums[i]
所以递推公式为dp[i] = max(dp[i-1]+nums[i], nums[i])

3 初始化dp数组
很明显，dp[0]=nums[0]

4 确定遍历顺序
由递推公式可知，dp[i]根据dp[i-1]得来，所以是正序遍历

5 举例推导dp数组
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]，对应的dp状态如下：
下标    0   1  2  3  4  5  6  7  8
dp[i]  -2  1 -2  4  3  5  6  1  5
*/

// MaxSubArray 时间复杂度O(N)，空间复杂度O(N)
func MaxSubArray(nums []int) int {
	n := len(nums)
	dp := make([]int, n)
	dp[0] = nums[0]
	max := nums[0]
	for i := 1; i < n; i++ {
		dp[i] = Utils.Max(dp[i-1]+nums[i], nums[i])
		if max < dp[i] {
			max = dp[i]
		}
	}
	return max
}

// MaxSubArraySimple 更简单的写法是下面这样 时间复杂度O(N)，空间复杂度O(1)
func MaxSubArraySimple(nums []int) int {
	max := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i]+nums[i-1] > nums[i] {
			nums[i] += nums[i-1]
		}
		if max < nums[i] {
			max = nums[i]
		}
	}
	return max
}

/*
1.7 判断子序列
给定字符串s和t ，判断s是否为t的子序列。
字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串(例如,"ace"是"abcde"的一个子序列，而"aec"不是）

示例1：
输入：s = "abc", t = "ahbgdc" 输出：true

示例2：
输入：s = "axc", t = "ahbgdc" 输出：false

提示：
0 <= s.length <= 100
0 <= t.length <= 10^4
两个字符串都只由小写字符组成。
*/

/*
思路:动态规划
本题与1.4 最长公共子序列实际上有许多相似之处

1 确定dp数组以及下标的含义
dp[i][j] 表示字符串s[:i](不包含i)，和字符串t[:j](不包含j)，相同子序列的长度为dp[i][j]。
注意这里是判断s是否为t的子序列。即t的长度是大于等于s的。

2 确定递推公式
在确定递推公式的时候，首先要考虑如下两种操作，整理如下：
如果s[i-1] == t[j-1])
t中找到了一个字符在s中也出现了，此时dp[i][j] = dp[i-1][j-1] + 1
如果s[i-1] != t[j-1])
相当于t要删除元素，t如果把当前元素t[j-1]删除，那么dp[i][j]的数值就是看s[i-1]与t[j-2]的比较结果了，
即：dp[i][j] = dp[i][j-1];

3 dp数组如何初始化
从递推公式可以看出dp[i][j]都是依赖于dp[i-1][j-1]和dp[i][j-1]，所以dp[0][0]和dp[i][0]是一定要初始化的。
根据dp数组定义，显然dp[i][0]和dp[j][0]都是0

4 确定遍历顺序
同理从从递推公式可以看出dp[i][j]都是依赖于dp[i-1][j-1] 和 dp[i][j-1]，那么遍历顺序也应该是从左到右

5 举例推导dp数组
略
*/

// IsSubSequence 时间复杂度O(M*N)，空间复杂度O(M*N)
func IsSubSequence(s, t string) bool {
	m, n := len(s), len(t)
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}

	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if s[i-1] == t[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = dp[i][j-1]
			}
		}
	}
	return dp[m][n] == len(s)
}

/*
思路:双指针法
本题询问的是，s是否是t的子序列，因此只要能找到任意一种ss在t中出现的方式，即可认为s是t的子序列。
而当我们从前往后匹配，可以发现每次贪心地匹配靠前的字符是最优决策。

假定当前需要匹配字符c，而字符c在t中的位置x1和x2出现（x1 < x2），那么贪心取x1是最优解，因为x2后面能取到的字符，x1也都能取到，并且通过x1
与x2之间的可选字符，更有希望能匹配成功。

这样，我们初始化两个指针i和j，分别指向s和t的初始位置。每次贪心地匹配，匹配成功则i和j同时右移，匹配s的下一个位置，匹配失败则j右移，i不变，尝试用t
的下一个字符匹配s。
最终如果i移动到s的末尾，就说明s是t的子序列。
*/

// IsSubSequenceSimple 时间复杂度O(N)，空间复杂度O(1)
func IsSubSequenceSimple(s, t string) bool {
	m, n := len(s), len(t)
	i, j := 0, 0
	for i < m && j < n {
		if s[i] == t[j] {
			i++
		}
		j++
	}
	return i == m
}

/*
1.8 不同的子序列
给定一个字符串s和一个字符串t ，计算在s的子序列中t出现的个数。
字符串的一个子序列是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如,"ACE"是"ABCDE"的一个子序列,而"AEC" 不是）

题目数据保证答案符合32位带符号整数范围。

示例1：
输入：s = "rabbbit", t = "rabbit"
输出：3
解释：有3种可以从s中得到"rabbit"的方案。

示例2：
输入：s = "babgbag", t = "bag"
输出：5
解释：有5种可以从s中得到"bag"的方案。
*/

/*
思路:动态规划

1 确定dp数组以及下标的含义
dp[i][j]：s[:i]中出现t[:j]的个数为dp[i][j]。

2 确定递推公式
这一类问题，基本是要分析两种情况
s[i-1]与t[j-1]相等以及s[i-1]与t[j-1]不相等
当s[i-1]与t[j-1]相等时，dp[i][j]可以由两部分组成。
一部分是用s[i-1]来匹配，那么个数为dp[i-1][j-1];一部分是不用s[i-1]来匹配，个数为dp[i-1][j]。
所以当s[i-1]与t[j-1]相等时，dp[i][j]=dp[i-1][j-1]+dp[i-1][j];

当s[i-1]与t[j-1]不相等时，dp[i][j]只由一部分组成，不用s[i-1]来匹配，即：dp[i-1][j]
所以递推公式为：dp[i][j] = dp[i-1][j];

3 dp数组如何初始化
从递推公式dp[i][j]=dp[i-1][j-1]+dp[i-1][j]; 和dp[i][j]=dp[i-1][j]; 中可以看出dp[i][0] 和dp[0][j]是一定要初始化的。

首先看dp[i][0]，dp[i][0]一定都是1，因为空字符串是任何字符串的子串。
再来看dp[0][j]，dp[0][j]一定都是0，非空字符串t无论如何也不可能出现在空字符串中。

最后就要看一个特殊位置了，即：dp[0][0]应该是多少。
dp[0][0]应该是1，空字符串s，可以删除0个元素，变成空字符串t。

4 确定遍历顺序
从递推公式中可以看出dp[i][j]都是根据左上方和正上方推出来的。
所以遍历的时候一定是从上到下，从左到右，这样保证dp[i][j]可以根据之前计算出来的数值进行计算。

5 举例推导dp数组
略
*/

// NumDistinct 时间复杂度O(M*N)，空间复杂度O(M*N)
func NumDistinct(s, t string) int {
	m, n := len(s), len(t)
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
		dp[i][0] = 1
	}
	for j := 1; j <= n; j++ {
		dp[0][j] = 0
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if s[i-1] == t[j-1] {
				dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
			} else {
				dp[i][j] = dp[i-1][j]
			}
		}
	}
	return dp[m][n]
}

/*
1.9 两个字符串的删除操作
给定两个单词word1和word2，找到使得word1和word2相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。

示例：
输入: "sea", "eat"
输出: 2 解释: 第一步将"sea"变为"ea"，第二步将"eat"变为"ea"
*/

/*
思路:动态规划

1 确定dp数组及其下标含义
dp[i][j]表示字符串word1[:i]和字符串word2[:j]要达到相同，所需要删除元素的最小次数(最小步数)

2 确定递推公式
dp[i][j]的值取决于word1[i-1]与word1[j-1]是否相等

如果word1[i-1]==word1[j-1]，则dp[i][j]=dp[i-1][j-1](两个字符串末尾元素相同，
所以dp[i][j]即为word1[:i-1]和word2[:j-1]相同的最小步数)

如果word1[i-1]!=word1[j-1],则dp[i][j]可以由以下三种情况推出:
一个是删除word1[i-1],此时dp[i][j]=dp[i-1][j]+1
再一个是删除word2[j-1],此时dp[i][j]=dp[i][j-1]+1
最后是同时删除word1[i-1]和word2[j-1],dp[i][j]=dp[i-1][j-1]+2
因为是求解最小步数，所以综上，此时dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+2)

3 初始化dp数组
从递推公式可知，我们需要初始化dp[i][0]和dp[0][j]，首先来看dp[i][0]，根据dp数组定义，要使得word1[:i]和空字符串达到相等，显然需要i步，
于是有dp[i][0]=i，同理可得dp[0][j]=j

4 确定遍历顺序
从递推公式可知，应该是从左到右顺序遍历

5 举例推导dp数组
略
*/

// MinDistance 时间复杂度O(M*N)，空间复杂度O(M*N)
func MinDistance(word1, word2 string) int {
	m, n := len(word1), len(word2)
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
		dp[i][0] = i
	}
	for j := 0; j <= n; j++ {
		dp[0][j] = j
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = Utils.Min(dp[i-1][j-1]+2, Utils.Min(dp[i-1][j]+1, dp[i][j-1]+1))
			}
		}
	}
	return dp[m][n]
}

/*
1.10 给你两个单词word1和word2，请你计算出将word1转换成word2所使用的最少操作数。

你可以对一个单词进行如下三种操作：
插入一个字符
删除一个字符
替换一个字符

示例1：
输入：word1 = "horse", word2 = "ros" 输出：3 解释：horse -> rorse (将'h'替换为'r') rorse -> rose (删除 'r') rose -> ros
(删除 'e')

示例2：
输入：word1 = "intention", word2 = "execution" 输出：5 解释：intention -> inention (删除 't') inention -> enention
(将'i'替换为'e') enention -> exention (将'n'替换为'x') exention -> exection (将'n'替换为'c') exection -> execution (插入'u')

提示：
0 <= word1.length, word2.length <= 500
word1和word2由小写英文字母组成
*/

/*
思路:动态规划
1 确定dp数组及其下标含义
dp[i][j]表示将字符串word1[:i]转换为字符串word2[:j]所使用的最小操作数(最小步数)

2 确定递推公式
dp[i][j]的值取决于word1[i-1]与word1[j-1]是否相等

如果word1[i-1]==word1[j-1]，则dp[i][j]=dp[i-1][j-1](两个字符串末尾元素已然相同，所以dp[i][j]即为word1[:i-1]转换为
字符串word2[:j-1]的最小步数)

如果word1[i-1]!=word1[j-1],则dp[i][j]可以由以下三种情况推出:
一个是删除word1[i-1],此时dp[i][j]=dp[i-1][j]+1
再一个是删除word2[j-1],此时dp[i][j]=dp[i][j-1]+1
最后是替换，可以将word1[i-1]替换为word2[j-1],也可以反过来将word2[j-1]替换为ord1[i-1]，此时dp[i][j]=dp[i-1][j-1]+1
因为是求解最小步数，所以综上，此时dp[i][j]=min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1

3 初始化dp数组
从递推公式可知，我们需要初始化dp[i][0]和dp[0][j]，首先来看dp[i][0]，根据dp数组定义，要将字符串word1[:i]转换为空字符串，显然需要i次删除，
于是有dp[i][0]=i，同理可得dp[0][j]=j

4 确定遍历顺序
从递推公式可知，应该是从左到右顺序遍历

5 举例推导dp数组
略
*/

// MinDistanceComplex 时间复杂度O(M*N)，空间复杂度O(M*N)
func MinDistanceComplex(word1, word2 string) int {
	m, n := len(word1), len(word2)
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
		dp[i][0] = i
	}
	for j := 0; j <= n; j++ {
		dp[0][j] = j
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = Utils.Min(dp[i-1][j-1], Utils.Min(dp[i-1][j], dp[i][j-1])) + 1
			}
		}
	}
	return dp[m][n]
}