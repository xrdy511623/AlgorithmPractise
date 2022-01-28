package Medium

import (
	"AlgorithmPractise/Utils"
)

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
考虑一个简单的贪心，如果我们要使上升子序列尽可能的长，则我们需要让序列上升得尽可能慢，因此我们希望每次在上升
子序列最后加上的那个数尽可能的小。

基于上面的贪心思路，我们维护一个数组d[i] ，表示长度为i的最长上升子序列的末尾元素的最小值，用len记录目前最长
上升子序列的长度，起始时len为1，d[1] =nums[0]。
同时我们可以注意到d[i]是关于i单调递增的。
根据d数组的单调性，我们可以使用二分查找寻找下标i，优化时间复杂度。

最后整个算法流程为：
设当前已求出的最长上升子序列的长度为len（初始时为1），从前往后遍历数组nums，在遍历到nums[i]时：
如果nums[i]>d[len]，则直接加入到d数组末尾，并更新len=len+1；
否则，在d数组中二分查找，找到第一个比nums[i]小的数d[k]，并更新d[k+1]=nums[i]。

以输入序列[0,8,4,12,2] 为例：
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
				// 如果找不到,也就是res数组中所有元素都比num大，此时更新res[0] = num
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
	maxLength := 1
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
对于数组A[:i]和数组B[:j]，如果有A[i-1]==B[j-1],那么数组A[:i]和数组B[:j]的最长重复子数组的长度就等于
数组A[:i-1]和数组B[:j-1]的最长重复子数组的长度+1，也就是dp[i][j]=dp[i-1][j-1]+1
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
一个字符串的子序列是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符
（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是
这两个字符串所共同拥有的子序列。

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
本题与1.3 最长重复子数组最大的不同是公共子序列不要求是连续的了，只需保持相对顺序即可
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
1.5 leetcode 1035 不相交的线
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
直线不能相交，这就是说明在字符串A中 找到一个与字符串B相同的子序列，且这个子序列不能改变相对顺序，只要相对顺序不改变，
链接相同数字的直线就不会相交。

其实也就是说A和B的最长公共子序列是[1,4]，长度为2。 这个公共子序列指的是相对顺序不变（即数字4在字符串A中数字1的后面，
那么数字4也应该在字符串B数字1的后面）

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
		if nums[i-1] > 0 {
			nums[i] += nums[i-1]
		}
		if max < nums[i] {
			max = nums[i]
		}
	}
	return max
}

/*
leetcode 918. 环形子数组的最大和
1.7 给定一个由整数数组A表示的环形数组C，求C的非空子数组的最大可能和。
在此处，环形数组意味着数组的末端将会与开头相连呈环状。（形式上，当0 <= i < A.length时C[i] = A[i]，
且当i >= 0时C[i+A.length] = C[i]）

此外，子数组最多只能包含固定缓冲区A中的每个元素一次。（形式上，对于子数组C[i], C[i+1], ..., C[j]，不存在
i <= k1, k2 <= j其中k1 % A.length= k2 % A.length）

示例1：
输入：[1,-2,3,-2]
输出：3
解释：从子数组 [3] 得到最大和3

示例2：
输入：[5,-3,5]
输出：10
解释：从子数组 [5,5] 得到最大和 5 + 5 = 10
*/

/*
思路:动态规划
如果数组nums中所有元素都是负数，那么此时数组元素之和sum与数组的最小子数组和minSub相等，此时的环形数组C的最大
子数组和应该是数组nums的最大子数组和maxSub(也就是绝对值最小的负数)，否则应该是数组nums的最大子数组和maxSub
以及sum-minSub两个值之间取较大值
*/

func MaxSubarraySumCircular(nums []int) int {
	n := len(nums)
	// curMax, curMin分别表示包含当前元素nums[i]的最大子数组和，最小子数组和
	// maxSub, minSub则分别表示数组nums的最大子数组和，最小子数组和
	// sum表示数组nums的元素之和
	curMax, curMin, maxSub, minSub, sum := nums[0], nums[0], nums[0], nums[0], nums[0]
	for i := 1; i < n; i++ {
		sum += nums[i]
		curMax = Utils.Max(curMax+nums[i], nums[i])
		maxSub = Utils.Max(curMax, maxSub)
		curMin = Utils.Min(curMin+nums[i], nums[i])
		minSub = Utils.Min(curMin, minSub)
	}
	if sum == minSub {
		return maxSub
	}
	return Utils.Max(maxSub, sum-minSub)
}

/*
1.8  乘积最大子数组
给你一个整数数组nums，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所
对应的乘积。


示例1:
输入: [2,3,-2,4]
输出: 6
解释:子数组 [2,3] 有最大乘积 6。

示例2:
输入: [-2,0,-1]
输出: 0
解释:结果不能为 2, 因为 [-2,-1] 不是子数组。
*/

/*
思路:动态规划
1 确定dp数组及其下标含义
dp[i]表示nums数组在[0, i](注意是左闭右闭区间)范围内(连续子数组)的最大乘积

2 确定递推公式
根据上题1.6 最大子序和我们似乎很容易联想到它的递推公式是:
dp[i] = max(dp[i-1]*nums[i], nums[i])
但是这个递推公式是错的，如果a={5,6,−3,4,−3}，那么据此递推公式，对应的dp数组应该是
5,30,-3,4,-3,似乎最大乘积就是30，但实际上最大乘积是数组所有元素的乘积1080，原因在于这里的定义并不满足
最优子结构，乘积不同于求和，最大乘积不一定就是前面的最大乘积*nums[i]，还有可能是前面的最小乘积*nums[i]，
因为最小乘积可能是一个绝对值很大的负数，如果nums[i]也是负数，那么负负得正，有可能就成了最大乘积了。
所以我们需要分开讨论，如果nums[i]为正数，我们希望以i-1结尾的子数组的乘积是一个尽可能大的正数；反之如果
nums[i]为负数，我们希望以i-1结尾的子数组的乘积是一个尽可能小(绝对值尽可能大)的负数，这样就得到了本题的
递推公式:
maxDp[i] = max(maxDp[i-1]*nums[i],max(minDp[i-1)*nums[i], nums[i])
minDp[i] = min(minDp[i-1]*nums[i],min(maxDp[i-1)*nums[i], nums[i])

3 初始化dp数组
由递推公式可知，dp[i]依赖于dp[i-1]，所以dp[0]一定要初始化，显然maxDp[0]和minDp[0]都是nums[0]

4 确定遍历顺序
由递推公式可知，应该是从前往后正序遍历

5 举例推导dp数组
如果a={5, 6, −3, 4, −3}，据递推公式，对应的两个dp数组应该是
maxDp 5  30 -3  4  1080
minDp 5  6 -90 -360 -12
最后返回1080
*/

// MaxProduct 时间复杂度O(3N)，空间复杂度O(2N)
func MaxProduct(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return nums[0]
	}
	maxDp := make([]int, n)
	minDp := make([]int, n)
	maxDp[0], minDp[0] = nums[0], nums[0]
	max := nums[0]
	for i := 1; i < n; i++ {
		maxDp[i] = Utils.Max(maxDp[i-1]*nums[i], Utils.Max(minDp[i-1]*nums[i], nums[i]))
		minDp[i] = Utils.Min(minDp[i-1]*nums[i], Utils.Min(maxDp[i-1]*nums[i], nums[i]))
		max = Utils.Max(max, maxDp[i])
	}
	return max
}

// MaxProductSimple 更简单的写法 时间复杂度O(3N)，空间复杂度O(1)
func MaxProductSimple(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return nums[0]
	}
	maxP, minP, max := nums[0], nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		mx, mn := maxP, minP
		maxP = Utils.Max(mx*nums[i], Utils.Max(mn*nums[i], nums[i]))
		minP = Utils.Min(mn*nums[i], Utils.Min(mx*nums[i], nums[i]))
		max = Utils.Max(max, maxP)
	}
	return max
}

/*
leetcode 1567. 乘积为正数的最长子数组长度
1.9 给你一个整数数组nums，请你求出乘积为正数的最长子数组的长度。
一个数组的子数组是由原数组中零个或者更多个连续数字组成的数组。
请你返回乘积为正数的最长子数组长度。

示例1：
输入：nums = [1,-2,-3,4]
输出：4
解释：数组本身乘积就是正数，值为 24 。

示例2：
输入：nums = [0,1,-2,-3,-4]
输出：3
解释：最长乘积为正数的子数组为 [1,-2,-3] ，乘积为6 。
注意，我们不能把 0 也包括到子数组中，因为这样乘积为0 ，不是正数。

示例3：
输入：nums = [-1,-2,-3,0,1]
输出：2
解释：乘积为正数的最长子数组是 [-1,-2] 或者 [-2,-3] 。

提示：
1 <= nums.length <= 10^5
-10^9 <= nums[i] <= 10^9
*/

/*
思路: 动态规划
本题跟乘积最大子数组一样，都需要维护两个动归数组，因为数组中可能有正数，也有可能有负数。下面我们详细展开。

1 确定dp数组及其下标含义
pos[i]表示以nums[i]结尾的子数组乘积为正数的最长子数组长度为pos[i]
ng[i]表示以nums[i]结尾的子数组乘积为负数的最长子数组长度为ng[i]
注意，从dp数组定义可以看出，pos[i]和ng[i]的最小值都是0，不可能是负数哈。

2 确定递推公式

a 当nums[i] > 0 时, nums[i]与nums[i-1]及其前面的元素相乘，不会改变前面元素乘积的正负数性质。
前面乘积是正数，现在乘以正数nums[i]，结果仍然是正数；
前面乘积是负数，现在乘以正数nums[i]，结果仍然负数。

我们先讨论乘积为正数的动归数组，递推公式是
pos[i] = pos[i-1]+1。
下面展开说一下:
如果pos[i-1]=0，说明nums:i-1范围内没有乘积为正数的子数组，那么此时乘积为正数的最长子数组就是[nums[i]],
所以pos[i] = pos[i-1]+1 = 0 + 1 = 1
如果pos[i-1]>0,说明nums[:i-1]范围内乘积为正数的子数组长度至少都是1，现在又来了一个正数，正数乘以正数仍然
是正数，所以pos[i] = pos[i-1] + 1

接下来我们讨论乘积为负数数的动归数组，递推公式是:
如果ng[i-1]>0, ng[i] = ng[i-1]+1; 否则ng[i] = 0

下面展开说一下:
如果ng[i-1]>0，说明前面有多个数乘积(至少有一个)为负数，现在来了一个正数nums[i], 负数乘以正数还是负数，
所以乘积为负数的最长子数组长度ng[i]要加1，即ng[i] = ng[i-1]+1。
否则ng[i-1]=0，说明前面没有任何一个子数组的乘积为负数，才会出现nums[:i-1]范围内乘积为负数的子数组长度为0
的情况，换句话说前面元素的乘积都是正数，那现在来了一个正数nums[i]，正数乘以正数还是正数，乘积为负数的子数组
还是没有，所以此时ng[i] = 0

b 当nums[i] < 0 时, nums[i]与nums[i-1]及其前面的元素相乘，就会改变前面元素乘积的正负数性质了。
前面乘积是正数，现在乘以负数nums[i]，结果会变成负数；
前面乘积是负数，现在乘以负数nums[i]，结果会变成正数。

我们先讨论乘积为正数的动归数组，递推公式是:
如果ng[i-1]>0, pos[i]=ng[i-1]+1, 否则pos[i] = 0

下面展开说一下:
如果ng[i-1]>0，说明前面有多个数乘积(至少有一个)为负数，现在来了一个负数nums[i], 负数乘以负数就变成了正数，
所以乘积为正数的最长子数组长度pos[i]要在ng[i-1]的基础上加1，即pos[i] = ng[i-1]+1。
否则ng[i-1]=0，说明前面没有任何一个子数组的乘积为负数，才会出现nums[:i-1]范围内乘积为负数的子数组长度为0
的情况，换句话说前面元素的乘积都是正数，那现在来了一个负数nums[i]，正数乘以负数变成了负数，乘积为正数的子数组
还是没有，所以此时pos[i] = 0

接下来我们讨论乘积为负数的动归数组，递推公式是:
ng[i] = pos[i-1]+1
下面展开说一下:
当pos[i-1]>0时，说明前面有多个数乘积(至少有一个)为正数，现在来了一个负数nums[i], 正数乘以负数就变成了负数，
所以乘积为负数的最长子数组长度ng[i]要在pos[i-1]的基础上加1，即ng[i] = pos[i-1]+1。
否则，那就是pos[i-1]=0，说明nums[:i-1]范围内没有乘积为正数的子数组，那么此时乘积为负数的最长子数组就是
[nums[i]], 所以ng[i] = pos[i-1] + 1 = 0 + 1 = 1

c 最后，当nums[i] = 0 时, nums[i]与nums[i-1]及其前面的元素的乘积都是0。
所以有: pos[i] = ng[i] = 0

3 初始化dp数组
pos和ng两个动归数组的长度都是len(nums)，从递推公式可知，dp[i]都依赖于dp[i-1]，所以我们必须初始化pos[0]
和ng[0]。
如果nums[0]>0, 那么pos[0] = 1;如果nums[0]<0, 那么ng[0] = 1; 如果nums[0] = 0, 那么
pos[0] = ng[0] = 0
乘积为正数的最长子数组长度maxLength便初始化为pos[0]

4 确定遍历顺序
从递归公式可知，dp[i]都依赖于dp[i-1]，所以我们应该从左到右遍历，而且dp[0]都已经初始化了，遍历下标可以从1
开始。

5 举例推导dp数组
略
*/

// GetMaxLen 时间复杂度O(N)，空间复杂度O(N)
func GetMaxLen(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	pos, ng := make([]int, n), make([]int, n)
	if nums[0] > 0 {
		pos[0] = 1
	} else if nums[0] < 0 {
		ng[0] = 1
	}
	maxLength := pos[0]
	for i := 1; i < n; i++ {
		if nums[i] > 0 {
			pos[i] = pos[i-1] + 1
			if ng[i-1] > 0 {
				ng[i] = ng[i-1] + 1
			} else {
				ng[i] = 0
			}
		} else if nums[i] < 0 {
			if ng[i-1] > 0 {
				pos[i] = ng[i-1] + 1
			} else {
				pos[i] = 0
			}
			ng[i] = pos[i-1] + 1
		} else {
			pos[i], ng[i] = 0, 0
		}
		maxLength = Utils.Max(maxLength, pos[i])
	}
	return maxLength
}

/*
从上面的代码可以看出，我们其实只需要维护pos和ng两个动归数组的最大值就好了，其实不需要建立两个数组，维护两个
最大状态变量就好了，这样可以将算法的空间复杂度降低到O(1),可以写成下面这样。
*/

// GetMaxLenSimple 时间复杂度O(N)，空间复杂度O(1)
func GetMaxLenSimple(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	pos, ng := 0, 0
	if nums[0] > 0 {
		pos = 1
	} else if nums[0] < 0 {
		ng = 1
	}
	maxLength := pos
	for i := 1; i < n; i++ {
		if nums[i] > 0 {
			pos += 1
			if ng > 0 {
				ng += 1
			} else {
				ng = 0
			}
		} else if nums[i] < 0 {
			newPos, newNg := 0, 0
			if ng > 0 {
				newPos = ng + 1
			} else {
				newPos = 0
			}
			newNg = pos + 1
			pos, ng = newPos, newNg
		} else {
			pos, ng = 0, 0
		}
		maxLength = Utils.Max(maxLength, pos)
	}
	return maxLength
}

/*
思路:贪心
子数组乘积为正数，即要求该段子数组中没有0且负数的个数为偶数，这样我们可以用三个变量：
pos:该段正数个数，初始化为0
ng:该段负数个数，初始化为0
first:第一个负数出现的位置，初始化为-1
来记录需要的数量，然后对数组进行遍历：
1.如果当前neg % 2 = 0，说明偶数个数为正该段组数组的所有元素相乘为正，
那么maxLength = max(maxLength, pos + neg)。
2.如果当前neg % 2 != 0，我们可以贪心的进行选取组数组，只要去掉该段字数组的一个负数便可以使负数个数为偶数，
即乘积为正，这时，即从第一个负数开始，后面的位置到当前位置所有数的乘积可以为正，
此时:maxLength = max(ans, 当前位置下标i- first).
3.如果遍历的当前元素为0，则将所有变量重新初始化，因为0不可能包含在任何子数组中，而使得乘积为正。
*/

func GetMaxLenTwo(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	pos, ng, first, maxLength := 0, 0, -1, 0
	for i := 0; i < n; i++ {
		if nums[i] == 0 {
			pos, ng, first = 0, 0, -1
		}
		if nums[i] > 0 {
			pos++
		}
		if nums[i] < 0 {
			if first == -1 {
				first = i
			}
			ng++
		}
		if ng%2 == 0 {
			maxLength = Utils.Max(maxLength, pos+ng)
		} else {
			maxLength = Utils.Max(maxLength, i-first)
		}
	}
	return maxLength
}

/*
1.10 判断子序列
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
1.11 不同的子序列
给定一个字符串s和一个字符串t ，计算在s的子序列中t出现的个数。
字符串的一个子序列是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。
（例如,"ACE"是"ABCDE"的一个子序列,而"AEC" 不是）

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
dp[i][j]表示s[:i]中出现t[:j]的个数为dp[i][j]。

2 确定递推公式
这一类问题，基本是要分析两种情况
s[i-1]与t[j-1]相等以及s[i-1]与t[j-1]不相等
当s[i-1]与t[j-1]相等时，dp[i][j]可以由两部分组成。
一部分是用s[i-1]来匹配，那么个数为dp[i-1][j-1];一部分是不用s[i-1]来匹配，个数为dp[i-1][j]。
所以当s[i-1]与t[j-1]相等时，dp[i][j]=dp[i-1][j-1]+dp[i-1][j];

当s[i-1]与t[j-1]不相等时，dp[i][j]只由一部分组成，不用s[i-1]来匹配，即：dp[i-1][j]
所以递推公式为：dp[i][j] = dp[i-1][j];

3 dp数组如何初始化
从递推公式dp[i][j]=dp[i-1][j-1]+dp[i-1][j]; 和dp[i][j]=dp[i-1][j]; 中可以看出dp[i][0] 和dp[0][j]
是一定要初始化的。

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
	if m < n {
		return 0
	}
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
1.12 两个字符串的删除操作
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

// MinDistance 时间复杂度O((M+1)*(N+1))，空间复杂度O(M*N)
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
思路:将问题转换为求两个字符串的最长公共子序列
给定两个字符串word1和word2，分别删除若干字符之后使得两个字符串相同，则剩下的字符为两个字符串的公共子序列。
为了使删除操作的次数最少，剩下的字符应尽可能多。因此当剩下的字符为两个字符串的最长公共子序列时，删除操作的
次数最少。因此，可以计算两个字符串的最长公共子序列的长度，然后分别计算两个字符串的长度和最长公共子序列的长度之差，
即为两个字符串分别需要删除的字符数，两个字符串各自需要删除的字符数之和即为最少的删除操作的总次数。
*/

// MinDistanceSimple 时间复杂度O(M*N)，空间复杂度O(M*N)
func MinDistanceSimple(word1, word2 string) int {
	m, n := len(word1), len(word2)
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = Utils.Max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	// 最长公共子序列长度为dp[m][n],所以结果为m-dp[m][n]+n-dp[m][n]
	return m + n - 2*dp[m][n]
}

/*
1.13 编辑距离
给你两个单词word1和word2，请你计算出将word1转换成word2所使用的最少操作数。

你可以对一个单词进行如下三种操作：
插入一个字符
删除一个字符
替换一个字符

示例1：
输入：word1 = "horse", word2 = "ros" 输出：3 解释：horse -> rorse (将'h'替换为'r') rorse -> rose
(删除 'r') rose -> ros(删除 'e')

示例2：
输入：word1 = "intention", word2 = "execution" 输出：5 解释：intention -> inention (删除 't')
inention -> enention(将'i'替换为'e') enention -> exention (将'n'替换为'x') exention -> exection
(将'n'替换为'c') exection -> execution (插入'u')

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
最后是替换，可以将word1[i-1]替换为word2[j-1],也可以反过来将word2[j-1]替换为word1[i-1]，此时dp[i][j]=dp[i-1][j-1]+1
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

/*
1.14 回文子串
给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。
具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

示例1：
输入："abc" 输出：3 解释：三个回文子串: "a", "b", "c"

示例2：
输入："aaa" 输出：6 解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"

提示：
输入的字符串长度不会超过1000。
*/

/*
思路:动态规划解决

1 确定dp数组以及下标的含义
布尔类型的dp[i][j]：表示区间范围[i,j]（注意是左闭右闭）的子串是否是回文子串，如果是, dp[i][j]为true，
否则为false。

2 确定递推公式
在确定递推公式时，就要分析如下几种情况。
整体上是两种，就是s[i]与s[j]相等，s[i]与s[j]不相等这两种。

当s[i]与s[j]不相等，那没啥好说的了，dp[i][j]一定是false。
当s[i]与s[j]相等时，这就复杂一些了，有如下三种情况

情况一：下标i与j相等，同一个字符例如a，当然是回文子串
情况二：下标i与j相差为1，例如aa，也是回文子串
情况三：下标：i与j相差大于1的时候，例如cabac，此时s[i]与s[j]已经相同了，我们看i到j区间是不是回文子串就看aba是不是回文就可以了，
那么aba的区间就是i+1到j-1区间，这个区间是不是回文就看dp[i+1][j-1]是否为true。

3 dp数组如何初始化
dp[i][j]可以初始化为true么？ 当然不行，怎能刚开始就全都匹配上了。
所以dp[i][j]初始化为false。

4 确定遍历顺序
遍历顺序可有点讲究了。
首先从递推公式中可以看出，情况三是根据dp[i+1][j-1]是否为true，再对dp[i][j]进行赋值true的。
dp[i+1][j-1]在dp[i][j]的左下角
如果这矩阵是从上到下，从左到右遍历，那么会用到没有计算过的dp[i+1][j-1]，也就是根据不确定是不是回文的区间[i+1,j-1]，
来判断了[i,j]是不是回文，那结果一定是不对的。
所以一定要从下到上，从左到右遍历，这样保证dp[i+1][j-1]都是经过计算的。

5 举例推导dp数组
略
*/

// CountSubStrings 时间复杂度O(N^2)，空间复杂度O(N^2)
func CountSubStrings(s string) int {
	n := len(s)
	dp := make([][]bool, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]bool, n)
	}
	num := 0
	for i := n - 1; i >= 0; i-- {
		for j := i; j < n; j++ {
			if s[i] == s[j] {
				if j-i <= 1 {
					num++
					dp[i][j] = true
				} else if dp[i+1][j-1] {
					num++
					dp[i][j] = true
				}
			}
		}
	}
	return num
}

/*
动态规划解法的空间复杂度偏高，若使用双指针法，空间复杂度可以大幅下降至O(1)
思路:中心扩展法
首先确定回文串，就是找中心然后想两边扩散看是不是对称的就可以了。由于s[i]本身肯定是回文串，所以left和right两个指针起始位置都是i，
而后分别向左向右移动，边界分别是0和n-1,只要满足s[left]==s[right]，就找到了一个回文子串。

但是在遍历中心点的时候，要注意中心点有两种情况。
一个元素可以作为中心点，两个元素也可以作为中心点，所以还需要计算以i和i+1两个元素作为中心点的情况
当然，三个元素还可以做中心点，但三个元素就可以由一个元素左右添加元素得到，四个元素则可以由两个元素左右添加元素得到。

所以我们在计算的时候，要注意一个元素为中心点和两个元素为中心点的情况。
*/

// CountSubStringsSimple 时间复杂度O(N^2)，空间复杂度O(1)
func CountSubStringsSimple(s string) int {
	sumNum := 0
	for i := 0; i < len(s); i++ {
		// 以i为中心
		sumNum += Extend(s, i, i, len(s))
		// 以和i+1为中心
		sumNum += Extend(s, i, i+1, len(s))
	}
	return sumNum
}

func Extend(s string, i, j, n int) int {
	num := 0
	for i >= 0 && j < n && s[i] == s[j] {
		i--
		j++
		num++
	}
	return num
}

/*
1.15 最长回文子序列
给定一个字符串s ，找到其中最长的回文子序列，并返回该序列的长度。可以假设s的最大长度为 1000 。
示例 1: 输入: "bbbab" 输出: 4 一个可能的最长回文子序列为 "bbbb"。
示例 2: 输入:"cbbd" 输出: 2 一个可能的最长回文子序列为 "bb"。
注意:这里的子序列不要求是连续的
提示：
1 <= s.length <= 1000
s 只包含小写英文字母
*/

/*
思路:动态规划

1 确定dp数组及其下标含义
dp[i][j]表示字符串s在[i:j](注意是左闭右闭区间)范围内的最长回文子序列长度为dp[i][j]
注意j>=i
2 确定递推公式
dp[i][j]的值，很明显取决于s[i]与s[j]是否相等。
如果s[i]==s[j]，那么dp[i][j] = dp[i+1][j-1]+2
a   b    a   a   b    a
i  i+1          j-1   j

如果s[i]!=s[j],说明同时加入s[i]与s[j]并不能增加s在[i:j]内的回文子串的长度，那么分别加入s[i],s[j]
看看哪一个可以组成最长的回文子序列
加入s[i]的最长的回文子序列为dp[i][j-1]
加入s[j]的最长的回文子序列为dp[i+1][j]
那么dp[i][j]一定取最大的,即dp[i][j] = max(dp[i][j-1], dp[i+1][j])

3 初始化dp数组
显然任何单个字符都是回文子串，所以dp[i][i]=1，其他初始化为0即可(递推公式不会被初始值覆盖)。

4 确定遍历顺序
从递推公式dp[i][j] = dp[i+1][j-1]+2,dp[i][j] = max(dp[i][j-1], dp[i+1][j])可以看出,dp[i][j]是依赖于
dp[i+1][j-1]和dp[i+1][j]的，所以外层遍历i只能是逆序从大到小遍历，内层遍历则是正序遍历

5 举例推导dp数组
略
*/

// LongestPalindromeSubSeq 时间复杂度O(N^2)，空间复杂度O(N^2)
func LongestPalindromeSubSeq(s string) int {
	n := len(s)
	if n <= 1 {
		return n
	}
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
		dp[i][i] = 1
	}
	for i := n - 1; i >= 0; i-- {
		// j>=i，dp[i][i]=1，所以j从i+1开始
		for j := i + 1; j < n; j++ {
			if s[i] == s[j] {
				dp[i][j] = dp[i+1][j-1] + 2
			} else {
				dp[i][j] = Utils.Max(dp[i+1][j], dp[i][j-1])
			}
		}
	}
	// 字符串s的最长回文子序列一定是在[0, len(s)-1]区间，也就是整个字符串范围内取得的。
	return dp[0][n-1]
}

/*
1.16 最长递增子序列的个数
给定一个未排序的整数数组，找到最长递增子序列的个数。

示例1:
输入: [1,3,5,4,7]
输出: 2
解释: 有两个最长递增子序列，分别是 [1, 3, 4, 7] 和[1, 3, 5, 7]。

示例2:
输入: [2,2,2,2,2]
输出: 5
解释: 最长递增子序列的长度是1，并且存在5个子序列的长度为1，因此输出5。

注意:给定的数组长度不超过2000并且结果一定是32位有符号整数。
*/

/*
本题可谓是1.1 最长递增子序列的进阶题目
思路:动态规划
1 确定dp数组以及下标的含义
这道题目我们要维护两个动规数组。
dp[i]表示以nums[i]为结尾的数组最长递增子序列的长度为dp[i]
count[i]表示以nums[i]为结尾的数组，最长递增子序列的个数为count[i]

2 确定递推公式
在1.1最长上升子序列 中，我们给出的状态转移方程是：
if (nums[i] > nums[j]) dp[i] = max(dp[i], dp[j] + 1);
即：位置i的最长递增子序列长度等于j从0到i-1各个位置的最长升序子序列+1的最大值。

本题就没那么简单了，我们要考虑两个维度，一个是dp[i]的更新，一个是count[i]的更新。
那么如何更新count[i]呢？

以nums[i]为结尾的数组，最长递增子序列的个数为count[i]。
那么在nums[i] > nums[j]前提下，如果在[0, i-1]的范围内，找到了j，使得dp[j] + 1 > dp[i]，说明找到了一个
更长的递增子序列。
那么以i为结尾的子数组的最长递增子序列的个数，就是最新的以j为结尾的子串的最长递增子序列的个数，即：
count[i] = count[j]。
在nums[i] > nums[j]前提下，如果在[0, i-1]的范围内，找到了j，使得dp[j] + 1 == dp[i]，说明找到了两个相同
长度的递增子序列。
那么以i为结尾的子串的最长递增子序列的个数，就应该加上以j为结尾的最长递增子序列的个数，即：
count[i] += count[j];

这里count[i]记录了以nums[i]为结尾的数组，最长递增子序列的个数。dp[i]记录了i之前（包括i）最长递增序列的长度。
题目要求最长递增序列的长度的个数，我们应该把最长长度记录下来。

3 dp数组如何初始化
再回顾一下dp[i]和count[i]的定义
count[i]记录了以nums[i]为结尾的数组，最长递增子序列的个数。
那么最少也就是1个，所以count[i]初始为1。
dp[i]记录了i之前（包括i）最长递增序列的长度。
最小的长度也是1，所以dp[i]初始为1。

4 确定遍历顺序
dp[i] 是由0到i-1各个位置的最长升序子序列推导而来，那么遍历i一定是从前向后遍历。
j其实就是0到i-1，遍历i的循环里外层，遍历j则在内层。
最后还要再遍历一遍dp[i]，把最长递增序列长度对应的count[i]累加下来就是结果了。

5. 举例推导dp数组
略
*/

// FindNumberOfLIS 时间复杂度O(N^2)，空间复杂度O(N)
func FindNumberOfLIS(nums []int) int {
	n := len(nums)
	if n <= 1 {
		return n
	}
	dp := make([]int, n)
	count := make([]int, n)
	for i := 0; i < n; i++ {
		dp[i] = 1
		count[i] = 1
	}
	maxLength := 1
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			if nums[i] > nums[j] {
				if dp[j]+1 > dp[i] {
					count[i] = count[j]
				} else if dp[j]+1 == dp[i] {
					count[i] += count[j]
				}
				dp[i] = Utils.Max(dp[i], dp[j]+1)
			}
		}
		if maxLength < dp[i] {
			maxLength = dp[i]
		}
	}
	sum := 0
	for i := 0; i < n; i++ {
		if dp[i] == maxLength {
			sum += count[i]
		}
	}
	return sum
}

/*
leetcode 413. 等差数列划分
1.17 如果一个数列至少有三个元素，并且任意两个相邻元素之差相同，则称该数列为等差数列。
例如，[1,3,5,7,9]、[7,7,7,7] 和 [3,-1,-5,-9] 都是等差数列。
给你一个整数数组nums ，返回数组nums中所有为等差数组的 子数组 个数。

子数组是数组中的一个连续序列。

示例1：
输入：nums = [1,2,3,4]
输出：3
解释：nums 中有三个子等差数组：[1, 2, 3]、[2, 3, 4] 和 [1,2,3,4] 自身。

示例2：
输入：nums = [1]
输出：0

提示：
1 <= nums.length <= 5000
-1000 <= nums[i] <= 1000
*/

/*
思路: 从左到右遍历数组，找到尽可能长的等差数列l，然后统计l中符合要求的连续等差子数组的个数，而后更新下一个等差数列的起始位置以及等差，直到遍历到数组末尾位置。
我们以数组nums := []int{1, 3, 5, 7, 9, 10, 13, 15, 17, 19, 20, 21, 22, 23, 24}为例。
等差df初始值为nums[0]-nums[1]=2, 第一个等差数列l的起始位置start为0，从下标2开始遍历，待遍历到下标5时，
发现nums[5]-nums[4]=1, 1!=df(2), 所以第一个尽可能长的等差数列l结束位置end就是i-1=4, 其长度length为
end-start+1=5，很明显5 > 3(最短等差子数组长度为3)，如果长度小于3，那就更新下一个等差数列l的起始位置
start和等差df就好了。

那么这个等差数列l[1, 3, 5, 7, 9]中符合条件的连续(注意是连续)等差子数组一共有多少个呢？
其实这是一个等差数列求和的问题，我们可以从最短的等差子数组开始：
如果我们要找的连续等差子数组长度为3，那么对于长度为length的等差数列l，我们从l的起始位置start开始可以选择
length-2个起点和紧挨着起点start的两个元素组成长度为3的连续等差数组，所以一共有length-2个长度为3的连续
等差子数组；
同理，如果我们要找的连续等差子数组长度为4，我们可以发现一共会有length-3个长度为4的连续等差子数组;
...
如果我们要找的连续等差子数组长度为length，我们发现只有一个子数组符合条件，那就是等差数列l本身。
所以等差数列l中长度大于等于3的连续子数组个数就是1+2+3+...+length-2。这就是一个等差数列求和的问题，显然有
sum(l) = (length-1)X(length-2)/2 = 4 X 3 / 2 = 6 个

此时我们需要更新下一个等差数列l的起始位置start为i-1=4, 更新等差df=nums[i]-nums[i-1]=nums[5]-nums[4]=1.
注意这里下一个等差子数列l的起始位置start是i-1=4，不是i=5，为什么？ 因为有可能nums[i-1],nums[i],nums[i+1]
三个数会形成一个等差数列。以这个数组为例，下一个等差数列l明显是[13,15,17,19], 接下来的等差数列
l[19,20,21,22,23,24]的起点start, nums[9]=19正好是上一个等差数列的终点end, nums[9]=19。需要注意的是
连续等差数组的长度至少为3，所以如果i+1>=n,也就是下标越界，就可以退出循环了(i-1,i,i+1三个元素)。

*/

// NumberOfArithmeticSlices 时间复杂度O(N)，空间复杂度O(1)
func NumberOfArithmeticSlices(nums []int) int {
	n := len(nums)
	// 处理特殊情况
	if n < 3 {
		return 0
	}
	// 等差df初始值为数组nums前两个元素之差
	df := nums[1] - nums[0]
	// 第一个等差子数组的起点start的位置是0
	start, count := 0, 0
	// 因为等差数组长度至少为3，所以从2开始遍历
	for i := 2; i < n; i++ {
		// 此时数组末尾元素为最后一个等差数组末尾元素
		if nums[i]-nums[i-1] == df && i == n-1 {
			length := i - start + 1
			if length >= 3 {
				count += (length - 1) * (length - 2) / 2
			}

		} else if nums[i]-nums[i-1] != df {
			// 此时找到了一个等差子数组
			end := i - 1
			// 计算该等差子数组的长度
			length := end - start + 1
			// 如果长度大于等于3，则计算该等差子数组中所有符合条件等差数组的数量
			if length >= 3 {
				count += (length - 1) * (length - 2) / 2
			}
			// 更新下一个等差子数组的起点start=i-1
			start = i - 1
			// 因为等差子数组长度至少得是3，所以如果i+1越界，就退出循环
			if i+1 >= n {
				break
			}
			// 更新下一个等差子数组的等差df
			df = nums[i] - nums[i-1]
			continue
		} else {
			continue
		}
	}
	return count
}

/*
leetcode 446. 等差数列划分II - 子序列
1.18 给你一个整数数组nums ，返回nums中所有等差子序列的数目。
如果一个序列中至少有三个元素 ，并且任意两个相邻元素之差相同，则称该序列为等差序列。
例如，[1, 3, 5, 7, 9]、[7, 7, 7, 7] 和 [3, -1, -5, -9] 都是等差序列。
再例如，[1, 1, 2, 5, 7] 不是等差序列。
数组中的子序列是从数组中删除一些元素（也可能不删除）得到的一个序列。

例如，[2,5,10] 是 [1,2,1,2,4,1,5,10] 的一个子序列。
题目数据保证答案是一个 32-bit 整数。
示例1：
输入：nums = [2,4,6,8,10]
输出：7
解释：所有的等差子序列为：
[2,4,6]
[4,6,8]
[6,8,10]
[2,4,6,8]
[4,6,8,10]
[2,4,6,8,10]
[2,6,10]

示例2：
输入：nums = [7,7,7,7,7]
输出：16
解释：数组中的任意子序列都是等差子序列。

提示：
1 <= nums.length <= 1000
-231 <= nums[i] <= 231 - 1
*/

// NumberOfArithmeticSlicesComplex 时间复杂度O(N^2)，空间复杂度O(N^2)
func NumberOfArithmeticSlicesComplex(nums []int) int {
	dp := make([]map[int]int, len(nums))
	count := 0
	for i, x := range nums {
		dp[i] = make(map[int]int)
		for j, y := range nums[:i] {
			d := x - y
			cnt := dp[j][d]
			// 为什么+=cnt? dp[j][d]表示以nums[j]为结尾，等差为d的子序列个数
			// dp[j][d]表示的子序列至少有俩元素等差为d，现在加上一个nums[i]
			// 就构成了一个长度至少为3的等差序列，所以有dp[j][d]个至少长度为2的等差序列
			// 那就意味着有dp[j][d]个至少长度为3的等差序列
			count += cnt
			// 为什么+1？因为[nums[j],nums[i]]也是等差为d的一个子序列
			dp[i][d] += cnt + 1
		}
	}
	return count
}

/*
leetcode 91. 解码方法
1.19 一条包含字母A-Z 的消息通过以下映射进行了编码 ：
'A' -> 1
'B' -> 2
...
'Z' -> 26
要解码已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射
为：
"AAJF" ，将消息分组为 (1 1 10 6)
"KJF" ，将消息分组为 (11 10 6)
注意，消息不能分组为 (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。
给你一个只含数字的非空字符串s ，请计算并返回解码方法的总数 。
题目数据保证答案肯定是一个32位的整数。

示例1：
输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）。

示例2：
输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。

示例3：
输入：s = "0"
输出：0
解释：没有字符映射到以 0 开头的数字。
含有 0 的有效映射是 'J' -> "10" 和 'T'-> "20" 。
由于没有字符，因此没有有效的方法对此进行解码，因为所有数字都需要映射。

示例4：
输入：s = "06"
输出：0
解释："06" 不能映射到 "F" ，因为字符串含有前导 0（"6" 和 "06" 在映射中并不等价）。

提示：
1 <= s.length <= 100
s 只包含数字，并且可能包含前导零。
*/

func NumDecoding(s string) int {
	if s[0] == '0' {
		return 0
	}
	n := len(s)
	dp := make([]int, n+1)
	dp[0] = 1
	for i := 1; i <= n; i++ {
		if s[i-1] != '0' {
			dp[i] += dp[i-1]
		}
		if i >= 2 && s[i-2] != '0' && (s[i-2]-'0')*10+s[i-1]-'0' <= 26 {
			dp[i] += dp[i-2]
		}
	}
	return dp[n]
}

/*
leetcode 264. 丑数II
1.20 给你一个整数n ，请你找出并返回第n个 丑数 。
丑数就是只包含质因数2、3 或5的正整数。

示例1：
输入：n = 10
输出：12
解释：[1, 2, 3, 4, 5, 6, 8, 9, 10, 12] 是由前10个丑数组成的序列。

示例2：
输入：n = 1
输出：1
解释：1通常被视为丑数。

提示：
1 <= n <= 1690
*/

func NthUglyNumber(n int) int {
	dp := make([]int, n+1)
	dp[1] = 1
	p2, p3, p5 := 1, 1, 1
	for i := 2; i <= n; i++ {
		// x2和x3以及x5三个数是有可能相等的,那么当它们相等时，它们对应的指针都需要向后移动
		// 否则dp数组中会出现重复数字(譬如x2=x3=6,此时p2=3,p3=2,p5=1，dp[6]=6)，dp[7]仍然是6
		// 因此不能使用if/else if/else的判断逻辑
		x2, x3, x5 := dp[p2]*2, dp[p3]*3, dp[p5]*5
		dp[i] = Utils.Min(x2, Utils.Min(x3, x5))
		if dp[i] == x2 {
			p2++
		}
		if dp[i] == x3 {
			p3++
		}
		if dp[i] == x5 {
			p5++
		}
	}
	return dp[n]
}

/*
leetcode 931. 下降路径最小和
1.21 给你一个n x n 的方形整数数组matrix ，请你找出并返回通过matrix的下降路径的最小和。
下降路径可以从第一行中的任何元素开始，并从每一行中选择一个元素。在下一行选择的元素和当前行所选元素最多相隔
一列（即位于正下方或者沿对角线向左或者向右的第一个元素）。具体来说，位置 (row, col) 的下一个元素应当是
(row + 1, col - 1)、(row + 1, col) 或者 (row + 1, col + 1) 。
输入：matrix = [[2,1,3],[6,5,4],[7,8,9]]
输出：13
*/

func minFallingPathSum(matrix [][]int) int {
	n := len(matrix)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
	}
	for j := 0; j < n; j++ {
		dp[n-1][j] = matrix[n-1][j]
	}
	for i := n - 2; i >= 0; i-- {
		for j := 0; j < n; j++ {
			dp[i][j] = dp[i+1][j]
			if j > 0 {
				dp[i][j] = Utils.Min(dp[i+1][j-1], dp[i][j])
			}
			if j+1 < n {
				dp[i][j] = Utils.Min(dp[i+1][j+1], dp[i][j])
			}
			dp[i][j] += matrix[i][j]
		}
	}
	return Utils.MinValueOfArray(dp[0])
}

/*
leetcode 120. 三角形最小路径和
1.22 给定一个三角形triangle ，找出自顶向下的最小路径和。
每一步只能移动到下一行中相邻的结点上。相邻的结点在这里指的是下标与上一层结点下标相同或者等于上一层结点下标+1
的两个结点。也就是说，如果正位于当前行的下标i ，那么下一步可以移动到下一行的下标i或i+1 。

示例1：
输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
输出：11
解释：如下面简图所示：
   2
  3 4
 6 5 7
4 1 8 3
自顶向下的最小路径和为11（即，2+3+5+1= 11）。
*/

func MinimumTotal(triangle [][]int) int {
	n := len(triangle)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, i+1)
	}
	for j := 0; j < n; j++ {
		dp[n-1][j] = triangle[n-1][j]
	}
	for i := n - 2; i >= 0; i-- {
		for j := 0; j < i+1; j++ {
			dp[i][j] = dp[i+1][j]
			if j+1 <= i+1 {
				dp[i][j] = Utils.Min(dp[i+1][j+1], dp[i][j])
			}
			dp[i][j] += triangle[i][j]
		}
	}
	return dp[0][0]
}