package bitcompute

/*
leetcode 136. 只出现一次的数字
1.4 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

说明：
你的算法应该具有线性时间复杂度。你可以不使用额外空间来实现吗？

示例 1:
输入: [2,2,1]
输出: 1

示例2:
输入: [4,1,2,1,2]
输出: 4
*/

/*
思路:使用位运算。对于这道题，可使用异或运算。异或运算有以下三个性质。
任何数和0做异或运算，结果仍然是原来的数，即a⊕0=a。
任何数和其自身做异或运算，结果是0，即a⊕a=0。
异或运算满足交换律和结合律，即a⊕b⊕a=b⊕a⊕a=b⊕(a⊕a)=b⊕0=b。
*/

func SingleNumberSimple(nums []int) int {
	res := 0
	for _, num := range nums {
		res ^= num
	}
	return res
}

/*
剑指Offer 56 - I. 数组中数字出现的次数
leetcode 260 只出现一次的数字III
1.5 一个整型数组nums里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度
是O(n)，空间复杂度是O(1)。

示例1：
输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]

示例2：
输入：nums = [1,2,10,4,1,4,3,3]
输出：[2,10] 或 [10,2]

限制：
2 <= nums.length <= 10000
*/

/*
二进制运算与，或，异或操作
与操作:当且仅当两个二进制位都是1的情况下，这个二进制位的运算结果才是1，其他情况运算结果为0。
或操作:两个二进制位只要有一个是1，这个二进制位的运算结果就是1。
异或操作:两个二进制位相同运算结果为0，不同为1，任何数与0异或结果为其本身。

*/

func SingleNumbers(nums []int) []int {
	res := 0
	// 因为相同的数字异或为0，任何数字与0异或结果是其本身。
	// 所以异或整个数组后得到的结果就是两个只出现一次的数字异或的结果：即 z = x ^ y
	for _, num := range nums {
		res ^= num
	}
	// 我们根据异或的性质可以知道：res中至少有一位是1，否则x与y就是相等的。
	// 我们通过一个辅助变量h来保存res中哪一位为1.（可能有多个位都为1，我们找到最低位的1即可）。
	// 举个例子：res = 10 ^ 2 = 1010 ^ 0010 = 1000, 第四位为1.
	// 我们将h初始化为1，如果（res & h）与操作的结果等于0说明res的最低位是0,因为h的最低位是1(0001)
	// 我们每次将h左移一位然后跟res做与操作，直到结果不为0.
	// 此时m应该等于1000，同res一样，第四位为1.
	h := 1
	for h&res == 0 {
		h <<= 1
	}
	x, y := 0, 0
	// 我们遍历数组，将每个数跟h进行与操作，结果为0的作为一组，结果不为0的作为一组
	// 例如对于数组：[1,2,10,4,1,4,3,3]，我们把每个数字跟1000做与操作，可以分为下面两组：
	// nums1存放结果为0的: [1, 2, 4, 1, 4, 3, 3]
	// nums2存放结果不为0的: [10] (碰巧nums2中只有一个10，如果原数组中的数字再大一些就不会这样了)
	// 此时我们发现问题已经转化为数组中有一个数字只出现了一次。
	// 分别对nums1和nums2异或就能得到我们预期的x和y。
	for _, num := range nums {
		if num&h == 0 {
			x ^= num
		} else {
			y ^= num
		}
	}
	return []int{x, y}
}

/*
剑指Offer 56 - II. 数组中数字出现的次数II
1.6 在一个数组nums中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

示例1：
输入：nums = [3,4,3,3]
输出：4

示例2：
输入：nums = [9,1,7,9,7,9,7]
输出：1


限制：
1 <= nums.length <= 10000
1 <= nums[i] < 2^31
*/

/*
如下图所示(位运算.png)，考虑数字的二进制形式，对于出现三次的数字，各二进制位出现的次数都是3的倍数。
因此，统计所有数字的各二进制位中1的出现次数，并对3求余，结果则为只出现一次的数字。
此解法为通用解法，即其他数字都出现4次，5次，N次啊，求只出现一次的数字，直接用4，5，N取余即可

*/

func singleNumber(nums []int) int {
	res := 0
	// 因为题意限定1<=nums[i]<2^31,所以设计为32位二进制数
	for i := 0; i < 32; i++ {
		bit := 0
		// 计算数组中所有元素在该二进制位i上之和
		for _, num := range nums {
			bit += num >> i & 1
		}
		// bit对3取余即为res在该二进制位的值
		res += bit % 3 << i
	}
	return res
}

/*
剑指offer 65 不用加减乘除做加法
写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。
*/

/*
思路: 位运算
加法的本质可以分解为两个部分：
无进位和：两个数的每一位相加，但不考虑进位。这可以用**异或（XOR）**操作实现，因为异或操作在二进制中恰好反映了“无进位加法”的逻辑。
进位：两个数的每一位相加时产生的进位。这可以用**与（AND）操作检测进位位，再通过左移（<<）**将进位移动到高一位。

通过反复计算“无进位和”和“进位”，并将两者合并，直到没有进位为止，最终得到两个数的和。
算法步骤
计算无进位和：使用异或操作（a ^ b）。
计算进位：使用与操作（a & b），然后左移一位（(a & b) << 1）。
将“无进位和”与“进位”合并：重复步骤1和2，直到进位为0。
返回结果：当进位为0时，“无进位和”即为最终结果。

示例：3 + 2
二进制表示：3 = 011，2 = 010

第一次：
无进位和：011 ^ 010 = 001（十进制1）
进位：(011 & 010) << 1 = 010 << 1 = 100（十进制4）

第二次：将001和100相加
无进位和：001 ^ 100 = 101（十进制5）
进位：(001 & 100) << 1 = 000 << 1 = 000（十进制0）
进位为0，结束，结果为101 = 5。

复杂度
时间复杂度：O(log n)，取决于整数的位数，每次进位左移会减少计算量。
空间复杂度：O(1)，只使用常数额外空间。
*/

func add(a, b int) int {
	// 当进位为0时，循环结束
	for b != 0 {
		// 计算进位：使用与操作检测进位位，再左移一位
		// 用 uint 转换为无符号整数，避免负数左移时的符号位问题
		carry := uint(a&b) << 1
		// 计算无进位和：使用异或操作模拟不带进位的加法
		a = a ^ b
		// 更新进位：将 carry 赋值给 b，准备下一次循环
		b = int(carry)
	}
	// 进位为0时，a 即为最终结果
	return a
}

/*
剑指offer 3 前n个数字二进制中1的个数
给定一个非负整数 n ，请计算 0 到 n 之间的每个数字的二进制表示中 1 的个数，并输出一个数组。

示例 1:
输入: n = 2
输出: [0,1,1]
解释:
0 --> 0
1 --> 1
2 --> 10

示例 2:
输入: n = 5
输出: [0,1,1,2,1,2]
解释:
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101

说明 :

0 <= n <= 105


进阶:
给出时间复杂度为 O(n*sizeof(integer)) 的解答非常容易。但你可以在线性时间 O(n) 内用一趟扫描做到吗？
要求算法的空间复杂度为 O(n) 。
*/

/*
思路:动态规划
为了实现 O(n) 的时间复杂度，我们需要利用数字之间的规律，避免逐位计算每个数的 1 个数。观察二进制数的递增模式，
可以发现一种动态规划（DP）的思想：每个数字的 1 个数可以基于之前的结果快速计算。

规律分析
0: 0 → 1 的个数 = 0
1: 1 → 1 的个数 = 1
2: 10 → 1 的个数 = 1
3: 11 → 1 的个数 = 2
4: 100 → 1 的个数 = 1
5: 101 → 1 的个数 = 2

注意到：
偶数（如 4 = 100）是某个较小数（如 2 = 10）左移一位的结果，1 的个数不变。
奇数（如 5 = 101）是前一个偶数（如 4 = 100）加 1，1 的个数是前一个数的 1 个数加 1。

更一般的规律：
对于数字 i，可以用位运算 i & (i-1) 去掉最低位的 1：
i & (i-1) 的结果是一个比 i 小的数，其 1 的个数比 i 少 1。
因此，i 的 1 个数 = (i & (i-1)) 的 1 个数 + 1。

另一种等价形式：
如果 i 是奇数，i 的 1 个数 = (i-1) 的 1 个数 + 1。
如果 i 是偶数，i 的 1 个数 = i/2 的 1 个数（因为 i/2 是右移一位）。
*/

func countBits(n int) []int {
	// 创建结果数组，长度为 n+1，存储 0 到 n 的每个数的 1 个数
	dp := make([]int, n+1)
	// 基数：0 的二进制中 1 的个数为 0
	dp[0] = 0
	// 从 1 到 n 遍历，计算每个数的 1 个数
	for i := 1; i <= n; i++ {
		// 递推公式：dp[i] = dp[i>>1] + (i & 1)
		// i>>1：右移一位，等价于 i/2，去掉最低位
		// i & 1：检查最低位是否为 1（奇数为 1，偶数为 0）
		dp[i] = dp[i>>1] + (i & 1)
	}
	return dp
}
