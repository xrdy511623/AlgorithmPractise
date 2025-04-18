package feature

import (
	"math"
	"math/rand"
	"sort"
	"time"
)

/*
leetcode 383. 赎金信
1.1给你两个字符串：ransomNote 和 magazine，判断ransomNote能不能由magazine里面的字符构成。
如果可以，返回true ；否则返回false 。

magazine中的每个字符只能在ransomNote中使用一次。

示例1：
输入：ransomNote = "a", magazine = "b"
输出：false

示例2：
输入：ransomNote = "aa", magazine = "ab"
输出：false

示例3：
输入：ransomNote = "aa", magazine = "aab"
输出：true
*/

/*
思路:哈希表比较词频
本题的意思是ransomNote中的字符，magazine中也必须有，而且这个字符在ransomNote中出现的次数必须小于
等于magazine中出现的次数，否则就不行。
*/

func CanConstruct(ransomNote string, magazine string) bool {
	record := make([]int, 26)
	for _, v := range magazine {
		record[v-'a']++
	}
	for _, v := range ransomNote {
		record[v-'a']--
		if record[v-'a'] < 0 {
			return false
		}
	}
	return true
}

/*
leetcode 349. 两个数组的交集
1.2 给定两个数组，编写一个函数来计算它们的交集。

示例1：
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2]

示例2：
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[9,4]
*/

// FindIntersection 时间复杂度O(M+N)，空间复杂度O(M)
func FindIntersection(nums1 []int, nums2 []int) []int {
	var res []int
	visited := make(map[int]bool)
	for _, v := range nums1 {
		visited[v] = true
	}
	for _, num := range nums2 {
		if visited[num] {
			res = append(res, num)
			// 去重
			delete(visited, num)
		}
	}
	return res
}

/*
leetcode 350. 两个数组的交集II
1.3 给你两个整数数组nums1和nums2 ，请你以数组形式返回两数组的交集。返回结果中每个元素出现的次数，应与元素
在两个数组中都出现的次数一致（如果出现次数不一致，则考虑取较小值）。可以不考虑输出结果的顺序。

示例1：
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]

示例2:
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]

提示：
1 <= nums1.length, nums2.length <= 1000
0 <= nums1[i], nums2[i] <= 1000

进阶：
如果给定的数组已经排好序呢？你将如何优化你的算法？
如果nums1的大小比nums2小，哪种方法更优？
如果nums2的元素存储在磁盘上，内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办？
*/

// Intersect 时间复杂度: O(M+N) 空间复杂度: O(min(M,N))
func Intersect(nums1 []int, nums2 []int) []int {
	var res []int
	visited := make(map[int]int)
	for _, num := range nums1 {
		visited[num]++
	}
	for _, num := range nums2 {
		if visited[num] > 0 {
			res = append(res, num)
			visited[num]--
		}
	}
	return res
}

/*
思路二:排序+双指针
如果两个数组是已经排好序的，那么使用双指针会更有效率，此时时间复杂度会降低到O(min(M,N)),空间复杂度也可以降低
到O(min(M,N))。
如果nums2的元素存储在磁盘上，磁盘内存是有限的，并且你不能一次加载所有的元素到内存中。那么就无法高效地对nums2
进行排序，因此推荐使用方法一而不是方法二。在方法一中，nums2只关系到查询操作，因此每次读取nums2中的一部分数据，
并进行处理即可。
*/

// IntersectTwo 未排序的情况下，时间复杂度: O(M*logM+N*logN+min(M,N)) 空间复杂度: O(min(M,N))
func IntersectTwo(nums1 []int, nums2 []int) []int {
	var res []int
	m, n := len(nums1), len(nums2)
	sort.Ints(nums1)
	sort.Ints(nums2)
	index1, index2 := 0, 0
	for index1 < m && index2 < n {
		if nums1[index1] < nums2[index2] {
			index1++
		} else if nums1[index1] > nums2[index2] {
			index2++
		} else {
			res = append(res, nums1[index1])
			index1++
			index2++
		}
	}
	return res
}

/*
leetcode 202. 快乐数
1.4 编写一个算法来判断一个数n是不是快乐数。
「快乐数」定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数
变为1，也可能是无限循环但始终变不到1。如果可以变为1，那么这个数就是快乐数。
如果n是快乐数就返回True；不是，则返回False 。

示例：
输入：19
输出：true
解释：
1^2 + 9^2 = 82
8^2 + 2^2 = 68
6^2 + 8^2 = 100
1^2 + 0^2 + 0^2 = 1
*/

/*
思路:利用哈希表检查循环
算法分为两部分，我们需要设计和编写代码。
给一个数字n，它的下一个数字是什么？
按照一系列的数字来判断我们是否进入了一个循环。
第1部分我们按照题目的要求做数位分离，求平方和。
第2部分可以使用哈希集合完成。每次生成链中的下一个数字时，我们都会检查它是否已经在哈希集合中。
如果它不在哈希集合中，我们应该添加它。
如果它在哈希集合中，这意味着我们处于一个死循环中，因此应该返回 false。
*/

// IsHappy 时间复杂度O(logN)，空间复杂度O(logN)
func IsHappy(n int) bool {
	occurred := make(map[int]bool)
	for {
		sum := GetSquareSum(n)
		if sum == 1 {
			return true
		}
		// 如果这个平方和重复出现，就证明陷入死循环，直接返回false
		if occurred[sum] {
			return false
		} else {
			// 否则，记录这个平方和出现过
			occurred[sum] = true
		}
		// 重置n的值为前一个平方和
		n = sum
	}
}

// GetSquareSum 求正整数每个位置上的数字的平方和
func GetSquareSum(n int) int {
	sum := 0
	for n > 0 {
		sum += (n % 10) * (n % 10)
		n = n / 10
	}
	return sum
}

/*
思路二:快慢双指针法
通过反复调用getNext(n) 得到的链是一个隐式的链表。隐式意味着我们没有实际的链表节点和指针，但数据仍然形成
链表结构。起始数字是链表的头“节点”，链中的所有其他数字都是节点。next 指针是通过调用 getNext(n) 函数获得。
意识到我们实际有个链表，那么这个问题就可以转换为检测一个链表是否有环。快慢指针法就派上了用场，如果链表有环，
也就是平方和重复出现，那就意味着快慢指针一定会相遇，此时返回false,否则不会相遇，那么只需要判断fast是否
等于1就可以了。
*/

// IsHappyNumber 时间复杂度O(logN)，空间复杂度O(1)
func IsHappyNumber(n int) bool {
	slow, fast := n, n
	var step func(int) int
	step = func(n int) int {
		sum := 0
		for n > 0 {
			sum += (n % 10) * (n % 10)
			n = n / 10
		}
		return sum
	}
	for fast != 1 {
		slow = step(slow)
		fast = step(step(fast))
		if slow == fast && slow != 1 {
			return false
		}
	}
	return fast == 1
}

/*
leetcode 1. 两数之和
1.5 给定一个整数数组nums和一个整数目标值target，请你在该数组中找出和为目标值target的那两个整数，并返回它们的
数组下标。
你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
你可以按任意顺序返回答案。

输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
*/

func TwoSum(nums []int, target int) []int {
	hashMap := make(map[int]int)
	for i, v := range nums {
		if k, ok := hashMap[target-v]; ok {
			return []int{k, i}
		}
		hashMap[v] = i
	}
	return []int{}
}

/*
leetcode 454
1.6 四数相加II
给你四个整数数组nums1、nums2、nums3 和 nums4 ，数组长度都是n ，请你计算有多少个元组 (i, j, k, l)
能满足：
0 <= i, j, k, l < n
nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0

输入：nums1 = [1,2], nums2 = [-2,-1], nums3 = [-1,2], nums4 = [0,2]
输出：2
解释：
两个元组如下：
1. (0, 0, 0, 1) -> nums1[0] + nums2[0] + nums3[0] + nums4[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> nums1[1] + nums2[1] + nums3[0] + nums4[0] = 2 + (-1) + (-1) + 0 = 0
*/

/*
思路:分组+哈希表
我们可以将四个数组分成两部分，A和B为一组，C和D为另外一组。
对于A和B，我们使用二重循环对它们进行遍历，得到所有A[i]+B[j]的值并存入哈希映射中。对于哈希映射中的每个键值对，
每个键表示一种A[i]+B[j]，对应的值为A[i]+B[j]出现的次数。

对于C和D，我们同样使用二重循环对它们进行遍历。当遍历到C[k]+D[l]时，如果−(C[k]+D[l]) 出现在哈希映射中，
那么将哈希表中key为−(C[k]+D[l])对应值累加进答案中。
最终即可得到满足A[i]+B[j]+C[k]+D[l]=0 的四元组数目。
*/

// FourSumCount 时间复杂度O(2*N^2)，空间复杂度O(N^2)
func FourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) int {
	sumMap := make(map[int]int)
	for _, n1 := range nums1 {
		for _, n2 := range nums2 {
			sumMap[n1+n2]++
		}
	}
	count := 0
	for _, n3 := range nums3 {
		for _, n4 := range nums4 {
			if sumMap[-(n3+n4)] != 0 {
				count += sumMap[-(n3 + n4)]
			}
		}
	}
	return count
}

/*
leetcode 128. 最长连续序列
1.7 给定一个未排序的整数数组nums，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
请你设计并实现时间复杂度为O(n)的算法解决此问题。

示例 1：
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。

示例 2：
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9
*/

/*
思路:哈希表
我们考虑枚举数组中的每个数x，考虑以其为起点，不断尝试匹配x+1,x+2,⋯ 是否存在，假设最长匹配到了x+y，那么以x为起点的最长连续序列即为x,x+1,x+2,⋯,
x+y，其长度为y+1，我们不断枚举并更新答案即可。
对于匹配的过程，暴力的方法是O(n) 遍历数组去看是否存在这个数，但其实更高效的方法是用一个哈希表存储数组中的数，这样查看一个数是否存在即能优化至O(1)
的时间复杂度。
仅仅是这样我们的算法时间复杂度最坏情况下还是会达到 O(n^2)（即外层需要枚举O(n)个数，内层需要暴力匹配O(n)次），无法满足题目的要求。但仔细分析这个过程，
我们会发现其中执行了很多不必要的枚举，如果已知有一个x,x+1,x+2,⋯,x+y 的连续序列，而我们却重新从x+1，x+2或者是x+y处开始尝试匹配，那么得到的结果肯定
不会优于枚举x为起点的答案，因此我们在外层循环的时候碰到这种情况跳过即可。

那么怎么判断是否跳过呢？由于我们要枚举的数x一定是在数组中不存在前驱数x-1的，不然按照上面的分析我们会从x−1开始尝试匹配，因此我们每次在哈希表中检查是
否存在x−1即能判断是否需要跳过了。

增加了判断跳过的逻辑之后，时间复杂度是多少呢？外层循环需要O(n) 的时间复杂度，只有当一个数是连续序列的第一个数的情况下才会进入内层循环，然后在内层循环
中匹配连续序列中的数，因此数组中的每个数只会进入内层循环一次。根据上述分析可知，总时间复杂度为O(n)，符合题目要求。
*/

func LongestConsecutive(nums []int) int {
	occurred := make(map[int]bool)
	// 哈希表去重
	for _, num := range nums {
		occurred[num] = true
	}
	maxLength := 0
	// 这里选择遍历哈希表而不是原数组nums，因为nums可能会有重复元素，导致重复判断以及后续连续子序列长度length的重复计算
	for num := range occurred {
		// 跳过num-1，否则会重复做无用功
		if !occurred[num-1] {
			cur := num
			length := 1
			// 不断枚举cur+1,cur+2 ...,判断是否存在于哈希表中
			for occurred[cur+1] {
				cur++
				// 如果每次cur累加1都满足，则以cur为起点的连续递增序列长度累加1
				length++
			}
			// 迭代maxLength
			if maxLength < length {
				maxLength = length
			}
		}
	}
	return maxLength
}

/*
leetcode 76. 最小覆盖子串
1.9 给你一个字符串s、一个字符串t。返回s中涵盖t所有字符的最小子串。如果s中不存在涵盖t所有字符的子串，则返回
空字符串""。

注意：
对于t中重复字符，我们寻找的子字符串中该字符数量必须不少于t中该字符数量。
如果s中存在这样的子串，我们保证它是唯一的答案。

示例1：
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"

示例2：
输入：s = "a", t = "a"
输出："a"

示例3:
输入: s = "a", t = "aa"
输出: ""
*/

/*
思路:滑动窗口
用l,r表示滑动窗口的左边界和右边界，通过改变l,r来扩展和收缩滑动窗口，可以想象成一个窗口在字符串上游走，当这个
窗口包含的元素满足条件，即包含字符串t的所有元素，记录下这个滑动窗口的长度r-l+1，这些长度中的最小值就是要求的
结果。

步骤一
不断增加r使滑动窗口增大，直到窗口包含了t的所有元素
步骤二
不断增加l使滑动窗口缩小，因为是要求最小字串，所以将不必要的元素移除，使长度减小，直到碰到一个必须包含的元素，
这个时候不能再移除了，再移除就不满足条件了，记录此时滑动窗口的长度，并保存最小值。
步骤三
让l向右移动一位，这个时候滑动窗口肯定不满足条件了，那么继续从步骤一开始执行，寻找新地满足条件的滑动窗口，
如此反复，直到r超出了字符串s的范围。

面临的问题：
如何判断滑动窗口包含了t的所有元素？
我们用一个数组need来表示当前滑动窗口中需要的各元素的数量，一开始滑动窗口为空，用t中各元素来初始化这个need，
当滑动窗口扩展或者收缩的时候，去维护这个need数组，例如当滑动窗口包含某个元素，我们就让need中这个元素的数量减1，
表示所需元素减少了1个；当滑动窗口移除某个元素，就让need中这个元素的数量加1。
记住一点：need始终记录着当前滑动窗口下，我们还需要的元素数量，我们在改变l,r时，需同步维护need。
值得注意的是，只要某个元素包含在滑动窗口中，我们就会在need中存储这个元素的数量，如果某个元素存储的是负数代表
这个元素是多余的。比如当need等于{'A':-2,'C':1}时，表示当前滑动窗口中，我们有2个A是多余的，同时还需要1个C。
这么做的目的就是为了步骤二中，移除不必要的元素，数量为负的就是不必要的元素，而数量为0表示刚刚好。
回到问题中来，那么如何判断滑动窗口包含了t的所有元素？结论就是当need中所有元素的数量都小于等于0时，表示当前
滑动窗口不再需要任何元素。
优化
如果每次判断滑动窗口是否包含了t的所有元素，都去遍历need数组看是否所有元素数量都小于等于0，这个会耗费O(k)的
时间复杂度，k代表字典长度，最坏情况下，k可能等于len(S)。其实这个是可以避免的，我们可以维护一个额外的变量
count来记录所需元素的总数量，当我们碰到一个所需元素c，不仅need[c]的数量减少1，同时count也要减少1，这样
我们通过count就可以知道是否满足条件，而无需遍历need数组了。前面也提到过，need记录了遍历到的所有元素，而
只有need[c]>0大于0时，代表c就是所需元素。
*/

func MinWindow(s, t string) string {
	// s的子串要覆盖t，那么这个子串就必须有t中的所有字符并且字符的个数不能少于t
	// count就表示t中所有字符出现的个数之和，也就是t的长度
	count := len(t)
	// 根据题意，两个字符串都是由英文字母组成，那么'z'-'A'=57,所以need数组长度应为58
	var need [58]int
	// 记录字符串t中每个字符的出现次数
	for _, v := range t {
		need[v-'A']++
	}
	// l表示滑动窗口的左边界
	l := 0
	// 滑动窗口的左边界和右边界初始值设置为0和math.MaxInt32，便于后续迭代
	// 因为根据题意两个字符串的最大长度不会超过10^5
	window := []int{0, math.MaxInt32}
	// r代表滑动窗口的右边界
	for r, v := range s {
		// 如果当前字符v在need中出现过，那么滑动窗口中所需的字符数便减一
		if need[v-'A'] > 0 {
			count--
		}
		// need数组中将当前字符的出现次数累减一
		need[v-'A']--
		// 如果count减到0，表明滑动窗口中已经包含了足够的字符来覆盖t
		if count == 0 {
			// 因为我们求的是最小覆盖子串，所以此时要尝试缩小滑动窗口
			// 也就是将滑动窗口的左边界l右移，移除不必要的字符
			for {
				c := s[l]
				// 如果移动过程中左边界指向的字符出现次数为0，表明该字符是滑动窗口必须包含的字符。
				// 此时需要退出当前循环
				if need[c-'A'] == 0 {
					break
				}
				// 将左边界字符的出现次数累加一
				need[c-'A']++
				// 左边界l右移
				l++
			}
			// 如果当前滑动窗口的长度小于之前滑动窗口的长度，那么就更新滑动窗口的左右边界
			if r-l < window[1]-window[0] {
				window = []int{l, r}
			}
			// 此时我们将左边界右移，尝试寻找新地能覆盖t的子串(滑动窗口)
			// 当前左边界l指向的字符是覆盖t的子串所必须包含的，现在要将它从窗口移除，那么
			// 我们所需要的字符数就要累加一了。
			count++
			need[s[l]-'A']++
			// 左边界右移
			l++
		}
	}
	if window[1] == math.MaxInt32 {
		return ""
	}
	return s[window[0] : window[1]+1]
}

/*
leetcode 560. 和为K的子数组
1.10 给你一个整数数组nums和一个整数k，请你统计并返回该数组中和为k的连续子数组的个数。

示例1：
输入：nums = [1,1,1], k = 2
输出：2

示例2：
输入：nums = [1,2,3], k = 3
输出：2
*/

/*
思路一:枚举法
考虑以i结尾和为k的连续子数组个数，我们需要统计符合条件的下标j的个数，其中0≤j≤i 且[j..i]这个子数组的和恰好为k 。
我们可以枚举 [0..i]里所有的下标j来判断是否符合条件。
*/

// SubarraySumSimple 时间复杂度O(N^2)，空间复杂度O(1)
func SubarraySumSimple(nums []int, k int) int {
	count := 0
	for i := 0; i < len(nums); i++ {
		sum := 0
		for end := i; end >= 0; end-- {
			sum += nums[end]
			if sum == k {
				count++
			}
		}
	}
	return count
}

/*
思路二:前缀和+哈希表优化
我们可以基于方法一利用数据结构进行进一步优化，我们知道方法一的瓶颈在于对每个i，我们需要枚举所有的j来判断是否符合
条件，这一步是否可以优化呢？答案是可以的。

我们定义pre[i]为[0..i]里所有数的和，则pre[i]可以由pre[i−1]递推而来，即：
pre[i]=pre[i−1]+nums[i]

那么[j..i]这个子数组和为k这个条件我们可以转化为
pre[i]−pre[j−1]=k

简单移项可得符合条件的下标j需要满足
pre[j−1]=pre[i]−k

所以我们考虑以i结尾的和为k的连续子数组个数时只要统计有多少个前缀和为pre[i]−k的pre[j]即可。我们建立哈希表
mp，以和为键，出现次数为对应的值，记录pre[i]出现的次数，从左往右边更新mp边计算答案，那么以i结尾的答案
mp[pre[i]−k]即可在O(1) 时间内得到。最后的答案即为所有下标结尾的和为k的子数组个数之和。

需要注意的是，从左往右边更新边计算的时候已经保证了mp[pre[i]−k]里记录的pre[j] 的下标范围是0≤j≤i 。同时，
由于pre[i]的计算只与前一项的答案有关，因此我们可以不用建立pre数组，直接用pre变量来记录pre[i-1]的答案即可。
*/

// SubarraySum 时间复杂度O(N)，空间复杂度O(N)
func SubarraySum(nums []int, k int) int {
	pre, count, n := 0, 0, len(nums)
	preSumMap := make(map[int]int)
	preSumMap[0] = 1
	for i := 0; i < n; i++ {
		pre += nums[i]
		if v, ok := preSumMap[pre-k]; ok {
			count += v
		}
		preSumMap[pre]++
	}
	return count
}

/*
leetcode 347. 前K个高频元素
1.11 给你一个整数数组nums和一个整数k ，请你返回其中出现频率前k高的元素。你可以按任意顺序返回答案。

示例1:
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]

示例2:
输入: nums = [1], k = 1
输出: [1]

提示：
1 <= nums.length <= 105
k 的取值范围是 [1, 数组中不相同的元素的个数]
题目数据保证答案唯一，换句话说，数组中前k个高频元素的集合是唯一的
进阶：你所设计算法的时间复杂度必须优于O(NlogN) ，其中n是数组大小。
*/

func TopKFrequent(nums []int, k int) []int {
	freqMap := make(map[int]int)
	maxFreq := 0
	for _, v := range nums {
		freqMap[v]++
		if freqMap[v] > maxFreq {
			maxFreq = freqMap[v]
		}
	}
	hashTop := make([][]int, maxFreq+1)
	for num, freq := range freqMap {
		hashTop[freq] = append(hashTop[freq], num)
	}
	res := make([]int, 0, len(freqMap))
	for freq := maxFreq; freq >= 0; freq-- {
		res = append(res, hashTop[freq]...)
		k -= len(hashTop[freq])
		if k == 0 {
			break
		}
	}
	return res
}

/*
692. 前K个高频单词
1.12 给一非空的单词列表，返回前k个出现次数最多的单词。
返回的答案应该按单词出现频率由高到低排序。如果不同的单词有相同出现频率，按字母顺序排序。

示例1：
输入: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
输出: ["i", "love"]
解析: "i" 和 "love" 为出现次数最多的两个单词，均为2次。
    注意，按字母顺序 "i" 在 "love" 之前。

示例2：
输入: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
输出: ["the", "is", "sunny", "day"]
解析: "the", "is", "sunny" 和 "day" 是出现次数最多的四个单词，
    出现次数依次为 4, 3, 2 和 1 次。

注意：
假定k总为有效值， 1 ≤ k ≤ 集合元素数。
输入的单词均由小写字母组成。
*/

func topKFrequent(words []string, k int) []string {
	freqMap := make(map[string]int)
	for _, word := range words {
		freqMap[word]++
	}
	uniqueWords := make([]string, len(freqMap))
	for key := range freqMap {
		uniqueWords = append(uniqueWords, key)
	}
	sort.Slice(uniqueWords, func(i, j int) bool {
		a, b := uniqueWords[i], uniqueWords[j]
		return freqMap[a] > freqMap[b] || freqMap[a] == freqMap[b] && a < b
	})
	return uniqueWords[:k]
}

/*
leetcode 460
1.13 LFU 缓存
请你为最不经常使用（LFU）缓存算法设计并实现数据结构。

实现LFUCache 类：
LFUCache(int capacity) - 用数据结构的容量capacity初始化对象
int get(int key)- 如果键key存在于缓存中，则获取键的值，否则返回 -1 。
void put(int key, int value)- 如果键key已存在，则变更其值；如果键不存在，请插入键值对。当缓存达到其容量
capacity时，则应该在插入新项之前，移除最不经常使用的项。在此问题中，当存在平局（即两个或更多个键具有相同使用频率）
时，应该去除最近最久未使用的键。
为了确定最不常使用的键，可以为缓存中的每个键维护一个使用计数器。使用计数最小的键是最久未使用的键。
当一个键首次插入到缓存中时，它的使用计数器被设置为 1 (由于 put 操作)。对缓存中的键执行 get 或 put 操作，
使用计数器的值将会递增。

函数get和put必须以 O(1) 的平均时间复杂度运行。
*/

/*
数据结构设计：
LFUCache 结构体包含了两个双向链表：freqList 和 cacheList，以及两个映射：freqMap 和 cacheMap。
freqList 用于维护所有频率桶的顺序，频率桶由 freqNode 表示。每个 freqNode 都包含一个 cacheList，
用于管理对应频率下的缓存项。
cacheMap 存储所有缓存项（cacheNode），以键值对的形式存储，便于快速查找和更新缓存项。

Get 方法：
查找缓存项时，首先检查缓存项是否存在。
如果缓存项存在，则将其从当前频率桶中移除，并更新其频率。然后将其插入到新频率桶中。
更新缓存项频率后，如果原来的频率桶为空，则删除该频率桶。

Put 方法：
如果缓存项已存在，更新其值并调用 Get 方法来更新频率。
如果缓存项不存在且缓存已满，移除最不常使用的缓存项。
*/

// LFUCache 结构体定义
type LFUCache struct {
	// 频率链表，维护所有频率的桶，每个 freqNode 通过 prev 和 next 指针与其他频率桶连接，
	// 形成一个频率链表（freqList）。这个链表的作用是帮助快速查找最小频率桶，并便于进行频率的更新。
	freqList *freqList
	// 频率 -> 频率节点映射
	freqMap map[int]*freqNode
	// 缓存链表，维护所有缓存项的插入顺序
	cacheList *cacheList
	// 缓存项 key -> 缓存节点映射
	cacheMap map[int]*cacheNode
	// 缓存容量
	capacity int
}

// Constructor LFUCache 构造函数，初始化缓存结构
func Constructor(capacity int) LFUCache {
	return LFUCache{
		freqList:  newFreqList(),
		freqMap:   make(map[int]*freqNode),
		cacheList: newCacheList(),
		cacheMap:  make(map[int]*cacheNode),
		capacity:  capacity,
	}
}

// Get 方法：获取指定 key 的值，若 key 不存在则返回 -1
func (lfu *LFUCache) Get(key int) int {
	// 如果缓存容量为 0，直接返回 -1
	if lfu.capacity == 0 {
		return -1
	}
	// 在 cacheMap 中查找 key 对应的缓存节点
	node, ok := lfu.cacheMap[key]
	// 若找不到该 key，则返回 -1
	if !ok {
		return -1
	}
	// key 存在，更新该缓存项的频率
	// 首先将该节点从当前缓存链表中移除
	node.remove()
	// 获取该节点的当前频率
	freq := node.frequency
	// 获取该频率桶
	fNode := lfu.freqMap[freq]
	// 尝试获取该频率 + 1 的频率桶
	newFreqNode, ok := lfu.freqMap[freq+1]
	if !ok {
		// 如果没有频率桶，则创建一个新的频率桶
		newFreqNode = &freqNode{
			frequency: freq + 1,
			data:      newCacheList(), // 新建一个缓存链表
		}
		// 将新频率桶插入到当前频率桶后面
		fNode.addBehind(newFreqNode)
		// 在频率映射中记录新频率桶
		lfu.freqMap[freq+1] = newFreqNode
	}
	// 增加缓存节点的频率
	node.frequency++
	// 将该节点插入到新缓存链表的头部
	newFreqNode.data.addToHead(node)
	// 如果原缓存链表变空，则删除该频率桶
	if fNode.data.isEmpty() {
		fNode.remove()
		delete(lfu.freqMap, freq)
	}
	// 返回缓存节点的值
	return node.value
}

// Put 方法：插入或更新缓存项，如果缓存满则移除最不常使用的项
func (lfu *LFUCache) Put(key int, value int) {
	// 如果缓存容量为 0，则不执行任何操作
	if lfu.capacity == 0 {
		return
	}

	node, ok := lfu.cacheMap[key]
	if ok {
		// 如果 key 已存在，更新值并调用 Get 更新频率
		lfu.Get(key)
		node.value = value
		return
	}

	// 如果缓存已满，需要删除最不常使用的缓存项
	if len(lfu.cacheMap) >= lfu.capacity {
		// 获取最小频率桶
		fNode := lfu.freqList.head.next
		// 获取该桶中最久未使用的缓存项
		delNode := fNode.data.tail.prev
		// 从缓存链表中删除该节点
		delNode.remove()
		// 从缓存映射中删除该缓存项
		delete(lfu.cacheMap, delNode.key)
		// 如果缓存链表为空且该频率大于 1，则删除该频率桶
		// 在 LFU 缓存中，对于频率为 1 的桶来说，我们不希望删除它，因为它是初始状态下所有缓存项的频率。
		// 即使所有项都被删除，也不应该删除频率为 1 的桶。
		if fNode.data.isEmpty() && fNode.frequency > 1 {
			delete(lfu.freqMap, fNode.frequency)
		}
	}

	// 插入新的缓存项
	fNode, ok := lfu.freqMap[1]
	if !ok {
		// 如果频率为 1 的桶不存在，创建一个新的频率桶
		fNode = &freqNode{
			frequency: 1,
			data:      newCacheList(),
		}
		// 将新的频率桶插入到频率链表的头部
		lfu.freqList.addToHead(fNode)
		// 在频率映射中记录该频率桶
		lfu.freqMap[1] = fNode
	}

	// 创建新的缓存节点并将其插入到频率为 1 的频率桶中
	newCacheNode := &cacheNode{
		key:       key,
		value:     value,
		frequency: 1,
	}
	// 将新缓存节点插入到缓存链表的头部
	fNode.data.addToHead(newCacheNode)
	// 在缓存映射中记录该缓存节点
	lfu.cacheMap[key] = newCacheNode
}

// 频率链表的实现，双向链表用于维护频率桶的顺序
type freqList struct {
	head *freqNode
	tail *freqNode
}

/*
freqNode, 频率节点，表示一个频率桶
data 字段是一个指向 cacheList 的指针。cacheList 用于存储在该频率桶下的所有缓存节点（cacheNode）。
它是一个双向链表，维护该频率桶下缓存项的顺序。
为什么要这样设计？
分组缓存项按频率：
在 LFU 缓存中，缓存项需要根据访问频率进行组织。每个缓存项都有一个访问频率。通过 freqNode 和 cacheList 的组合，
我们能够把所有具有相同频率的缓存项分组在一起。freqNode 用于表示频率桶，而 cacheList 用于存储具有相同频率的缓存项。
频率桶帮助我们将缓存项按频率排序，便于查找和删除最少使用的缓存项。

频率更新：
当一个缓存项的频率发生变化时（例如被访问或更新），它需要从当前的频率桶中移除，并插入到新的频率桶中。
cacheList 作为双向链表，允许我们高效地在一个桶中进行插入和删除操作。而 freqNode 通过 data 指向这个链表，使得
每个频率桶都能动态管理和更新其对应的缓存项。

提高效率：
将缓存项按频率划分到不同的桶（freqNode）中，且每个频率桶内的缓存项通过 cacheList 组织，使得频率的更新、删除最少使用的
项等操作都能够在常数时间内完成。
例如，若缓存已满，我们需要删除最不常用的缓存项。在这个设计中，最不常用的项通常位于频率最小的桶中（freqNode 的 frequency 最小）。
通过 cacheList，我们可以在该频率桶中轻松获取和删除最久未访问的缓存项。

维护频率链表的顺序：
freqList 是一个双向链表，维护了频率桶的顺序（从最低频率到最高频率）。当某个缓存项的访问频率发生变化时，它会被从当前的
cacheList 移动到另一个频率桶的 cacheList 中。
freqNode 通过指向 cacheList 来管理和更新同一频率下的所有缓存项，cacheList 在此过程中作为一个管理容器，允许高效
的增删操作。
*/
type freqNode struct {
	frequency int
	data      *cacheList
	prev      *freqNode
	next      *freqNode
}

// 创建新的频率链表
func newFreqList() *freqList {
	headNode := &freqNode{}
	tailNode := &freqNode{}
	headNode.next = tailNode
	tailNode.prev = headNode
	return &freqList{
		head: headNode,
		tail: tailNode,
	}
}

// 从频率链表中移除该频率节点
func (f *freqNode) remove() {
	f.prev.next = f.next
	f.next.prev = f.prev
}

// 将频率节点插入到当前频率节点之后
func (f *freqNode) addBehind(node *freqNode) {
	node.prev = f
	node.next = f.next
	f.next.prev = node
	f.next = node
}

// 将频率节点插入到链表头部
func (fl *freqList) addToHead(node *freqNode) {
	node.prev = fl.head
	node.next = fl.head.next
	fl.head.next.prev = node
	fl.head.next = node
}

// 缓存链表，双向链表用于维护缓存项的顺序
type cacheList struct {
	head *cacheNode
	tail *cacheNode
}

// 缓存节点，表示一个缓存项
type cacheNode struct {
	key       int
	value     int
	frequency int
	prev      *cacheNode
	next      *cacheNode
}

// 创建新的缓存链表
func newCacheList() *cacheList {
	headNode := &cacheNode{}
	tailNode := &cacheNode{}
	headNode.next = tailNode
	tailNode.prev = headNode
	return &cacheList{
		head: headNode,
		tail: tailNode,
	}
}

// 从缓存链表中移除该缓存节点
func (c *cacheNode) remove() {
	c.prev.next = c.next
	c.next.prev = c.prev
}

// 将缓存节点插入到链表头部
func (cl *cacheList) addToHead(node *cacheNode) {
	node.prev = cl.head
	node.next = cl.head.next
	cl.head.next.prev = node
	cl.head.next = node
}

// 判断缓存链表是否为空
func (cl *cacheList) isEmpty() bool {
	return cl.head.next == cl.tail
}

/*
leetcode 380 O(1)时间复杂度插入，删除和获取随机元素

实现RandomizedSet 类：
RandomizedSet() 初始化 RandomizedSet 对象
bool insert(int val) 当元素 val 不存在时，向集合中插入该项，并返回 true ；否则，返回 false 。
bool remove(int val) 当元素 val 存在时，从集合中移除该项，并返回 true ；否则，返回 false 。
int getRandom() 随机返回现有集合中的一项（测试用例保证调用此方法时集合中至少存在一个元素）。每个元素应该有相同的概率被返回。
你必须实现类的所有函数，并满足每个函数的 平均 时间复杂度为 O(1) 。


示例：
输入
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
输出
[null, true, false, true, 2, true, false, 2]

解释
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // 向集合中插入 1 。返回 true 表示 1 被成功地插入。
randomizedSet.remove(2); // 返回 false ，表示集合中不存在 2 。
randomizedSet.insert(2); // 向集合中插入 2 。返回 true 。集合现在包含 [1,2] 。
randomizedSet.getRandom(); // getRandom 应随机返回 1 或 2 。
randomizedSet.remove(1); // 从集合中移除 1 ，返回 true 。集合现在包含 [2] 。
randomizedSet.insert(2); // 2 已在集合中，所以返回 false 。
randomizedSet.getRandom(); // 由于 2 是集合中唯一的数字，getRandom 总是返回 2 。

提示：
-231 <= val <= 231 - 1
最多调用 insert、remove 和 getRandom 函数 2 * 105 次
在调用 getRandom 方法时，数据结构中 至少存在一个 元素。
*/

/*
思路:哈希表+动态数组
设计思路：

插入 (insert)：
如果元素不在集合中，则将其添加到数组的末尾，并将该元素和数组索引映射到哈希表。
时间复杂度：O(1)。

删除 (remove)：
如果元素在集合中，先找到其在数组中的索引，然后用数组最后一个元素替代该位置的元素，这样就能保持数组的连续性。
然后更新哈希表，删除该元素的映射。
时间复杂度：O(1)。

获取随机数 (getRandom)：
从数组中随机选择一个元素，这可以通过 rand.Intn(len(array)) 轻松实现，时间复杂度 O(1)。
*/

type RandomizedSet struct {
	// 哈希表，用来存储元素值到数组索引的映射
	indexMap map[int]int
	// 数组，用来存储集合的元素
	nums []int
}

func Construct() RandomizedSet {
	// 初始化随机数生成器
	rand.Seed(time.Now().UnixNano())
	return RandomizedSet{
		indexMap: make(map[int]int),
		nums:     []int{},
	}
}

func (rs *RandomizedSet) Insert(val int) bool {
	// val 已经存在
	if _, exists := rs.indexMap[val]; exists {
		return false
	}
	// 将 val 添加到数组末尾
	rs.nums = append(rs.nums, val)
	// 更新哈希表，存储 val 在数组中的索引
	rs.indexMap[val] = len(rs.nums) - 1
	return true
}

func (rs *RandomizedSet) Remove(val int) bool {
	if index, exists := rs.indexMap[val]; exists {
		lastIdx := len(rs.nums) - 1
		lastVal := rs.nums[lastIdx]
		// 将数组最后一个元素移到要删除的位置
		rs.nums[index] = lastVal
		// 更新哈希表中 lastVal 的索引
		rs.indexMap[lastVal] = index
		//  删除 val
		rs.nums = rs.nums[:lastIdx]
		// 从哈希表中删除 val 的映射
		delete(rs.indexMap, val)
		return true
	}
	return false
}

func (rs *RandomizedSet) GetRandom() int {
	// 获取一个随机索引
	randIdx := rand.Intn(len(rs.nums))
	// 返回对应的元素
	return rs.nums[randIdx]
}

/*
万能字符单词拼写
有一个字符串数组 words 和一个字符串 chars。假如可以用 chars 中的字母拼写出 words 中的某个"单词"（字符串)，那么我们就认为
你掌握了这个单词。

words 的字符仅由 a-z 英文小写宁母组成，例如“abc”。
chars 由 a- z 英文小写字母和“?”组成，其中英文“?"表示万能字符，能够在拼写时当作任意一个英文字母。例如“?"可以当作"a"等字母。

注意: 每次拼写时，chars 中的每个字母和万能字符都只能使用一次。
输出词汇表 words 中你掌握的所有单词的个数。没有掌握任何单词，则输出0。

输入描述
第一行: 输入数组 words 的个数，记作N。
第二行~第N+1行: 依次输入数组words的每个字符串元素。
第N+2行: 输入字符串 chars

输出描述
输出一个整数，表示词汇表 words 中你掌握的单词个数

备注
1 <= words.length <= 100
1 <= words[i].length, chars.length <= 100
所有字符串中都仅包含小写英文字母、英文问号

示例1
输入
4
cat
bt
hat
tree
atach??

输出
3
说明:可以掌握的单词 "cat”、“bt"和"hat"。
*/

func countFrequency(s string) [26]int {
	freq := [26]int{}
	for _, char := range s {
		if char >= 'a' && char <= 'z' {
			freq[char-'a']++
		}
	}
	return freq
}

func masterWordsCount(words []string, chars string) int {
	charFreq := countFrequency(chars)
	// 统计chars中的万能字符(?)的数量
	markCount := 0
	for _, char := range chars {
		if char == '?' {
			markCount++
		}
	}
	masterCount := 0
	for _, word := range words {
		fillCount := 0
		wordFreq := countFrequency(word)
		for i := 0; i < 26; i++ {
			if wordFreq[i] > charFreq[i] {
				fillCount += wordFreq[i] - charFreq[i]
			}
		}
		// 如果需要补的字符数不超过chars中的万能字符(?)的数量，则表示已掌握该单词word
		if fillCount <= markCount {
			masterCount++
		}
	}
	return masterCount
}

/*
小朋友至少有几个
幼儿园组织活动，老师布置了一个任务：
每个小朋友去了解与自己同一个小区的小朋友还有几个。
我们将这些数量汇总到数组 garden 中。
请根据这些小朋友给出的信息，计算小朋友至少有几个?

输入描述
输入：garden[] = {2, 2, 3}

说明：
garden 数组长度最大为 999
每个小区的小朋友数量最多 1000 人，也就是 garden[i] 的范围为 [0, 999]

输出描述
输出：7

示例1
输入：
2 2 3

输出：
7

说明：
第一个小朋友反馈有两个小朋友和自己同一小区，即此小区有3个小朋友。
第二个小朋友反馈有两个小朋友和自己同一小区，即此小区有3个小朋友。
这两个小朋友，可能是同一小区的，且此小区的小朋友只有3个人。
第三个小区反馈还有3个小朋友与自己同一小区，则这些小朋友只能是另外一个小区的。这个小区有4个小朋友。
*/

func countKids(garden []int) int {
	n := len(garden)
	freqMark := make(map[int]int, n)
	for i := 0; i < n; i++ {
		freqMark[garden[i]]++
	}
	totalKids := 0
	for k, freq := range freqMark {
		if k == 0 {
			totalKids += freq
		} else {
			numGroups := int(math.Ceil(float64(freq) / float64(k+1)))
			totalKids += numGroups * (k + 1)
		}
	}
	return totalKids
}

/*
内存冷热标记
现代计算机系统通常存在多级的存储设备，针对海量的负载优化的一种思路是将热点内存页优化先放到快速存储层级，这就需要对内存页
进行冷热标记。
一种典型的方案是基于内存页的访问频次进行标记，如果统计窗口内访问次数大于等于设定阈值，要实现基于频次的冷热标记。内存页使用
页框号作为标识。

输入描述
第一行输入为 N, 表示访存序列的记录条数， 0 < N ≤ 10000。
第二行为访问内存序列，空格间隔的 N 个内存页框号，页面号范围 0 ~ 65535，同一个页框号可能重复出现，出现的次数即为对应框号的频次。
第三行为热内存的频次阈值 T ，正整数范围 1 ≤ T ≤ 10000。

输出描述
第一行为输出标记为热内存的内存页个数，如果没有被标记为热内存的，则输出 0。
如果第一行大于 0，则接下来按照访问频次降序输出内存页框号，一行一个，频次一样的页框号，页框号小的排前面。

示例1
输入：
10
1 2 1 2 1 2 1 2 1 2
5

输出：
2
1
2

说明：
内存页 1 和内存页 2 均被访问了5 次，达到了阈值5 ，因此热内存页有2个。内存页1 和内存页 2 的访问频次相等，页框号小的排前面。

示例2
输入：
5
1 2 3 4 5
3

输出：
0

说明：
从访问跟踪里面访问频次没有超过 3 的，因此热内存个数为 0。
*/

// PageFreq 定义页面号和频次的结构体
type PageFreq struct {
	// 页面号
	Page int
	// 访问频次
	Freq int
}

func markHotAndColdMemory(T int, pages []int) []PageFreq {
	n := len(pages)
	freq := make(map[int]int, n)
	for _, page := range pages {
		freq[page]++
	}
	// 筛选热内存页面,预分配容量
	hotPages := make([]PageFreq, 0, len(freq))
	for page, cnt := range freq {
		if cnt >= T {
			hotPages = append(hotPages, PageFreq{Page: page, Freq: cnt})
		}
	}
	sort.Slice(hotPages, func(i, j int) bool {
		if hotPages[i].Freq == hotPages[j].Freq {
			return hotPages[i].Page < hotPages[j].Page
		}
		return hotPages[i].Freq > hotPages[j].Freq
	})
	return hotPages
}

/*
剑指offer 50  第一个只出现一次的字符
在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。
示例 1：

输入：arr = "abbccdeff"
输出：'a'
示例 2：

输入：arr = "ccdd"
输出：' '

限制：
0 <= arr.length <= 50000
*/

/*
使用一个map，遍历一遍字符串s，计算出每个字符出现的次数
再遍历一遍字符串s，找到第一个出现次数为1的字符
*/

func firstAppearOnceChar(s string) byte {
	n := len(s)
	mark := make(map[byte]int, n)
	for i := 0; i < n; i++ {
		mark[s[i]]++
	}
	for i := 0; i < n; i++ {
		if mark[s[i]] == 1 {
			return s[i]
		}
	}
	return ' '
}
