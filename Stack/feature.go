package stack

import (
	"algorithm-practise/utils"
	"math"
	"strconv"
	"strings"
)

/*
leetcode 20. 有效的括号
1.1  有效的括号
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串s ，判断字符串是否有效。
有效字符串需满足：
左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。

示例1：
输入：s = "()"
输出：true
示例2：

输入：s = "()[]{}"
输出：true
示例3：

输入：s = "(]"
输出：false
示例4：

输入：s = "([)]"
输出：false
示例5：

输入：s = "{[]}"
输出：true
*/

/*
栈+哈希表解决
我们遍历给定的字符串s。当我们遇到一个左括号时，我们会期望在后续的遍历中，有一个相同类型的右括号将其闭合。由于后
遇到的左括号要先闭合，因此我们可以将这个左括号放入栈顶。当我们遇到一个右括号时，我们需要一个相同类型的左括号闭合。
此时，我们可以取出栈顶的左括号并判断它们是否是相同类型的括号。如果不是相同的类型，或者栈中并没有左括号，那么字符串
s无效，返回false。为了快速判断括号的类型，我们可以使用哈希表存储每一种括号。哈希表的键为右括号，值为相同类型的
左括号。在遍历结束后，如果栈中没有左括号，说明我们将字符串s中的所有左括号闭合，返回true，否则返回false。
注意到有效字符串的长度一定为偶数，因此如果字符串的长度为奇数，我们可以直接返回false，省去后续的遍历判断过程。
*/

func IsValid(s string) bool {
	length := len(s)
	// 如果字符串的长度为奇数，我们可以直接返回False
	if length%2 == 1 {
		return false
	}
	pairs := map[byte]byte{')': '(', ']': '[', '}': '{'}
	var stack []byte
	for i := 0; i < length; i++ {
		if s[i] == '(' || s[i] == '[' || s[i] == '{' {
			stack = append(stack, s[i])
		} else if len(stack) > 0 && stack[len(stack)-1] == pairs[s[i]] {
			stack = stack[:len(stack)-1]
		} else {
			return false
		}
	}
	return len(stack) == 0
}

/*
leetcode 232. 用栈实现队列
1.2 请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：

实现MyQueue类：
void push(int x) 将元素 x 推到队列的末尾
int pop() 从队列的开头移除并返回元素
int peek() 返回队列开头的元素
boolean empty() 如果队列为空，返回true；否则返回false
说明：
你只能使用标准的栈操作 —— 也就是只有push to top,peek/pop from top,size和is empty操作是合法的。
你所使用的语言也许不支持栈。你可以使用list或者deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。

进阶：
你能否实现每个操作均摊时间复杂度为O(1)的队列？换句话说，执行n个操作的总时间复杂度为 O(n) ，即使
其中一个操作可能花费较长时间。
*/

type MyQueue struct {
	InputStack  []int
	OutputStack []int
}

func Constructor() *MyQueue {
	return &MyQueue{
		InputStack:  make([]int, 0),
		OutputStack: make([]int, 0),
	}
}

// Push element x to the back of queue.
func (q *MyQueue) Push(x int) {
	q.InputStack = append(q.InputStack, x)
}

// Pop Removes the element from in front of queue and returns that element.
func (q *MyQueue) Pop() int {
	if len(q.OutputStack) == 0 {
		for len(q.InputStack) > 0 {
			q.OutputStack = append(q.OutputStack, q.InputStack[0])
			q.InputStack = q.InputStack[1:]
		}
	} else {
		if len(q.InputStack) > 0 {
			q.OutputStack = append(q.OutputStack, q.InputStack[0])
		}
	}
	value := q.OutputStack[0]
	q.OutputStack = q.OutputStack[1:]
	return value
}

// Peek Get the front element.
func (q *MyQueue) Peek() int {
	if len(q.OutputStack) == 0 {
		for len(q.InputStack) > 0 {
			q.OutputStack = append(q.OutputStack, q.InputStack[0])
			q.InputStack = q.InputStack[1:]
		}
	} else {
		if len(q.InputStack) > 0 {
			q.OutputStack = append(q.OutputStack, q.InputStack[0])
		}
	}
	return q.OutputStack[0]
}

// Empty Returns whether the queue is empty.
func (q *MyQueue) Empty() bool {
	if len(q.InputStack) == 0 && len(q.OutputStack) == 0 {
		return true
	}
	return false
}

/*
leetcode 155. 最小栈
1.3 设计一个支持 push，pop，top操作，并能在常数时间内检索到最小元素的栈。

push(x) —— 将元素 x 推入栈中。
pop()—— 删除栈顶的元素。
top()—— 获取栈顶元素。
getMin() —— 检索栈中的最小元素。

pop、top和getMin操作总是在非空栈上调用。
*/

type MinStack struct {
	stack    []int
	minStack []int
}

func Construct() MinStack {
	return MinStack{
		stack:    []int{},
		minStack: []int{math.MaxInt64},
	}
}

func (ms *MinStack) Push(val int) {
	ms.stack = append(ms.stack, val)
	top := ms.minStack[len(ms.minStack)-1]
	ms.minStack = append(ms.minStack, utils.Min(top, val))
}

func (ms *MinStack) Pop() {
	ms.stack = ms.stack[:len(ms.stack)-1]
	ms.minStack = ms.minStack[:len(ms.minStack)-1]
}

func (ms *MinStack) Top() int {
	return ms.stack[len(ms.stack)-1]
}

func (ms *MinStack) GetMin() int {
	return ms.minStack[len(ms.minStack)-1]
}

/*
leetcode 1047. 删除字符串中的所有相邻重复项
1.4 给出由小写字母组成的字符串S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。
在S上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

示例：
输入："abbaca"
输出："ca"
解释：
例如，在 "abbaca" 中，我们可以删除"bb"由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们
得到字符串 "aaca"，其中又只有"aa"可以执行重复项删除操作，所以最后的字符串为"ca"。
*/

func RemoveDuplicates(s string) string {
	var stack []byte
	n := len(s)
	for i := 0; i < n; i++ {
		if len(stack) > 0 && stack[len(stack)-1] == s[i] {
			stack = stack[:len(stack)-1]
		} else {
			stack = append(stack, s[i])
		}
	}
	return string(stack)
}

/*
leetcode 1209. 删除字符串中的所有相邻重复项II
1.5 给你一个字符串s，「k 倍重复项删除操作」将会从s中选择k个相邻且相等的字母，并删除它们，使被删去的字符串的
左侧和右侧连在一起。
你需要对s重复进行无限次这样的删除操作，直到无法继续为止。
在执行完所有删除操作后，返回最终得到的字符串。
本题答案保证唯一。

示例1：
输入：s = "abcd", k = 2
输出："abcd"
解释：没有要删除的内容。

示例2：
输入：s = "deeedbbcccbdaa", k = 3
输出："aa"
解释：
先删除 "eee" 和 "ccc"，得到 "ddbbbdaa"
再删除 "bbb"，得到 "dddaa"
最后删除 "ddd"，得到 "aa"

示例3：
输入：s = "pbbcggttciiippooaais", k = 2
输出："ps"

提示：
1 <= s.length <= 10^5
2 <= k <= 10^4
s中只含有小写英文字母。
*/

/*
思路:栈+双指针
初始慢指针j等于0。使用快指针i遍历字符串：
进入循环后第一个操作是令s[i] = s[j]。
如果j > 0 && s[j] = s[j - 1]，则栈顶元素加1。否则，栈中压入1。
如果计数器等于k，j = j - k，并弹出栈顶元素。
返回字符串前j个字符。
*/

func RemoveDuplicatesComplex(s string, k int) string {
	n := len(s)
	if n < k {
		return s
	}
	var stack []int
	j := 0
	ss := []byte(s)
	for i := 0; i < n; i++ {
		ss[j] = ss[i]
		if j == 0 || ss[j] != ss[j-1] {
			stack = append(stack, 1)
		} else {
			length := len(stack)
			increment := stack[length-1] + 1
			if increment == k {
				j -= k
				stack = stack[:length-1]
			} else {
				stack[length-1] = increment
			}
		}
		j++
	}
	return string(ss[:j])
}

/*
leetcode 150. 逆波兰表达式求值
1.6 根据逆波兰表示法，求表达式的值。
有效的算符包括+、-、*、/。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

说明：
整数除法只保留整数部分。
给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。


示例1：
输入：tokens = ["2","1","+","3","*"]
输出：9
解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9

示例2：
输入：tokens = ["4","13","5","/","+"]
输出：6
解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6

示例3：
输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
输出：22
解释：
该算式转化为常见的中缀算术表达式为：
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22


提示：
1 <= tokens.length <= 104
tokens[i] 要么是一个算符（"+"、"-"、"*" 或 "/"），要么是一个在范围 [-200, 200] 内的整数

逆波兰表达式：
逆波兰表达式是一种后缀表达式，所谓后缀就是指算符写在后面。
平常使用的算式则是一种中缀表达式，如 ( 1 + 2 ) * ( 3 + 4 ) 。
该算式的逆波兰表达式写法为 ( ( 1 2 + ) ( 3 4 + ) * ) 。
逆波兰表达式主要有以下两个优点：
去掉括号后表达式无歧义，上式即便写成 1 2 + 3 4 + * 也可以依据次序计算出正确结果。
适合用栈操作运算：遇到数字则入栈；遇到算符则取出栈顶两个数字进行计算，并将结果压入栈中。
*/

// EvalRPN 按照题意写代码即可
func EvalRPN(tokens []string) int {
	var stack []int
	for _, v := range tokens {
		num, err := strconv.Atoi(v)
		if err == nil {
			// 遇到数字入栈
			stack = append(stack, num)
		} else {
			// 遇到运算符，取出栈顶的两个数字进行计算，并将结果压入栈中
			n := len(stack)
			n1, n2 := stack[n-2], stack[n-1]
			stack = stack[:n-2]
			switch v {
			case "+":
				stack = append(stack, n1+n2)
			case "-":
				stack = append(stack, n1-n2)
			case "*":
				stack = append(stack, n1*n2)
			case "/":
				stack = append(stack, n1/n2)
			}
		}

	}
	// 返回栈中所剩的最后一个元素
	return stack[0]
}

/*
leetcode 32. 最长有效括号
1.7 最长有效括号
给你一个只包含 '('和 ')'的字符串，找出最长有效（格式正确且连续）括号子串的长度。

示例1：
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"

示例2：
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"

示例3：
输入：s = ""
输出：0

提示：
0 <= s.length <= 3 * 104
s[i] 为 '(' 或 ')'
*/

/*
通过栈，我们可以在遍历给定字符串的过程中去判断到目前为止扫描的子串的有效性，同时能得到最长有效括号的长度。
具体做法是我们始终保持栈底元素为当前已经遍历过的元素中「最后一个没有被匹配的右括号的下标」，这样的做法主要是
考虑了边界条件的处理，栈里其他元素维护左括号的下标：

对于遇到的每个‘(’ ，我们将它的下标放入栈中
对于遇到的每个‘)’ ，我们先弹出栈顶元素表示匹配了当前右括号：
如果栈为空，说明当前的右括号为没有被匹配的右括号，我们将其下标放入栈中来更新我们之前提到的「最后一个没有被匹配的
右括号的下标」
如果栈不为空，当前右括号的下标减去栈顶元素即为「以该右括号为结尾的最长有效括号的长度」
我们从前往后遍历字符串并更新答案即可。

需要注意的是，如果一开始栈为空，第一个字符为左括号的时候我们会将其放入栈中，这样就不满足提及的「最后一个没有
被匹配的右括号的下标」，为了保持统一，我们在一开始的时候往栈中放入一个值为−1 的元素。
*/

// LongestValidParentheses 时间复杂度O(N), 空间复杂度O(N)
func LongestValidParentheses(s string) int {
	maxLength, n := 0, len(s)
	// 创建一个栈，并压入 -1 作为初始标记（-1 是一个边界值，帮助计算长度）。
	stack := []int{-1}
	for i := 0; i < n; i++ {
		// 遇到左括号，将其索引压入栈中。
		if s[i] == '(' {
			stack = append(stack, i)
		} else {
			// 遇到右括号，弹出栈顶元素与右括号匹配
			stack = stack[:len(stack)-1]
			// 连续性保证：算法的目标是计算最长的连续有效括号子串。压入无效右括号的索引后，栈顶总能正确反映当前子串的
			// 左边界前一个位置，确保计算不会跨越无效位置。
			// 如果弹出后栈为空，说明没有匹配的左括号，将当前右括号的索引压入栈中，作为新的无效位置标记。
			if len(stack) == 0 {
				stack = append(stack, i)
			} else {
				/*
					在括号匹配问题中，我们通常使用栈来跟踪括号的位置。这里，i 是当前右括号 ')' 的索引，而
					stack[len(stack)-1] 是栈顶元素的索引。栈顶元素可能是：
					一个未匹配的左括号 '(' 的索引；
					或者是一个无效的右括号 ')' 的索引（之前未能配对时压入的）。
					当我们遇到一个右括号 ')' 并成功匹配（即弹出栈顶的左括号后），需要计算当前有效子串的长度。
					计算公式 currentLen := i - stack[len(stack)-1] 的含义是：
					stack[len(stack)-1] 表示当前有效子串的“左边界”前一个位置；
					i 是当前有效子串的右边界；
					因此，i - stack[len(stack)-1] 直接给出了从左边界后一个位置到右边界的元素个数，也就是有效子串的长度。
				*/
				maxLength = utils.Max(maxLength, i-stack[len(stack)-1])
			}
		}
	}
	return maxLength
}

/*
第二种解法:贪心
我们利用两个计数器left和right 。首先，我们从左到右遍历字符串，对于遇到的每个‘(’，我们增加left计数器，对于遇到的每个
‘)’ ，我们增加right计数器。每当left计数器与right计数器相等时，我们计算当前有效字符串的长度，并且记录目前为止找到
的最长子字符串。当right计数器比left计数器大时，我们将left和right计数器同时重置为0。

这样的做法贪心地考虑了以当前字符下标结尾的有效括号长度，每次当右括号数量多于左括号数量的时候之前的字符我们都扔掉
不再考虑，重新从下一个字符开始计算，但这样会漏掉一种情况，就是遍历的时候左括号的数量始终大于右括号的数量，即 (()，
这种时候最长有效括号是求不出来的。

解决的方法也很简单，我们只需要从右往左遍历用类似的方法计算即可，只是这个时候判断条件反了过来：
当left计数器比right计数器大时，我们将left和right计数器同时重置为0。
当left 计数器与right计数器相等时，我们计算当前有效字符串的长度，并且记录目前为止找到的最长子字符串
这样我们就能涵盖所有情况从而求解出答案。
*/

// LongestValidParenthesesSimple 时间复杂度O(N), 空间复杂度O(1)
func LongestValidParenthesesSimple(s string) int {
	left, right, maxLength, n := 0, 0, 0, len(s)
	for i := 0; i < n; i++ {
		if s[i] == '(' {
			left++
		} else {
			right++
		}
		if left == right {
			maxLength = utils.Max(maxLength, 2*right)
		} else if right > left {
			left, right = 0, 0
		}
	}
	left, right = 0, 0
	for i := n - 1; i >= 0; i-- {
		if s[i] == '(' {
			left++
		} else {
			right++
		}
		if left == right {
			maxLength = utils.Max(maxLength, 2*right)
		} else if left > right {
			left, right = 0, 0
		}
	}
	return maxLength
}

/*
第三种解法:动态规划
*/

// LongestValidParenthesesComplex 时间复杂度O(N), 空间复杂度O(N)
func LongestValidParenthesesComplex(s string) int {
	n := len(s)
	dp := make([]int, n)
	maxLength := 0
	for i := 1; i < n; i++ {
		if s[i] == ')' {
			if s[i-1] == '(' {
				if i >= 2 {
					dp[i] = dp[i-2] + 2
				} else {
					dp[i] = 2
				}
			} else if i-dp[i-1] > 0 && s[i-dp[i-1]-1] == '(' {
				if i-dp[i-1] >= 2 {
					dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2
				} else {
					dp[i] = dp[i-1] + 2
				}
			}
			maxLength = utils.Max(maxLength, dp[i])
		}
	}
	return maxLength
}

/*
leetcode 678. 有效的括号字符串
1.8 给定一个只包含三种字符的字符串：（，),和 *，写一个函数来检验这个字符串是否为有效字符串。有效字符串具有如下规则：
任何左括号(必须有相应的右括号)。
任何右括号)必须有相应的左括号(。
左括号(必须在对应的右括号之前)。
*可以被视为单个右括号)，或单个左括号(，或一个空字符串。
一个空字符串也被视为有效字符串。

示例1:
输入: "()"
输出: True

示例2:
输入: "(*)"
输出: True

示例3:
输入: "(*))"
输出: True

注意:
字符串大小将在 [1，100] 范围内。
*/

/*
括号匹配的问题可以用栈求解。
如果字符串中没有星号，则只需要一个栈存储左括号，在从左到右遍历字符串的过程中检查括号是否匹配。
在有星号的情况下，需要两个栈分别存储左括号和星号。从左到右遍历字符串，进行如下操作。
如果遇到左括号，则将当前下标存入左括号栈。
如果遇到星号，则将当前下标存入星号栈。
如果遇到右括号，则需要有一个左括号或星号和右括号匹配，由于星号也可以看成右括号或者空字符串，因此当前的右括号应
优先和左括号匹配，没有左括号时再和星号匹配。

所以有:
如果左括号栈不为空，则从左括号栈弹出栈顶元素；
如果左括号栈为空且星号栈不为空，则从星号栈弹出栈顶元素；
如果左括号栈和星号栈都为空，则表示没有字符可以和当前的右括号匹配，返回false。

遍历结束之后，左括号栈和星号栈可能还有元素。为了将每个左括号匹配，需要将星号看成右括号，且每个左括号必须出现在其
匹配的星号之前。当两个栈都不为空时，每次从左括号栈和星号栈分别弹出栈顶元素，对应左括号下标和星号下标，判断是否
可以匹配，匹配的条件是左括号下标小于星号下标，如果左括号下标大于星号下标则返回false。
最终判断左括号栈是否为空。如果左括号栈为空，则左括号全部匹配完毕，剩下的星号都可以看成空字符串，此时s是有效的
括号字符串，返回true。如果左括号栈不为空，则还有左括号无法用有括号或星号来匹配，此时s不是有效的括号字符串，
返回false。
*/

// CheckValidString 时间复杂度O(N), 空间复杂度O(N)
func CheckValidString(s string) bool {
	var leftStack, starStack []int
	for i, v := range s {
		if v == '(' {
			leftStack = append(leftStack, i)
		} else if v == '*' {
			starStack = append(starStack, i)
		} else {
			// 优先从左括号栈中弹出栈顶元素与当前右括号匹配
			if len(leftStack) > 0 {
				leftStack = leftStack[:len(leftStack)-1]
				// 若左括号栈为空，则从星号栈中弹出栈顶元素与当前右括号匹配
			} else if len(starStack) > 0 {
				starStack = starStack[:len(starStack)-1]
				// 若左括号栈和星号栈都弹出所有元素了，此时还有右括号，则当前右括号无法匹配
				// 返回false
			} else {
				return false
			}
		}
	}
	m, n := len(leftStack), len(starStack)
	// 若此时左括号栈为空，则无论星号栈是否为空，都返回true
	// 因为星号可以当成空字符串，而空字符串也被视为有效括号字符串
	if m == 0 {
		return true
	}
	i := m - 1
	for j := n - 1; i >= 0 && j >= 0; i, j = i-1, j-1 {
		// 将星号栈中的星号当做右括号与左括号匹配
		// 如果左括号出现在星号之后，则无法匹配
		if leftStack[i] > starStack[j] {
			return false
		}
	}
	// 如果左括号栈中的所有左括号都匹配完毕，那么i会减到-1
	return i == -1
}

/*
思路:贪心
使用贪心的思想，可以将空间复杂度降到 O(1)。
从左到右遍历字符串，遍历过程中，未匹配的左括号数量可能会出现如下变化：
如果遇到左括号，则未匹配的左括号数量加1；
如果遇到右括号，则需要有一个左括号和右括号匹配，因此未匹配的左括号数量减1；
如果遇到星号，由于星号可以看成左括号、右括号或空字符串，因此未匹配的左括号数量可能加1、减1或不变。

基于上述结论，可以在遍历过程中维护未匹配的左括号数量可能的最小值和最大值，根据遍历到的字符更新最小值和最大值：
如果遇到左括号，则将最小值和最大值分别加1；
如果遇到右括号，则将最小值和最大值分别减1；
如果遇到星号，则将最小值减1，将最大值加1。

任何情况下，未匹配的左括号数量必须非负，因此当最大值变成负数时，说明没有左括号可以和右括号匹配，返回false。
当最小值为0时，不应将最小值继续减少，以确保最小值非负。
遍历结束时，所有的左括号都应和右括号匹配，因此只有当最小值为0时，字符串s才是有效的括号字符串。
*/

// CheckValidStringSimple 时间复杂度O(N), 空间复杂度O(1)
func CheckValidStringSimple(s string) bool {
	maxCount, minCount := 0, 0
	for _, ch := range s {
		if ch == '(' {
			maxCount++
			minCount++
		} else if ch == ')' {
			minCount = utils.Max(minCount-1, 0)
			maxCount--
			if maxCount < 0 {
				return false
			}
		} else {
			minCount = utils.Max(minCount-1, 0)
			maxCount++
		}
	}
	return minCount == 0
}

/*
leetcode 84 柱状图中最大的矩形
1.9 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
求在该柱状图中，能够勾勒出来的矩形的最大面积。

示例1:
参见images目录下的 柱状图.jpeg
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
*/

/*
思路1:暴力解法,时间复杂度O(N^2)
我们需要在柱状图中找出最大的矩形，因此我们可以考虑枚举矩形的宽和高，其中「宽」表示矩形贴着柱状图底边的宽度，「高」
表示矩形在柱状图上的高度。
如果我们枚举「宽」，我们可以使用两重循环枚举矩形的左右边界以固定宽度w，此时矩形的高度h，就是所有包含在内的柱子的
最小高度，对应的面积为 w×h
*/

func largestRectangleAreaBrutal(heights []int) int {
	n := len(heights)
	res := 0
	// 枚举矩形的左边界
	for l := 0; l < n; l++ {
		minHeight := math.MaxInt32
		// 枚举矩形的右边界
		for r := l; r < n; r++ {
			// 确定矩形最小高度
			minHeight = utils.Min(minHeight, heights[r])
			// 计算矩形面积，迭代最大面积
			res = utils.Max(res, minHeight*(r-l+1))
		}
	}
	return res
}

/*
或者我们也可以枚举「高」，我们可以使用一重循环枚举某一根柱子，将其固定为矩形的高度h。随后我们从这跟柱子开始向两侧延伸，
直到遇到高度小于h的柱子，就确定了矩形的左右边界。如果左右边界之间的宽度为w，那么对应的面积为 w×h
*/
func largestRectangleAreaBrutalTwo(heights []int) int {
	n := len(heights)
	res := 0
	for i := 0; i < n; i++ {
		h := heights[i]
		l, r := i, i
		for l-1 >= 0 && heights[l-1] >= h {
			l--
		}
		for r+1 < n && heights[r+1] >= h {
			r++
		}
		res = utils.Max(res, h*(r-l+1))
	}
	return res
}

/*
思路2：单调栈
我们用一个具体的例子 [6,7,5,2,4,5,9,3]来帮助理解单调栈。我们需要求出每一根柱子的左侧且最近的小于其高度的柱子。
初始时的栈为空。

我们枚举6，因为栈为空，所以6左侧的柱子是「哨兵」，位置为 -1。随后我们将6入栈。

栈：[6(0)]。（这里括号内的数字表示柱子在原数组中的位置）
我们枚举7，由于6<7，因此不会移除栈顶元素，所以7左侧的柱子是6，位置为0。随后我们将7入栈。

栈：[6(0), 7(1)]
我们枚举5，由于7≥5，因此移除栈顶元素7。同样地，6≥5，再移除栈顶元素6。此时栈为空，所以5左侧的柱子是「哨兵」，
位置为−1。随后我们将5入栈。

栈：[5(2)]
接下来的枚举过程也大同小异。我们枚举2，移除栈顶元素5，得到2左侧的柱子是「哨兵」，位置为−1。将2入栈。

栈：[2(3)]
我们枚举4，5 和9，都不会移除任何栈顶元素，得到它们左侧的柱子分别是2，4 和5，位置分别为3，4 和5。将它们入栈。

栈：[2(3), 4(4), 5(5), 9(6)]
我们枚举3，依次移除栈顶元素9，5 和4，得到3左侧的柱子是2，位置为3。将3入栈。

栈：[2(3), 3(7)]
这样一来，我们得到它们左侧的柱子编号分别为 [−1,0,−1,−1,3,4,5,3]
用相同的方法，我们从右向左进行遍历，也可以得到它们右侧的柱子编号分别为 [2,2,3,8,7,7,7,8]
这里我们将位置8看作「哨兵」。

在得到了柱子左右两侧最近的且小于其高度的柱子之后，我们就可以计算出每根柱子对应的左右边界，并求出答案了。

*/

// largestRectangleArea 时间复杂度和空间复杂度都是O(N)
func largestRectangleArea(heights []int) int {
	n := len(heights)
	left, right := make([]int, n), make([]int, n)
	s := []int{}
	// 首先从左向右遍历，寻找每根柱子左侧最近的小于其高度的柱子位置
	for i := 0; i < n; i++ {
		// 如果栈不为空，且栈顶柱子高度大于等于当前柱子高度，说明它不是我们想要的左边界，将栈顶元素弹出
		for len(s) > 0 && heights[s[len(s)-1]] >= heights[i] {
			s = s[:len(s)-1]
		}
		// 如果栈长度为空，说明该柱子左侧没有比当前柱子更高的柱子，我们将哨兵-1入栈
		if len(s) == 0 {
			left[i] = -1
		} else {
			// 否则，栈顶元素即为左边界，即当前柱子左侧最近的且高度小于自己的柱子
			left[i] = s[len(s)-1]
		}
		// 将当前柱子位置入栈
		s = append(s, i)
	}

	// 左边界确定后，重置栈s为空，以备下一次遍历确定右边界
	s = []int{}
	// 然后从右向左遍历，寻找每根柱子右侧最近的小于其高度的柱子位置
	for i := n - 1; i >= 0; i-- {
		// 如果栈不为空，且栈顶柱子高度大于等于当前柱子高度，说明它不是我们想要的右边界，将栈顶元素弹出
		for len(s) > 0 && heights[s[len(s)-1]] >= heights[i] {
			s = s[:len(s)-1]
		}
		// 如果栈长度为空，说明该柱子左侧没有比当前柱子更高的柱子，我们将哨兵n入栈
		if len(s) == 0 {
			right[i] = n
		} else {
			// 否则，栈顶元素即为左边界，即当前柱子右侧最近的且高度小于自己的柱子
			right[i] = s[len(s)-1]
		}
		// 将当前柱子位置入栈
		s = append(s, i)
	}

	res := 0
	// 最后从左到右遍历，确定以每个柱子的高度为高的矩形的面积
	for i := 0; i < n; i++ {
		// 面积即为当前柱子的高度*宽度(也就是右边界-左边界-1)
		// 减一是因为左右边界的柱子高度都是低于当前柱子高度的，所以左右边界都不包含
		// 迭代最大矩形面积
		res = utils.Max(res, heights[i]*(right[i]-left[i]-1))
	}
	return res
}

func largestRectangleAreaSimple(heights []int) int {
	n := len(heights)
	left, right := make([]int, n), make([]int, n)
	for i := 0; i < n; i++ {
		right[i] = n
	}
	s := []int{}
	for i := 0; i < n; i++ {
		for len(s) > 0 && heights[s[len(s)-1]] >= heights[i] {
			right[s[len(s)-1]] = i
			s = s[:len(s)-1]
		}
		if len(s) == 0 {
			left[i] = -1
		} else {
			left[i] = s[len(s)-1]
		}
		s = append(s, i)
	}
	res := 0
	for i := 0; i < n; i++ {
		res = utils.Max(res, heights[i]*(right[i]-left[i]-1))
	}
	return res
}

/*
leetcode 85 最大矩形
1.10 给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。

输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：6

示例 2：
输入：matrix = [["0"]]
输出：0

示例 3：
输入：matrix = [["1"]]
输出：1

提示：
rows == matrix.length
cols == matrix[0].length
1 <= row, cols <= 200
matrix[i][j] 为 '0' 或 '1'
解释：最大矩形如(images目录下的最大矩形.jpeg)所示。
*/

func maximalRectangle(matrix [][]byte) int {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return 0
	}
	rows, cols := len(matrix), len(matrix[0])
	// 初始化高度数组
	heights := make([]int, cols)
	maxArea := 0
	// 遍历每一行
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// 更新高度数组
			if matrix[i][j] == '1' {
				heights[j] += 1
			} else {
				heights[j] = 0
			}
		}
		// 计算当前行的最大矩形面积
		maxArea = utils.Max(maxArea, largestRectangleArea(heights))
	}
	return maxArea
}

/*
leetcode 227 基本计算器II

给你一个字符串表达式s ，请你实现一个基本计算器来计算并返回它的值。
整数除法仅保留整数部分。

你可以假设给定的表达式总是有效的。所有中间结果将在 [-231, 231 - 1] 的范围内。
注意：不允许使用任何将字符串作为数学表达式计算的内置函数，比如 eval() 。

示例 1：
输入：s = "3+2*2"
输出：7

示例 2：
输入：s = " 3/2 "
输出：1

示例 3：
输入：s = " 3+5 / 2 "
输出：5

提示：
1 <= s.length <= 3 * 105
s 由整数和算符 ('+', '-', '*', '/') 组成，中间由一些空格隔开
s 表示一个 有效表达式
表达式中的所有整数都是非负整数，且在范围 [0, 231 - 1] 内
题目数据保证答案是一个 32-bit 整数
*/

func calculate(s string) int {
	stack := []int{}
	operator := '+'
	num := 0
	for i, v := range s {
		isDigit := '0' <= v && v <= '9'
		// 如果是数字，连续构造完整的数字
		if isDigit {
			num = num*10 + int(v-'0')
		}
		// 如果是操作符或最后一个字符，处理当前数字和操作符
		if !isDigit && v != ' ' || i == len(s)-1 {
			switch operator {
			case '+':
				// 直接压入栈
				stack = append(stack, num)
			case '-':
				// 压入负值
				stack = append(stack, -num)
			case '*':
				// 栈顶与当前数字相乘
				stack[len(stack)-1] *= num
			case '/':
				// 栈顶与当前数字相除
				stack[len(stack)-1] /= num
			}
			// 更新操作符
			operator = v
			// 重置当前数字
			num = 0
		}
	}
	// 计算最终结果
	res := 0
	for _, v := range stack {
		res += v
	}
	return res
}

/*
leetcode 224 基本计算器
给你一个字符串表达式 s ，请你实现一个基本计算器来计算并返回它的值。
注意:不允许使用任何将字符串作为数学表达式计算的内置函数，比如 eval() 。

示例 1：
输入：s = "1 + 1"
输出：2

示例 2：
输入：s = " 2-1 + 2 "
输出：3

示例 3：
输入：s = "(1+(4+5+2)-3)+(6+8)"
输出：23

提示：
1 <= s.length <= 3 * 105
s 由数字、'+'、'-'、'('、')'、和 ' ' 组成
s 表示一个有效的表达式
'+' 不能用作一元运算(例如， "+1" 和 "+(2 + 3)" 无效)
'-' 可以用作一元运算(即 "-1" 和 "-(2 + 3)" 是有效的)
输入中不存在两个连续的操作符
每个数字和运行的计算将适合于一个有符号的 32位 整数
*/

func calculateComplex(s string) int {
	result, num, sign := 0, 0, 1
	// 栈用于存储当前的 result 和 sign，以便在处理括号嵌套时能够正确还原上下文
	stack := []int{}
	for _, v := range s {
		if v == ' ' {
			continue
			// 累计数字：遍历字符串时，如果遇到数字字符，将其累积到 num 中，处理多位数字的情况。
		} else if v >= '0' && v <= '9' {
			num = num*10 + int(v-'0')
			// 符号处理：碰到 + 或 - 时，利用当前的 sign 将 num 累加到 result，然后更新符号。
		} else if v == '+' {
			result += sign * num
			sign = 1
			num = 0
		} else if v == '-' {
			result += sign * num
			sign = -1
			num = 0
			// 括号处理：遇到 '(' 时，将当前计算状态保存到栈中。
		} else if v == '(' {
			stack = append(stack, result, sign)
			result, sign = 0, 1
		} else {
			// 遇到 ')' 时，将当前括号内的计算结果与栈中的上下文合并。
			result += sign * num
			num = 0
			// 遇到 ')'，将当前计算结果乘以栈顶的sign，并与栈顶的 result 相加
			sign = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			result = stack[len(stack)-1] + sign*result
			stack = stack[:len(stack)-1]
		}
	}
	result += sign * num
	return result
}

/*
leetcode 739 每日温度
给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，
下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。

示例 1:
输入: temperatures = [73,74,75,71,69,72,76,73]
输出: [1,1,4,2,1,1,0,0]

示例 2:
输入: temperatures = [30,40,50,60]
输出: [1,1,1,0]

示例 3:
输入: temperatures = [30,60,90]
输出: [1,1,0]

提示：
1 <= temperatures.length <= 105
30 <= temperatures[i] <= 100
*/

/*
算法思路
使用一个栈来存储当前温度的下标，栈中的温度是递减的。栈中元素代表的是尚未找到比它更高温度的日子。
从左到右遍历数组，每遇到一个新的温度时：
比较当前温度与栈顶温度的大小。如果当前温度更高，说明找到了栈顶元素的答案：
弹出栈顶元素，并计算它与当前天数的差值。
将当前温度的下标压入栈中。
最终，栈中剩下的元素的答案为 0，因为没有找到比它更高的温度。
时间复杂度:𝑂(n) 每个元素最多入栈一次，出栈一次。
空间复杂度:O(n)，栈的大小最多为n
*/

func dailyTemperatures(temperatures []int) []int {
	n := len(temperatures)
	// 用于存储每一天到下一个更高温度的天数
	answer := make([]int, n)
	// 用于存储索引，栈顶是尚未找到下一个更高温度的元素(元素的位置下标)
	stack := []int{}
	for i := 0; i < n; i++ {
		// 如果当前温度比栈顶的温度高，那么找到了答案
		// 弹出栈顶元素，计算天数差并更新答案
		// temperatures = [73,74,75,71,69,72,76,73]
		for len(stack) > 0 && temperatures[i] > temperatures[stack[len(stack)-1]] {
			idx := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			answer[idx] = i - idx
		}
		// 将当前温度的索引压入栈中
		stack = append(stack, i)
	}
	// 栈中剩下的元素不需要处理，它们的答案已初始化为0
	return answer
}

/*
leetcode 402 移掉K位数字
给你一个以字符串表示的非负整数 num 和一个整数 k ，移除这个数中的 k 位数字，使得剩下的数字最小。请你以字符串形式返回
这个最小的数字。

示例 1 ：
输入：num = "1432219", k = 3
输出："1219"
解释：移除掉三个数字 4, 3, 和 2 形成一个新的最小的数字 1219 。

示例 2 ：
输入：num = "10200", k = 1
输出："200"
解释：移掉首位的 1 剩下的数字为 200. 注意输出不能有任何前导零。

示例 3 ：
输入：num = "10", k = 2
输出："0"
解释：从原数字移除所有的数字，剩余为空就是 0 。

提示：
1 <= k <= num.length <= 105
num 仅由若干位数字（0 - 9）组成
除了 0 本身之外，num 不含任何前导零
*/

/*
题目要求我们从一个非负整数中删除 k 位数字，得到一个最小的数字。为了实现这个目标，我们需要理解如何去除数字，并且保持
剩余的数字尽可能小。直观来看，最小化结果意味着要保持较小的数字靠前，并删除掉较大的数字。

我们可以将这个问题理解为：如何保持数字的 "递增" 顺序，尽可能去除不必要的数字，以便保留最小的数字。

解题步骤
思路分析：
我们需要删除 k 位数字，这意味着最终的数字长度为 len(num) - k。
我们可以将原始数字看作一个栈，逐位遍历数字，并且尝试将大的数字移除，保持小的数字在前。
对于每个数字，若当前数字小于栈顶的数字，我们可以移除栈顶的数字（即删除较大的数字），以便得到更小的结果。

栈的使用：
我们遍历 num 中的每一位数字，对于每一个数字：
如果栈不为空，并且栈顶的数字大于当前数字，就弹出栈顶的数字。
每次弹出时，计数器 k 减少，表示删除了一位数字。
最终，栈中存储的是移除 k 位数字后的结果。

特殊情况：
如果删除完 k 位数字后栈中有多余的数字（比如 k 没有完全删除），则将栈中的前 len(num) - k 个数字取出。
结果需要处理前导零，即如果最终结果有前导零，应该去掉。

时间复杂度：
遍历一次字符串，每个元素最多被入栈和出栈一次，时间复杂度为 O(n)，其中 n 是字符串 num 的长度。
空间复杂度：
使用栈存储结果，空间复杂度为 O(n)。

*/

func removeKdigits(num string, k int) string {
	n := len(num)
	stack := []byte{}
	for i := 0; i < n; i++ {
		for k > 0 && len(stack) > 0 && stack[len(stack)-1] > num[i] {
			stack = stack[:len(stack)-1]
			k--
		}
		stack = append(stack, num[i])
	}
	stack = stack[:len(stack)-k]
	res := string(stack)
	res = strings.TrimLeft(res, "0")
	if res == "" {
		return "0"
	}
	return res
}

/*
leetcode 316/1081 去除重复字母/不同字符的最小子序列
给你一个字符串s ，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证返回结果的字典序最小
（要求不能打乱其他字符的相对位置）。

示例 1：
输入：s = "bcabc"
输出："abc"

示例 2：
输入：s = "cbacdcbc"
输出："acdb"

提示：
1 <= s.length <= 104
s 由小写英文字母组成
*/

/*
思路:单调栈
本题要求去除重复字母，并保持字典序最小。可以使用单调栈来高效解决这个问题。核心思想是：
统计字符出现次数： 首先遍历字符串，统计每个字符出现的次数，方便后续判断字符是否可以被移除。
使用栈维护结果： 使用一个栈来维护最终的结果字符串。遍历输入字符串，对于每个字符：
如果该字符已经在栈中，则跳过。
否则，比较栈顶字符和当前字符：
如果栈顶字符的字典序大于当前字符，并且栈顶字符在后续还会出现（即统计次数大于0），则弹出栈顶字符，直到栈为空或不满足上述条件。
将当前字符压入栈。
构建结果字符串： 遍历完成后，栈中的字符即为最终结果，将其转换为字符串返回。
之所以使用单调栈，是因为它能够有效地维护一个递增（或递减）的序列，从而帮助我们找到字典序最小的组合。
*/

func removeDuplicateLetters(s string) string {
	n := len(s)
	if n <= 1 {
		return s
	}
	// 统计每个字符出现的次数
	count := make([]int, 26)
	for i := 0; i < n; i++ {
		count[s[i]-'a']++
	}
	// 栈用于维护结果
	stack := make([]byte, 0)
	// 记录字符是否在栈中
	inStack := make([]bool, 26)

	for i := 0; i < n; i++ {
		char := s[i]
		index := char - 'a'
		// 如果字符已经在栈中，则跳过
		if inStack[index] {
			count[index]--
			continue
		}
		// 维护单调栈
		for len(stack) > 0 && stack[len(stack)-1] > char && count[stack[len(stack)-1]-'a'] > 0 {
			inStack[stack[len(stack)-1]-'a'] = false
			stack = stack[:len(stack)-1]
		}
		// 将当前字符入栈
		stack = append(stack, char)
		// 标记该字符已在栈中
		inStack[index] = true
		// 剩余次数减一
		count[index]--
	}
	return string(stack)
}

/*
leetcode 503 下一个更大的元素II
给定一个循环数组 nums （ nums[nums.length - 1] 的下一个元素是 nums[0] ），返回 nums 中每个元素的 下一个更大元素 。
数字 x 的 下一个更大的元素 是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。
如果不存在，则输出 -1 。

示例 1:
输入: nums = [1,2,1]
输出: [2,-1,2]
解释: 第一个 1 的下一个更大的数是 2；
数字 2 找不到下一个更大的数；
第二个 1 的下一个最大的数需要循环搜索，结果也是 2。

示例 2:
输入: nums = [1,2,3,4,3]
输出: [2,3,4,-1,4]

提示:
1 <= nums.length <= 104
-109 <= nums[i] <= 109
*/

/*
思路:单调栈
要解决这个问题，我们可以使用单调栈（Monotonic Stack）的方法。通过栈的数据结构，我们可以有效地解决这个问题，避免暴力搜索
带来的时间复杂度。

问题分析
我们要找出每个元素的下一个更大的元素。但是数组是循环的，这意味着最后一个元素的下一个元素是数组的第一个元素，所以我们需要考虑循环的情况。

思路
我们可以将数组的长度 n 加倍，通过模拟一个“非循环”的场景，简化问题。具体来说，将数组 nums 重复一次，得到一个新数组，这样我们就可以
通过一次遍历找到每个元素的下一个更大的元素。
我们使用一个栈来存储下一个更大的元素的索引。当栈顶的元素比当前元素小（即当前元素是栈顶元素的下一个更大的元素）时，我们就更新栈顶元素的结果。
为了避免超出原数组的范围，我们可以限制栈中的索引范围，确保我们只查找循环数组中的元素。

算法步骤
初始化：
创建一个大小为 n 的结果数组 res，并初始化为 -1（表示没有下一个更大的元素）。
使用一个栈来存储数组元素的索引。

遍历：
从 0 到 2n-1 遍历这个数组，模拟循环数组。
对于每个元素 nums[i]，我们不断检查栈顶的元素是否小于 nums[i]，如果小于，则说明 nums[i] 是栈顶元素的下一个更大的元素，更新 res 数组，
并弹出栈顶元素。
将当前元素的索引推入栈中。

返回结果：
遍历结束后，res 数组就是我们要的答案。
*/

func nextGreaterElements(nums []int) []int {
	n := len(nums)
	// 初始化res数组为-1，表示默认情况下都没有下一个更大的元素
	res := make([]int, n)
	for i := 0; i < n; i++ {
		res[i] = -1
	}
	// 用来存放数组元素的索引
	s := []int{}
	// 遍历两倍的数组长度，模拟循环数组
	for i := 0; i < n*2; i++ {
		// 当前元素的索引
		index := i % n
		// 当前元素
		current := nums[index]
		// 如果栈非空，并且栈顶元素小于当前元素，说明找到了栈顶元素的下一个更大元素
		for len(s) > 0 && nums[s[len(s)-1]] < current {
			topIndex := s[len(s)-1]
			res[topIndex] = current
			// 弹出栈顶元素
			s = s[:len(s)-1]
		}
		// 如果i小于n，就可以将当前元素的索引压入栈中
		if i < n {
			s = append(s, index)
		}
	}
	return res
}

/*
剑指offer 31 栈的压入，弹出序列
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。
例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2}
就不可能是该压栈序列的弹出序列。

2. 示例
示例1：
输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1

示例2：
输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。
*/

/*
此题的解题思路是模仿出栈

步骤：
新建一个空栈（临时栈）用来执行入栈操作
遍历入栈，每次一个元素压入临时栈，判断临时栈顶元素是否与出栈栈顶元素相等
相等，临时栈顶元素出栈，出栈后移一位，循环判断
最后，返回临时栈，若为空说明顺序一致
*/

func checkStackSequence(pushed, popped []int) bool {
	if len(pushed) != len(popped) {
		return false
	}
	stack := make([]int, 0, len(pushed))
	i := 0
	for _, num := range pushed {
		stack = append(stack, num)
		for len(stack) > 0 && stack[len(stack)-1] == popped[i] {
			stack = stack[:len(stack)-1]
			i++
		}
	}
	return len(stack) == 0
}
