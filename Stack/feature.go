package Stack

import (
	"AlgorithmPractise/Utils"
	"math"
	"strconv"
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
ss无效，返回false。为了快速判断括号的类型，我们可以使用哈希表存储每一种括号。哈希表的键为右括号，值为相同类型的
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
		for len(q.InputStack) != 0 {
			q.OutputStack = append(q.OutputStack, q.InputStack[0])
			q.InputStack = q.InputStack[1:]
		}
	} else {
		if len(q.InputStack) != 0 {
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
		for len(q.InputStack) != 0 {
			q.OutputStack = append(q.OutputStack, q.InputStack[0])
			q.InputStack = q.InputStack[1:]
		}
	} else {
		if len(q.InputStack) != 0 {
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
	ms.minStack = append(ms.minStack, Utils.Min(top, val))
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
	for i := 0; i < len(s); i++ {
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
初始慢指针j等于 0。
使用快指针i遍历字符串：
进入循环后第一个操作是
令s[i] = s[j]。
如果j > 0 && s[j] = s[j - 1]，则栈顶元素加 1。
否则，栈中压入 1。
如果计数器等于k，j = j - k，并弹出栈顶元素。
返回字符串的前j个字符。
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
			increment := stack[len(stack)-1] + 1
			if increment == k {
				j -= k
				stack = stack[:len(stack)-1]
			} else {
				stack[len(stack)-1] = increment
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
			// 遇到运算符，取出顶两个数字进行计算，并将结果压入栈中
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
	maxLength := 0
	stack := []int{-1}
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			stack = append(stack, i)
		} else {
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				stack = append(stack, i)
			} else {
				maxLength = Utils.Max(maxLength, i-stack[len(stack)-1])
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
	left, right, maxLength := 0, 0, 0
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			left++
		} else {
			right++
		}
		if left == right {
			maxLength = Utils.Max(maxLength, 2*right)
		} else if right > left {
			left, right = 0, 0
		}
	}
	left, right = 0, 0
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == '(' {
			left++
		} else {
			right++
		}
		if left == right {
			maxLength = Utils.Max(maxLength, 2*right)
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
			maxLength = Utils.Max(maxLength, dp[i])
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
			minCount = Utils.Max(minCount-1, 0)
			maxCount--
			if maxCount < 0 {
				return false
			}
		} else {
			minCount = Utils.Max(minCount-1, 0)
			maxCount++
		}
	}
	return minCount == 0
}