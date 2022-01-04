package Stack

import (
	"AlgorithmPractise/Utils"
	"math"
	"strconv"
)

/*
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
1.2 用栈实现队列
请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：

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
1.3 最小栈
设计一个支持 push，pop，top操作，并能在常数时间内检索到最小元素的栈。

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
1.4 删除字符串中的所有相邻重复项
给出由小写字母组成的字符串S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。
在S上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。


示例：
输入："abbaca"
输出："ca"
解释：
例如，在 "abbaca" 中，我们可以删除"bb"由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们得到
字符串 "aaca"，其中又只有"aa"可以执行重复项删除操作，所以最后的字符串为"ca"。
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
1.5 逆波兰表达式求值
根据逆波兰表示法，求表达式的值。
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
1.6 滑动窗口最大值
给你一个整数数组nums，有一个大小为k的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的k个数字。
滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

示例1：
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

提示：
1 <= nums.length <= 105
-104 <= nums[i] <= 104
1 <= k <= nums.length
*/

// maxSlidingWindow 暴力解法 时间复杂度O(N*K), 空间复杂度O(N)
func maxSlidingWindow(nums []int, k int) []int {
	var res []int
	for i := 0; i+k <= len(nums); i++ {
		res = append(res, Utils.MaxValueOfArray(nums[i:i+k]))
	}
	return res
}

/*
思路:实现一个严格单调递减队列sq，在sq里维护有可能成为窗口里最大值的元素就可以了，同时保证sq里的元素是单调递减的。
设计单调队列的时候，Pop和Push操作要保持如下规则：
Pop(value)：如果窗口移除的元素value等于单调队列的出口元素，那么队列将弹出元素，否则不用任何操作
Push(value)：如果窗口Push的元素value大于单调队列入口元素的数值，那么就将队列入口的元素弹出，直到Push的元素小于
等于队列入口元素的数值为止。
保持如上规则，每次窗口移动的时候，只要调用sq.Peek()就可以返回当前窗口的最大值。
*/

type StrictQueue struct {
	Queue []int
}

func NewStrictQueue() *StrictQueue {
	return &StrictQueue{
		Queue: make([]int, 0),
	}
}

func (sq *StrictQueue) Pop(value int) {
	// 如果队列不为空且要移除的元素为队列中最大值，则将其弹出，否则不做任何操作
	// 因为我们关心的是队列中的最大值
	if !sq.IsEmpty() && sq.Queue[0] == value {
		sq.Queue = sq.Queue[1:]
	}
}

func (sq *StrictQueue) Push(value int) {
	// 如果队列不为空且要添加的元素大于队列入口元素，则将队列入口元素弹出
	// 直到添加的元素小于等于队列入口元素为止，以保证队列是严格单调递减的。
	for !sq.IsEmpty() && sq.Queue[sq.Size()-1] < value {
		sq.Queue = sq.Queue[:sq.Size()-1]
	}
	sq.Queue = append(sq.Queue, value)
}

func (sq *StrictQueue) Peek() int {
	// 返回队列中的最大值
	return sq.Queue[0]
}

func (sq *StrictQueue) IsEmpty() bool {
	return len(sq.Queue) == 0
}

func (sq *StrictQueue) Size() int {
	return len(sq.Queue)
}

// MaxSlidingWindow 时间复杂度O(N), 空间复杂度O(N)
func MaxSlidingWindow(nums []int, k int) []int {
	var res []int
	sq := NewStrictQueue()
	for i := 0; i < k; i++ {
		sq.Push(nums[i])
	}
	res = append(res, sq.Peek())
	for i := k; i < len(nums); i++ {
		sq.Pop(nums[i-k])
		sq.Push(nums[i])
		res = append(res, sq.Peek())
	}
	return res
}