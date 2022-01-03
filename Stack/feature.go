package Stack

import (
	"AlgorithmPractise/Utils"
	"math"
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
我们遍历给定的字符串s。当我们遇到一个左括号时，我们会期望在后续的遍历中，有一个相同类型的右括号将其闭合。由于后遇到的左括号要先闭合，因此
我们可以将这个左括号放入栈顶。当我们遇到一个右括号时，我们需要将一个相同类型的左括号闭合。此时，我们可以取出栈顶的左括号并判断它们是否是
相同类型的括号。如果不是相同的类型，或者栈中并没有左括号，那么字符串ss无效，返回False。为了快速判断括号的类型，我们可以使用哈希表存储
每一种括号。哈希表的键为右括号，值为相同类型的左括号。
在遍历结束后，如果栈中没有左括号，说明我们将字符串s中的所有左括号闭合，返回True，否则返回False。
注意到有效字符串的长度一定为偶数，因此如果字符串的长度为奇数，我们可以直接返回False，省去后续的遍历判断过程。
*/

func IsValid(s string) bool {
	length := len(s)
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
	}else{
		if len(q.InputStack) != 0{
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
	}else{
		if len(q.InputStack) != 0{
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
