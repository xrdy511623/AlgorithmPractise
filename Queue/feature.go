package Queue

/*
1.1 用队列实现栈
请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通栈的全部四种操作（push、top、pop 和 empty）。
实现MyStack类：
void push(int x) 将元素x压入栈顶。
int pop() 移除并返回栈顶元素。
int top() 返回栈顶元素。
boolean empty() 如果栈是空的，返回 true；否则，返回 false 。

注意：
你只能使用队列的基本操作 —— 也就是push to back、peek/pop from front、size 和is empty这些操作。
你所使用的语言也许不支持队列。你可以使用list（列表）或者 deque（双端队列来模拟一个队列, 只要是标准的
队列操作即可。
*/

type MyStack struct {
	InputQueue  []int
	OutputQueue []int
}

func Constructor() MyStack {
	return MyStack{
		InputQueue:  make([]int, 0),
		OutputQueue: make([]int, 0),
	}
}

func (s *MyStack) Push(x int) {
	s.InputQueue = append(s.InputQueue, x)
}

func (s *MyStack) Pop() int {
	if len(s.OutputQueue) == 0 {
		for len(s.InputQueue) != 0 {
			s.OutputQueue = append(s.OutputQueue, s.InputQueue[0])
			s.InputQueue = s.InputQueue[1:]
		}
	} else {
		if len(s.InputQueue) != 0 {
			s.OutputQueue = append(s.OutputQueue, s.InputQueue[0])
		}
	}
	value := s.OutputQueue[len(s.OutputQueue)-1]
	s.OutputQueue = s.OutputQueue[:len(s.OutputQueue)-1]
	return value
}

func (s *MyStack) Top() int {
	if len(s.OutputQueue) == 0 {
		for len(s.InputQueue) != 0 {
			s.OutputQueue = append(s.OutputQueue, s.InputQueue[0])
			s.InputQueue = s.InputQueue[1:]
		}
	} else {
		if len(s.InputQueue) != 0 {
			s.OutputQueue = append(s.OutputQueue, s.InputQueue[0])
		}
	}
	return s.OutputQueue[len(s.OutputQueue)-1]
}

func (s *MyStack) Empty() bool {
	if len(s.InputQueue) == 0 && len(s.OutputQueue) == 0 {
		return true
	}
	return false
}