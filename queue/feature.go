package queue

import (
	"algorithmpractise/utils"
	"math"
)

/*
leetcode 225. 用队列实现栈
1.1 请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通栈的全部四种操作（push、top、pop 和 empty）。
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
	// 主队列
	q1 []int
	// 辅助队列
	q2 []int
}

func Constructor() MyStack {
	return MyStack{
		q1: make([]int, 0),
		q2: make([]int, 0),
	}
}

// Push 将元素压入栈顶
func (s *MyStack) Push(x int) {
	// 将新元素加入辅助队列
	s.q2 = append(s.q2, x)
	// 将主队列的所有元素转移到辅助队列
	for len(s.q1) > 0 {
		s.q2 = append(s.q2, s.q1[0])
		s.q1 = s.q1[1:]
	}
	// 交换 q1 和 q2
	s.q1, s.q2 = s.q2, s.q1
}

// Pop 移除并返回栈顶元素
func (s *MyStack) Pop() int {
	// 空栈处理
	if len(s.q1) == 0 {
		return -1
	}
	// 直接移除 q1 的队首元素
	val := s.q1[0]
	s.q1 = s.q1[1:]
	return val
}

// Top 返回栈顶元素
func (s *MyStack) Top() int {
	// 空栈处理
	if len(s.q1) == 0 {
		return -1
	}
	// q1 的队首元素即为栈顶
	return s.q1[0]
}

// Empty 判断栈是否为空
func (s *MyStack) Empty() bool {
	return len(s.q1) == 0
}

/*
leetcode 239. 滑动窗口最大值
剑指Offer 59 - I. 滑动窗口的最大值
1.2 给你一个整数数组nums，有一个大小为k的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内
的k个数字。滑动窗口每次只向右移动一位。

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
	for i, n := 0, len(nums); i+k <= n; i++ {
		res = append(res, utils.MaxValueOfArray(nums[i:i+k]))
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
	// 因为我们只关心队列中的最大值
	if !sq.IsEmpty() && value == sq.Peek() {
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
	return sq.Size() == 0
}

func (sq *StrictQueue) Size() int {
	return len(sq.Queue)
}

// MaxSlidingWindow 时间复杂度O(N), 空间复杂度O(N)
func MaxSlidingWindow(nums []int, k int) []int {
	var res []int
	n := len(nums)
	// 处理特殊情况
	if k > n {
		return res
	}
	sq := NewStrictQueue()
	// 首先将前k个元素添加到队列中
	for i := 0; i < k; i++ {
		sq.Push(nums[i])
	}
	// 将前k个元素中的最大值添加到结果集合中
	res = append(res, sq.Peek())
	for i := k; i < n; i++ {
		// 队列移除最前面元素
		sq.Pop(nums[i-k])
		// 向队列添加新元素，也就是当前元素nums[i]
		sq.Push(nums[i])
		// 将当前队列中的最大值添加到结果集合中
		res = append(res, sq.Peek())
	}
	return res
}

/*
leetcode 862. 和至少为K的最短子数组
1.3 给你一个整数数组nums和一个整数k ，找出nums中和至少为k的最短非空子数组，并返回该子数组的长度。如果不存在
这样的子数组，返回-1 。
子数组是数组中连续的一部分。
*/

/*
思路:滑动窗口
我们用数组P表示数组A的前缀和，即 P[i] = A[0] + A[1] + ... + A[i - 1]。我们需要找到x和y，
使得P[y] - P[x] >= K 且 y - x 最小, 也就是将问题转化为前缀和数组中满足P[y] - P[x] >= K的最短距离。
为此，我们需要维护一个前缀和的单调递增队列，所以当我们遇到了一个新的下标y时，我们会在队尾移除若干元素，
直到 P[x0], P[x1], ..., P[y] 单调递增。同时，我们会在队首也移除若干元素，如果 P[y] >= P[x0] + K，
则将队首元素移除，以尝试缩小窗口，得到更短的距离。
*/

func ShortestSubarray(nums []int, k int) int {
	n := len(nums)
	sumPrefix := make([]int, n+1)
	for i, v := range nums {
		sumPrefix[i+1] = sumPrefix[i] + v
	}
	monoQueue := make([]int, 0)
	minLength := math.MaxInt32
	for i, v := range sumPrefix {
		// 维护前缀和的单调递增队列
		for len(monoQueue) > 0 && v <= sumPrefix[monoQueue[len(monoQueue)-1]] {
			monoQueue = monoQueue[:len(monoQueue)-1]
		}
		// 尝试将队首元素移除，以缩小窗口，得到更短的距离
		for len(monoQueue) > 0 && v-sumPrefix[monoQueue[0]] >= k {
			minLength = utils.Min(minLength, i-monoQueue[0])
			monoQueue = monoQueue[1:]
		}
		monoQueue = append(monoQueue, i)
	}
	if minLength == math.MaxInt32 {
		return -1
	}
	return minLength
}

/*
leetcode 622 设计循环队列
设计你的循环队列实现。 循环队列是一种线性数据结构，其操作表现基于 FIFO（先进先出）原则并且队尾被连接在队首之后以形成一个循环。
它也被称为“环形缓冲器”。

循环队列的一个好处是我们可以利用这个队列之前用过的空间。在一个普通队列里，一旦一个队列满了，我们就不能插入下一个元素，即使
在队列前面仍有空间。但是使用循环队列，我们能使用这些空间去存储新的值。

你的实现应该支持如下操作：
MyCircularQueue(k): 构造器，设置队列长度为 k 。
Front: 从队首获取元素。如果队列为空，返回 -1 。
Rear: 获取队尾元素。如果队列为空，返回 -1 。
enQueue(value): 向循环队列插入一个元素。如果成功插入则返回真。
deQueue(): 从循环队列中删除一个元素。如果成功删除则返回真。
isEmpty(): 检查循环队列是否为空。
isFull(): 检查循环队列是否已满。

示例：
MyCircularQueue circularQueue = new MyCircularQueue(3); // 设置长度为 3
circularQueue.enQueue(1);  // 返回 true
circularQueue.enQueue(2);  // 返回 true
circularQueue.enQueue(3);  // 返回 true
circularQueue.enQueue(4);  // 返回 false，队列已满
circularQueue.Rear();  // 返回 3
circularQueue.isFull();  // 返回 true
circularQueue.deQueue();  // 返回 true
circularQueue.enQueue(4);  // 返回 true
circularQueue.Rear();  // 返回 4

提示：
所有的值都在 0 至 1000 的范围内；
操作数将在 1 至 1000 的范围内；
请不要使用内置的队列库。
*/

type MyCircularQueue struct {
	Capacity int
	Queue    []int
}

func Construct(k int) MyCircularQueue {
	return MyCircularQueue{
		Capacity: k,
		Queue:    []int{},
	}
}

func (mc *MyCircularQueue) EnQueue(value int) bool {
	if mc.IsFull() {
		return false
	}
	mc.Queue = append(mc.Queue, value)
	return true
}

func (mc *MyCircularQueue) DeQueue() bool {
	if mc.IsEmpty() {
		return false
	}
	mc.Queue = mc.Queue[1:]
	return true
}

func (mc *MyCircularQueue) Front() int {
	if mc.IsEmpty() {
		return -1
	}
	return mc.Queue[0]
}

func (mc *MyCircularQueue) Rear() int {
	if mc.IsEmpty() {
		return -1
	}
	return mc.Queue[len(mc.Queue)-1]
}

func (mc *MyCircularQueue) IsEmpty() bool {
	return len(mc.Queue) == 0
}

func (mc *MyCircularQueue) IsFull() bool {
	return len(mc.Queue) == mc.Capacity
}
