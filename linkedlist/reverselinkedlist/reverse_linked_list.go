package reverselinkedlist

import "algorithmpractise/linkedlist/entity"

/*
反转链表专题
*/

/*
Leetcode 24. 两两交换链表中的节点
1.1 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
你不能只是单纯地改变节点内部的值，而是需要实际的进行节点交换。
譬如1-2-3-4-5-6
应返回2-1-4-3-6-5
*/

func SwapPairs(head *entity.ListNode) *entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	dummy := &entity.ListNode{Next: head}
	pre, cur := dummy, head
	for cur != nil && cur.Next != nil {
		firstNode, secondNode := cur, cur.Next
		pre.Next = secondNode
		firstNode.Next = secondNode.Next
		secondNode.Next = firstNode
		// 更新pre和cur指针
		pre, cur = cur, cur.Next
	}
	return dummy.Next
}

/*
其实这个问题可以转化为K个一组反转链表的特例，此时K=2，所以可以像下面这样写
*/

func SwapPairsTwo(head *entity.ListNode) *entity.ListNode {
	dummy := &entity.ListNode{Next: head}
	prev := dummy
	for head != nil {
		tail := prev
		for i := 0; i < 2; i++ {
			tail = tail.Next
			if tail == nil {
				return dummy.Next
			}
		}
		ndx := tail.Next
		head, tail = reverse(head, tail, 2)
		prev.Next = head
		tail.Next = ndx
		prev = tail
		head = ndx
	}
	return dummy.Next
}

/*
leetcode 206. 反转链表
1.2 基础题:反转一个链表
譬如1-2-3-4-5-6
应返回6-5-4-3-2-1
*/

// Reverse  思路:迭代法，时间复杂度O(n),空间复杂度O(1)
func Reverse(head *entity.ListNode) *entity.ListNode {
	var prev *entity.ListNode
	cur := head
	for cur != nil {
		cur.Next, prev, cur = prev, cur, cur.Next
	}
	return prev
}

/*
leetcode 25. K个一组翻转链表
1.3 进阶 给你一个链表，每k个节点一组进行翻转，请你返回翻转后的链表。
k是一个正整数，它的值小于或等于链表的长度。
如果节点总数不是k的整数倍，那么请将最后剩余的节点保持原有顺序。
进阶：
你可以设计一个只使用常数额外空间的算法来解决此问题吗？
你不能只是单纯地改变节点内部的值，而是需要实际进行节点交换。
*/

/*
本题的关键是找到k个一组组成的子链表的头尾结点head和tail,以及头结点的前一个结点prev和tail的下一个结点
ndx。反转完后，要将反转后的子链表接入到原链表中，只需要prev.Next=head, tail.Next=ndx即可。第一次
反转时，head显然就是链表的头结点，prev显然应该指向伪头结点dummy, tail显然就是dummy走k步后指向的结点。
每次反转结束后需要更新prev和head指针，很明显下一次需要反转的子链表的头结点应该是ndx, prev应该指向
上一次反转后子链表的尾结点tail。
*/

func ReverseKGroup(head *entity.ListNode, k int) *entity.ListNode {
	dummy := &entity.ListNode{Next: head}
	prev := dummy
	for head != nil {
		tail := prev
		// 如果链表中结点个数少于k个，那么也不用反转了，返回头结点即可
		for i := 0; i < k; i++ {
			tail = tail.Next
			if tail == nil {
				return dummy.Next
			}
		}
		// 保存反转前k个结点组成子链表尾结点的下一个结点
		ndx := tail.Next
		// 调用反转链表函数得到反转后的头尾结点
		head, tail = reverse(head, tail, k)
		// 将头结点和尾结点接入到原来的链表中
		prev.Next = head
		tail.Next = ndx

		// 更新prev和head指针，此时分别指向反转后的k个结点组成的链表的尾结点和反转前k个结点组成的链表尾结点的下一个结点
		prev = tail
		head = ndx
	}
	return dummy.Next
}

func reverse(head, tail *entity.ListNode, k int) (*entity.ListNode, *entity.ListNode) {
	var prev *entity.ListNode
	cur := head
	for i := 0; i < k; i++ {
		cur.Next, prev, cur = prev, cur, cur.Next
	}
	return tail, head
}

/*
leetcode 92. 反转链表II
1.4 变形题
给你单链表的头指针head和两个整数left和right ，其中left <= right 。请你反转从位置left到位置right的链表
节点，返回反转后的链表。
例如对于链表1-2-3-4-5
left=2,right=4
最后应该返回1-4-3-2-5

提示：
链表中节点数目为n
1 <= n <= 500
-500 <= Node.val <= 500
1 <= left <= right <= n
*/

/*
思路:k个一组反转链表的简化变形题，找到left位置前的节点pre,left位置的节点start,right位置的节点end,
以及right位置的下一个节点ndx，将start,end，k=(right-left)+1作为参数传递给reverse(反转k个节点组成的链表)
得到返回的head和tail节点，将pre节点的Next指针指向head节点，将tail节点的Next指针指向ndx节点即可。
*/

func ReverseBetween(head *entity.ListNode, left int, right int) *entity.ListNode {
	if left == right {
		return head
	}
	k := right - left + 1
	dummy := &entity.ListNode{Next: head}
	prev := dummy
	for i := 0; i < left-1; i++ {
		prev = prev.Next
		if prev == nil {
			return dummy.Next
		}
	}
	tail := prev
	// s标记left位置结点，也是需要反转的left-right位置子链表部分的头结点
	s := prev.Next
	// tail初始位置是prev, 标记left位置前一个结点，向后移动k次后，便指向right位置结点
	for i := 0; i < k; i++ {
		tail = tail.Next
		if tail == nil {
			return dummy.Next
		}
	}
	// ndx标记right位置结点的下一个结点
	ndx := tail.Next
	// 将left-right位置子链表部分翻转
	s, tail = reverse(s, tail, k)
	// 将反转后的子链表接入到原链表中
	prev.Next = s
	tail.Next = ndx
	return dummy.Next
}

/*
leetcode 143. 重排链表
1.5 重排链表
给定一个单链表L的头节点 head ，单链表L表示为：
L0 → L1 → … → Ln - 1 → Ln

请将其重新排列后变为：

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

输入：head = [1,2,3,4]
输出：[1,4,2,3]

输入：head = [1,2,3,4,5]
输出：[1,5,2,4,3]
*/

// ReOrderLinkedList 用线性表存储该链表，利用线性表可以按下标访问的特性，按顺序访问指定元素，重建链表即可
// 重排后链表的特点是前后两个节点的下标和为n-1,时间复杂度O(N),空间复杂度O(N)。
func ReOrderLinkedList(head *entity.ListNode) {
	if head == nil {
		return
	}
	var nodes []*entity.ListNode
	for head != nil {
		nodes = append(nodes, head)
		head = head.Next
	}
	i, j := 0, len(nodes)-1
	for i < j {
		nodes[i].Next = nodes[j]
		i++
		if i == j {
			break
		}
		nodes[j].Next = nodes[i]
		j--
	}
	nodes[i].Next = nil
}

/*
寻找链表中点 + 链表逆序 + 合并链表
注意到目标链表即为将原链表的左半端和反转后的右半端合并后的结果。

这样我们的任务即可划分为三步：
找到原链表的中点。
我们可以使用快慢指针来O(N)地找到链表的中间节点。
将原链表的右半端反转。
我们可以使用迭代法实现链表的反转。
将原链表的两端合并。
因为两链表长度相差不超过1，因此直接合并即可。
时间复杂度O(N),空间复杂度O(1)
*/

func ReOrderLinkedListSimple(head *entity.ListNode) {
	if head == nil || head.Next == nil {
		return
	}
	// 找到链表中间节点，确定链表右半部分
	mid := GetMiddleNode(head)
	r := mid.Next
	// 将原链表从mid结点截断
	mid.Next = nil
	// 将链表右半部分反转
	r = Reverse(r)
	// 将链表左半部分与右半部分依次合并
	MergeLists(head, r)
}

// GetMiddleNode 寻找链表中间节点
func GetMiddleNode(head *entity.ListNode) *entity.ListNode {
	fast, slow := head, head
	for fast.Next != nil && fast.Next.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	return slow
}

// MergeLists 合并链表
func MergeLists(l1, l2 *entity.ListNode) {
	var l1Tmp, l2Tmp *entity.ListNode
	for l1 != nil && l2 != nil {
		l1Tmp = l1.Next
		l2Tmp = l2.Next
		l1.Next = l2
		l1 = l1Tmp
		l2.Next = l1
		l2 = l2Tmp
	}
}
