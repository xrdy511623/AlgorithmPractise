package ReverseLinkedList

import "AlgorithmPractise/LinkedList/Entity"

/*
反转链表
*/

/*
1.1 两两交换链表中的节点
给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
譬如1-2-3-4-5-6
应返回2-1-4-3-6-5
*/

func SwapPairs(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	dummy := &Entity.ListNode{0, head}
	pre, cur := dummy, head
	for cur != nil && cur.Next != nil {
		firstNode := cur
		secondNode := cur.Next
		pre.Next = secondNode
		firstNode.Next = secondNode.Next
		secondNode.Next = firstNode

		// 更新pre和cur指针
		pre = cur
		cur = cur.Next
	}
	return dummy.Next
}

/*
1.2 基础题:反转一个链表
譬如1-2-3-4-5-6
应返回6-5-4-3-2-1
*/

// ReverseLinkedList 思路:迭代法，时间复杂度O(n),空间复杂度O(1)
func ReverseLinkedList(head *Entity.ListNode) *Entity.ListNode {
	var prev *Entity.ListNode
	cur := head
	for cur != nil {
		cur.Next, prev, cur = prev, cur, cur.Next
	}
	return prev
}

/*
1.3 进阶 k个一组反转链表
给你一个链表，每k个节点一组进行翻转，请你返回翻转后的链表。
k是一个正整数，它的值小于或等于链表的长度。
如果节点总数不是k的整数倍，那么请将最后剩余的节点保持原有顺序。
进阶：
你可以设计一个只使用常数额外空间的算法来解决此问题吗？
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
*/

func ReverseKGroup(head *Entity.ListNode, k int) *Entity.ListNode {
	dummy := &Entity.ListNode{0, head}
	pre := dummy
	for head != nil {
		tail := pre
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
		pre.Next = head
		tail.Next = ndx

		// 更新pre和head指针，此时分别指向反转后的k个结点组成的链表的尾结点和反转前k个结点组成的链表尾结点的下一个结点
		pre = tail
		head = ndx
	}
	return dummy.Next
}

func reverse(head, tail *Entity.ListNode, k int) (*Entity.ListNode, *Entity.ListNode) {
	var pre *Entity.ListNode
	cur := head
	for i := 0; i < k; i++ {
		cur.Next, pre, cur = pre, cur, cur.Next
	}
	return tail, head
}

/*
1.4 变形题
给你单链表的头指针head和两个整数left和right ，其中left <= right 。请你反转从位置left到位置right的链表节点，返回反转后的链表。
例如对于链表1-2-3-4-5
left=2,right=4
最后应该返回1-4-3-2-5
思路:k个一组反转链表的简化变形题，找到left位置前的节点pre,left位置的节点start,right位置的节点end,以及right位置的下一个节点ndx，
将start,end，k=(right-left)+1作为参数传递给reverse(反转k个节点组成的链表)得到返回的head和tail节点，将pre节点的Next指针
指向head节点，将tail节点的Next指针指向ndx节点即可。
*/

func ReverseBetween(head *Entity.ListNode, left int, right int) *Entity.ListNode {
	k := right - left + 1
	dummy := &Entity.ListNode{0, head}
	cur := dummy
	for i := 0; i < left-1; i++ {
		cur = cur.Next
	}
	pre := cur
	start := cur.Next
	for i := 0; i < k; i++ {
		cur = cur.Next
	}
	end := cur
	ndx := cur.Next
	head, tail := reverse(start, end, k)
	pre.Next = head
	tail.Next = ndx
	return dummy.Next
}