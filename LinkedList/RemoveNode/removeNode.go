package RemoveNode

import "AlgorithmPractise/LinkedList/Entity"

/*
1.1
给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
返回删除后的链表的头节点。
示例 1:
输入: head = [4,5,1,9], val = 5
输出: [4,1,9]
解释: 给定你链表中值为5的第二个节点，那么在调用了你的函数之后，该链表应变为4 -> 1 -> 9.
示例 2:
输入: head = [4,5,1,9], val = 1
输出: [4,5,9]
解释: 给定你链表中值为1的第三个节点，那么在调用了你的函数之后，该链表应变为4 -> 5 -> 9.
*/

// DeleteNode 思路:找到值等于val的目标节点target及其前驱节点pre，将pre.Next指向target的Next节点即可

func DeleteNode(head *Entity.ListNode, val int) *Entity.ListNode {
	if head == nil {
		return nil
	}
	dummy := &Entity.ListNode{0, head}
	pre, cur := dummy, head
	for cur.Val != val {
		pre = pre.Next
		cur = cur.Next
	}
	pre.Next = cur.Next
	return dummy.Next
}

/*
1.2 给你一个链表，删除链表的倒数第n个结点，并且返回链表的头结点。
进阶：你能尝试使用一趟扫描实现吗？
*/

// RemoveNthFromEnd 思路与1.1大体类似，那就是找到该节点的前一个节点，将其Next指针指向该节点的Next节点即可
func RemoveNthFromEnd(head *Entity.ListNode, n int) *Entity.ListNode {
	dummy := &Entity.ListNode{0, head}
	pre := dummy
	length := GetLengthOfLinkedList(head)
	for i := 0; i < length-n; i++ {
		pre = pre.Next
	}
	pre.Next = pre.Next.Next
	return dummy.Next
}

func GetLengthOfLinkedList(head *Entity.ListNode) int {
	length := 0
	if head == nil {
		return length
	}
	for head != nil {
		length++
		head = head.Next
	}
	return length
}

// RemoveNthNodeFromEnd 双指针法
func RemoveNthNodeFromEnd(head *Entity.ListNode, n int) *Entity.ListNode {
	dummy := &Entity.ListNode{0, head}
	pre, fast := dummy, head
	for i := 0; i < n; i++ {
		fast = fast.Next
	}
	for fast != nil {
		fast = fast.Next
		pre = pre.Next
	}
	pre.Next = pre.Next.Next
	return dummy.Next
}

/*
1.3：删除排序链表中的重复元素
存在一个按升序排列的链表，给你这个链表的头节点head，请你删除所有重复的元素，使每个元素只出现一次 。
返回同样按升序排列的结果链表。
譬如1-2-3-3-5
返回1-2-3-5
*/

// DeleteDuplicatesSimple  从头结点开始遍历链表，遇有与其Next指针指向结点值相等的，则将该节点
// Next指针指向Next.Next，这样便跳过了重复结点。
func DeleteDuplicatesSimple(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	cur := head
	for cur.Next != nil {
		if cur.Val == cur.Next.Val {
			cur.Next = cur.Next.Next
		} else {
			cur = cur.Next
		}
	}
	return head
}

/*
1.4 变形题 删除排序链表中的重复元素
存在一个按升序排列的链表，给你这个链表的头节点head，请你删除链表中所有存在数字重复情况的节点，只保留原始链表中没有重复出现的数字。
返回同样按升序排列的结果链表。
譬如1-2-3-3-5
返回1-2-5
思路:只要当前节点重复，就将节点的Next指针向后移动，判断第一个重复节点的前驱节点pre的Next指针指向的节点是否是当前节点cur，如果是，证明没有重复节点，
pre节点向后顺序移动即可，否则pre.Next便指向当前节点的Next节点，这样就跳过了所有重复节点
*/

func DeleteDuplicates(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	dummy := &Entity.ListNode{0, head}
	pre, cur := dummy, head
	for cur != nil {
		for cur.Next != nil && cur.Val == cur.Next.Val {
			cur = cur.Next
		}

		if pre.Next == cur {
			pre = pre.Next
		} else {
			pre.Next = cur.Next
		}

		cur = cur.Next
	}
	return dummy.Next
}

/*
1.5 进阶:移除未排序链表中的重复节点，只保留最开始出现的节点。
譬如1-2-3-3-2-1
返回1-2-3
思路:哈希表去重,不能再使用1.3中的方法，因为这次是未排序链表，前面出现过的节点值在后面任意位置都可能再次出现
*/

func RemoveDuplicateNodes(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	cur := head
	occurred := make(map[int]int)
	occurred[head.Val] = 1
	for cur.Next != nil {
		node := cur.Next
		if _, ok := occurred[node.Val]; !ok {
			occurred[node.Val]++
			cur = cur.Next
		} else {
			cur.Next = cur.Next.Next
		}
	}
	return head
}
