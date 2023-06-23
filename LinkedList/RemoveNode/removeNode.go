package RemoveNode

import "AlgorithmPractise/LinkedList/Entity"

/*
剑指Offer 18. 删除链表的节点
1.1 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
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
	dummy := &Entity.ListNode{Next: head}
	prev, cur := dummy, head
	for cur.Val != val {
		prev = prev.Next
		cur = cur.Next
	}
	prev.Next = cur.Next
	return dummy.Next
}

/*
leetcode 203. 移除链表元素
1.2 给你一个链表的头节点head和一个整数val，请你删除链表中所有满足Node.val == val的节点，并返回新的头节点。

示例1：
输入：head = [1,2,6,3,4,5,6], val = 6
输出：[1,2,3,4,5]

示例2：
输入：head = [], val = 1
输出：[]

示例3：
输入：head = [7,7,7,7], val = 7
输出：[]
*/

/*
找到值等于val的目标节点target及其前驱节点pre，将pre.Next指向target的Next节点.
*/

// RemoveElements 时间复杂度O(N),空间复杂度O(1)
func RemoveElements(head *Entity.ListNode, val int) *Entity.ListNode {
	dummy := &Entity.ListNode{Next: head}
	prev, cur := dummy, head
	for cur != nil {
		if cur.Val == val {
			prev.Next = cur.Next
		} else {
			prev = prev.Next
		}
		cur = cur.Next
	}
	return dummy.Next
}

/*
leetcode 19. 删除链表的倒数第N个结点
剑指Offer II 021. 删除链表的倒数第n个结点
1.3 给你一个链表，删除链表的倒数第n个结点，并且返回链表的头结点。
进阶：你能尝试使用一趟扫描实现吗？

提示：
链表中结点的数目为 sz
1 <= sz <= 30
0 <= Node.val <= 100
1 <= n <= sz
*/

/*
若链表长度为length, 删除链表的倒数第n个结点，等价于删除链表的正数第length-n+1个结点，那么我们要做的
其实很简单，在head结点之前建立一个伪头结点dummy,令prev:=dummy,从prev开始遍历length-n次(prev=prev.Next)，
此时prev指向的即为目标结点的前一个结点，我们只需要将prev.Next 指向它的Next.Next指向的结点即可删除
目标结点了。
*/

// RemoveNthFromEnd 思路与1.1大体类似，那就是找到该节点的前一个节点，将其Next指针指向该节点的Next节点即可
func RemoveNthFromEnd(head *Entity.ListNode, n int) *Entity.ListNode {
	length := GetLengthOfLinkedList(head)
	// 若头结点为空或链表结点数小于n，则直接返回头结点
	if length == 0 || length-n < 0 {
		return head
	}
	dummy := &Entity.ListNode{Next: head}
	pre := dummy
	for i := 0; i < length-n; i++ {
		pre = pre.Next
	}
	pre.Next = pre.Next.Next
	return dummy.Next
}

func GetLengthOfLinkedList(head *Entity.ListNode) int {
	length := 0
	for head != nil {
		length++
		head = head.Next
	}
	return length
}

/*
为了便于处理链表只有一个头结点或者头结点为nil的情况，创建一个伪头结点dummy,令其Next指针指向head.
若链表长度为length, 删除链表的倒数第n个结点，等价于删除链表的正数第length-n+1个结点. 所以我们
准备两个指针fast和slow，初始位置分别为head和dummy。我们先让快指针fast移动n步。然后fast, slow
同时向后移动，那么fast剩下可移动的步数为length-n步，此时slow指向的结点即为目标结点的前一个结点，
我们只需要设置slow.Next = slow.Next.Next便可删除目标结点了。
*/

// RemoveNthNodeFromEnd 双指针法
func RemoveNthNodeFromEnd(head *Entity.ListNode, n int) *Entity.ListNode {
	dummy := &Entity.ListNode{Next: head}
	slow, fast := dummy, head
	for i := 0; i < n; i++ {
		if fast == nil {
			return dummy.Next
		}
		fast = fast.Next
	}
	for fast != nil {
		fast = fast.Next
		slow = slow.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}

/*
leetcode 83. 删除排序链表中的重复元素
1.4 存在一个按升序排列的链表，给你这个链表的头节点head，请你删除所有重复的元素，使每个元素只出现一次 。
返回同样按升序排列的结果链表。
譬如1-2-3-3-5
返回1-2-3-5
*/

/*
从头结点开始遍历链表，遇有与其Next指针指向结点值相等的，则将该节点Next指针指向Next.Next，这样便跳过了重复
结点。
*/

func DeleteDuplicate(head *Entity.ListNode) *Entity.ListNode {
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
leetcode 82. 删除排序链表中的重复元素II
1.5 变形题
存在一个按升序排列的链表，给你这个链表的头节点head，请你删除链表中所有存在数字重复情况的节点，只保留原始链表
中没有重复出现的数字。返回同样按升序排列的结果链表。
譬如1-2-3-3-5
返回1-2-5
*/

/*
思路:只要当前节点cur与其Next节点值重复，就将当前节点cur向后移动，判断前驱节点prev的Next指针指向的节点
是否是当前节点cur，如果是，证明cur没有向后移动，意味着没有重复节点，此时prev = prev.Next,否则
prev.Next便指向当前节点cur的Next节点，这样就跳过了所有重复节点。
*/

func DeleteDuplicates(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	dummy := &Entity.ListNode{Next: head}
	prev, cur := dummy, head
	for cur != nil {
		for cur.Next != nil && cur.Val == cur.Next.Val {
			cur = cur.Next
		}
		if prev.Next == cur {
			prev = prev.Next
		} else {
			prev.Next = cur.Next
		}

		cur = cur.Next
	}
	return dummy.Next
}

/*
1.6 进阶:移除未排序链表中的重复节点，只保留最开始出现的节点。
譬如1-2-3-3-2-1
返回1-2-3
思路:哈希表去重,不能再使用1.4中的方法，因为这次是未排序链表，前面出现过的节点值在后面任意位置都可能再次出现
*/

func RemoveDuplicateNodes(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	occurred := make(map[int]bool)
	cur := head
	occurred[cur.Val] = true
	for cur.Next != nil {
		node := cur.Next
		if !occurred[node.Val] {
			occurred[node.Val] = true
			cur = cur.Next
		} else {
			cur.Next = cur.Next.Next
		}
	}
	return head
}
