package RemoveNode

type ListNode struct {
	Val  int
	Next *ListNode
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

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

func DeleteNode(head *ListNode, val int) *ListNode {
	if head == nil {
		return nil
	}
	dummy := &ListNode{0, head}
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

// removeNthFromEnd, 思路与1.1大体类似，那就是找到该节点的前一个节点，将其Next指针指向该节点的Next节点即可
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{0, head}
	pre := dummy
	length := GetLengthOfLinkedList(head)
	for i := 0; i < length-n; i++ {
		pre = pre.Next
	}
	pre.Next = pre.Next.Next
	return dummy.Next
}

func GetLengthOfLinkedList(head *ListNode) int {
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

/*
1.3：删除排序链表中的重复元素
存在一个按升序排列的链表，给你这个链表的头节点head ，请你删除所有重复的元素，使每个元素只出现一次 。
返回同样按升序排列的结果链表。
譬如1-2-3-3-5
返回1-2-3-5
*/
func DeleteDuplicatesSimple(head *ListNode) *ListNode {
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