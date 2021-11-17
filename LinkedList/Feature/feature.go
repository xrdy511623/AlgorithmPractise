package Feature

import "AlgorithmPractise/LinkedList/Entity"

/*
1.1 判断链表是否有环
给定一个链表，判断链表中是否有环。
如果链表中有某个节点，可以通过连续跟踪next指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数pos来表示链表尾连接到链表中的位置
（索引从 0 开始）。 如果pos是-1，则在该链表中没有环。注意：pos不作为参数进行传递，仅仅是为了标识链表的实际情况。
如果链表中存在环，则返回true 。 否则返回false 。
*/

/*
思路一:快慢双指针，最优解，不易想到
设置快、慢两种指针，快指针每次跨两步，慢指针每次跨一步，如果快指针没有与慢指针相遇而是顺利到达链表尾部
说明没有环；否则，存在环
返回:
True:有环
False:没有环
时间复杂度O(n),空间复杂度O(1)
*/

func CheckRing(head *Entity.ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	fast, slow := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
		if fast == slow {
			return true
		}
	}
	return false
}

/*
思路二: 利用哈希表来检测链表中是否有环，易想到，但空间复杂度更高，不推荐。
遍历链表，如果该节点在集合中已经存在，直接将其返回，否则将该结点添加到哈希表中
时间复杂度O(n),空间复杂度O(n)
*/

func CheckRingUseHashTable(head *Entity.ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	visited := make(map[*Entity.ListNode]int)
	for head != nil {
		if _, ok := visited[head]; ok {
			return true
		} else {
			visited[head]++
			head = head.Next
		}
	}
	return false
}

/*
1.2 进阶 环形链表
给定一个链表，返回链表开始入环的第一个节点。如果链表无环，则返回null。
为了表示给定链表中的环，我们使用整数pos来表示链表尾连接到链表中的位置（索引从0开始）。 如果pos是 -1，则在该链表中没有环。注意，pos
仅仅是用于标识环的情况，并不会作为参数传递到函数中。
说明：不允许修改给定的链表。
*/

// DetectCycleUseHashTable 思路1，从头结点开始遍历链表，若哈希表中不存在此结点，则将其添加到哈希表中，否则意味着遇到了开始入环的
//第一个结点 (重复节点)
func DetectCycleUseHashTable(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return nil
	}
	seen := make(map[*Entity.ListNode]struct{})
	for head != nil {
		if _, ok := seen[head]; ok {
			return head
		} else {
			seen[head] = struct{}{}
			head = head.Next
		}
	}
	return nil
}

// DetectCycle 思路2，双指针法，不易想到，但是是最优解
func DetectCycle(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return nil
	}
	fast, slow := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
		if slow == fast {
			p := head
			for p != slow {
				p = p.Next
				slow = slow.Next
			}
			return p
		}
	}
	return nil
}

/*
1.3 两个链表的第一个公共节点
输入两个链表，找出它们的第一个公共节点。
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表A为 [4,1,8,4,5]，链表B为 [5,0,1,8,4,5]。
在A中，相交节点前有2个节点；在B中，相交节点前有3个节点。
*/

/*
思路:双指针解决
用双指针解决两个链表的第一个公共节点
设第一个公共节点为node ，链表headA的节点数量为a，链表headB的节点数量为b，两链表的公共尾部的节点数量为c
则有：头节点headA到node前，共有a−c个节点；
头节点headB到node前，共有b−c个节点；
考虑构建两个节点指针A, B 分别指向两链表头节点headA , headB ，做如下操作：
指针A先遍历完链表headA，再开始遍历链表headB ，当走到node时，共走步数为：
a+(b−c)
指针B先遍历完链表headB，再开始遍历链表headA ，当走到node时，共走步数为：
b+(a−c)
如下式所示，此时指针 A , B 重合：
a+(b−c)=b+(a−c)
*/

func getIntersectionNode(headA, headB *Entity.ListNode) *Entity.ListNode {
	a, b := headA, headB
	for a != b {
		if a == nil {
			a = headB
		} else {
			a = a.Next
		}
		if b == nil {
			b = headA
		} else {
			b = b.Next
		}
	}
	return a
}

/*
1.4 链表排序
给你链表的头结点head，请将其按升序排列并返回排序后的链表 。
进阶：
你可以在O(nlogn) 时间复杂度和常数级空间复杂度下，对链表进行排序吗？
输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
 */

// SortLinkedList 思路，归并排序解决
func SortLinkedList(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	// 利用快慢指针寻找链表中间结点，将链表一分为二
	slow, fast := head, head.Next
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	mid := slow.Next
	slow.Next = nil
	left, right := SortLinkedList(head), SortLinkedList(mid)
	res := &Entity.ListNode{0, nil}
	h := res
	for left != nil && right != nil {
		if left.Val < right.Val{
			h.Next = left
			left = left.Next
		} else {
			h.Next = right
			right = right.Next
		}
		h = h.Next
	}
	if left != nil {
		h.Next = left
	} else {
		h.Next = right
	}

	return res.Next
	//return MergeLinkedList(left, right)
}

func MergeLinkedList(l1,l2 *Entity.ListNode) *Entity.ListNode {
	dummy := &Entity.ListNode{0, nil}
	cur := dummy
	for l1 != nil && l2 != nil {
		if l1.Val < l2.Val{
			cur.Next = l1
			l1 = l1.Next
		} else {
			cur.Next = l2
			l2 = l2.Next
		}
		cur = cur.Next
	}
	if l1 != nil {
		cur.Next = l1
	} else {
		cur.Next = l2
	}

	return dummy.Next
}

/*
1.5 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
*/


func mergeTwoLists(l1 *Entity.ListNode, l2 *Entity.ListNode) *Entity.ListNode {
	dummy := &Entity.ListNode{0, nil}
	cur := dummy
	if l1 != nil && l2 != nil{
		for l1 != nil && l2 != nil{
			if l1.Val <= l2.Val{
				cur.Next = l1
				l1 = l1.Next
			} else{
				cur.Next = l2
				l2 = l2.Next
			}
			cur = cur.Next
		}

		if l1 != nil{
			cur.Next = l1
		} else{
			cur.Next = l2
		}

		return dummy.Next
	}

	if l1 != nil{
		return l1
	} else{
		return l2
	}
}

/*
1.6 分隔链表
给你一个链表的头节点 head 和一个特定值x ，请你对链表进行分隔，使得所有小于x的节点都出现在大于或等于x的节点之前。
你应当保留两个分区中每个节点的初始相对位置。
 */

/*
题目本质上就是将链表分为：
1.小于x部分的链表按照原始顺序记为p
2.大于等于x部分的链表按照原始顺序记为
3.拼接两个链表，p --> q
 */


func PartitionlinkedList(head *Entity.ListNode, x int) *Entity.ListNode {
	before_head, after_head := &Entity.ListNode{0, nil}, &Entity.ListNode{0, nil}
	before, after := before_head, after_head
	for head != nil {
		if head.Val < x {
			before.Next = head
			before = before.Next
		} else {
			after.Next = head
			after = after.Next
		}
		head = head.Next
	}
	// 拼接两个链表
	after.Next = nil
	before.Next = after_head.Next
	return before_head.Next
}