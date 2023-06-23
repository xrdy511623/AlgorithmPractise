package Feature

import "AlgorithmPractise/LinkedList/Entity"

/*
leetcode 141. 环形链表
1.1 给定一个链表，判断链表中是否有环。
如果链表中有某个节点，可以通过连续跟踪next指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数
pos来表示链表尾连接到链表中的位置（索引从0开始）。 如果pos是-1，则在该链表中没有环。注意：pos不作为参数进行
传递，仅仅是为了标识链表的实际情况。如果链表中存在环，则返回true。 否则返回false 。
*/

/*
思路一:快慢双指针，最优解，不易想到
设置快、慢两个指针，快指针每次跨两步，慢指针每次跨一步，如果快指针没有与慢指针相遇而是顺利到达链表尾部
说明没有环；否则，存在环。原因是因为每走1轮，fast与slow的间距+1，fast终会追上slow。
*/

// CheckRing 时间复杂度O(n),空间复杂度O(1)
func CheckRing(head *Entity.ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	fast, slow := head, head
	for fast.Next != nil && fast.Next.Next != nil {
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

// CheckRingUseHashTable 时间复杂度O(n),空间复杂度O(n)
func CheckRingUseHashTable(head *Entity.ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	visited := make(map[*Entity.ListNode]bool)
	for head != nil {
		if visited[head] {
			return true
		}
		visited[head] = true
		head = head.Next
	}
	return false
}

/*
leetcode 142. 环形链表II
1.2 进阶
给定一个链表，返回链表开始入环的第一个节点。如果链表无环，则返回null。
为了表示给定链表中的环，我们使用整数pos来表示链表尾连接到链表中的位置（索引从0开始)。 如果pos是 -1，
则在该链表中没有环。注意，pos仅仅是用于标识环的情况，并不会作为参数传递到函数中。
说明：不允许修改给定的链表。
*/

/*
思路1，从头结点开始遍历链表，若哈希表中不存在此结点，则将其添加到哈希表中，否则意味着遇到了开始入环的
第一个结点 (重复节点)
*/

// DetectCycleUseHashTable 时间复杂度O(N),空间复杂度O(N)
func DetectCycleUseHashTable(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return nil
	}
	seen := make(map[*Entity.ListNode]bool)
	for head != nil {
		if seen[head] {
			return head
		}
		seen[head] = true
		head = head.Next
	}
	return nil
}

// DetectCycle 思路2，双指针法，不易想到，但是是最优解, 时间复杂度仍然是O(N),但空间复杂度下降为O(1)
func DetectCycle(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return nil
	}
	fast, slow := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
		// 快慢指针相遇后，慢指针和从头节点出发的p指针同时向后移动，相遇节点即为开始入环的第一个节点
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
leetcode 160. 相交链表
剑指Offer 52. 两个链表的第一个公共节点
1.3 输入两个链表，找出它们的第一个公共节点。
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
剑指Offer 22. 链表中倒数第k个节点
面试题 02.02. 返回倒数第k个节点
1.4 输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数
第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第3个节点
是值为4的节点。
示例：
给定一个链表: 1->2->3->4->5, 和 k = 2.
返回链表 4->5.
*/

// GetKthFromEnd 顺序查找，倒数第k个节点即为正数第n-k+1个节点
func GetKthFromEnd(head *Entity.ListNode, k int) *Entity.ListNode {
	n := 0
	cur := head
	for cur != nil {
		n++
		cur = cur.Next
	}
	// 如果链表长度小于k, 说明链表中节点总数小于k个, 则返回nil
	if n < k {
		return nil
	}
	// head头节点是正数第1个节点，要走到正数第n-k+1个节点，需要走n-k步
	for i := 0; i < n-k; i++ {
		head = head.Next
	}
	return head
}

/*
思路: 双指针法，假设链表总共有n个节点(n>=k) 。那么求倒数第k个节点，即是求正数第n-k+1个节点。设快慢指针初始
位置都是头节点，快指针先走k步移动至k+1个节点，此时快慢指针相隔k个节点，然后双方以相同速度同时向后移动，则当快
指针移动至链表尾部空节点时，快指针走过了n-k步，此时慢指针也走了n-k步，正好是正数第n-k+1个节点。
(从头节点出发，所以要加1), 即为所求节点。
*/

func GetKthNodeFromEnd(head *Entity.ListNode, k int) *Entity.ListNode {
	fast, slow := head, head
	for i := 0; i < k; i++ {
		// 若链表中节点总数少于k个，则返回nil
		if fast == nil {
			return nil
		}
		fast = fast.Next
	}
	for fast != nil {
		fast = fast.Next
		slow = slow.Next
	}
	return slow
}

/*
leetcode 21. 合并两个有序链表
1.5 将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
*/

func MergeTwoLists(l1 *Entity.ListNode, l2 *Entity.ListNode) *Entity.ListNode {
	dummy := new(Entity.ListNode)
	cur := dummy
	for l1 != nil && l2 != nil {
		if l1.Val < l2.Val {
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
leetcode 148. 排序链表
剑指Offer II 077. 链表排序
1.6 给你链表的头结点head，请将其按升序排列并返回排序后的链表 。
进阶：
你可以在O(nlogn) 时间复杂度和常数级空间复杂度下，对链表进行排序吗？
输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
*/

// SortLinkedList 思路:归并排序解决
func SortLinkedList(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	// 利用快慢指针寻找链表中间结点，将链表一分为二
	slow, fast := head, head
	for fast.Next != nil && fast.Next.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	mid := slow.Next
	slow.Next = nil
	left, right := SortLinkedList(head), SortLinkedList(mid)
	return MergeTwoLists(left, right)
}

/*
leetcode 86. 分隔链表
1.7 给你一个链表的头节点head和一个特定值x，请你对链表进行分隔，使得所有小于x的节点都出现在大于或等于x的
节点之前。你应当保留两个分区中每个节点的初始相对位置。
*/

/*
题目本质上就是将链表分为：
1.小于x部分的链表按照原始顺序记为p
2.大于等于x部分的链表按照原始顺序记为q
3.拼接两个链表，p --> q
*/

func PartitionLinkedList(head *Entity.ListNode, x int) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	p, q := new(Entity.ListNode), new(Entity.ListNode)
	c1, c2 := p, q
	for head != nil {
		if head.Val < x {
			c1.Next = head
			c1 = c1.Next
		} else {
			c2.Next = head
			c2 = c2.Next
		}
		head = head.Next
	}
	// 拼接两个链表
	// 将c2的Next指针指向nil，这一点一定要记住
	c2.Next = nil
	c1.Next = q.Next
	return p.Next
}

/*
leetcode 147. 对链表进行插入排序
1.8 对链表进行插入排序。
插入排序算法：
插入排序是迭代的，每次只移动一个元素，直到所有元素可以形成一个有序地输出列表。
每次迭代中，插入排序只从输入数据中移除一个待排序的元素，找到它在序列中适当的位置，并将其插入。
重复直到所有输入数据插入完为止。

输入: 4->2->1->3
输出: 1->2->3->4
*/

func insertionSortList(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	dummy := &Entity.ListNode{Next: head}
	// 有序链表的尾结点，待插入结点初始值分别为头结点以及头结点的下一个结点
	lastSorted, cur := head, head.Next
	for cur != nil {
		// 如果尾结点的值小于等于待插入结点值，则保持原序即可
		if lastSorted.Val <= cur.Val {
			lastSorted = lastSorted.Next
		} else {
			// 否则，需要在有序链表部分找到待插入结点的位置
			// 也就是小于等于待插入结点值(cur)的最大结点prev
			prev := dummy
			for prev.Next.Val <= cur.Val {
				prev = prev.Next
			}
			// 将待插入结点cur插入到prev和lastSorted之间
			lastSorted.Next = cur.Next
			cur.Next = prev.Next
			prev.Next = cur
		}
		cur = lastSorted.Next
	}
	return dummy.Next
}
