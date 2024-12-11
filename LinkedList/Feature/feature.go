package feature

import (
	"algorithm-practise/linkedlist/entity"
)

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
说明没有环；否则，存在环。因为每走1轮，fast与slow的间距+1，fast终会追上slow。
*/

// CheckRing 时间复杂度O(n),空间复杂度O(1)
func CheckRing(head *entity.ListNode) bool {
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
func CheckRingUseHashTable(head *entity.ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	visited := make(map[*entity.ListNode]bool)
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
第一个结点 (重复结点)
*/

// DetectCycleUseHashTable 时间复杂度O(N),空间复杂度O(N)
func DetectCycleUseHashTable(head *entity.ListNode) *entity.ListNode {
	if head == nil || head.Next == nil {
		return nil
	}
	seen := make(map[*entity.ListNode]bool)
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
func DetectCycle(head *entity.ListNode) *entity.ListNode {
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

func getIntersectionNode(headA, headB *entity.ListNode) *entity.ListNode {
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
func GetKthFromEnd(head *entity.ListNode, k int) *entity.ListNode {
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

func GetKthNodeFromEnd(head *entity.ListNode, k int) *entity.ListNode {
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

func MergeTwoLists(l1 *entity.ListNode, l2 *entity.ListNode) *entity.ListNode {
	dummy := new(entity.ListNode)
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
func SortLinkedList(head *entity.ListNode) *entity.ListNode {
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

func PartitionLinkedList(head *entity.ListNode, x int) *entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	p, q := new(entity.ListNode), new(entity.ListNode)
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

func insertionSortList(head *entity.ListNode) *entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	dummy := &entity.ListNode{Next: head}
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

/*
剑指 Offer 35. 复杂链表的复制
1.9 请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个
random 指针指向链表中的任意节点或者 null。
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]

leetcode 138 随机链表的复制
给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next
指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针
都不应指向原链表中的节点 。

例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。
返回复制链表的头节点。

用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：
val：一个表示 Node.val 的整数。
random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
你的代码 只 接受原链表的头节点 head 作为传入参数。

输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]

提示：
0 <= n <= 1000
-104 <= Node.val <= 104
Node.random 为 null 或指向链表中的节点。
*/

type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

func copyRandomList(head *Node) *Node {
	cache := make(map[*Node]*Node)
	var deepCopy func(*Node) *Node
	deepCopy = func(node *Node) *Node {
		if node == nil {
			return nil
		}
		if Node, done := cache[node]; done {
			return Node
		}
		newNode := &Node{Val: node.Val}
		cache[node] = newNode
		newNode.Next = deepCopy(node.Next)
		newNode.Random = deepCopy(node.Random)
		return newNode
	}
	return deepCopy(head)
}

/*
思路：迭代 + 节点拆分
复制链表节点：我们首先遍历原链表，在每个节点后插入一个新的节点，这个新节点是该原节点的复制。这样，我们在同一个链表中形成了
一个“交替”结构：
原节点 -> 新节点 -> 原节点 -> 新节点 -> …

例如，对于原链表 head -> A -> B -> C，复制后的链表结构将是：
head -> A' -> A -> B' -> B -> C' -> C

复制 random 指针：由于新节点是紧跟在原节点后的，我们可以通过原节点的 random 指针来直接访问新节点。具体做法是：

对于每个节点的原 random 指针，如果原节点的 random 指向节点 X，那么新节点的 random 就应该指向 X 的下一个节点。
也就是说，对于原节点 node 和新节点 node'，如果 node.random 指向 otherNode，那么 node'.random 应该指向 otherNode.next。

分离两个链表：在完成复制后，原链表和新链表是交替插入的。接下来，我们需要将它们分开，使得新链表成为一个独立的链表。

具体步骤：
在原链表节点后插入复制节点。
设置复制节点的 random 指针。
将原链表和复制链表分开。
时间复杂度：
每次操作都只遍历链表一次，因此时间复杂度是 O(n)。
空间复杂度：
我们只使用了常数空间来处理指针和临时变量，因此空间复杂度是 O(1)。
*/

func copyRandomListSimple(head *Node) *Node {
	if head == nil {
		return nil
	}
	// 第一步：在每个节点后面插入一个新的节点
	cur := head
	for cur != nil {
		newNode := &Node{
			Val:  cur.Val,
			Next: cur.Next,
		}
		cur.Next = newNode
		cur = newNode.Next
	}
	// 第二步：复制 random 指针
	cur = head
	for cur != nil {
		if cur.Random != nil {
			cur.Next.Random = cur.Random.Next
		}
		cur = cur.Next.Next
	}
	// 第三步：分离原链表和新链表
	newHead := head.Next
	curOld := head
	curNew := newHead
	for curNew != nil {
		curOld.Next = curOld.Next.Next // 还原原链表
		if curNew.Next != nil {
			curNew.Next = curNew.Next.Next // 设置新链表的 next
		}
		curOld = curOld.Next
		curNew = curNew.Next
	}

	return newHead
}

/*
leetcode 234 回文链表
给你一个单链表的头节点 head ，请你判断该链表是否为回文链表，如果是，返回 true ；否则，返回 false 。

输入：head = [1,2,2,1]
输出：true

输入：head = [1,2]
输出：false

提示：
链表中节点数目在范围[1, 105] 内
0 <= Node.val <= 9
进阶：你能否用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题？
*/

func isPalindrome(head *entity.ListNode) bool {
	l := []int{}
	for head != nil {
		l = append(l, head.Val)
		head = head.Next
	}
	for i, n := 0, len(l); i < n/2; i++ {
		if l[i] != l[n-1-i] {
			return false
		}
	}
	return true
}

func isPalindromeAdvanced(head *entity.ListNode) bool {
	tail := getMid(head)
	mid := tail.Next
	tail.Next = nil
	l1 := head
	l2 := reverse(mid)
	res := true
	for res && l2 != nil {
		if l1.Val != l2.Val {
			res = false
			break
		}
		l1 = l1.Next
		l2 = l2.Next
	}
	return res
}

func getMid(head *entity.ListNode) *entity.ListNode {
	fast, slow := head, head
	for fast.Next != nil && fast.Next.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	return slow
}

func reverse(head *entity.ListNode) *entity.ListNode {
	var prev *entity.ListNode
	cur := head
	for cur != nil {
		cur.Next, prev, cur = prev, cur, cur.Next
	}
	return prev
}
