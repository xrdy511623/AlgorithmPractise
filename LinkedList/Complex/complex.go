package Complex

import (
	"algorithm-practise/linkedlist/Entity"
	"algorithm-practise/linkedlist/ReverseLinkedList"
)

/*
leetcode 146 (面试题 16.25)
1.1 LRU缓存机制
运用你所掌握的数据结构，设计和实现一个LRU (最近最少使用) 缓存机制 。
实现LRUCache类：
LRUCache(int capacity) 以正整数作为容量capacity初始化LRU缓存
get(int key) 如果关键字key存在于缓存中，则返回关键字的值，否则返回-1 。
put(int key, int value)如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组[关键字-值]。
当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。
要求：在O(1) 时间复杂度内完成这两种操作？
*/

/*
LRU缓存机制可以通过哈希表辅以双向链表实现，我们用一个哈希表和一个双向链表维护所有在缓存中的键值对。
双向链表按照被使用的顺序存储了这些键值对，靠近头部的键值对是最近使用的，而靠近尾部的键值对是最久未使用的。
哈希表即为普通的哈希映射（HashMap），通过缓存数据的键映射到其在双向链表中的位置。
这样一来，我们首先使用哈希表进行定位，找出缓存项在双向链表中的位置，随后将其移动到双向链表的头部，即可在O(1)
的时间内完成get或者put操作。
具体的方法如下：
对于get操作，首先判断key是否存在：
如果key不存在，则返回−1；
如果key存在，则key对应的节点是最近被使用的节点。通过哈希表定位到该节点在双向链表中的位置，并将其移动到
双向链表的头部，最后返回该节点的值。

对于put操作，首先判断key是否存在：
如果key不存在，使用key和value创建一个新的节点，在双向链表的头部添加该节点，并将key和该节点添加进哈希表中。
然后判断双向链表的节点数是否超出容量，如果超出容量，则删除双向链表的尾部节点，并删除哈希表中对应的项；
如果key存在，则与get操作类似，先通过哈希表定位，再将对应的节点的值更新为value，并将该节点移到双向链表的头部。
上述各项操作中，访问哈希表的时间复杂度为O(1)，在双向链表的头部添加节点、在双向链表的尾部删除节点的复杂度也为
O(1)。而将一个节点移到双向链表的头部，可以分成「删除该节点」和「在双向链表的头部添加节点」两步操作，
都可以在O(1)时间内完成。

在双向链表的实现中，使用一个伪头部（dummy head）和伪尾部（dummy tail）标记界限，这样在添加节点和删除节点
的时候就不需要检查相邻的节点是否存在。
*/

// 综合使用双向链表和哈希表实现

type LRUCache struct {
	Size, Capacity int
	Cache          map[int]*DoubleLinkedListNode
	Head, Tail     *DoubleLinkedListNode
}

type DoubleLinkedListNode struct {
	Key, Value int
	Prev, Next *DoubleLinkedListNode
}

func InitDoubleLinkedListNode(key, value int) *DoubleLinkedListNode {
	return &DoubleLinkedListNode{
		Key:   key,
		Value: value,
	}
}

func Constructor(capacity int) LRUCache {
	l := LRUCache{
		Capacity: capacity,
		Cache:    map[int]*DoubleLinkedListNode{},
		Head:     InitDoubleLinkedListNode(0, 0),
		Tail:     InitDoubleLinkedListNode(0, 0),
	}
	l.Head.Next = l.Tail
	l.Tail.Prev = l.Head
	return l
}

func (l *LRUCache) Get(key int) int {
	if _, ok := l.Cache[key]; !ok {
		return -1
	}
	node := l.Cache[key]
	l.MoveToHead(node)
	return node.Value
}

func (l *LRUCache) Put(key, value int) {
	if _, ok := l.Cache[key]; !ok {
		node := InitDoubleLinkedListNode(key, value)
		l.Cache[key] = node
		l.AddToHead(node)
		l.Size++
		if l.Size > l.Capacity {
			removed := l.RemoveTail()
			delete(l.Cache, removed.Key)
			l.Size--
		}
	} else {
		node := l.Cache[key]
		node.Value = value
		l.MoveToHead(node)
	}
}

func (l *LRUCache) MoveToHead(node *DoubleLinkedListNode) {
	l.RemoveNode(node)
	l.AddToHead(node)
}

func (l *LRUCache) AddToHead(node *DoubleLinkedListNode) {
	node.Prev = l.Head
	node.Next = l.Head.Next
	l.Head.Next.Prev = node
	l.Head.Next = node
}

func (l *LRUCache) RemoveNode(node *DoubleLinkedListNode) {
	node.Prev.Next = node.Next
	node.Next.Prev = node.Prev
}

func (l *LRUCache) RemoveTail() *DoubleLinkedListNode {
	node := l.Tail.Prev
	l.RemoveNode(node)
	return node
}

/*
leetcode 23. 合并K个升序链表
1.2 给你一个链表数组，每个链表都已经按升序排列。
请你将所有链表合并到一个升序链表中，返回合并后的链表。

示例 1：
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6

示例 2：
输入：lists = []
输出：[]

示例 3：
输入：lists = [[]]
输出：[]
*/

/*
首先简化问题，合并两个有序链表是容易做到的，见MergeTwoLists代码
那么合并多个有序链表，我们第一个想到的办法就是用一个变量ans来维护合并的链表，第i次循环把第i个链表和ans合并，
答案保存到ans中。但这样做时间复杂度高达O(k^2n),联想到数组的归并排序，我们不难想到用分治合并的思路进行优化。
将k个链表配对并将同一对中的链表合并；
第一轮合并以后， k个链表被合并成了k/2个链表，平均长度为2n/k, 然后是k/4个链表,k/8个链表等等；
重复这一过程，直到我们得到了最终的有序链表。
故渐进时间复杂度为O(kn×logK)。
空间复杂度：递归会使用到O(logK) 空间代价的栈空间。
*/

func MergeKLists(lists []*Entity.ListNode) *Entity.ListNode {
	return Merge(lists, 0, len(lists)-1)
}

func Merge(lists []*Entity.ListNode, l, r int) *Entity.ListNode {
	if l == r {
		return lists[l]
	}
	if l > r {
		return nil
	}
	mid := (l + r) / 2
	return MergeTwoLists(Merge(lists, l, mid), Merge(lists, mid+1, r))
}

func MergeTwoLists(l1, l2 *Entity.ListNode) *Entity.ListNode {
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
leetcode 2. 两数相加
1.3 给你两个非空的链表，表示两个非负的整数。它们每位数字都是按照逆序的方式存储的，并且每个节点只能存储一位数字。
请你将两个数相加，并以相同形式返回一个表示和的链表。
你可以假设除了数字0之外，这两个数都不会以0开头。
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
*/

// AddTwoNumbers 从低位也就是链表头开始相加,时间复杂度O(max(m,n)), m和n分别为两个链表的长度，空间复杂度O(1)
func AddTwoNumbers(l1, l2 *Entity.ListNode) *Entity.ListNode {
	dummy := new(Entity.ListNode)
	cur := dummy
	// 进位值初始值为0
	carry := 0
	for l1 != nil || l2 != nil || carry > 0 {
		sum := 0
		if l1 != nil {
			sum += l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			sum += l2.Val
			l2 = l2.Next
		}
		sum += carry
		cur.Next = &Entity.ListNode{Val: sum % 10}
		cur = cur.Next
		carry = sum / 10
	}
	return dummy.Next
}

/*
leetcode 445. 两数相加II
剑指OfferII 025. 链表中的两数相加
1.4 给你两个非空链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加
会返回一个新的链表。

你可以假设除了数字0之外，这两个数字都不会以零开头。

输入：l1 = [7,2,4,3], l2 = [5,6,4]
输出：[7,8,0,7]
*/

/*
思路:利用栈先进后出的特性解决
本题的主要难点在于链表中数位的顺序与我们做加法的顺序是相反的，为了逆序处理所有数位，我们可以使用栈：
把所有数字压入栈中，再依次取出相加。计算过程中需要注意进位的情况。
*/

func AddTwoNumbersComplex(l1, l2 *Entity.ListNode) (head *Entity.ListNode) {
	var s1, s2 []int
	for l1 != nil {
		s1 = append(s1, l1.Val)
		l1 = l1.Next
	}
	for l2 != nil {
		s2 = append(s2, l2.Val)
		l2 = l2.Next
	}
	// 初始进位值carry为0
	carry := 0
	for len(s1) > 0 || len(s2) > 0 || carry > 0 {
		sum := 0
		if len(s1) > 0 {
			sum += s1[len(s1)-1]
			s1 = s1[:len(s1)-1]
		}
		if len(s2) > 0 {
			sum += s2[len(s2)-1]
			s2 = s2[:len(s2)-1]
		}
		sum += carry
		cur := &Entity.ListNode{Val: sum % 10}
		cur.Next = head
		head = cur
		carry = sum / 10
	}
	return
}

/*
leetcode 61. 旋转链表
1.5 给你一个链表的头节点head，旋转链表，将链表每个节点向右移动k个位置。
输入：head = [1,2,3,4,5], k = 2
输出：[4,5,1,2,3]
*/

/*
思路:闭合为环
记给定链表的长度为n，注意到当向右移动的次数k≥n 时，我们仅需要向右移动k%n 次即可。因为每n次移动都会让
链表变为原状。这样我们可以知道，新链表的最后一个节点为原链表的第 n - k % n 个节点（从1开始计数）。
这样，我们可以先将给定的链表连接成环，然后从指定位置断开。

具体代码中，我们首先计算出链表的长度n，并找到该链表的尾节点，将其与头节点相连。这样就得到了闭合为环的链表。
然后我们找到新链表的最后一个节点（即原链表的第 n - k % n 个节点），将当前闭合为环的链表断开，即可得到我们
所需要的结果。
特别地，当链表长度不大于1，或者k为n的倍数时，新链表将与原链表相同，我们无需进行任何处理。

*/

func RotateRight(head *Entity.ListNode, k int) *Entity.ListNode {
	if head == nil || head.Next == nil || k == 0 {
		return head
	}
	n := 1
	cur := head
	// 统计链表的长度
	for cur.Next != nil {
		cur = cur.Next
		n++
	}
	k = k % n
	// 如果k等于0, 证明k为n的倍数，那么保持原链表不变即可
	if k == 0 {
		return head
	}
	// 链表成环，将原链表的尾结点的Next指向头结点即成为首尾相连的新链表
	cur.Next = head
	// 新链表的尾结点end是原链表的第n-k个结点，那么从头结点移动到end，需要移动n-k-1步
	// 那么从原链表的尾结点移动到end，就需要多移动一步，即为n-k步。
	for i := 0; i < n-k; i++ {
		cur = cur.Next
	}
	// 因为是有环的链表，那么新链表的尾结点的Next指向的结点即为所求的头结点
	ret := cur.Next
	// 将新链表从尾部断开，去掉环。
	cur.Next = nil
	return ret
}

/*
leetcode 328. 奇偶链表
1.6 给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的
奇偶性，而不是节点的值的奇偶性。
请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes为节点总数。
*/

/*
思路:分离奇偶结点后再合并
最简单的做法是在从头遍历链表过程中，逐个完成奇数结点链表和偶数结点链表的构建，最后将奇数结点链表尾部结点的
Next指向偶数结点链表的头结点即可。
*/

// OddEvenListSimple 时间复杂度为O(N), 空间复杂度为O(1)
func OddEvenListSimple(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	// 奇数结点链表和偶数结点链表的头结点分别为head和head.Next
	odd, evenHead := head, head.Next
	even := evenHead
	for even != nil && even.Next != nil {
		// 奇偶结点是相邻的，下面的写法很容易了
		odd.Next = even.Next
		odd = odd.Next
		even.Next = odd.Next
		even = even.Next
	}
	// 将奇数结点链表尾结点的Next指向偶数结点链表的头结点
	odd.Next = evenHead
	return head
}

// oddEvenList 根据结点编号奇偶性分别迭代出奇数结点链表和偶数结点链表最后再合并
func oddEvenList(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	beforeOdd, beforeEven := new(Entity.ListNode), new(Entity.ListNode)
	odd, even := beforeOdd, beforeEven
	num := 1
	for head != nil {
		if num%2 == 1 {
			odd.Next = head
			odd = odd.Next
		} else {
			even.Next = head
			even = even.Next
		}
		num++
		head = head.Next
	}
	odd.Next = beforeEven.Next
	even.Next = nil
	return beforeOdd.Next
}

/*
1.7 排序奇升偶降链表
给定一个奇数位升序，偶数位降序的链表，将其重新排序为一个升序链表。

输入: 1->8->3->6->5->4->7->2
输出: 1->2->3->4->5->6->7->8
*/

/*
本题是leetcode 328.奇偶链表的变形题，更是一道综合题，其实如果把反转链表，奇偶链表，合并两个升序链表三道题目
弄懂了，本题并不难解。首先从头到尾遍历链表，逐个完成奇数结点链表和偶数结点链表的构建。接着我们将
降序排列的偶数结点链表反转，这样我们就得到了两个升序链表，最后将它们俩合并为一个升序链表即可
*/

func SortOddAscEvenDescList(head *Entity.ListNode) *Entity.ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	odd, evenHead := head, head.Next
	even := evenHead
	// 在遍历链表过程中构建奇数结点链表和偶数结点链表
	for even != nil && even.Next != nil {
		odd.Next = even.Next
		odd = odd.Next
		even.Next = odd.Next
		even = even.Next
	}
	odd.Next = nil
	l1 := head
	// 将偶数结点链表反转得到升序链表
	l2 := ReverseLinkedList.Reverse(evenHead)
	// 合并奇数结点链表和偶数结点链表这两个升序链表
	return MergeTwoLists(l1, l2)
}

/*
leetcode 1171. 从链表中删去总和值为零的连续节点
1.8 给你一个链表的头节点head，请你编写代码，反复删去链表中由总和值为0的连续节点组成的序列，直到不存在这样的
序列为止。
删除完毕后，请你返回最终结果链表的头节点。
你可以返回任何满足题目要求的答案。

输入：head = [1,2,-3,3,1]
输出：[3,1]
提示：答案 [1,2,1] 也是正确的。
*/

func RemoveZeroSumSubLists(head *Entity.ListNode) *Entity.ListNode {
	dummy := &Entity.ListNode{Next: head}
	occurred := make(map[int]*Entity.ListNode)
	occurred[0] = dummy
	sum := 0
	for head != nil {
		sum += head.Val
		if _, ok := occurred[sum]; ok {
			// 如果重复出现一个连续结点和sum，只能说明中间这一段的结点和为0
			// sum+0还是sum本身嘛，所以这一段是要跳过的。
			occurred[sum].Next = head.Next
			// 然后再去删除后的链表看是否还有总和值为0的连续结点需要删除
			return RemoveZeroSumSubLists(dummy.Next)
		} else {
			occurred[sum] = head
			head = head.Next
		}
	}
	return dummy.Next
}

/*
输入：head = [1,2,-3,3,1]
输出：[3,1]
提示：答案 [1,2,1] 也是正确的。
*/

// RemoveZeroSumSubListsSimple 更简单的写法是下面这样，时间复杂度为O(2N), 空间复杂度为O(N)
func RemoveZeroSumSubListsSimple(head *Entity.ListNode) *Entity.ListNode {
	dummy := &Entity.ListNode{Val: 0, Next: head}
	sum := 0
	dict := make(map[int]*Entity.ListNode)
	for p := dummy; p != nil; p = p.Next {
		sum += p.Val
		dict[sum] = p
	}
	sum = 0
	for p := dummy; p != nil; p = p.Next {
		sum += p.Val
		p.Next = dict[sum].Next
	}
	return dummy.Next
}
