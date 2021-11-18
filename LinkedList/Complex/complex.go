package Complex

import "AlgorithmPractise/LinkedList/Entity"

/*
1.1 LRU缓存机制
运用你所掌握的数据结构，设计和实现一个LRU (最近最少使用) 缓存机制 。
实现LRUCache类：
LRUCache(int capacity) 以正整数作为容量capacity初始化LRU缓存
get(int key) 如果关键字key存在于缓存中，则返回关键字的值，否则返回-1 。
put(int key, int value)如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组[关键字-值]。当缓存容量达到上限时，
它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。
要求：在O(1) 时间复杂度内完成这两种操作？
*/

/*
LRU 缓存机制可以通过哈希表辅以双向链表实现，我们用一个哈希表和一个双向链表维护所有在缓存中的键值对。
双向链表按照被使用的顺序存储了这些键值对，靠近头部的键值对是最近使用的，而靠近尾部的键值对是最久未使用的。
哈希表即为普通的哈希映射（HashMap），通过缓存数据的键映射到其在双向链表中的位置。
这样以来，我们首先使用哈希表进行定位，找出缓存项在双向链表中的位置，随后将其移动到双向链表的头部，即可在O(1)的时间内完成get或者put操作。
具体的方法如下：
对于get操作，首先判断key是否存在：
如果key不存在，则返回−1；
如果key存在，则key对应的节点是最近被使用的节点。通过哈希表定位到该节点在双向链表中的位置，并将其移动到双向链表的头部，最后返回该节点的值。

对于put操作，首先判断key是否存在：
如果key不存在，使用key和value创建一个新的节点，在双向链表的头部添加该节点，并将key和该节点添加进哈希表中。然后判断双向链表的节点数是否
超出容量，如果超出容量，则删除双向链表的尾部节点，并删除哈希表中对应的项；
如果key存在，则与get操作类似，先通过哈希表定位，再将对应的节点的值更新为value，并将该节点移到双向链表的头部。
上述各项操作中，访问哈希表的时间复杂度为O(1)，在双向链表的头部添加节点、在双向链表的尾部删除节点的复杂度也为O(1)。而将一个节点移到
双向链表的头部，可以分成「删除该节点」和「在双向链表的头部添加节点」两步操作，都可以在O(1)时间内完成。
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
1.2 合并k个排序链表
给你一个链表数组，每个链表都已经按升序排列。
请你将所有链表合并到一个升序链表中，返回合并后的链表。
给你一个链表数组，每个链表都已经按升序排列。

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
那么合并多个有序链表，我们第一个想到的办法就是用一个变量ans来维护合并的链表，第 i次循环把第i个链表和ans合并，答案保存到ans中。
但这样做时间复杂度高达O(k^2n),联想到数组的归并排序，我们不难想到用分治合并的思路进行优化。
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
	mid := l + (r-l)/2
	return MergeTwoLists(Merge(lists, l, mid), Merge(lists, mid+1, r))
}

func MergeTwoLists(a, b *Entity.ListNode) *Entity.ListNode {
	if a == nil || b == nil {
		if a != nil {
			return a
		} else {
			return b
		}
	}
	dummy := &Entity.ListNode{Val: 0, Next: nil}
	cur := dummy
	l1, l2 := a, b
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
	}
	if l2 != nil {
		cur.Next = l2
	}
	return dummy.Next
}
