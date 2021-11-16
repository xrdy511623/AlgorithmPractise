package Complex

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
