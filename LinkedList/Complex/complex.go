package Complex

/*
1.0 LRU缓存机制
运用你所掌握的数据结构，设计和实现一个LRU (最近最少使用) 缓存机制 。
实现LRUCache类：
LRUCache(int capacity) 以正整数作为容量capacity初始化LRU缓存
get(int key) 如果关键字key存在于缓存中，则返回关键字的值，否则返回-1 。
put(int key, int value)如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组[关键字-值]。当缓存容量达到上限时，
它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。
要求：在O(1) 时间复杂度内完成这两种操作？
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
