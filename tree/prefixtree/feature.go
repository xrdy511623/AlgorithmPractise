package prefixtree

/*
leetcode 208 实现Trie(前缀树)
Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用
情景，例如自动补全和拼写检查。

请你实现 Trie 类：

Trie() 初始化前缀树对象。
void insert(String word) 向前缀树中插入字符串 word 。
boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。


示例：
输入
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出
[null, null, true, false, true, null, true]

解释
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True

提示：
1 <= word.length, prefix.length <= 2000
word 和 prefix 仅由小写英文字母组成
insert、search 和 startsWith 调用次数 总计 不超过 3 * 104 次
*/

/*
解题思路
Trie 树结构：
每个节点存储一个字符，以及该字符对应的子节点。
需要标记一个布尔值表示是否有单词在当前节点结束。
操作实现：

插入：
遍历单词的每个字符，在树中逐层插入。
如果某个字符对应的子节点不存在，则创建一个新的子节点。
插入结束后，标记当前节点为单词结尾。

搜索：
遍历单词的每个字符，在树中逐层查找。
如果某个字符对应的子节点不存在，则返回 false。
检查最后一个字符对应的节点是否为单词结尾。

检查前缀：
遍历前缀的每个字符，在树中逐层查找。
如果某个字符对应的子节点不存在，则返回 false。
如果遍历完成，返回 true。

性能优化：
使用一个固定大小为 26 的数组代替哈希表存储子节点，优化空间和查找速度（假设只处理小写字母）。
*/

// TrieNode 定义 Trie 树的节点
type TrieNode struct {
	// 子节点，固定为 26 个字母
	children []*TrieNode
	// 是否是单词结尾
	isEnd bool
}

// Trie 定义 Trie 树
type Trie struct {
	root *TrieNode
}

// Constructor 初始化 Trie 树对象
func Constructor() Trie {
	return Trie{
		root: &TrieNode{
			children: make([]*TrieNode, 26),
			isEnd:    false,
		},
	}
}

// Insert 向 Trie 树中插入字符串 word
func (t *Trie) Insert(word string) {
	node := t.root
	for _, ch := range word {
		// 将字符转换为数组索引
		index := ch - 'a'
		if node.children[index] == nil {
			node.children[index] = &TrieNode{
				children: make([]*TrieNode, 26),
				isEnd:    false,
			}
		}
		node = node.children[index]
	}
	// 标记单词结尾
	node.isEnd = true
}

// Search 搜索完整单词 word 是否存在
func (t *Trie) Search(word string) bool {
	node := t.root
	for _, ch := range word {
		index := ch - 'a'
		// 如果某个字符不存在，则单词不存在
		if node.children[index] == nil {
			return false
		}
		node = node.children[index]
	}
	// 只有到达单词结尾且标记为 isEnd 才返回 true
	return node.isEnd
}

func (t *Trie) StartsWith(prefix string) bool {
	node := t.root
	for _, ch := range prefix {
		index := ch - 'a'
		// 如果某个字符不存在，则前缀不存在
		if node.children[index] == nil {
			return false
		}
		node = node.children[index]
	}
	// 遍历完成即表示前缀存在
	return true
}
