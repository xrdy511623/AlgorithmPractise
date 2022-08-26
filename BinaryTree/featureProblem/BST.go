package featureProblem

import (
	"AlgorithmPractise/BinaryTree/Entity"
	Entity2 "AlgorithmPractise/LinkedList/Entity"
	"math"
)

/*
二叉搜索树专题
BST二叉搜索树(Binary Search Tree)是一种特殊类型的二叉树，它的特点是根节点值为所有节点值排序序列
的中位数，任一节点值均大于等于其左子树所有节点值，且均小于等于其右子树所有节点值。因此它的中序遍历
序列是一个升序的有序数组。
*/

/*
leetcode 701. 二叉搜索树中的插入操作
1.1 给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。
输入数据保证，新值和原始二叉搜索树中的任意节点值都不同。
注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。你可以返回任意有效的结果。
*/

// InsertIntoBST 迭代法 时间复杂度O(N)，空间复杂度O(1)
func InsertIntoBST(root *Entity.TreeNode, val int) *Entity.TreeNode {
	if root == nil {
		return &Entity.TreeNode{Val: val}
	}
	cur := root
	newNode := &Entity.TreeNode{Val: val}
	for cur != nil {
		if val < cur.Val {
			if cur.Left == nil {
				cur.Left = newNode
				break
			}
			cur = cur.Left
		} else {
			if cur.Right == nil {
				cur.Right = newNode
				break
			}
			cur = cur.Right
		}
	}
	return root
}

// InsertIntoBSTSimple 递归法 时间复杂度O(N)，空间复杂度O(1)
func InsertIntoBSTSimple(root *Entity.TreeNode, val int) *Entity.TreeNode {
	if root == nil {
		root = &Entity.TreeNode{Val: val}
		return root
	}
	if val < root.Val {
		root.Left = InsertIntoBSTSimple(root.Left, val)
	} else {
		root.Right = InsertIntoBSTSimple(root.Right, val)
	}
	return root
}

/*
leetcode 450. 删除二叉搜索树中的节点
1.2 给定一个二叉搜索树的根节点root和一个值 key，删除二叉搜索树中的key对应的节点，并保证二叉搜索树的性质不变。
返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

首先找到需要删除的节点；
如果找到了，删除它。

输入：root = [5,3,6,2,4,null,7], key = 3
输出：[5,4,6,2,null,null,7]
解释：给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。
一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。
另一个正确答案是 [5,2,6,null,4,null,7]。
*/

/*
思路:分类讨论，递归解决
删除BST中的节点，无外乎以下三种情况:
1 如果key < root.Val，说明要删除的节点在BST的左子树，那么递归的去左子树删除即可
2 如果key > root.Val，说明要删除的节点在BST的右子树，那么递归的去右子树删除即可
3 如果key = root.Val，说明要删除的节点就是本节点，这一类又分以下三种情况
a 要删除的节点是叶子节点，那很简单，直接将当前节点删除，置为nil即可;
b 要删除的节点有右子节点，那么为了维持BST的特性，我们需要找到该节点的后继节点post(BST中大于它的最小节点)，
将该节点的值更新为后继节点post的值，然后递归的在当前节点的右子树中删除该后继节点post;
c 要删除的节点有左子节点，那么为了维持BST的特性，我们需要找到该节点的前驱节点pre(BST中小于于它的最大节点)，
将该节点的值更新为前驱节点pre的值，然后递归的在当前节点的左子树中删除该前驱节点pre;
最后返回当前节点的引用即可。
*/

// DeleteNodeInBST 时间复杂度O(logN)，空间复杂度O(H)
func DeleteNodeInBST(root *Entity.TreeNode, key int) *Entity.TreeNode {
	if root == nil {
		return nil
	} else if key < root.Val {
		root.Left = DeleteNodeInBST(root.Left, key)
	} else if key > root.Val {
		root.Right = DeleteNodeInBST(root.Right, key)
	} else {
		if root.Left == nil && root.Right == nil {
			root = nil
		} else if root.Right != nil {
			root.Val = Successor(root).Val
			root.Right = DeleteNodeInBST(root.Right, root.Val)
		} else {
			root.Val = Predecessor(root).Val
			root.Left = DeleteNodeInBST(root.Left, root.Val)
		}
	}
	return root
}

// Predecessor 在二叉搜索树(BST)中寻找当前节点的前驱节点
func Predecessor(cur *Entity.TreeNode) *Entity.TreeNode {
	pre := cur.Left
	if pre == nil {
		return pre
	}
	for pre.Right != nil {
		pre = pre.Right
	}
	return pre
}

// Successor 在二叉搜索树(BST)中寻找当前节点的后继节点
func Successor(cur *Entity.TreeNode) *Entity.TreeNode {
	post := cur.Right
	if post == nil {
		return post
	}
	for post.Left != nil {
		post = post.Left
	}
	return post
}

/*
leetcode 98. 验证二叉搜索树
1.3 给你一个二叉树的根节点root，判断其是否是一个有效的二叉搜索树。

有效二叉搜索树定义如下：
节点的左子树只包含小于当前节点的数。
节点的右子树只包含大于当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。
*/

// CheckIsValidBST 一个很容易想到的思路是中序遍历二叉树，如果它是BST，就会得到一个升序序列，否则就不是BST
func CheckIsValidBST(root *Entity.TreeNode) bool {
	minValue := math.MinInt64
	stack := []*Entity.TreeNode{}
	for len(stack) > 0 || root != nil {
		if root != nil {
			stack = append(stack, root)
			root = root.Left
		} else {
			root = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if root.Val <= minValue {
				return false
			}
			minValue = root.Val
			root = root.Right
		}
	}
	return true
}

// IsValidBST 利用二叉搜索树的特征递归解决,时间复杂度和空间复杂度都是O(N)
func IsValidBST(root *Entity.TreeNode) bool {
	min := math.MinInt64
	max := math.MaxInt64
	return helper(root, min, max)
}

func helper(root *Entity.TreeNode, min, max int) bool {
	if root == nil {
		return true
	}
	// 二叉搜索树的核心特征就是根节点的值分布在其左右子节点值区间内[root.Left.Val, root.Right.Val]，否则就不是BST
	if root.Val <= min || root.Val >= max {
		return false
	}
	return helper(root.Left, min, root.Val) && helper(root.Right, root.Val, max)
}

/*
leetcode 700. 二叉搜索树中的搜索
1.4 给定二叉搜索树（BST）的根节点和一个值。你需要在BST中找到节点值等于给定值的节点。返回以该节点为根的子树。
如果节点不存在，则返回NULL。
*/

// SearchBST DFS递归
func SearchBST(root *Entity.TreeNode, val int) *Entity.TreeNode {
	if root == nil || root.Val == val {
		return root
	}
	if root.Val < val {
		return SearchBST(root.Right, val)
	} else {
		return SearchBST(root.Left, val)
	}
}

// SearchBSTSimple 迭代法
func SearchBSTSimple(root *Entity.TreeNode, val int) *Entity.TreeNode {
	for root != nil {
		if root.Val < val {
			root = root.Right
		} else if root.Val > val {
			root = root.Left
		} else {
			return root
		}
	}
	return root
}

/*
面试题 04.06. 后继者
剑指Offer II 053. 二叉搜索树中的中序后继
1.5 设计一个算法，找出二叉搜索树中指定节点的“下一个”节点（也即中序后继）。
如果指定节点没有对应的“下一个”节点，则返回null。
*/

/*
第一种方案，对二叉搜索树(BST)进行中序遍历，即可得到升序排列的集合list，在集合中找到指定节点p的位置pos,
返回list[index+1]对应的节点即可，若列表长度-1<=pos，则证明指定节点p为最大节点，也就没有了下一个节点，返回空
此方案简单易懂，但是效率太差，需要先遍历一遍整个BST，然后for循环list找到p节点的位置，整体时间复杂度为O(n+pos),
空间复杂度为O(n),因此不是最佳解决方案。
*/

func InorderSuccessor(root, p *Entity.TreeNode) *Entity.TreeNode {
	var travel func(node *Entity.TreeNode) []*Entity.TreeNode
	travel = func(node *Entity.TreeNode) []*Entity.TreeNode {
		res := []*Entity.TreeNode{}
		if node == nil {
			return res
		}
		res = append(res, travel(node.Left)...)
		res = append(res, node)
		res = append(res, travel(node.Right)...)
		return res
	}
	nodesList := travel(root)
	if len(nodesList) == 0 {
		return nil
	}
	pos := 0
	for index, node := range nodesList {
		if node == p {
			pos = index
			break
		}
	}
	if len(nodesList)-1 > pos {
		return nodesList[pos+1]
	} else {
		return nil
	}
}

/*
在二叉搜索树BST中寻找p的中序后继节点无非三种情形，一是p有右子树，那么大于p的最小节点一定是它的
右子树中最左边的节点，如图bst1.jpg所示；
第二种情况是p没有右子树，此时我们需要从p的父亲节点往上寻找，若p是其父亲节点的左子节点，此时p的
后继节点显然就是其父节点，如图bst1.jpg所示;若p是其父亲节点的右子节点，此时p的后继节点显然就是
其祖父节点,如图bst3.jpg所示
最后一种情况就是p没有右子树，且p是其父亲节点的左子节点的右子节点，也就是p是该BST中最大的一个节点，
此时p是没有中序后继节点的。
所以，针对情形一，我们需要在p的右子树中寻找最左边的节点；针对情形二，我们需要找到p的父亲节点
第三种情形，直接返回null
*/

// InorderSuccessorUseIteration 迭代法解决，时间复杂度降低为O(pos),空间复杂度降低为O(1)
func InorderSuccessorUseIteration(root, p *Entity.TreeNode) *Entity.TreeNode {
	if root == nil {
		return nil
	}
	var prev *Entity.TreeNode
	for root.Val != p.Val {
		if root.Val < p.Val {
			root = root.Right
		} else {
			// prev为大于p的父节点
			prev = root
			root = root.Left
		}
	}
	// 此时在BST中找到了节点p
	// 如果p节点没有右子节点，则其中序后继节点为prev
	if p.Right == nil {
		return prev
	} else {
		// 否则可在其右子树中寻找其中序后继节点
		post := p.Right
		for post.Left != nil {
			post = post.Left
		}
		return post
	}
}

/*
同类题，二叉搜索树中的中序前驱
1.6 设计一个算法，找出二叉搜索树中指定节点的“上一个”节点（也即中序前驱）。
如果指定节点没有对应的“上一个”节点，则返回null。
*/

func InorderPredecessor(root, p *Entity.TreeNode) *Entity.TreeNode {
	if root == nil {
		return nil
	}
	var prev *Entity.TreeNode
	for root.Val != p.Val {
		if p.Val > root.Val {
			prev = root
			root = root.Right
		} else {
			root = root.Left
		}
	}
	if p.Left == nil {
		return prev
	}
	pre := p.Left
	for pre.Right != nil {
		pre = pre.Right
	}
	return pre
}

/*
leetcode 938. 二叉搜索树的范围和
1.7 给定二叉搜索树的根结点root，返回值位于范围[low, high] 之间的所有结点的值的和。

示例:
输入：root = [10,5,15,3,7,null,18], low = 7, high = 15
输出：32
*/

// RangeSumBST 递归 时间复杂度O(N)，空间复杂度O(N)
func RangeSumBST(root *Entity.TreeNode, low int, high int) int {
	if root == nil {
		return 0
	}
	if root.Val < low {
		return RangeSumBST(root.Right, low, high)
	} else if root.Val > high {
		return RangeSumBST(root.Left, low, high)
	} else {
		return root.Val + RangeSumBST(root.Right, low, high) + RangeSumBST(root.Left, low, high)
	}
}

// RangeSumBSTSimple 迭代 时间复杂度O(N)，空间复杂度O(N)
func RangeSumBSTSimple(root *Entity.TreeNode, low int, high int) int {
	sum := 0
	if root == nil {
		return sum
	}
	queue := []*Entity.TreeNode{root}
	for len(queue) != 0 {
		node := queue[0]
		queue = queue[1:]
		if node == nil {
			continue
		}
		if node.Val < low {
			queue = append(queue, node.Right)
		} else if node.Val > high {
			queue = append(queue, node.Left)
		} else {
			sum += node.Val
			queue = append(queue, node.Left, node.Right)
		}
	}
	return sum
}

/*
leetcode 538. 把二叉搜索树转换为累加树
1.8 给出二叉搜索树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点
node的新值等于原树中大于或等于node.val的值之和。

提醒一下，二叉搜索树满足下列约束条件：
节点的左子树仅包含键小于节点键的节点。 节点的右子树仅包含键大于节点键的节点。 左右子树也必须是二叉
搜索树。

示例1:
输入：[4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
输出：[30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]

示例2：
输入：root = [0,null,1]
输出：[1,null,1]

示例3：
输入：root = [1,0,2]
输出：[3,3,2]

示例4：
输入：root = [3,2,4,1]
输出：[7,9,4,10]

提示：
树中的节点数介于0和104之间。
每个节点的值介于-104和104之间。
树中的所有值互不相同 。
给定的树为二叉搜索树。
*/

/*
思路:反序中序遍历
本题中要求我们将每个节点的值修改为原来的节点值加上所有大于它的节点值之和。这样我们只需要反序中序遍历
该二叉搜索树，记录过程中的节点值之和，并不断更新当前遍历到的节点的节点值，即可得到题目要求的累加树。
*/

// ConvertBST 时间复杂度O(N)，空间复杂度O(N)
func ConvertBST(root *Entity.TreeNode) *Entity.TreeNode {
	sum := 0
	var dfs func(*Entity.TreeNode)
	dfs = func(node *Entity.TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Right)
		sum += node.Val
		node.Val = sum
		dfs(node.Left)
	}
	dfs(root)
	return root
}

/*
leetcode 108
1.9 将有序数组转换为二叉搜索树
给你一个整数数组nums ，其中元素已经按升序排列，请你将其转换为一棵高度平衡二叉搜索树。
高度平衡二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过1的二叉树。

输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案
*/

/*
思路:按照BST中序遍历会得到一个升序序列的特点，递归的构建BST即可，具体来说，有序数组中间位置mid元素
为根节点，nums[:mid]为左子树范围，nums[mid+1:]为右子树范围，这样左右子树范围大体相当，节点数相差
不会超过1，肯定是平衡二叉树，根节点的左子节点显然也是nums[:mid]中间位置的元素，根节点的右子节点显然
是nums[mid+1:]中间位置的元素,如此根节点左子树和右子树也是平衡二叉搜索树了.总之，以有序数组中间位置
为切割点，一定能保证左右子树范围大体相当，是平衡树。
*/

func SortedArrayToBST(nums []int) *Entity.TreeNode {
	// 递归终止条件，当有序数组为空时，返回nil
	if len(nums) == 0 {
		return nil
	}
	// 当前根节点永远是有序数组中间位置元素
	mid := len(nums) / 2
	root := &Entity.TreeNode{Val: nums[mid]}
	root.Left = SortedArrayToBST(nums[:mid])
	root.Left = SortedArrayToBST(nums[mid+1:])
	return root
}

/*
leetcode 109. 有序链表转换二叉搜索树
1.10 给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。
本题中，一个高度平衡二叉树是指一个二叉树每个节点的左右两个子树的高度差的绝对值不超过1。

示例:
给定的有序链表： [-10, -3, 0, 5, 9],
一个可能的答案是：[0, -3, 9, -10, null, 5], 它可以表示下面这个高度平衡二叉搜索树：

      0
     / \
   -3   9
   /   /
 -10  5
*/

/*
思路:本题与1.29 将有序数组转换为二叉搜索树本质上是一样的，不过是多了一个顺序遍历有序链表得到升序
数组的过程。
*/

func sortedListToBST(head *Entity2.ListNode) *Entity.TreeNode {
	if head == nil {
		return nil
	}
	var sortedArray []int
	for head != nil {
		sortedArray = append(sortedArray, head.Val)
		head = head.Next
	}
	var dfs func([]int) *Entity.TreeNode
	dfs = func(nums []int) *Entity.TreeNode {
		if len(nums) == 0 {
			return nil
		}
		mid := len(nums) / 2
		root := &Entity.TreeNode{Val: nums[mid]}
		root.Left = dfs(nums[:mid])
		root.Right = dfs(nums[mid+1:])
		return root
	}
	return dfs(sortedArray)
}

/*
剑指Offer 33. 二叉搜索树的后序遍历序列
1.11 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回true，否则返回false。假设输入
的数组的任意两个数字都互不相同。
输入: [1,6,3,2,5]
输出: false

输入: [1,3,2,6,5]
输出: true
*/

/*
后序遍历定义： [ 左子树 | 右子树 | 根节点 ] ，即遍历顺序为 “左、右、根” 。
二叉搜索树定义：左子树中所有节点的值<根节点的值；右子树中所有节点的值>根节点的值；其左、右子树也分别为二叉搜索树。
方法一：递归分治
根据二叉搜索树的定义，可以通过递归，判断所有子树的正确性（即其后序遍历是否满足二叉搜索树的定义） ，若所有子树都正确，则此序列为
二叉搜索树的后序遍历。
递归解析:i为后序遍历数组中第一个元素，j为最后一个元素
终止条件:当 i≥j ，说明此子树节点数量≤1 ，无需判别正确性，因此直接返回true；
递归工作:
划分左右子树：遍历后序遍历的 [i, j]区间元素，寻找第一个大于根节点的节点，索引记为m 。此时，可划分出左子树区间[i,m-1] 、右子树区间
[m, j-1], 根节点索引j 。
判断是否为二叉搜索树：
左子树区间[i,m−1]内的所有节点都应 < postorder[j] 。而划分左右子树步骤已经保证左子树区间的正确性，因此只需要判断右子树区间即可。
右子树区间[m,j−1] 内的所有节点都应 > postorder[j] 。实现方式为遍历，当遇到 ≤ postorder[j] 的节点则跳出；则可通过p=j判断
是否为二叉搜索树。

返回值:所有子树都需正确才可判定正确，因此使用与逻辑符&&连接。
p=j： 判断此树是否正确。
recur(i,m−1):判断此树的左子树是否正确。
recur(m,j−1):判断此树的右子树是否正确。
复杂度分析：
时间复杂度 O(N):每次调用recur(i,j) 减去一个根节点，因此递归占用O(N) ；最差情况下（即当树退化为链表），每轮递归都需遍历树所有节点，
占用 O(N^2)。
空间复杂度 O(N):最差情况下（即当树退化为链表），递归深度将达到N 。
*/

func VerifyPostOrder(postOrder []int) bool {
	var recur func([]int, int, int) bool
	recur = func(array []int, start, stop int) bool {
		if start >= stop {
			return true
		}
		p := start
		for array[p] < array[stop] {
			p += 1
		}
		m := p
		for array[p] > array[stop] {
			p += 1
		}
		return p == stop && recur(array, start, m-1) && recur(array, m, stop-1)
	}

	return recur(postOrder, 0, len(postOrder)-1)
}

/*
1.11.1 拓展 验证二叉搜索树的前序遍历序列
输入一个整数数组，判断该数组是不是某二叉搜索树的前序遍历结果。如果是则返回true，否则返回false。假设输入
的数组的任意两个数字都互不相同。
输入: [8，5，1，7，10，12]
输出: true
*/

func VerifyPreOrder(preOrder []int) bool {
	var recur func([]int, int, int) bool
	recur = func(nums []int, start, stop int) bool {
		if start >= stop {
			return true
		}
		p := start + 1
		for nums[p] < nums[start] {
			p++
		}
		m := p
		for nums[p] > nums[start] && p < stop {
			p++
		}
		return p == stop && recur(nums, start+1, m-1) && recur(nums, m, stop)
	}
	return recur(preOrder, 0, len(preOrder)-1)
}

/*
leetcode 235. 二叉搜索树的最近公共祖先
1.12 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
*/

// LowestCommonAncestor 迭代法解决
func LowestCommonAncestor(root, p, q *Entity.TreeNode) *Entity.TreeNode {
	for root != nil {
		if p.Val < root.Val && q.Val < root.Val {
			root = root.Left
		} else if p.Val > root.Val && q.Val > root.Val {
			root = root.Right
		} else {
			return root
		}
	}
	return nil
}

// LowestCommonAncestorUseRecursion 递归法解决
func LowestCommonAncestorUseRecursion(root, p, q *Entity.TreeNode) *Entity.TreeNode {
	if p.Val < root.Val && q.Val < root.Val {
		return LowestCommonAncestorUseRecursion(root.Left, p, q)
	} else if p.Val > root.Val && q.Val > root.Val {
		return LowestCommonAncestorUseRecursion(root.Right, p, q)
	} else {
		return root
	}
}

/*
剑指Offer II 052. 展平二叉搜索树
1.13 给你一棵二叉搜索树，请按中序遍历 将其重新排列为一棵递增顺序搜索树，使树中最左边的节点成为树的根节点，
并且每个节点没有左子节点，只有一个右子节点。
*/

// IncreasingBST 时间复杂度O(2*N)，空间复杂度O(N)
func IncreasingBST(root *Entity.TreeNode) *Entity.TreeNode {
	var res []*Entity.TreeNode
	var dfs func(*Entity.TreeNode)
	dfs = func(node *Entity.TreeNode) {
		if node != nil {
			dfs(node.Left)
			res = append(res, node)
			dfs(node.Right)
		}
	}
	dfs(root)
	dummy := new(Entity.TreeNode)
	cur := dummy
	for _, node := range res {
		cur.Right = node
		node.Left, node.Right = nil, nil
		cur = cur.Right
	}
	return dummy.Right
}

// IncreasingSimpleBST 更好的做法是在中序遍历的过程中直接改变节点指向，时间复杂度下降为O(N)
func IncreasingSimpleBST(root *Entity.TreeNode) *Entity.TreeNode {
	dummy := new(Entity.TreeNode)
	cur := dummy
	var helper func(*Entity.TreeNode)
	helper = func(node *Entity.TreeNode) {
		if node == nil {
			return
		}
		// 在中序遍历的过程中修改节点指向
		helper(node.Left)
		cur.Right = node
		node.Left = nil
		cur = node
		helper(node.Right)
	}
	helper(root)
	return dummy.Right
}

/*
leetcode 173. 二叉搜索树迭代器
1.13 实现一个二叉搜索树迭代器类BSTIterator，表示一个按中序遍历二叉搜索树（BST）的迭代器：
BSTIterator(TreeNode root) 初始化BSTIterator类的一个对象。BST的根节点root会作为构造函数的一部分给出。
指针应初始化为一个不存在于BST中的数字，且该数字小于BST中的任何元素。
boolean hasNext() 如果向指针右侧遍历存在数字，则返回true ；否则返回false 。
int next()将指针向右移动，然后返回指针处的数字。
注意，指针初始化为一个不存在于BST中的数字，所以对next()的首次调用将返回 BST 中的最小元素。
你可以假设next()调用总是有效的，也就是说，当调用 next()时，BST的中序遍历中至少存在一个下一个数字。
*/

type BSTIterator struct {
	Nums []int
	Root *Entity.TreeNode
}

func Constructor(root *Entity.TreeNode) BSTIterator {
	nums := make([]int, 0)
	Inorder(root, &nums)
	return BSTIterator{
		Nums: nums,
		Root: root,
	}
}

func (this *BSTIterator) Next() int {
	val := this.Nums[0]
	this.Nums = this.Nums[1:]
	return val
}

func (this *BSTIterator) HasNext() bool {
	return len(this.Nums) > 0
}

func Inorder(node *Entity.TreeNode, nums *[]int) {
	if node == nil {
		return
	}
	Inorder(node.Left, nums)
	*nums = append(*nums, node.Val)
	Inorder(node.Right, nums)
}

/*
leetcode 95. 不同的二叉搜索树
1.14 给你一个整数n，请你生成并返回所有由n个节点组成且节点值从1到n互不相同的不同二叉搜索树。
可以按任意顺序返回答案。
输入：n = 3
输出：[[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]
*/

func GenerateTrees(n int) []*Entity.TreeNode {
	if n == 0 {
		return []*Entity.TreeNode{}
	}
	return Helper(1, n)
}

func Helper(start, end int) []*Entity.TreeNode {
	if start > end {
		return []*Entity.TreeNode{nil}
	}
	var allTrees []*Entity.TreeNode
	// 枚举可行根节点
	for i := start; i <= end; i++ {
		// 获得所有可行的左子树集合
		leftTrees := Helper(start, i-1)
		// 获得所有可行的右子树集合
		rightTrees := Helper(i+1, end)
		// 从左子树集合中选出一棵左子树，从右子树集合中选出一棵右子树，拼接到根节点上
		for _, left := range leftTrees {
			for _, right := range rightTrees {
				curTree := &Entity.TreeNode{Val: i, Left: nil, Right: nil}
				curTree.Left = left
				curTree.Right = right
				allTrees = append(allTrees, curTree)
			}
		}
	}
	return allTrees
}

/*
leetcode 530. 二叉搜索树的最小绝对差
1.15 给你一个二叉搜索树的根节点root ，返回树中任意两不同节点值之间的最小差值。
差值是一个正数，其数值等于两值之差的绝对值。

示例:
输入：root = [4,2,6,1,3]
输出：1

提示：
树中节点的数目范围是 [2, 104]
0 <= Node.val <= 105
*/

// GetMinimumDifference 中序遍历得到升序数组，然后迭代取相邻元素差值的最小值即可
func GetMinimumDifference(root *Entity.TreeNode) int {
	var dfs func(*Entity.TreeNode) []int
	dfs = func(node *Entity.TreeNode) (res []int) {
		if node == nil {
			return res
		}
		res = append(res, dfs(node.Left)...)
		res = append(res, node.Val)
		res = append(res, dfs(node.Right)...)
		return res
	}
	sortedArray := dfs(root)
	min := math.MaxInt32
	for i := 1; i < len(sortedArray); i++ {
		if value := sortedArray[i] - sortedArray[i-1]; value < min {
			min = value
		}
	}
	return min
}

// GetMinimumDifferenceSimple 也可以直接在dfs中序遍历中迭代这个最小差值
func GetMinimumDifferenceSimple(root *Entity.TreeNode) int {
	var prev *Entity.TreeNode
	min := math.MaxInt32
	var dfs func(*Entity.TreeNode)
	dfs = func(node *Entity.TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		if prev != nil && node.Val-prev.Val < min {
			min = node.Val - prev.Val
		}
		prev = node
		dfs(node.Right)
	}
	dfs(root)
	return min
}

/*
leetcode 501. 二叉搜索树中的众数
1.16 给定一个有相同值的二叉搜索树（BST），找出BST中的所有众数（出现频率最高的元素）。

假定BST有如下定义：
结点左子树中所含结点的值小于等于当前结点的值
结点右子树中所含结点的值大于等于当前结点的值
左子树和右子树都是二叉搜索树

例如：
给定BST [1,null,2,2]
返回[2].

提示：如果众数超过1个，不需考虑输出顺序
进阶：你可以不使用额外的空间吗？（假设由递归产生的隐式调用栈的开销不被计算在内）
*/

/*
BST(二叉搜索树)统计节点值出现频率，那就通过中序遍历形成有序数组，然后相邻两个元素作比较，
就把出现频率最高的元素输出就可以了。
*/

// FindMode  时间复杂度O(N)，空间复杂度O(N)
func FindMode(root *Entity.TreeNode) []int {
	var res []int
	var prev *Entity.TreeNode
	count, maxCount := 1, 1
	var dfs func(*Entity.TreeNode)
	dfs = func(node *Entity.TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		// 与前一个节点值相等，出现频次累加1
		if prev != nil && prev.Val == node.Val {
			count++
		} else {
			// 否则，出现频次重置为1
			count = 1
		}
		prev = node
		// 如果出现频次与最大出现频次相等，说明当前节点值为众数
		if count == maxCount {
			res = append(res, node.Val)
		}
		// 如果当前节点值出现频次大于最大出现频次，那就需要更新最大出现频次maxCount
		// 清空原来存储众数的数组(之前存储的都不对，因为出现频次不够大)，将当前节点值存到数组中
		if count > maxCount {
			maxCount = count
			res = []int{}
			res = append(res, node.Val)
		}
		dfs(node.Right)
	}
	dfs(root)
	return res
}

/*
leetcode 669. 修剪二叉搜索树
1.17 给你二叉搜索树的根节点root ，同时给定最小边界low和最大边界high。通过修剪二叉搜索树，使得所有节点的值
在[low, high]中。修剪树不应该改变保留在树中的元素的相对结构（即，如果没有被移除，原有的父代子代关系
都应当保留）。 可以证明，存在唯一的答案。

所以结果应当返回修剪好的二叉搜索树的新的根节点。注意，根节点可能会根据给定的边界发生改变。

示例:
输入：root = [1,0,2], low = 1, high = 2
输出：[1,null,2]
*/

/*
当node.val>high，那么修剪后的二叉树必定出现在节点的左边。
类似地，当node.val<low，那么修剪后的二叉树出现在节点的右边。否则，我们将会修剪树的两边。
*/

func TrimBST(root *Entity.TreeNode, low, high int) *Entity.TreeNode {
	if root == nil {
		return root
	}
	if root.Val < low {
		// 去右子树修剪
		return TrimBST(root.Right, low, high)
	} else if root.Val > high {
		// 去左子树修剪
		return TrimBST(root.Left, low, high)
	} else {
		// 左右子树都需要修剪
		root.Left = TrimBST(root.Left, low, high)
		root.Right = TrimBST(root.Right, low, high)
		return root
	}
}

/*
leetcode 1305. 两棵二叉搜索树中的所有元素
1.18 给你root1和root2这两棵二叉搜索树。
请你返回一个列表，其中包含两棵树中的所有整数并按升序排序。

示例1:
输入：root1 = [2,1,4], root2 = [1,0,3]
输出：[0,1,1,2,3,4]
*/

// GetAllElements 中序遍历+归并排序
func GetAllElements(root1, root2 *Entity.TreeNode) []int {
	var dfs func(*Entity.TreeNode) []int
	dfs = func(node *Entity.TreeNode) (res []int) {
		if node == nil {
			return res
		}
		res = append(res, dfs(node.Left)...)
		res = append(res, node.Val)
		res = append(res, dfs(node.Right)...)
		return res
	}
	l1, l2 := dfs(root1), dfs(root2)
	if len(l1) == 0 {
		return l2
	}
	if len(l2) == 0 {
		return l1
	}
	var ret []int
	s1, s2 := 0, 0
	for s1 < len(l1) && s2 < len(l2) {
		if l1[s1] < l2[s2] {
			ret = append(ret, l1[s1])
			s1++
		} else {
			ret = append(ret, l2[s2])
			s2++
		}
	}
	ret = append(ret, l1[s1:]...)
	ret = append(ret, l2[s2:]...)
	return ret
}

/*
1008. 前序遍历构造二叉搜索树
1.19 返回与给定前序遍历preorder相匹配的二叉搜索树（binary search tree）的根结点。
题目保证，对于给定的测试用例，总能找到满足要求的二叉搜索树。

示例：
输入：[8,5,1,7,10,12]
输出：[8,5,10,1,7,null,12]
*/

func BstFromPreorder(preorder []int) *Entity.TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	root := &Entity.TreeNode{Val: preorder[0]}
	var left, right []int
	for _, v := range preorder[1:] {
		if v < root.Val {
			left = append(left, v)
		} else {
			right = append(right, v)
		}
	}
	root.Left = BstFromPreorder(left)
	root.Right = BstFromPreorder(right)
	return root
}

/*
1.20 拓展: 后序遍历构造二叉搜索树
*/

func BstFromPostorder(postorder []int) *Entity.TreeNode {
	if len(postorder) == 0 {
		return nil
	}
	root := &Entity.TreeNode{Val: postorder[len(postorder)-1]}
	var left, right []int
	for _, v := range postorder[:len(postorder)-1] {
		if v < root.Val {
			left = append(left, v)
		} else {
			right = append(right, v)
		}
	}
	root.Left = BstFromPreorder(left)
	root.Right = BstFromPreorder(right)
	return root
}

/*
剑指 Offer II 056. 二叉搜索树中两个节点之和
1.21 给定一个二叉搜索树的 根节点 root 和一个整数 k , 请判断该二叉搜索树中是否存在两个节点它们的值之和等于 k 。
假设二叉搜索树中节点的值均唯一。

示例1：
输入: root = [8,6,10,5,7,9,11], k = 12
输出: true
解释: 节点 5 和节点 7 之和等于 12

示例2：
输入: root = [8,6,10,5,7,9,11], k = 22
输出: false
解释: 不存在两个节点值之和为 22 的节点
*/

// FindTarget 先序遍历+哈希表
func FindTarget(root *Entity.TreeNode, k int) bool {
	hashtable := make(map[int]bool)
	var dfs func(*Entity.TreeNode) bool
	dfs = func(node *Entity.TreeNode) bool {
		if node == nil {
			return false
		}
		if hashtable[k-node.Val] {
			return true
		}
		hashtable[node.Val] = true
		return dfs(node.Left) || dfs(node.Right)
	}
	return dfs(root)
}

// findTarget 中序遍历+双指针
func findTarget(root *Entity.TreeNode, k int) bool {
	var sortedArray []int
	var dfs func(*Entity.TreeNode)
	dfs = func(node *Entity.TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		sortedArray = append(sortedArray, node.Val)
		dfs(node.Right)
	}
	dfs(root)
	l, r := 0, len(sortedArray)-1
	for l < r {
		if sortedArray[l]+sortedArray[r] == k {
			return true
		} else if sortedArray[l]+sortedArray[r] > k {
			r--
		} else {
			l++
		}
	}
	return false
}
