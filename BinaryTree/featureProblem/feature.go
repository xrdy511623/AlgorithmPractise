package featureProblem

import (
	"AlgorithmPractise/LinkedList/Entity"
	"AlgorithmPractise/Utils"
)

/*
1.1 反转二叉树
*/

// ReverseBinaryTree 利用递归解决
func ReverseBinaryTree(root *Entity.TreeNode) *Entity.TreeNode {
	if root == nil {
		return root
	}
	root.Left, root.Right = root.Right, root.Left
	ReverseBinaryTree(root.Left)
	ReverseBinaryTree(root.Right)
	return root
}

/*
1.2 二叉树的最大深度
给定一个二叉树，找出其最大深度。
二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
说明: 叶子节点是指没有子节点的节点。
示例：
给定二叉树 [3,9,20,null,null,15,7]，

    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度3 。
*/

/*
利用BFS解决,每遍历一层，maxDepth++
时间复杂度：O(n)，其中n为二叉树的节点个数。每个节点只会被访问一次。
空间复杂度：此方法空间的消耗取决于队列存储的元素数量，其在最坏情况下会达到 O(n)。
*/

func MaxDepth(root *Entity.TreeNode) int {
	var maxDepth int
	if root == nil {
		return maxDepth
	}
	queue := []*Entity.TreeNode{root}
	for len(queue) != 0 {
		levelSize := len(queue)
		for i := 0; i < levelSize; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		maxDepth++
	}
	return maxDepth
}

/*
DFS递归法求解
时间复杂度：O(n)，其中n为二叉树节点的个数。每个节点在递归中只被遍历一次。
空间复杂度：O(height)，其中height表示二叉树的高度。递归函数需要栈空间，而栈空间取决于递归的深度，因此空间复杂度等价于二叉树的高度。
*/

func MaxDepthUseDfs(root *Entity.TreeNode) int {
	if root == nil {
		return 0
	}
	return 1 + Utils.Max(MaxDepthUseDfs(root.Left), MaxDepthUseDfs(root.Right))
}

/*
1.3 二叉搜索树的最近公共祖先
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
1.4 进阶:二叉树的最近公共祖先
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
提示：
树中节点数目在范围 [2, 105] 内。
-109 <= Node.val <= 109
所有 Node.val互不相同 。
p != q
p和q均存在于给定的二叉树中。
*/

// NearestCommonAncestor 递归解决
func NearestCommonAncestor(root, p, q *Entity.TreeNode) *Entity.TreeNode {
	if root == nil || p == root || q == root {
		return root
	}
	left := NearestCommonAncestor(root.Left, p, q)
	right := NearestCommonAncestor(root.Right, p, q)
	if left != nil && right != nil {
		return root
	}
	if left != nil {
		return left
	} else {
		return right
	}
}

/*
利用类似哈希表求交集的思路解决
我们可以用哈希表存储所有节点的父节点.然后对于p节点我们就可以利用节点的父节点信息向上遍历它的祖先节点
(从父亲节点到爷爷节点一直到根节点),并用集合visited记录已经访问过的节点.然后再从q节点也利用其父节点
的信息向上跳,如果集合visited中碰到已经访问过的节点,那么该节点就是我们要找的最近公共祖先
*/

func NearestCommonAncestorUseIteration(root, p, q *Entity.TreeNode) *Entity.TreeNode {
	if root == nil {
		return root
	}
	parentDict := make(map[int]*Entity.TreeNode)
	visited := make(map[int]int)
	var dfs func(node *Entity.TreeNode)
	dfs = func(node *Entity.TreeNode) {
		if node.Left != nil {
			parentDict[node.Left.Val] = node
			dfs(node.Left)
		}
		if node.Right != nil {
			parentDict[node.Right.Val] = node
			dfs(node.Right)
		}
	}
	dfs(root)
	for p != nil {
		visited[p.Val]++
		p = parentDict[p.Val]
	}
	for q != nil {
		if _, ok := visited[q.Val]; ok {
			return q
		}
		q = parentDict[q.Val]
	}
	return nil
}

/*
1.5 计算二叉树的直径
给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的
最大值。这条路径可能穿过也可能不穿过根结点。
注意：两结点之间的路径长度是以它们之间边的数目表示。
*/

/*
思路:dfs深度优先遍历+迭代法解决
最大直径maxDia初始值为0
计算以指定节点为根的子树(二叉树)的最大直径maxDia,其实就是该二叉树左子树的
最大高度+右子树的最大高度, 因此我们对二叉树从根节点root开始进行深度优先遍历，每遍历一个节点，
迭代maxDia的值(将maxDia与以该节点为根的二叉树的最大直径即lh+rh之和比较，取较大值)，遍历结束后
的maxDia即为该二叉树的最大直径。dfs深度优先遍历时返回以该节点为根的二叉树的最大高度，也就是
1+max(lh+rh),时间复杂度度O(n),空间复杂度O(h),n为该二叉树节点个数，h为该二叉树高度
*/

func DiameterOfBinaryTree(root *Entity.TreeNode) int {
	maxDia := 0
	var dfs func(node *Entity.TreeNode) int
	dfs = func(node *Entity.TreeNode) int {
		if node == nil {
			return 0
		}
		lh := dfs(node.Left)
		rh := dfs(node.Right)
		maxDia = Utils.Max(maxDia, lh+rh)
		return 1 + Utils.Max(lh, rh)
	}
	dfs(root)
	return maxDia
}

/*
1.6 中序后继结点
设计一个算法，找出二叉搜索树中指定节点的“下一个”节点（也即中序后继）。
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
		var res []*Entity.TreeNode
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

// InorderSuccessorUseIteration 迭代法解决，时间复杂度降低为O(pos),空间复杂度降低为O(1)
func InorderSuccessorUseIteration(root, p *Entity.TreeNode) *Entity.TreeNode {
	curr := root
	var ans *Entity.TreeNode
	for curr != nil {
		// 当后继存在于经过的节点中时（找到一个>val的最小点）
		if curr.Val > p.Val {
			if ans == nil || ans.Val > p.Val {
				ans = curr
			}
		}
		// 找到p节点
		if curr.Val == p.Val {
			// 如果有右子树,说明后继节点一定在右子树的最左边
			if curr.Right != nil {
				curr = curr.Right
				for curr.Left != nil {
					curr = curr.Left
				}
				return curr
			}
			break
		}
		if curr.Val > p.Val {
			curr = curr.Left
		} else {
			curr = curr.Right
		}
	}
	return ans
}

/*
1.7 验证二叉搜索树的后序遍历序列
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回true，否则返回false。假设输入的数组的
任意两个数字都互不相同。
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
	var recur func(array []int, start, stop int) bool
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
1.8 二叉树的右视图
给定一个二叉树的根节点root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
*/

// RightSideView BFS(广度优先遍历解决)，时间复杂度O(N),空间复杂度O(N)
func RightSideView(root *Entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	queue := []*Entity.TreeNode{root}
	for len(queue) != 0 {
		var temp []*Entity.TreeNode
		for _, node := range queue {
			if node.Left != nil {
				temp = append(temp, node.Left)
			}
			if node.Right != nil {
				temp = append(temp, node.Right)
			}
		}
		res = append(res, queue[len(queue)-1].Val)
		queue = temp
	}
	return res
}