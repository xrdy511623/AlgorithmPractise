package featureProblem

import (
	"AlgorithmPractise/BinaryTree/pathSumProblem"
	"AlgorithmPractise/LinkedList/Entity"
)

/*
1.1 反转二叉树
*/

// 利用递归解决
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
	return 1 + pathSumProblem.Max(MaxDepthUseDfs(root.Left), MaxDepthUseDfs(root.Right))
}

/*
1.3 二叉搜索树的最近公共祖先
*/

// 迭代法解决
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

// 递归法解决
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

// 递归解决
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
		p, _ = parentDict[p.Val]
	}
	for q != nil {
		if _, ok := visited[q.Val]; ok {
			return q
		}
		q, _ = parentDict[q.Val]
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
	var dfs func(node *Entity.TreeNode)int
	dfs = func(node *Entity.TreeNode)int{
		if node == nil{
			return 0
		}
		lh := dfs(node.Left)
		rh := dfs(node.Right)
		maxDia = pathSumProblem.Max(maxDia, lh+rh)
		return 1 + pathSumProblem.Max(lh, rh)
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
	var travel func(node *Entity.TreeNode)[]*Entity.TreeNode
	travel = func(node *Entity.TreeNode)[]*Entity.TreeNode{
		var res []*Entity.TreeNode
		if node == nil{
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
	for index, node := range nodesList{
		if node == p {
			pos = index
			break
		}
	}
	if len(nodesList) - 1 > pos {
		return nodesList[pos+1]
	} else {
		return nil
	}
}

// 迭代法解决，时间复杂度降低为O(pos),空间复杂度降低为O(1)
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