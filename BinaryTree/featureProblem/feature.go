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

func LowestCommonAncestorUseRecursion(root, p, q *Entity.TreeNode) *Entity.TreeNode {
	if p.Val < root.Val && q.Val < root.Val {
		return LowestCommonAncestorUseRecursion(root.Left, p, q)
	} else if p.Val > root.Val && q.Val > root.Val {
		return LowestCommonAncestorUseRecursion(root.Right, p, q)
	} else {
		return root
	}
}