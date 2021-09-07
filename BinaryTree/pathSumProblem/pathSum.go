package pathSumProblem

import (
	"AlgorithmPractise/BinaryTree/Entitly"
	"math"
)

/*
路径和问题
*/

/*
1.1 路径和:给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和
的路径。(返回列表)
说明: 叶子节点是指没有子节点的节点。
*/

// PathSum BFS解决，简单易懂效率高
func PathSum(root *Entitly.TreeNode, target int) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	var queue []*group
	queue = append(queue, &group{root, []int{root.Val}})
	for len(queue) != 0 {
		node := queue[0].Node
		temp := queue[0].Val
		queue = queue[1:]
		if node.Left == nil && node.Right == nil && sumOfArray(temp) == target {
			res = append(res, temp)
		}
		copyTemp := copySlice(temp)
		if node.Left != nil {
			temp1 := copyTemp
			temp1 = append(temp1, node.Left.Val)
			queue = append(queue, &group{node.Left, temp1})
		}

		if node.Right != nil {
			temp2 := copyTemp
			temp2 = append(temp2, node.Right.Val)
			queue = append(queue, &group{node.Right, temp2})
		}

	}
	return res
}

// PathSumUseDfs, DFS递归也能解决
func PathSumUseDfs(root *Entitly.TreeNode, target int) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	var dfs func(root *Entitly.TreeNode, path []int, target int)
	dfs = func(node *Entitly.TreeNode, path []int, target int) {
		if node.Left == nil && node.Right == nil && sumOfArray(path) == target {
			res = append(res, path)
		}
		temp := copySlice(path)
		if node.Left != nil {
			temp1 := append(temp, node.Left.Val)
			dfs(node.Left, temp1, target)
		}
		if node.Right != nil {
			temp2 := append(temp, node.Right.Val)
			dfs(node.Right, temp2, target)
		}
	}
	path := []int{root.Val}
	dfs(root, path, target)
	return res
}

type group struct {
	Node *Entitly.TreeNode
	Val  []int
}

func sumOfArray(array []int) int {
	sum := 0
	for _, v := range array {
		sum += v
	}

	return sum
}

func copySlice(src []int) []int {
	dst := make([]int, 0, len(src))
	for _, v := range src {
		dst = append(dst, v)
	}

	return dst
}

/*
1.2 路径和等于目标值的条数
给定一个二叉树，它的每个结点都存放着一个整数值。
找出路径和等于给定数值的路径总数。
路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点
到子节点）。
二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。
示例：
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

          10
         /  \
        5   -3
       / \    \
      3   2   11
     / \   \
    3  -2   1

    返回3, 因为和等于8的路径有3条:

    1.  5 -> 3
    2.  5 -> 2 -> 1
    3.  -3 -> 11
*/

func NumberOfPathSum(root *Entitly.TreeNode, target int) int {
	if root == nil {
		return 0
	}
	res := 0
	queue := []group{group{root, []int{root.Val}}}
	for len(queue) != 0 {
		node := queue[0].Node
		temp := queue[0].Val
		res += countTarget(temp, target)
		queue = queue[1:]
		// 增加0是因为当前节点本身的值也有可能等于目标值target,此时路径就是节点本身
		temp = append(temp, 0)
		if node.Left != nil {
			var temp1 []int
			for _, v := range temp {
				temp1 = append(temp1, node.Left.Val+v)
			}
			queue = append(queue, group{node.Left, temp1})
		}
		if node.Right != nil {
			var temp2 []int
			for _, v := range temp {
				temp2 = append(temp2, node.Right.Val+v)
			}
			queue = append(queue, group{node.Right, temp2})
		}
	}
	return res
}

func countTarget(s []int, target int) int {
	count := 0
	for _, v := range s {
		if v == target {
			count++
		}
	}
	return count
}

/*
1.3 出现次数最多的子树元素和
给你一个二叉树的根结点，请你找出出现次数最多的子树元素和。一个结点的子树元素和定义
为以该结点为根的二叉树上所有结点的元素之和（包括结点本身）.
你需要返回出现次数最多的子树元素和。如果有多个元素出现的次数相同，返回所有出现次数最多
的子树元素和（不限顺序）。

示例 1：
输入:
  5
 /  \
2   -3
返回[2, -3, 4]，所有的值均只出现一次，以任意顺序返回所有值。

示例2：
输入：
  5
 /  \
2   -5
返回[2]，只有2出现两次，-5只出现1次。
*/

func FindFrequentTreeSum(root *Entitly.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}

	treeSum := make(map[int]int)
	var subTreeSum func(node *Entitly.TreeNode) int
	subTreeSum = func(node *Entitly.TreeNode) int {
		if node == nil {
			return 0
		}
		left := subTreeSum(node.Left)
		right := subTreeSum(node.Right)
		sum := left + right + node.Val
		treeSum[sum]++
		return sum
	}
	subTreeSum(root)
	mostFrequent := 0
	for _, v := range treeSum {
		if v > mostFrequent {
			mostFrequent = v
		}
	}
	for k, v := range treeSum {
		if v == mostFrequent {
			res = append(res, k)
		}
	}
	return res
}

/*
1.4 最大路径和
给定一个非空二叉树，返回其最大路径和。
本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径至少包含一个节点，
且不一定经过根节点。
示例 1:

输入: [1,2,3]

	   1
	  / \
	 2   3

输出: 6

示例2:
输入: [-10,9,20,null,null,15,7]

		  -10
		  / \
		 9  20
		   / \
		  15  7

输出: 42
*/

/*
思路:DFS递归解决
路径每到一个节点,有3种选择：1.停下不走 2.走到左子节点 3.走到右子节点
走到子节点,又有3种选择：1.停下不走 2.走到左子节点 3.走到右子节点
不能走进一个分支,又掉头走另一个分支,不符合要求.
怎么定义递归函数？
我们关心：如果路径走入一个子树,能从中捞取的最大收益,不关心具体路径.
这就是一种属于递归的,自顶而下的思考方式。
定义dfs函数：返回当前子树能向父节点“提供”的最大路径和。即一条从父节点延伸下来的路径,
能在当前子树中获得的最大收益。它分为三种情况,取其中最大的：
停在当前子树的root,最大收益：root.val.
走入左子树,最大收益：root.val + dfs(root.left)
走入右子树,最大收益：root.val + dfs(root.right)
当遍历到null节点时，返回0，代表此处收益为0。
*/

func maxPathSum(root *Entitly.TreeNode) int {
	maxSum := math.MinInt32
	var dfs func(node *Entitly.TreeNode) int
	dfs = func(node *Entitly.TreeNode) int {
		if node == nil {
			return 0
		}
		leftPath := dfs(node.Left)
		rightPath := dfs(node.Right)
		// 更新maxSum的值，左子树最大收益+右子树最大收益+节点本身的值
		maxSum = max(maxSum, max(leftPath, 0)+max(rightPath, 0)+node.Val)
		if leftPath > rightPath {
			return max(0, leftPath) + node.Val
		} else {
			return max(0, rightPath) + node.Val
		}
	}
	dfs(root)
	return maxSum
}

func max(a, b int) int {
	if a > b {
		return a
	} else {
		return b
	}
}

/*
1.5 左叶子之和
计算给定二叉树的所有左叶子之和。
示例：
    3
   / \
  9  20
    /  \
   15   7

在这个二叉树中，有两个左叶子，分别是9和15，所以返回24
*/

// sumOfLeftLeaves easy
func sumOfLeftLeaves(root *Entitly.TreeNode) int {
	var res int
	if root == nil {
		return res
	}
	queue := []*Entitly.TreeNode{root}
	for len(queue) != 0 {
		node := queue[0]
		queue = queue[1:]
		if node.Left != nil && node.Left.Left == nil && node.Left.Right == nil {
			res += node.Left.Val
		}
		if node.Left != nil {
			queue = append(queue, node.Left)
		}
		if node.Right != nil {
			queue = append(queue, node.Right)
		}
	}
	return res
}