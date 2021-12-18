package pathSumProblem

import (
	"AlgorithmPractise/BinaryTree/Entity"
	"AlgorithmPractise/Utils"
	"math"
	"strconv"
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
func PathSum(root *Entity.TreeNode, target int) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	var queue []Group
	queue = append(queue, Group{root, []int{root.Val}})
	for len(queue) != 0 {
		node := queue[0].Node
		tempPath := queue[0].Path
		queue = queue[1:]
		if node.Left == nil && node.Right == nil && sumOfArray(tempPath) == target {
			res = append(res, tempPath)
		}
		copyTemp := copySlice(tempPath)
		if node.Left != nil {
			temp1 := copyTemp
			temp1 = append(temp1, node.Left.Val)
			queue = append(queue, Group{node.Left, temp1})
		}

		if node.Right != nil {
			temp2 := copyTemp
			temp2 = append(temp2, node.Right.Val)
			queue = append(queue, Group{node.Right, temp2})
		}

	}
	return res
}

// PathSumUseDfs DFS递归也能解决
func PathSumUseDfs(root *Entity.TreeNode, target int) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	var dfs func(root *Entity.TreeNode, path []int, target int)
	dfs = func(node *Entity.TreeNode, path []int, target int) {
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

type Group struct {
	Node *Entity.TreeNode
	Path []int
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

func NumberOfPathSum(root *Entity.TreeNode, target int) int {
	if root == nil {
		return 0
	}
	res := 0
	queue := []Group{{root, []int{root.Val}}}
	for len(queue) != 0 {
		node := queue[0].Node
		temp := queue[0].Path
		res += CountTarget(temp, target)
		queue = queue[1:]
		// 增加0是因为当前节点本身的值也有可能等于目标值target,此时路径就是节点本身
		temp = append(temp, 0)
		if node.Left != nil {
			var temp1 []int
			for _, v := range temp {
				temp1 = append(temp1, node.Left.Val+v)
			}
			queue = append(queue, Group{node.Left, temp1})
		}
		if node.Right != nil {
			var temp2 []int
			for _, v := range temp {
				temp2 = append(temp2, node.Right.Val+v)
			}
			queue = append(queue, Group{node.Right, temp2})
		}
	}
	return res
}

func CountTarget(s []int, target int) int {
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

func FindFrequentTreeSum(root *Entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}

	treeSum := make(map[int]int)
	var subTreeSum func(node *Entity.TreeNode) int
	subTreeSum = func(node *Entity.TreeNode) int {
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

func MaxPathSum(root *Entity.TreeNode) int {
	maxSum := math.MinInt32
	var dfs func(node *Entity.TreeNode) int
	dfs = func(node *Entity.TreeNode) int {
		if node == nil {
			return 0
		}
		leftPath := dfs(node.Left)
		rightPath := dfs(node.Right)
		// 更新maxSum的值，左子树最大收益+右子树最大收益+节点本身的值
		maxSum = Utils.Max(maxSum, Utils.Max(leftPath, 0)+Utils.Max(rightPath, 0)+node.Val)
		if leftPath > rightPath {
			return Utils.Max(0, leftPath) + node.Val
		} else {
			return Utils.Max(0, rightPath) + node.Val
		}
	}
	dfs(root)
	return maxSum
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
func sumOfLeftLeaves(root *Entity.TreeNode) int {
	var res int
	if root == nil {
		return res
	}
	queue := []*Entity.TreeNode{root}
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

/*
1.6 求根节点到叶节点数字之和
给你一个二叉树的根节点root ，树中每个节点都存放有一个0到9之间的数字。
每条从根节点到叶节点的路径都代表一个数字：

例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123 。
计算从根节点到叶节点生成的所有数字之和 。

叶节点是指没有子节点的节点。

输入：root = [1,2,3]
输出：25
解释：
从根到叶子节点路径 1->2 代表数字 12
从根到叶子节点路径 1->3 代表数字 13
因此，数字总和 = 12 + 13 = 25

输入：root = [4,9,0,5,1]
输出：1026
解释：
从根到叶子节点路径 4->9->5 代表数字495
从根到叶子节点路径 4->9->1 代表数字491
从根到叶子节点路径 4->0 代表数字40
因此，数字总和 = 495 + 491 + 40 = 1026
*/

type LogicNode struct {
	Node *Entity.TreeNode
	Val  string
}

// SumNumbers BFS解决, 时间复杂度O(N),空间复杂度O(H),H为二叉树的高度
func SumNumbers(root *Entity.TreeNode) int {
	res := 0
	if root == nil {
		return res
	}
	stack := []LogicNode{{root, strconv.Itoa(root.Val)}}
	for len(stack) != 0 {
		logicNode := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		sum := logicNode.Val
		if logicNode.Node.Left == nil && logicNode.Node.Right == nil {
			pathSum, _ := strconv.Atoi(sum)
			res += pathSum
		}
		if logicNode.Node.Right != nil {
			stack = append(stack, LogicNode{logicNode.Node.Right, sum + strconv.Itoa(logicNode.Node.Right.Val)})
		}
		if logicNode.Node.Left != nil {
			stack = append(stack, LogicNode{logicNode.Node.Left, sum + strconv.Itoa(logicNode.Node.Left.Val)})
		}
	}
	return res
}

/*
1.7 二叉树的所有路径
给你一个二叉树的根节点root，按任意顺序 ，返回所有从根节点到叶子节点的路径。
叶子节点是指没有子节点的节点。
示例1：
输入：root = [1,2,3,null,5]
输出：["1->2->5","1->3"]
*/


type NodePath struct{
	Node  *Entity.TreeNode
	Path  string
}

// BinaryTreePaths BFS解决
func BinaryTreePaths(root *Entity.TreeNode) []string {
	var res []string
	if root == nil{
		return res
	}
	stack := []NodePath{NodePath{root, strconv.Itoa(root.Val)}}
	for len(stack) != 0{
		np := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if np.Node.Left == nil && np.Node.Right == nil{
			res = append(res, np.Path)
		}
		if np.Node.Right != nil{
			path := np.Path + "->" + strconv.Itoa(np.Node.Right.Val)
			stack = append(stack, NodePath{np.Node.Right, path})
		}
		if np.Node.Left != nil{
			path := np.Path + "->" + strconv.Itoa(np.Node.Left.Val)
			stack = append(stack, NodePath{np.Node.Left, path})
		}

	}
	return res
}


// DFS解决

func BinaryTreePathsUseDFS(root *Entity.TreeNode) []string {
	var paths []string
	var dfs func(*Entity.TreeNode, string)
	dfs = func(root *Entity.TreeNode, path string){
		if root != nil{
			curPath := path
			curPath += strconv.Itoa(root.Val)
			if root.Left == nil && root.Right == nil{
				paths = append(paths, curPath)
			} else{
				curPath += "->"
				dfs(root.Left, curPath)
				dfs(root.Right, curPath)
			}
		}
	}
	dfs(root, "")
	return paths
}