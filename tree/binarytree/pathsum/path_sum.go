package pathsum

import (
	"algorithm-practise/tree/binarytree/entity"
	"algorithm-practise/utils"
	"math"
	"strconv"
)

/*
路径和问题
*/

/*
leetcode 112. 路径总和
1.0 给你二叉树的根节点root和一个表示目标和的整数targetSum。判断该树中是否存在根节点到叶子节点的路径，
使得这条路径上所有节点值相加等于目标和targetSum 。如果存在，返回true ；否则，返回false 。
叶子节点是指没有子节点的节点。

示例:
输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
输出：true
*/

/*
思路一:BFS
*/

func HasPathSum(root *entity.TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}
	queue := []*entity.TreeNode{root}
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		if node.Left == nil && node.Right == nil && node.Val == targetSum {
			return true
		}
		if node.Left != nil {
			node.Left.Val += node.Val
			queue = append(queue, node.Left)
		}
		if node.Right != nil {
			node.Right.Val += node.Val
			queue = append(queue, node.Right)
		}
	}
	return false
}

/*
思路二:DFS
*/

func HasPathSumSimple(root *entity.TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}
	// 递归终止条件，当前节点为叶子节点且该叶子节点路径和等于目标路径和targetSum
	if root.Left == nil && root.Right == nil && root.Val == targetSum {
		return true
	}
	// 只要左子树或右子树上有一个能返回true就行
	return HasPathSumSimple(root.Left, targetSum-root.Val) || HasPathSumSimple(root.Right, targetSum-root.Val)
}

/*
剑指Offer 34. 二叉树中和为某一值的路径
113. 路径总和II
1.1 给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和
的路径。(返回列表)
说明: 叶子节点是指没有子节点的节点。
*/

// PathSum BFS解决，简单易懂效率高
func PathSum(root *entity.TreeNode, target int) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	queue := []Group{{root, []int{root.Val}}}
	for len(queue) != 0 {
		node := queue[0].Node
		path := queue[0].Path
		queue = queue[1:]
		if node.Left == nil && node.Right == nil && sumOfArray(path) == target {
			res = append(res, path)
		}
		if node.Left != nil {
			leftPath := make([]int, len(path))
			copy(leftPath, path)
			leftPath = append(leftPath, node.Left.Val)
			queue = append(queue, Group{node.Left, leftPath})
		}
		if node.Right != nil {
			rightPath := make([]int, len(path))
			copy(rightPath, path)
			rightPath = append(rightPath, node.Right.Val)
			queue = append(queue, Group{node.Right, rightPath})
		}

	}
	return res
}

// PathSumUseDfs DFS递归也能解决
func PathSumUseDfs(root *entity.TreeNode, target int) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	var dfs func(*entity.TreeNode, []int)
	dfs = func(node *entity.TreeNode, path []int) {
		if node.Left == nil && node.Right == nil && sumOfArray(path) == target {
			res = append(res, path)
		}
		if node.Left != nil {
			leftPath := make([]int, len(path))
			copy(leftPath, path)
			leftPath = append(leftPath, node.Left.Val)
			dfs(node.Left, leftPath)
		}
		if node.Right != nil {
			rightPath := make([]int, len(path))
			copy(rightPath, path)
			rightPath = append(rightPath, node.Right.Val)
			dfs(node.Right, rightPath)
		}
	}
	dfs(root, []int{root.Val})
	return res
}

type Group struct {
	Node *entity.TreeNode
	Path []int
}

func sumOfArray(array []int) int {
	sum := 0
	for _, v := range array {
		sum += v
	}

	return sum
}

/*
leetcode 437 路径总和III
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

func NumberOfPathSum(root *entity.TreeNode, target int) int {
	if root == nil {
		return 0
	}
	res := 0
	queue := []Group{{root, []int{root.Val}}}
	for len(queue) != 0 {
		node := queue[0].Node
		path := queue[0].Path
		res += CountTarget(path, target)
		queue = queue[1:]
		// 增加0是因为当前节点本身的值也有可能等于目标值target,此时路径就是节点本身
		path = append(path, 0)
		temp := make([]int, len(path))
		copy(temp, path)
		if node.Left != nil {
			var leftPath []int
			for _, v := range temp {
				leftPath = append(leftPath, node.Left.Val+v)
			}
			queue = append(queue, Group{node.Left, leftPath})
		}
		if node.Right != nil {
			var rightPath []int
			for _, v := range temp {
				rightPath = append(rightPath, node.Right.Val+v)
			}
			queue = append(queue, Group{node.Right, rightPath})
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
哈希表+前缀和
*/
func pathSum(root *entity.TreeNode, targetSum int) int {
	preMap := make(map[int]int)
	// 处理包含根节点的情况，如果root.Val==targetSum,统计结果不能忽略它
	preMap[0] = 1
	count := 0
	var dfs func(*entity.TreeNode, int)
	dfs = func(node *entity.TreeNode, preSum int) {
		if node == nil {
			return
		}
		preSum += node.Val
		if v, ok := preMap[preSum-targetSum]; ok {
			count += v
		}
		preMap[preSum]++
		// 递归
		dfs(node.Left, preSum)
		dfs(node.Right, preSum)
		// 回溯
		preMap[preSum]--
		return
	}
	dfs(root, 0)
	return count
}

/*
leetcode 508. 出现次数最多的子树元素和
1.3 给你一个二叉树的根结点，请你找出出现次数最多的子树元素和。一个结点的子树元素和定义
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

/*
DFS递归解决
1 明确递归函数的参数和返回值
参数为二叉树根节点指针，返回值为以该节点为根节点的二叉树所有节点值之和
2 确定递归终止条件
当遇到空节点时，返回0
3 确定单层递归逻辑
以当前节点为根节点的二叉树所有节点值之和即为其左子树所有节点之和+其右子树所有节点之和+当前节点值

在递归函数中统计子树元素和sum的出现频次，在递归结束后即得到所有子树元素和的出现频次。
最后将出现频次最多的子树元素和添加到结果集合中即可。
*/

// FindFrequentTreeSum 先序遍历+DFS
func FindFrequentTreeSum(root *entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}

	treeSum := make(map[int]int)
	var subTreeSum func(*entity.TreeNode) int
	subTreeSum = func(node *entity.TreeNode) int {
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
	mostFrequent := 1
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
leetcode 124. 二叉树中的最大路径和
1.4 给定一个非空二叉树，返回其最大路径和。
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

/*
1 明确递归函数的参数和返回值
参数为二叉树节点指针，返回值为如果从该节点走入一个子树,能从中捞取的最大收益
2 确定递归终止条件
当遇到空节点时，返回0，表明获得的最大收益为0
3 确定单层递归逻辑
获取当前节点左子树的最大收益与0相比取大值lp，获取当前节点右子树的最大收益与0相比取大值rp
迭代最大路径和pathSum=Max(pathSum, lp+rp+node.Val)
*/

func MaxPathSum(root *entity.TreeNode) int {
	if root == nil {
		return 0
	}
	maxSum := math.MinInt32
	var dfs func(*entity.TreeNode) int
	dfs = func(node *entity.TreeNode) int {
		if node == nil {
			return 0
		}
		leftPath := utils.Max(dfs(node.Left), 0)
		rightPath := utils.Max(dfs(node.Right), 0)
		// 更新maxSum的值，左子树最大收益+右子树最大收益+节点本身的值
		maxSum = utils.Max(maxSum, leftPath+rightPath+node.Val)
		return utils.Max(leftPath, rightPath) + node.Val
	}
	dfs(root)
	return maxSum
}

/*
leetcode 404. 左叶子之和
1.5 计算给定二叉树的所有左叶子之和。

示例：
    3
   / \
  9  20
    /  \
   15   7

在这个二叉树中，有两个左叶子，分别是9和15，所以返回24
*/

// SumOfLeftLeaves BFS
func SumOfLeftLeaves(root *entity.TreeNode) int {
	sum := 0
	if root == nil {
		return sum
	}
	queue := []*entity.TreeNode{root}
	for len(queue) != 0 {
		node := queue[0]
		queue = queue[1:]
		if node.Left != nil && node.Left.Left == nil && node.Left.Right == nil {
			sum += node.Left.Val
		}
		if node.Left != nil {
			queue = append(queue, node.Left)
		}
		if node.Right != nil {
			queue = append(queue, node.Right)
		}
	}
	return sum
}

// SumOfLeftLeavesSimple DFS
func SumOfLeftLeavesSimple(root *entity.TreeNode) int {
	sum := 0
	var findLeftLeaves func(*entity.TreeNode)
	findLeftLeaves = func(node *entity.TreeNode) {
		if node == nil {
			return
		}
		if node.Left != nil && node.Left.Left == nil && node.Left.Right == nil {
			sum += node.Left.Val
		}
		if node.Left != nil {
			findLeftLeaves(node.Left)
		}
		if node.Right != nil {
			findLeftLeaves(node.Right)
		}
		return
	}
	findLeftLeaves(root)
	return sum
}

/*
leetcode 129. 求根节点到叶节点数字之和
1.6 给你一个二叉树的根节点root ，树中每个节点都存放有一个0到9之间的数字。
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

// SumNumbers BFS解决, 时间复杂度O(N),空间复杂度O(H),H为二叉树的高度
func SumNumbers(root *entity.TreeNode) int {
	sum := 0
	if root == nil {
		return sum
	}
	queue := []*entity.TreeNode{root}
	for len(queue) != 0 {
		node := queue[0]
		queue = queue[1:]
		if node.Left == nil && node.Right == nil {
			sum += node.Val
		}
		if node.Left != nil {
			node.Left.Val += node.Val * 10
			queue = append(queue, node.Left)
		}
		if node.Right != nil {
			node.Right.Val += node.Val * 10
			queue = append(queue, node.Right)
		}
	}
	return sum
}

/*
递归三部曲:
1 确定递归函数的参数和返回值
参数为当前二叉树根节点的指针以及不含当前节点的路径和pathSum，返回值为当前二叉树所有叶子节点路径和
2 明确递归终止条件
遇到空节点，返回0；遇到叶子节点，返回该叶子节点路径和。
3 确定单层递归逻辑
当前节点路径和sum为不含当前节点的路径和pathSum*10+node.Val,如果当前节点为叶子节点，返回当前节点路径和sum
否则到其左子树和右子树寻找叶子节点，将左子树和右子树所有叶子节点路径和累加返回。
*/

// SumNumbersSimple DFS递归解决
func SumNumbersSimple(root *entity.TreeNode) int {
	var dfs func(*entity.TreeNode, int) int
	dfs = func(node *entity.TreeNode, pathSum int) int {
		if node == nil {
			return 0
		}
		sum := node.Val + pathSum*10
		if node.Left == nil && node.Right == nil {
			return sum
		}
		return dfs(node.Left, sum) + dfs(node.Right, sum)
	}
	return dfs(root, 0)
}

/*
leetcode 257
1.7 二叉树的所有路径
给你一个二叉树的根节点root，按任意顺序，返回所有从根节点到叶子节点的路径。
叶子节点是指没有子节点的节点。
示例1：
输入：root = [1,2,3,null,5]
输出：["1->2->5","1->3"]
*/

type NodePath struct {
	Node *entity.TreeNode
	Path string
}

// BinaryTreePaths BFS解决
func BinaryTreePaths(root *entity.TreeNode) []string {
	var res []string
	if root == nil {
		return res
	}
	queue := []NodePath{NodePath{root, strconv.Itoa(root.Val)}}
	for len(queue) != 0 {
		np := queue[0]
		queue = queue[1:]
		if np.Node.Left == nil && np.Node.Right == nil {
			res = append(res, np.Path)
		}
		if np.Node.Left != nil {
			path := np.Path + "->" + strconv.Itoa(np.Node.Left.Val)
			queue = append(queue, NodePath{np.Node.Left, path})
		}
		if np.Node.Right != nil {
			path := np.Path + "->" + strconv.Itoa(np.Node.Right.Val)
			queue = append(queue, NodePath{np.Node.Right, path})
		}

	}
	return res
}

// BinaryTreePathsUseDFS DFS+先序遍历解决
func BinaryTreePathsUseDFS(root *entity.TreeNode) []string {
	var paths []string
	var dfs func(*entity.TreeNode, string)
	dfs = func(root *entity.TreeNode, path string) {
		if root != nil {
			path += strconv.Itoa(root.Val)
			if root.Left == nil && root.Right == nil {
				paths = append(paths, path)
			} else {
				path += "->"
				dfs(root.Left, path)
				dfs(root.Right, path)
			}
		}
	}
	dfs(root, "")
	return paths
}

/*
leetcode 1315. 祖父节点值为偶数的节点和
1.8 给你一棵二叉树，请你返回满足以下条件的所有节点的值之和：
该节点的祖父节点的值为偶数。（一个节点的祖父节点是指该节点的父节点的父节点。）
如果不存在祖父节点值为偶数的节点，那么返回0 。
*/

/*
思路:先判断当前节点的值是否为偶数，然后依次对孙子节点的值进行累加
*/

// SumEvenGrandparent DFS
func SumEvenGrandparent(root *entity.TreeNode) int {
	if root == nil {
		return 0
	}
	sum := 0
	if root.Val%2 == 0 {
		if root.Left != nil {
			if root.Left.Left != nil {
				sum += root.Left.Left.Val
			}
			if root.Left.Right != nil {
				sum += root.Left.Right.Val
			}
		}
		if root.Right != nil {
			if root.Right.Left != nil {
				sum += root.Right.Left.Val
			}
			if root.Right.Right != nil {
				sum += root.Right.Right.Val
			}
		}
	}
	return sum + SumEvenGrandparent(root.Left) + SumEvenGrandparent(root.Right)
}

// SumEvenGrandparentSimple BFS
func SumEvenGrandparentSimple(root *entity.TreeNode) int {
	if root == nil {
		return 0
	}
	sum := 0
	queue := []*entity.TreeNode{root}
	for len(queue) != 0 {
		node := queue[0]
		queue = queue[1:]
		if node.Val%2 == 0 {
			if node.Left != nil {
				if node.Left.Left != nil {
					sum += node.Left.Left.Val
				}
				if node.Left.Right != nil {
					sum += node.Left.Right.Val
				}
			}
			if node.Right != nil {
				if node.Right.Left != nil {
					sum += node.Right.Left.Val
				}
				if node.Right.Right != nil {
					sum += node.Right.Right.Val
				}
			}
		}
		if node.Left != nil {
			queue = append(queue, node.Left)
		}
		if node.Right != nil {
			queue = append(queue, node.Right)
		}
	}
	return sum
}

/*
leetcode 1026. 节点与其祖先之间的最大差值
1.9 给定二叉树的根节点root，找出存在于不同节点A和B之间的最大值V，其中V = |A.val - B.val|，
且A是B的祖先。
(如果 A 的任何子节点之一为 B，或者 A 的任何子节点是 B 的祖先，那么我们认为 A 是 B 的祖先)

示例1:
输入：root = [8,3,10,1,6,null,14,null,null,4,7,13]
输出：7
解释：
我们有大量的节点与其祖先的差值，其中一些如下：
|8 - 3| = 5
|3 - 7| = 4
|8 - 1| = 7
|10 - 13| = 3
在所有可能的差值中，最大值 7 由 |8 - 1| = 7 得出。
*/

// MaxAncestorDiff DFS递归解决
func MaxAncestorDiff(root *entity.TreeNode) int {
	if root == nil {
		return 0
	}
	var dfs func(*entity.TreeNode, int, int) int
	dfs = func(node *entity.TreeNode, min, max int) int {
		if node == nil {
			return max - min
		}
		if node.Val < min {
			min = node.Val
		}
		if node.Val > max {
			max = node.Val
		}
		leftMax := dfs(node.Left, min, max)
		rightMax := dfs(node.Right, min, max)
		return utils.Max(leftMax, rightMax)
	}
	return dfs(root, root.Val, root.Val)
}

type MaxDiff struct {
	Node *entity.TreeNode
	// Diff中存储两个数，分别是当前节点与其所有祖先节点值中的最大值和最小值
	Diff []int
}

// MaxAncestorDiffSimple BFS
func MaxAncestorDiffSimple(root *entity.TreeNode) int {
	maxVal := 0
	if root == nil {
		return maxVal
	}
	queue := []MaxDiff{MaxDiff{root, []int{root.Val, root.Val}}}
	for len(queue) != 0 {
		node := queue[0].Node
		diff := queue[0].Diff
		max, min := diff[0], diff[1]
		queue = queue[1:]
		// 迭代最大差值maxVal
		maxVal = utils.Max(maxVal, max-min)
		if node.Left != nil {
			// 迭代当前节点左子节点Diff中的最大值和最小值
			// 就是将左子节点的值与其父亲节点，也就是当前节点的值进行比较
			leftMax, leftMin := max, min
			if leftMax < node.Left.Val {
				leftMax = node.Left.Val
			}
			if leftMin > node.Left.Val {
				leftMin = node.Left.Val
			}
			leftDiff := []int{leftMax, leftMin}
			queue = append(queue, MaxDiff{node.Left, leftDiff})
		}
		if node.Right != nil {
			// 迭代当前节点右子节点Diff中的最大值和最小值
			// 就是将右子节点的值与其父亲节点，也就是当前节点的值进行比较
			rightMax, rightMin := max, min
			if rightMax < node.Right.Val {
				rightMax = node.Right.Val
			}
			if rightMin > node.Right.Val {
				rightMin = node.Right.Val
			}
			rightDiff := []int{rightMax, rightMin}
			queue = append(queue, MaxDiff{node.Right, rightDiff})
		}
	}
	return maxVal
}

/*
leetcode 863
1.10 二叉树中所有距离为K的节点
给定一个二叉树（具有根节点root），一个目标节点target，和一个整数值K 。
返回到目标结点target距离为K的所有结点的值的列表。 答案可以以任何顺序返回。

示例1：
输入：root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, K = 2
输出：[7,4,1]
解释：
所求结点为与目标结点（值为5）距离为2的节点，值分别为7，4，以及1

		   3
         /  \
        5    1
       / \  / \
      6  2  0  8
        / \
       7  4

提示：
给定的树是非空的。
树上的每个结点都具有唯一的值0 <= node.val <= 500。
目标结点target是树上的结点。
0 <= K <= 1000.
*/

/*
思路:如果能知道节点的父节点,那么就可以知道所有与该节点距离为1的节点
(也就是该节点的左右子节点以及父节点三个节点).之后就可以从target开始做广度优先搜索
算法:先遍历一遍二叉树，记录除根节点外每个节点的父节点，之后做广度优先搜索,找到所有距离target
节点K距离的节点。

特别需要注意去重:假设节点p与target距离为n,那么其父节点par与target距离为n+1,那么par的父节点和
左右子节点此时与target距离应为n+2,但是此时p就是par的子节点，它的距离不应该被更新为n+2,应该还是n
它也不应被添加到队列中。
*/

// Element 结构体Element由两部分组成，Node记录二叉树当前节点指针，Distance表示当前节点与target的距离
type Element struct {
	Node     *entity.TreeNode
	Distance int
}

func DistanceK(root, target *entity.TreeNode, k int) []int {
	var res []int
	// parentMap记录二叉树根节点外所有节点的父节点
	parentMap := make(map[*entity.TreeNode]*entity.TreeNode)
	// seen记录已经出现过的节点，用于去重。
	seen := make(map[*entity.TreeNode]bool)
	// 因为是从target节点开始做广度优先搜索，所以target就是第一个出现的节点
	seen[target] = true
	var dfs func(*entity.TreeNode)
	// dfs函数遍历整个二叉树，记录根节点外所有节点的父节点
	dfs = func(node *entity.TreeNode) {
		if node == nil {
			return
		}
		if node.Left != nil {
			parentMap[node.Left] = node
			dfs(node.Left)
		}
		if node.Right != nil {
			parentMap[node.Right] = node
			dfs(node.Right)
		}
	}
	dfs(root)
	// 从target节点开始bfs搜索,那么此时与target的距离就是0
	queue := []Element{{target, 0}}
	for len(queue) > 0 {
		if queue[0].Distance == k {
			// 因为是queue是先进先出的队列，所以如果第一个元素与target的距离为k
			// 代表队列中所有元素距离都是k
			for _, element := range queue {
				res = append(res, element.Node.Val)
			}
			// 此时已经找到了所有满足条件的节点，退出循环
			// 如果继续遍历队列添加元素，添加的元素与target的距离肯定是大于k的
			break
		}
		// 先进先出
		node, d := queue[0].Node, queue[0].Distance
		queue = queue[1:]
		// 将与target距离相等的节点(当前节点的父节点，左子节点和右子节点)放到同一个数组中
		array := []*entity.TreeNode{parentMap[node], node.Left, node.Right}
		for _, td := range array {
			// 如果这个节点非空，且没有出现在哈希表seen中
			// 就可以将该节点添加到队列中，同时该节点与target距离即为当前节点与target距离+1
			if td != nil && !seen[td] {
				seen[td] = true
				queue = append(queue, Element{td, d + 1})
			}
		}
	}
	return res
}
