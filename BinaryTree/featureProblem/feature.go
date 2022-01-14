package featureProblem

import (
	"AlgorithmPractise/BinaryTree/Entity"
	Entity2 "AlgorithmPractise/LinkedList/Entity"
	"AlgorithmPractise/Utils"
	"fmt"
	"math"
	"reflect"
)

/*
leetcode 226. 翻转二叉树
1.1 翻转一棵二叉树。
*/

/*
递归解决
1 确定递归函数的参数和返回值
参数和返回值都是二叉树的节点的指针。其实递归函数做的就是反转以当前节点node为根节点的二叉树，
然后返回当前节点node。
2 明确递归终止条件
当前节点为空时，返回nil
3 确定单层递归逻辑
如果是先序遍历，就是先交换左右子节点，再反转左子树和右子树
如果是后序遍历，那就是先反转左子树和右子树，再交换左右子节点
*/

// InvertBinaryTree 后序遍历递归版
func InvertBinaryTree(root *Entity.TreeNode) *Entity.TreeNode {
	if root == nil {
		return nil
	}
	left := InvertBinaryTree(root.Left)
	right := InvertBinaryTree(root.Right)
	root.Left, root.Right = right, left
	return root
}

// InvertBinaryTreeTwo 或者也可以写成这样 先序遍历递归版
func InvertBinaryTreeTwo(root *Entity.TreeNode) *Entity.TreeNode {
	if root == nil {
		return nil
	}
	root.Left, root.Right = root.Right, root.Left
	InvertBinaryTreeTwo(root.Left)
	InvertBinaryTreeTwo(root.Right)
	return root
}

// InvertTreeUseIteration 迭代法(先序遍历)
func InvertTreeUseIteration(root *Entity.TreeNode) *Entity.TreeNode {
	if root == nil {
		return nil
	}
	stack := []*Entity.TreeNode{root}
	for len(stack) != 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		node.Left, node.Right = node.Right, node.Left
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
	}
	return root
}

// InvertTreeUseBFS BFS(层序遍历)
func InvertTreeUseBFS(root *Entity.TreeNode) *Entity.TreeNode {
	if root == nil {
		return nil
	}
	queue := []*Entity.TreeNode{root}
	for len(queue) != 0 {
		node := queue[0]
		node.Left, node.Right = node.Right, node.Left
		queue = queue[1:]
		if node.Left != nil {
			queue = append(queue, node.Left)
		}
		if node.Right != nil {
			queue = append(queue, node.Right)
		}

	}
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
	maxDepth := 0
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
空间复杂度：O(height)，其中height表示二叉树的高度。递归函数需要栈空间，而栈空间取决于递归的深度，因此空间
复杂度等价于二叉树的高度。
递归三部曲:
1 确定递归函数的参数和返回值
参数为当前二叉树根节点的指针，返回值为当前二叉树的最大深度。
2 明确递归终止条件
遇到空节点，返回0，表明上一层已经是叶子节点，深度不会再增加了。
3 确定单层递归逻辑
当前二叉树的最大深度,如果当前二叉树根节点为空，则返回0，否则返回其左子树的最大深度ld和右子树的最大深度rd的
较大值+1(1代表根节点这一层的深度)。

*/

func MaxDepthUseDfs(root *Entity.TreeNode) int {
	if root == nil {
		return 0
	}
	return 1 + Utils.Max(MaxDepthUseDfs(root.Left), MaxDepthUseDfs(root.Right))
}

/*
leetcode 236. 二叉树的最近公共祖先
1.3 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

提示：
树中节点数目在范围 [2, 105] 内。
-109 <= Node.val <= 109
所有 Node.val互不相同 。
p != q
p和q均存在于给定的二叉树中。
*/

// NearestCommonAncestor 递归解决, 参看二叉树的最近公共祖先.png
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
	visited := make(map[int]bool)
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
		visited[p.Val] = true
		p = parentDict[p.Val]
	}
	for q != nil {
		if visited[q.Val] {
			return q
		}
		q = parentDict[q.Val]
	}
	return nil
}

/*
leetcode 543. 二叉树的直径
1.4 给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个节点路径长度中的
最大值。这条路径可能穿过也可能不穿过根节点。
注意：两节点之间的路径长度是以它们之间边的数目表示。
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
leetcode 199. 二叉树的右视图
1.5 给定一个二叉树的根节点root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
*/

/*
本题其实与leetcode 102 二叉树的层序遍历本质上是一样的，所谓右视图不过就是遍历的每一层最右侧节点集合而已
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

/*
leetcode 110. 平衡二叉树
1.6 给定一个二叉树，判断它是否是高度平衡的二叉树。
本题中，一棵高度平衡二叉树定义为：
一个二叉树每个节点的左右两个子树的高度差的绝对值不超过1 。
*/

/*
思路:自顶向下的递归
利用二叉树的前序遍历，即对于当前遍历到的节点，首先计算左右子树的高度，如果左右子树的高度差不超过1，再分别
递归地遍历左右子节点，并判断左子树和右子树是否平衡。这是一个自顶向下的递归的过程。
时间复杂度：O(n^2)，其中n是二叉树中的节点个数。
最坏情况下，二叉树是满二叉树，需要遍历二叉树中的所有节点，时间复杂度是 O(n)。
对于节点p，如果它的高度是d，则height(p)最多会被调用d次（即遍历到它的每一个祖先节点时）。对于平均的情况，
一棵树的高度h满足O(h)=O(logn)，因为d≤h，所以总时间复杂度为O(nlogn)。对于最坏的情况，二叉树形成
链式结构，高度为O(n)，此时总时间复杂度为O(n^2)
空间复杂度：O(n)，其中n是二叉树中的节点个数。空间复杂度主要取决于递归调用的层数，递归调用的层数不会超过n。
*/

// IsBalanced 时间复杂度O(N*N),空间复杂度O(N)
func IsBalanced(root *Entity.TreeNode) bool {
	if root == nil {
		return true
	}
	return Utils.Abs(GetHeightOfBinaryTree(root.Left)-GetHeightOfBinaryTree(root.Right)) <= 1 && IsBalanced(root.Left) && IsBalanced(root.Right)
}

// 计算以root为根节点的二叉树的高度
func GetHeightOfBinaryTree(root *Entity.TreeNode) int {
	if root == nil {
		return 0
	}
	return 1 + Utils.Max(GetHeightOfBinaryTree(root.Left), GetHeightOfBinaryTree(root.Right))
}

/*
思路二: 自底向上的递归
方法一由于是自顶向下递归，因此对于同一个节点，求二叉树高度的函数height会被重复调用，导致时间复杂度较高。如果
使用自底向上的做法，则对于每个节点，函数height只会被调用一次。自底向上递归的做法类似于后序遍历，对于当前遍历
到的节点，先递归地判断其左右子树是否平衡，再判断以当前节点为根的子树是否平衡。如果一棵子树是平衡的，则返回其
高度（高度一定是非负整数），否则返回−1。如果存在一棵子树不平衡，则整个二叉树一定不平衡。
*/

// IsBalancedSimple 时间复杂度O(N), 空间复杂度O(N)
func IsBalancedSimple(root *Entity.TreeNode) bool {
	return Height(root) >= 0
}

func Height(root *Entity.TreeNode) int {
	if root == nil {
		return 0
	}
	lh := Height(root.Left)
	rh := Height(root.Right)
	if lh == -1 || rh == -1 || Utils.Abs(lh-rh) > 1 {
		return -1
	}
	return 1 + Utils.Max(lh, rh)
}

/*
958. 二叉树的完全性检验
1.7 给定一个二叉树，确定它是否是一个完全二叉树。
百度百科中对完全二叉树的定义如下：
若设二叉树的深度为h，除第h层外，其它各层 (1～h-1) 的节点数都达到最大个数，第h层所有的节点都连续集中在
最左边，这就是完全二叉树。（注：第h层可能包含 1~2h个节点。）
*/

type Element struct {
	Node   *Entity.TreeNode
	Number int
}

// IsCompleteTree 时间复杂度O(N),空间复杂度O(N)
func IsCompleteTree(root *Entity.TreeNode) bool {
	if root == nil {
		return true
	}
	queue := []Element{{root, 1}}
	seq, count := 0, 0
	for len(queue) != 0 {
		node := queue[0].Node
		seq = queue[0].Number
		count++
		queue = queue[1:]
		if node.Left != nil {
			queue = append(queue, Element{node.Left, 2 * seq})
		}
		if node.Right != nil {
			queue = append(queue, Element{node.Right, 2*seq + 1})
		}
	}
	return seq == count
}


/*
1.8 相同的二叉树
给你两棵二叉树的根节点p和q ，编写一个函数来检验这两棵树是否相同。
如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
*/

// IsSameTree DFS递归解决 时间复杂度O(min(M, N)),空间复杂度O(min(M, N))
func IsSameTree(p, q *Entity.TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p == nil || q == nil {
		return false
	}
	return p.Val == q.Val && IsSameTree(p.Left, q.Left) && IsSameTree(p.Right, q.Right)
}

/*
1.9 二叉树的最大宽度
给定一个二叉树，编写一个函数来获取这个树的最大宽度。树的宽度是所有层中的最大宽度。这个二叉树与满二叉树
（full binary tree）结构相同，但一些节点为空。

每一层的宽度被定义为两个端点（该层最左和最右的非空节点，两端点间的null节点也计入长度）之间的长度。

示例1:

输入:

           1
         /   \
        3     2
       / \     \
      5   3     9

输出: 4
解释: 最大值出现在树的第 3 层，宽度为 4 (5,3,null,9)。
*/

/*
思路:题目中的二叉树在同一层的最左边和最右边节点之间允许有空节点，但是计算宽度时空节点也包括在内，那我们可以
模拟把这些空节点填充上，具体来说就是在每一层从左到右给每个节点编号(单调递增1)，这样一来，每层的宽度就等于
最右边节点的编号-最右边节点的编号+1.
*/

// WidthOfBinaryTree 时间复杂度O(N),空间复杂度O(N)
func WidthOfBinaryTree(root *Entity.TreeNode) int {
	maxWidth := 0
	if root == nil {
		return maxWidth
	}
	// 根节点编号设置为1
	queue := []Element{Element{root, 1}}
	for len(queue) != 0 {
		var temp []Element
		for _, element := range queue {
			if element.Node.Left != nil {
				temp = append(temp, Element{element.Node.Left, element.Number * 2})
			}
			if element.Node.Right != nil {
				temp = append(temp, Element{element.Node.Right, element.Number*2 + 1})
			}
		}
		maxWidth = Utils.Max(maxWidth, queue[len(queue)-1].Number-queue[0].Number+1)
		queue = temp
	}
	return maxWidth
}

/*
1.10 对称二叉树
给定一个二叉树，检查它是否是镜像对称的。


例如，二叉树[1,2,2,3,4,4,3] 是对称的。

    1
   / \
  2   2
 / \ / \
3  4 4  3


但是下面这个[1,2,2,null,3,null,3] 则不是镜像对称的:

    1
   / \
  2   2
   \   \
   3    3

*/

/*
思路一: DFS,深度优先遍历
观察[1,2,2,3,4,4,3]这棵对称的二叉树不难发现，它的左子树的先序遍历序列[2,3,4]正好是右子树后序遍历序列
[4,3,2]反转之后的结果，找到这个规律问题就好解决了。不过，还需要注意的是，按照题意如果子树中有空节点，也需要
填充到遍历后的序列中，否则会出错。以[1,2,2,null,3,null,3]这棵二叉树为例，如果忽略掉空节点，其左子树的先序
遍历序列[2,3]正好也是其右子树后序遍历序列[3, 2]反转后的结果，得出结论这棵二叉树也是对称的，显然是不对的，所以
这里我们对空节点一律以0填充。这样，其左子树的先序遍历序列preOrder就变成了[2, 0, 3, 0, 0],右子树后序遍历序列
postOrder为[0, 0, 0, 3, 2], postOrder反转后是[2, 3, 0, 0, 0]，显然与preOrder不同，问题得以解决。
*/

func IsSymmetric(root *Entity.TreeNode) bool {
	if root == nil {
		return true
	}
	leftPreOrder := PreOrder(root.Left)
	rightPostOrder := PostOrder(root.Right)
	// 因为slice不能直接比较，所以借助反射包中的方法比较
	return reflect.DeepEqual(leftPreOrder, Utils.ReverseArray(rightPostOrder))
}

func PreOrder(node *Entity.TreeNode) []int {
	var res []int
	if node == nil {
		res = append(res, 0)
		return res
	}
	res = append(res, node.Val)
	res = append(res, PreOrder(node.Left)...)
	res = append(res, PreOrder(node.Right)...)
	return res
}

func PostOrder(node *Entity.TreeNode) []int {
	var res []int
	if node == nil {
		res = append(res, 0)
		return res
	}
	res = append(res, PostOrder(node.Left)...)
	res = append(res, PostOrder(node.Right)...)
	res = append(res, node.Val)
	return res
}

/*
思路二:BFS(广度优先遍历)
首先判断根节点是否为空,如果是返回真;然后判断根节点的左右子节点是否都为空,如果是
返回真,如果根节点的左右子节点有一个为空,返回假;如果节点的左右子节点都不为空,判断
左右子节点的值是否相等,如果不相等则返回假,如果相等则继续递归判断;由于左右子节点的子节点一共有4个
(包括空节点),所以将node1.left与node2.right进行比较判断, 将node1.right与node2.right
进行比较判断
*/

func IsSymmetricUseBFS(root *Entity.TreeNode) bool {
	if root == nil {
		return true
	}
	return BFS(root.Left, root.Right)
}

func BFS(node1, node2 *Entity.TreeNode) bool {
	// 两个节点均为空节点，返回true
	if node1 == nil && node2 == nil {
		return true
		// 有一个节点为空，而另一个节点不为空，返回false
	} else if node1 == nil || node2 == nil {
		return false
	} else {
		// 两个节点均不为空，但节点的值不等，也返回false
		if node1.Val != node2.Val {
			return false
		}
		// 继续递归比较判断
		return BFS(node1.Left, node2.Right) && BFS(node1.Right, node2.Left)
	}
}

/*
1.11 另一棵树的子树
给你两棵二叉树root和subRoot 。检验root中是否包含和subRoot具有相同结构和节点值的子树。如果存在，返回
true；否则，返回false 。

二叉树tree的一棵子树包括tree的某个节点和这个节点的所有后代节点。tree也可以看做它自身的一棵子树。
*/

/*
思路一:本题与1.18 对称二叉树其实很类似，我们只需要广度优先遍历的比较两棵树就好了
*/

// IsSubTree BFS解决
func IsSubTree(s, t *Entity.TreeNode) bool {
	if s == nil {
		return false
	}
	return Check(s, t) || IsSubTree(s.Left, t) || IsSubTree(s.Right, t)
}

func Check(node1, node2 *Entity.TreeNode) bool {
	if node1 == nil && node2 == nil {
		return true
	}
	if node1 == nil || node2 == nil {
		return false
	}
	if node1.Val != node2.Val {
		return false
	}
	return Check(node1.Left, node2.Left) && Check(node1.Right, node2.Right)
}

/*
思路二:这个方法需要我们先了解一个「小套路」：一棵子树上的点在深度优先搜索序列（即先序遍历）中是连续的。了解了
这个「小套路」之后，我们可以确定解决这个问题的方向就是：把s和t先转换成深度优先搜索序列，然后看t的深度
优先搜索序列是否是s的深度优先搜索序列的「子串」。

这样做正确吗？假设s由两个节点组成，1是根，2是1的左孩子；t也由两个节点组成，1是根，2是1的右孩子。
这样一来s和t的深度优先搜索序列相同，可是t并不是s的某一棵子树。由此可见s的深度优先搜索序列包含t的深度优先搜索
序列」是t是s子树的必要不充分条件，所以单纯这样做是不正确的。

为了解决这个问题，我们可以引入两个空值lNull和rNull，当一个节点的左孩子或者右孩子为空的时候，就插入这两个空值，
这样深度优先搜索序列就唯一对应一棵树。处理完之后，就可以通过判断s的深度优先搜索序列包含t的深度优先搜索序列
来判断答案。
*/

func IsSubTreeSimple(s, t *Entity.TreeNode) bool {
	max := math.MinInt32
	GetMaxElement(s, &max)
	GetMaxElement(t, &max)
	lN, rN := max+1, max+2
	sList := GetPreOrder(s, []int{}, lN, rN)
	tList := GetPreOrder(t, []int{}, lN, rN)
	sLen, tLen := len(sList), len(tList)
	// 这个匹配算法效率比较低，以后再优化
	for i := 0; i <= sLen-tLen; i++ {
		if reflect.DeepEqual(sList[i:i+tLen], tList) {
			return true
		}
	}
	return false
}

func GetPreOrder(node *Entity.TreeNode, list []int, lN, rN int) []int {
	if node == nil {
		return list
	}
	list = append(list, node.Val)
	if node.Left != nil {
		list = GetPreOrder(node.Left, list, lN, rN)
	} else {
		list = append(list, lN)
	}
	if node.Right != nil {
		list = GetPreOrder(node.Right, list, lN, rN)
	} else {
		list = append(list, rN)
	}
	return list
}

func GetMaxElement(root *Entity.TreeNode, max *int) {
	if root == nil {
		return
	}
	if root.Val > *max {
		*max = root.Val
	}
	GetMaxElement(root.Left, max)
	GetMaxElement(root.Right, max)
}

/*
1.12 二叉树的最小深度
给定一个二叉树，找出其最小深度。
最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
说明: 叶子节点是指没有子节点的节点。

示例:
给定二叉树 [3,9,20,null,null,15,7],
返回它的最小深度 2
*/

/*
思路一:DFS递归
注意本题有一个误区，最小深度指的是从根节点到最近叶子节点的最短路径上的节点数量，所谓叶子节点指的是没有左右
子节点的节点，所以如果一棵二叉树没有左子树但有右子树时，其最小深度应该是1+右子树的最小深度，而不是1，如下图
这棵二叉树所示，根节点没有左子节点，但有右子节点，所以根节点并非叶子节点，最小深度不是1，应该是3。同理，如果
一棵二叉树没有右子树但有左子树时，其最小深度应该是1+左子树的最小深度，而不是1。

		   5
         /  \
            8
           / \
		  13  4
         	 / \
         	5   1

明白了这一点，问题就很好解决了。
递归三部曲
1 确定递归函数的参数和返回值
参数为当前二叉树根节点的指针，返回值为当前二叉树的最小深度
2 明确递归终止条件
遍历到空节点时，返回深度0
3 明确单层递归逻辑
分以下三种情况:
a 当前二叉树没有左子树但有右子树时，其最小深度应该是1+右子树的最小深度
b 当前二叉树没有右子树但有左子树时，其最小深度应该是1+左子树的最小深度
c 返回1+min(左子树的最小深度,右子树的最小深度)
c其实包含了两种情况，一是当前二叉树左右子树均为空时，返回1+min(ld,rd)=1+min(0,0)=1，正确
二是二叉树左右子树均不为空，返回1+min(ld,rd)，也是正确的，所以这两种情况最小深度的计算可以合并。
*/

func MinDepth(root *Entity.TreeNode) int {
	if root == nil {
		return 0
	}
	if root.Left != nil && root.Right == nil {
		return 1 + MinDepth(root.Left)
	}
	if root.Right != nil && root.Left == nil {
		return 1 + MinDepth(root.Right)
	}
	return 1 + Utils.Min(MinDepth(root.Left), MinDepth(root.Right))
}

func MinDepthUseBFS(root *Entity.TreeNode) int {
	if root == nil {
		return 0
	}
	// 此时根节点不为空，则最小深度至少为1
	depth := 1
	queue := []*Entity.TreeNode{root}
	for len(queue) != 0 {
		size := len(queue)
		for i := 0; i < size; i++ {
			node := queue[0]
			queue = queue[1:]
			// 因为是按照队列先进先出顺序遍历，所以此时遍历到的叶子节点，一定是深度最小的
			if node.Left == nil && node.Right == nil {
				return depth
			}
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		// 每遍历一层，深度累加1
		depth++
	}
	return depth
}

/*
1.13 完全二叉树的节点个数
给你一棵完全二叉树的根节点root，求出该树的节点个数。
完全二叉树的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层
的节点都集中在该层最左边的若干位置。若最底层为第h层，则该层包含 1~2^h个节点。
*/

/*
时间复杂度O(N)的解法就不说了，DFS和BFS都可以。这里为了提高算法效率，需要好好利用完全二叉树的特性
完全二叉树只有两种情况，情况一：就是满二叉树，情况二：最后一层叶子节点没有满。
对于情况一，可以直接用 2^树深度 - 1 来计算，注意这里根节点深度为1。
对于情况二，分别递归左孩子，和右孩子，递归到某一深度一定会有左孩子或者右孩子为满二叉树，然后依然可以
按照情况1来计算。
*/

// CountNodes 时间复杂度O(logN*logN),空间复杂度O(logN)
func CountNodes(root *Entity.TreeNode) int {
	if root == nil {
		return 0
	}
	left, right := root.Left, root.Right
	lh, rh := 0, 0
	for left != nil {
		lh++
		left = left.Left
	}
	for right != nil {
		rh++
		right = right.Right
	}
	if lh == rh {
		return (2 << lh) - 1
	}
	return CountNodes(root.Left) + CountNodes(root.Right) + 1
}

/*
1.14 找树左下角的值
给定一个二叉树的根节点root，请找出该二叉树的最底层最左边节点的值。
假设二叉树中至少有一个节点。
*/

/*
思路一:广度优先遍历(BFS),用一个二维数组保存二叉树每一层节点的值，最后返回最后一层最左边的元素即可
*/

func FindBottomLeftValue(root *Entity.TreeNode) int {
	if root.Left == nil && root.Right == nil {
		return root.Val
	}
	var res [][]int
	queue := []*Entity.TreeNode{root}
	for len(queue) != 0 {
		var curLevel []int
		size := len(queue)
		for _, node := range queue {
			curLevel = append(curLevel, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		queue = queue[size:]
		res = append(res, curLevel)
	}
	return res[len(res)-1][0]
}

/*
思路二: 深度优先遍历(DFS),当达到最大深度时，找到最左边的叶子节点的值(先序遍历)
*/

func FindBottomLeftValueSimple(root *Entity.TreeNode) int {
	if root.Left == nil && root.Right == nil {
		return root.Val
	}
	maxDepth, res := 0, 0
	var helper func(*Entity.TreeNode, int)
	helper = func(root *Entity.TreeNode, depth int) {
		if root.Left == nil && root.Right == nil {
			if depth > maxDepth {
				maxDepth = depth
				res = root.Val
			}
			if root.Left != nil {
				depth++
				helper(root.Left, depth)
				// 回溯,如果root有右子树，depth需要与遍历root左子树之前保持一致
				depth--
			}
			if root.Right != nil {
				depth++
				helper(root.Right, depth)
				// 回溯
				depth--
			}
		}
	}
	helper(root, 0)
	return res
}

/*
1.15 合并二叉树
给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，
否则不为NULL的节点将直接作为新二叉树的节点。

输入
		   1
         /  \
        3    2
       /
      5


输入		   2
         /  \
        1    3
        \    \
         4    7

输出
合并后的树
           3
         /  \
        4    5
       / \    \
      5  4     7


注意: 合并必须从两个树的根节点开始。
*/

// DFS递归解决
func MergeTrees(root1 *Entity.TreeNode, root2 *Entity.TreeNode) *Entity.TreeNode {
	if root1 == nil && root2 == nil {
		return nil
	}
	if root2 == nil && root1 != nil {
		return root1
	}
	if root1 == nil && root2 != nil {
		return root2
	}
	root := &Entity.TreeNode{root1.Val + root2.Val, nil, nil}
	root.Left = MergeTrees(root1.Left, root2.Left)
	root.Right = MergeTrees(root1.Right, root2.Right)
	return root
}

/*
1.16 最长同值路径
给定一个二叉树，找到最长的路径，这个路径中的每个节点具有相同值。 这条路径可以经过也可以不经过根节点。

注意：两个节点之间的路径长度由它们之间的边数表示。

示例 1:

输入:

              5
             / \
            4   5
           / \   \
          1   1   5
输出:
2
*/

/*
思路:dfs递归解决
1 明确递归函数参数和返回值
参数为当前二叉树根节点指针，返回值为左最长同值路径长度和右最长同值路径长度的最大值
2 确定递归终止条件
当遇到空节点时，返回0
3 确定单层递归逻辑
若当前根节点存在左子节点和根节点同值，更新左最长同值路径长度+1;否则左最长同值路径长度重置为0
若当前根节点存在右子节点和根节点同值，更新右最长同值路径长度+1;否则右最长同值路径长度重置为0
同时迭代更新最长同值路径最大长度(左最长同值路径长度+右最长同值路径长度)
*/

// LongestSameValuePath 时间复杂度O(N),空间复杂度O(H)
func LongestSameValuePath(root *Entity.TreeNode) int {
	longestLength := 0
	var dfs func(*Entity.TreeNode) int
	dfs = func(node *Entity.TreeNode) int {
		if node == nil {
			return 0
		}
		leftLength := dfs(node.Left)
		rightLength := dfs(node.Right)
		// 如果存在左子节点和根节点同值，更新左最长路径;否则左最长路径为0
		if node.Left != nil && node.Left.Val == node.Val {
			leftLength++
		} else {
			leftLength = 0
		}
		// 如果存在右子节点和根节点同值，更新右最长路径;否则右最长路径为0
		if node.Right != nil && node.Right.Val == node.Val {
			rightLength++
		} else {
			rightLength = 0
		}
		// 迭代最长同值路径的值，左最长路径+右最长路径
		longestLength = Utils.Max(longestLength, leftLength+rightLength)
		return Utils.Max(leftLength, rightLength)
	}
	dfs(root)
	return longestLength
}

/*
1.17 寻找重复的子树
给定一棵二叉树，返回所有重复的子树。对于同一类的重复子树，你只需要返回其中任意一棵的根节点即可。
两棵树重复是指它们具有相同的结构以及相同的节点值。

示例 1：
        1
       / \
      2   3
     /   / \
    4   2   4
       /
      4
下面是两个重复的子树：

      2
     /
    4

和

  4

*/

/*
思路:遍历每一个节点,构造该节点可能的子树序列,并存储到哈希表中
如果有重复,则将该节点添加到结果列表中.
*/

func FindDuplicateSubtrees(root *Entity.TreeNode) []*Entity.TreeNode {
	var res []*Entity.TreeNode
	if root == nil {
		return res
	}
	subTreeMap := make(map[string]int)
	var dfs func(*Entity.TreeNode) string
	dfs = func(node *Entity.TreeNode) string {
		if node == nil {
			return "#"
		}
		subTree := fmt.Sprintf("%v:%v:%v", node.Val, dfs(node.Left), dfs(node.Right))
		subTreeMap[subTree]++
		if subTreeMap[subTree] == 2 {
			res = append(res, node)
		}
		return subTree
	}
	dfs(root)
	return res
}

/*
1.18 二叉树中的列表
给你一棵以root为根的二叉树和一个head为第一个节点的链表。
如果在二叉树中，存在一条一直向下的路径，且每个点的数值恰好一一对应以head为首的链表中每个节点的值，那么
请你返回True ，否则返回 False 。
一直向下的路径的意思是：从树中某个节点开始，一直连续向下的路径。

示例:
输入：head = [4,2,8], root = [1,4,4,null,2,2,null,1,null,6,8,null,null,null,null,1,3]
输出：true
*/

// IsSubPath 递归解决
func IsSubPath(head *Entity2.ListNode, root *Entity.TreeNode) bool {
	if root == nil {
		return false
	}
	var dfs func(*Entity2.ListNode, *Entity.TreeNode) bool
	dfs = func(head *Entity2.ListNode, root *Entity.TreeNode) bool {
		if head == nil {
			return true
		}
		if root == nil {
			return false
		}
		if head.Val != root.Val {
			return false
		}
		return dfs(head.Next, root.Left) || dfs(head.Next, root.Right)
	}
	return dfs(head, root) || IsSubPath(head, root.Left) || IsSubPath(head, root.Right)
}

/*
1.19 二叉树展开为链表
给你二叉树的根节点root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用TreeNode ，其中right子指针指向链表中下一个节点，而左子指针始终为null 。
展开后的单链表应该与二叉树先序遍历 顺序相同。

示例:
输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]
*/

// Flatten dfs先序遍历解决
func Flatten(root *Entity.TreeNode) {
	var dfs func(*Entity.TreeNode) []*Entity.TreeNode
	dfs = func(node *Entity.TreeNode) []*Entity.TreeNode {
		var res []*Entity.TreeNode
		if node == nil {
			return res
		}
		res = append(res, node)
		res = append(res, dfs(node.Left)...)
		res = append(res, dfs(node.Right)...)
		return res
	}
	nodesList := dfs(root)
	for i := 1; i < len(nodesList); i++ {
		prev, cur := nodesList[i-1], nodesList[i]
		prev.Left = nil
		prev.Right = cur
	}
}

/*
1.20 填充每个节点的下一个右侧节点指针
给定一个完美二叉树，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个next指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将next指针设置为NULL。
初始状态下，所有next指针都被设置为NULL。
*/

type Node struct {
	Val   int
	Left  *Node
	Right *Node
	Next  *Node
}

// Connect 第一种方案, 递归 时间复杂度O(2N), 空间复杂度O(H)
func Connect(root *Node) *Node {
	if root == nil {
		return nil
	}
	var dfs func(*Node, *Node)
	dfs = func(l, r *Node) {
		if l == nil && r == nil {
			return
		}
		l.Next = r
		dfs(l.Left, l.Right)
		dfs(l.Right, r.Left)
		dfs(r.Left, r.Right)
	}
	dfs(root.Left, root.Right)
	return root
}

// ConnectNext 第二种方案, BFS 时间复杂度O(N), 空间复杂度O(N)
func ConnectNext(root *Node) *Node {
	if root == nil {
		return nil
	}
	queue := []*Node{root}
	for len(queue) != 0 {
		size := len(queue)
		for i := 0; i < size; i++ {
			node := queue[i]
			// 确保queue[i+1]不是空节点
			if i+1 < size {
				node.Next = queue[i+1]
			}
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		queue = queue[size:]
	}
	return root
}

// ConnectNextSimple 第三种方案, 利用已建立的Next指针 时间复杂度O(N), 空间复杂度O(1)
func ConnectNextSimple(root *Node) *Node {
	if root == nil {
		return nil
	}
	// 每次循环从该层的最左侧节点开始
	for leftMost := root; leftMost.Left != nil; leftMost = leftMost.Left {
		// 通过Next指针遍历同一层节点，为下一层的节点更新Next指针
		for cur := leftMost; cur != nil; cur = cur.Next {
			cur.Left.Next = cur.Right
			if cur.Next != nil {
				cur.Right.Next = cur.Next.Left
			}
		}
	}
	return root
}

/*
1.21 填充每个节点的下一个右侧节点指针II
给定一个二叉树

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将next指针设置为NULL。
初始状态下，所有next指针都被设置为NULL。
*/

// ConnectNextComplex 每层单链表法 时间复杂度O(N), 空间复杂度O(1)
func ConnectNextComplex(root *Node) *Node {
	if root == nil {
		return nil
	}
	head := root
	for head != nil {
		// 创建每一层链表的伪头节点
		dummy := new(Node)
		q := dummy
		// 每一层都从左向右遍历
		for p := head; p != nil; p = p.Next {
			if p.Left != nil {
				q.Next = p.Left
				q = q.Next
			}
			if p.Right != nil {
				q.Next = p.Right
				q = q.Next
			}
		}
		// 更新每层的头节点
		head = dummy.Next
	}
	return root
}

/*
1.22 特定深度节点链表
给定一棵二叉树，设计一个算法，创建含有某一深度上所有节点的链表（比如，若一棵树的深度为D，则会创建出D个链表）。
返回一个包含所有深度的链表的数组。

 		   5
         /  \
        4    8
       / \  / \
      11   13  4
     / \   	  /  \
    7  2   	 5    1

输入[5,4,8,11,null,13,4,7,2,null,null,5,1]
输出[[5],[4,8],[11,13,4],[7,2,5,1]]
*/

// ListOfDepth BFS解决
func ListOfDepth(root *Entity.TreeNode) []*Entity2.ListNode {
	var res []*Entity2.ListNode
	if root == nil {
		return res
	}
	queue := []*Entity.TreeNode{root}
	for len(queue) != 0 {
		size := len(queue)
		dummy := &Entity2.ListNode{0, nil}
		p := dummy
		for i := 0; i < size; i++ {
			node := queue[i]
			listNode := &Entity2.ListNode{node.Val, nil}
			p.Next = listNode
			p = p.Next
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		res = append(res, dummy.Next)
		queue = queue[size:]
	}
	return res
}

/*
1.23 删除给定值的叶子节点
给你一棵以root为根的二叉树和一个整数target,请你删除所有值为target的叶子节点.
注意,一旦删除值为target的叶子节点，它的父节点就可能变成叶子节点；如果新叶子节点的值
恰好也是target,那么这个节点也应该被删除.也就是说，你需要重复此过程直到不能继续删除.
*/

// RemoveLeafNodes 后序遍历，递归解决
func RemoveLeafNodes(root *Entity.TreeNode, target int) *Entity.TreeNode {
	if root == nil {
		return nil
	}
	root.Left = RemoveLeafNodes(root.Left, target)
	root.Right = RemoveLeafNodes(root.Right, target)
	if root.Left == nil && root.Right == nil && root.Val == target {
		return nil
	}
	return root
}

/*
1.24 层数最深叶子节点的和
给你一棵二叉树的根节点root ，请你返回 层数最深的叶子节点的和。

示例:
输入：root = [1,2,3,4,5,null,6,7,null,null,null,null,8]
输出：15

提示：
树中节点数目在范围 [1, 104] 之间。
1 <= Node.val <= 100
*/

type Nd struct {
	Node  *Entity.TreeNode
	Depth int
}

// DeepestLeavesSum BFS解决
func DeepestLeavesSum(root *Entity.TreeNode) int {
	sum := 0
	if root == nil {
		return sum
	}
	maxDepth := 0
	queue := []Nd{Nd{root, 1}}
	for len(queue) != 0 {
		size := len(queue)
		// 既然是层序遍历，那么最后一层的叶子节点肯定是最后被添加到队列中的
		// 所以叶子节点一定是优先从队列末尾处得到。
		nd := queue[size-1]
		queue = queue[:size-1]
		node, depth := nd.Node, nd.Depth
		// 如果此时最大深度小于当前层的深度，那么需要更新最大深度和sum的值
		// 也就是说只要没遍历到深度最大的层，就会一直更新maxDepth和sum
		if maxDepth < depth {
			maxDepth, sum = depth, node.Val
			// 若此时最大深度等于当前层深度，证明已经到了最深层，sum累加当前节点值
		} else if maxDepth == depth {
			sum += node.Val
		}
		// 当前节点若有子节点，则其子节点的深度为父节点深度+1
		if node.Left != nil {
			queue = append(queue, Nd{node.Left, depth + 1})
		}
		if node.Right != nil {
			queue = append(queue, Nd{node.Right, depth + 1})
		}
	}
	return sum
}

// DeepestLeavesSumSimple DFS解决
func DeepestLeavesSumSimple(root *Entity.TreeNode) int {
	sum := 0
	if root == nil {
		return sum
	}
	maxDepth := 0
	var dfs func(*Entity.TreeNode, int)
	dfs = func(node *Entity.TreeNode, depth int) {
		if node == nil {
			return
		}
		if maxDepth < depth {
			maxDepth, sum = depth, node.Val
		} else if maxDepth == depth {
			sum += node.Val
		}
		dfs(node.Left, depth+1)
		dfs(node.Right, depth+1)
	}
	dfs(root, 1)
	return sum
}

/*
1.25 二叉树的坡度
给你一个二叉树的根节点root，计算并返回整个树的坡度 。
一个树的节点的坡度定义即为，该节点左子树的节点之和和右子树节点之和的差的绝对值 。如果没有左子树的话，
左子树的节点之和为0 ；没有右子树的话也是一样。空节点的坡度是0 。
整个树的坡度就是其所有节点的坡度之和。

示例:
输入：root = [1,2,3]
输出：1
解释：
节点 2 的坡度：|0-0| = 0（没有子节点）
节点 3 的坡度：|0-0| = 0（没有子节点）
节点 1 的坡度：|2-3| = 1（左子树就是左子节点，所以和是 2 ；右子树就是右子节点，所以和是 3 ）
坡度总和：0 + 0 + 1 = 1
*/

/*
思路:DFS后序遍历递归解决
递归三部曲
1 明确递归函数的参数和返回值
参数为当前节点的指针，返回值为以当前节点为根的二叉树所有节点值之和
2 确定递归终止条件
当遍历到空节点时，返回0
3 明确单层递归逻辑
为了避免重复运算，需要从底层向上到根节点遍历，也就是后序遍历，当前节点左子树元素和+右子树元素和+
当前节点值即为返回值。
在遍历过程中累加当前节点的坡度即可
*/

func FindTilt(root *Entity.TreeNode) int {
	res := 0
	if root == nil {
		return res
	}
	var dfs func(*Entity.TreeNode) int
	dfs = func(node *Entity.TreeNode) int {
		if node == nil {
			return 0
		}
		l := dfs(node.Left)
		r := dfs(node.Right)
		// 当前节点坡度即为其左子树元素和与右子树元素和之差的绝对值
		res += Utils.Abs(l - r)
		return l + r + node.Val
	}
	dfs(root)
	return res
}

/*
1.26 最深叶节点的最近公共祖先
给你一个有根节点的二叉树，找到它最深的叶节点的最近公共祖先。

回想一下：

叶节点是二叉树中没有子节点的节点
树的根节点的深度为0，如果某一节点的深度为d，那它的子节点的深度就是d+1
如果我们假定A是一组节点S的最近公共祖先，S中的每个节点都在以A为根节点的子树中，且A的深度达到此条件下
可能的最大值。

示例1：
输入：root = [3,5,1,6,2,0,8,null,null,7,4]
输出：[2,7,4]
解释：
我们返回值为2的节点.

示例2：
输入：root = [1]
输出：[1]
解释：根节点是树中最深的节点，它是它本身的最近公共祖先。

示例3：
输入：root = [0,1,3,null,2]
输出：[2]
解释：树中最深的叶节点是2 ，最近公共祖先是它自己。
*/

/*
思路:dfs递归解决
由于最深的叶子节点可能有多个（如果只有一个最深的叶节点，那么它的最近公共祖先就是它自己),
我们观察它们最近公共祖先的性质.首先,我们可以看出最近公共祖先的两个子树是等高的,如果不等高,
那么高度较小的那个子树叶节点必然不是最深。所以我们可以设计这样的深度优先搜索算法,每一层返回
值有两部分组成：一部分是以该节点为根的子树中，最深叶子节点的公共祖先，另一部分是该层的高度
（也就是该节点到其最深叶子节点的深度）.然后我们可以递归比较：
如果一个节点的左子树和右子树高度相等，那么其左子树的最深节点和右子树的最深节点,都是以
这个节点为根的最深叶子节点,那么我们就返回这个节点,和这个节点的高度（左子树高度或右子树
高度加1）；
如果一个节点的左子树高度大于右子树，那么以这个节点为根的树，其最深叶子节点一定在左子树中
那么我们就返回其左子树中最深节点的最近公共祖先，和当前节点的高度（左子树高度加1）；
如果一个节点的右子树高度大于左子树，那么我们处理情况和情况2相反，返回右子树中最深节点
的最近公共祖先，和当前节点的高度（右子树高度加1).
*/

func LcaDeepestLeaves(root *Entity.TreeNode) *Entity.TreeNode {
	if root == nil {
		return nil
	}
	var dfs func(*Entity.TreeNode) (*Entity.TreeNode, int)
	dfs = func(node *Entity.TreeNode) (*Entity.TreeNode, int) {
		if node == nil {
			return nil, 0
		}
		lAns, lh := dfs(node.Left)
		rAns, rh := dfs(node.Left)
		if lh > rh {
			return lAns, lh + 1
		} else if lh < rh {
			return rAns, rh + 1
		} else {
			return node, lh + 1
		}
	}
	node, _ := dfs(root)
	return node
}

/*
1.27 叶子相似的树
请考虑一颗二叉树上所有的叶子,这些叶子的值按从左到右的顺序排列形成一个叶值序列.
如果有两颗二叉树的叶值序列是相同,那么我们就认为它们是叶相似的.

示例:
输入：root1 = [3,5,1,6,2,9,8,null,null,7,4],
root2 = [3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]
输出：true
*/

/*
思路：深度优先搜索
我们可以使用深度优先搜索的方法得到一棵树的「叶值序列」。
具体地，在深度优先搜索的过程中，我们总是先搜索当前节点的左子节点，再搜索当前节点的右子节点。如果我们搜索
到一个叶节点，就将它的值放入序列中。
在得到了两棵树分别的「叶值序列」后，我们比较它们是否相等即可。
*/

func LeafSimilar(root1, root2 *Entity.TreeNode) bool {
	var res []int
	var dfs func(*Entity.TreeNode)
	dfs = func(node *Entity.TreeNode) {
		if node == nil {
			return
		}
		if node.Left == nil && node.Right == nil {
			res = append(res, node.Val)
		}
		dfs(node.Left)
		dfs(node.Right)
	}
	dfs(root1)
	l1 := res
	res = []int{}
	dfs(root2)
	return reflect.DeepEqual(l1, res)
}
