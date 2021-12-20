package featureProblem

import (
	"AlgorithmPractise/BinaryTree/Entity"
	"AlgorithmPractise/Utils"
	"math"
	"reflect"
)

/*
1.1 翻转二叉树
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
	root.Left = right
	root.Right = left
	return root
}

// InvertBinaryTreeTwo 或者也可以写成这样 先序遍历递归版
func InvertBinaryTreeTwo(root *Entity.TreeNode) *Entity.TreeNode {
	if root == nil {
		return nil
	}
	temp := root.Left
	root.Left = root.Right
	root.Right = temp
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
空间复杂度：O(height)，其中height表示二叉树的高度。递归函数需要栈空间，而栈空间取决于递归的深度，因此空间
复杂度等价于二叉树的高度。
递归三部曲:
1 确定递归函数的参数和返回值
参数为当前二叉树根节点的指针，返回值为当前二叉树的最大深度
2 明确递归终止条件
遇到空节点，返回0，表明上一层已经是叶子结点，深度不会再增加了
3 确定单层递归逻辑
当前二叉树的最大深度,如果当前二叉树根节点为空，则返回0，否则返回其左子树的最大深度ld和右子树的最大深度rd的
较大值+1(1代表根节点这一层的深度)

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
	cur := root
	var ans *Entity.TreeNode
	for cur != nil {
		// 当后继存在于经过的节点中时（找到一个>val的最小点）
		if cur.Val > p.Val {
			if ans == nil || ans.Val > p.Val {
				ans = cur
			}
		}
		// 找到p节点
		if cur.Val == p.Val {
			// 如果有右子树,说明后继节点一定在右子树的最左边
			if cur.Right != nil {
				cur = cur.Right
				for cur.Left != nil {
					cur = cur.Left
				}
				return cur
			}
			break
		}
		if cur.Val > p.Val {
			cur = cur.Left
		} else {
			cur = cur.Right
		}
	}
	return ans
}

func FindInorderSuccessor(root, p *Entity.TreeNode) *Entity.TreeNode {
	var ans *Entity.TreeNode
	ans = nil
	for root != nil {
		if root.Val > p.Val {
			ans = root
			root = root.Left
		} else {
			root = root.Right
		}
	}
	return ans
}

/*
1.7 验证二叉搜索树
给你一个二叉树的根节点root，判断其是否是一个有效的二叉搜索树。

有效二叉搜索树定义如下：
节点的左子树只包含小于当前节点的数。
节点的右子树只包含大于当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。
*/

// CheckIsValidBST 一个很容易想到的思路是中序遍历二叉树，如果它是BST，就会得到一个升序序列，否则就不是BST
func CheckIsValidBST(root *Entity.TreeNode) bool {
	if root == nil {
		return false
	}
	var dfs func(root *Entity.TreeNode) []int
	dfs = func(root *Entity.TreeNode) (res []int) {
		if root == nil {
			return res
		}
		res = append(res, dfs(root.Left)...)
		res = append(res, root.Val)
		res = append(res, dfs(root.Right)...)
		return res
	}
	sortedArray := dfs(root)
	for i := 1; i < len(sortedArray); i++ {
		if sortedArray[i] <= sortedArray[i-1] {
			return false
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
1.8 验证二叉搜索树的后序遍历序列
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
1.9 二叉树的右视图
给定一个二叉树的根节点root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
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
1.10 平衡二叉树
给定一个二叉树，判断它是否是高度平衡的二叉树。
本题中，一棵高度平衡二叉树定义为：
一个二叉树每个节点的左右两个子树的高度差的绝对值不超过1 。
*/

/*
思路:自顶向下的递归
利用二叉树的前序遍历，即对于当前遍历到的节点，首先计算左右子树的高度，如果左右子树的高度差是否不超过1，再分别
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
1.11 展平二叉搜索树
给你一棵二叉搜索树，请按中序遍历 将其重新排列为一棵递增顺序搜索树，使树中最左边的节点成为树的根节点，并且每个节点没有左子节点，
只有一个右子节点。
*/

// IncreasingBST 时间复杂度O(2*N)，空间复杂度O(N)
func IncreasingBST(root *Entity.TreeNode) *Entity.TreeNode {
	var res []int
	var dfs func(*Entity.TreeNode)
	dfs = func(root *Entity.TreeNode) {
		if root != nil {
			dfs(root.Left)
			res = append(res, root.Val)
			dfs(root.Right)
		}
	}
	dfs(root)
	dummyNode := &Entity.TreeNode{}
	cur := dummyNode
	for _, val := range res {
		cur.Right = &Entity.TreeNode{Val: val}
		cur = cur.Right
	}
	return dummyNode.Right
}

// IncreasingSimpleBST 更好的做法是在中序遍历的过程中改变节点指向，时间复杂度下降为O(N)
func IncreasingSimpleBST(root *Entity.TreeNode) *Entity.TreeNode {
	dummyNode := &Entity.TreeNode{}
	cur := dummyNode
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
	return dummyNode.Right
}

/*
1.12 二叉搜索树迭代器
实现一个二叉搜索树迭代器类BSTIterator，表示一个按中序遍历二叉搜索树（BST）的迭代器：
BSTIterator(TreeNode root) 初始化BSTIterator类的一个对象。BST的根节点root会作为构造函数的一部分给出。指针应初始化为一个不存在于
BST中的数字，且该数字小于BST中的任何元素。
boolean hasNext() 如果向指针右侧遍历存在数字，则返回true ；否则返回false 。
int next()将指针向右移动，然后返回指针处的数字。
注意，指针初始化为一个不存在于BST中的数字，所以对next()的首次调用将返回 BST 中的最小元素。
你可以假设next()调用总是有效的，也就是说，当调用 next()时，BST的中序遍历中至少存在一个下一个数字。
*/

type BSTIterator struct {
	Nums []int
	Root *Entity.TreeNode
}

func Constructor(root *Entity.TreeNode) *BSTIterator {
	nums := make([]int, 0)
	Inorder(root, &nums)
	return &BSTIterator{
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
1.13 给你一个整数n，请你生成并返回所有由n个节点组成且节点值从1到n互不相同的不同二叉搜索树 。
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
1.14 二叉树的完全性检验
给定一个二叉树，确定它是否是一个完全二叉树。
百度百科中对完全二叉树的定义如下：
若设二叉树的深度为 h，除第 h 层外，其它各层 (1～h-1) 的结点数都达到最大个数，第 h 层所有的结点都连续集中在
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
	stack := []Element{{root, 1}}
	var res []int
	p := 1
	for len(stack) != 0 {
		node := stack[0].Node
		p = stack[0].Number
		res = append(res, node.Val)
		stack = stack[1:]
		if node.Left != nil {
			stack = append(stack, Element{node.Left, 2 * p})
		}
		if node.Right != nil {
			stack = append(stack, Element{node.Right, 2*p + 1})
		}
	}
	return p == len(res)
}

/*
1.15 相同的二叉树
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
1.16 删除二叉搜索树中的节点
给定一个二叉搜索树的根节点root和一个值 key，删除二叉搜索树中的key对应的节点，并保证二叉搜索树的性质不变。
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
a 要删除的节点是叶子节点，那很简单，直接将当前节点删除，置为nil即可
b 要删除的节点有右子节点，那么为了维持BST的特性，我们需要找到该节点的后继节点post(BST中大于它的最小节点)，
将该节点的值更新为后继节点post的值，然后递归的在当前节点的右子树中删除该后继节点post
c 要删除的节点有左子节点，那么为了维持BST的特性，我们需要找到该节点的前驱节点pre(BST中小于于它的最大节点)，
将该节点的值更新为前驱节点pre的值，然后递归的在当前节点的左子树中删除该前驱节点pre
最后返回当前节点的引用即可
*/

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
func Predecessor(node *Entity.TreeNode) *Entity.TreeNode {
	pre := node.Left
	for pre.Right != nil {
		pre = pre.Right
	}
	return pre
}

// Successor 在二叉搜索树(BST)中寻找当前节点的后继节点
func Successor(node *Entity.TreeNode) *Entity.TreeNode {
	post := node.Right
	for post.Left != nil {
		post = post.Left
	}
	return post
}

/*
1.17 二叉搜索树的最大宽度
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
1.18 对称二叉树
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
1.19 另一棵树的子树
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
1.20 二叉树的最小深度
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
这棵二叉树所示，根节点没有左子节点，但有右子节点，所以根节点并非叶子结点，最小深度不是1，应该是3。同理，如果
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
			// 因为是按照队列先进先出顺序遍历，所以此时遍历到的叶子结点，一定是深度最小的
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
1.21 完全二叉树的节点个数
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
1.22 找树左下角的值
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