package travel

import (
	"algorithmpractise/tree/binarytree/entity"
	"algorithmpractise/utils"
)

/*
二叉树的遍历问题,一般都可以通过DFS(递归)和BFS(迭代)解决
144. 二叉树的前序遍历
94. 二叉树的中序遍历
145. 二叉树的后序遍历
*/

/*
1.1 先序，中序和后序遍历,分别采用DFS(深度优先遍历,递归)和BFS(广度优先遍历，迭代)解决
二叉树示例如下:
		   5
         /  \
        4    8
       / \  / \
      11   13 4
     / \   	 / \
    7  2   	5   1
*/

func PreOrderTravel(root *entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}

	res = append(res, root.Val)
	res = append(res, PreOrderTravel(root.Left)...)
	res = append(res, PreOrderTravel(root.Right)...)
	return res
}

// PreOrderTravelUseIteration 用BFS解决，入栈是根右左，出栈添加到结果数组中则是根左右
func PreOrderTravelUseIteration(root *entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	stack := []*entity.TreeNode{root}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		res = append(res, node.Val)
		stack = stack[:len(stack)-1]
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
	}
	return res
}

func InOrderTravel(root *entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	res = append(res, InOrderTravel(root.Left)...)
	res = append(res, root.Val)
	res = append(res, InOrderTravel(root.Right)...)
	return res
}

func InOrderTravelUseIteration(root *entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	var stack []*entity.TreeNode
	for len(stack) != 0 || root != nil {
		if root != nil {
			stack = append(stack, root)
			root = root.Left
		} else {
			root = stack[len(stack)-1]
			res = append(res, root.Val)
			stack = stack[:len(stack)-1]
			root = root.Right
		}
	}
	return res
}

func PostOrderTravel(root *entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	res = append(res, PostOrderTravel(root.Left)...)
	res = append(res, PostOrderTravel(root.Right)...)
	res = append(res, root.Val)
	return res
}

// PostOrderTravelUseIteration 入栈左右根，出栈根右左，逆序后即为满足要求的左右根
func PostOrderTravelUseIteration(root *entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	stack := []*entity.TreeNode{root}
	for len(stack) != 0 {
		node := stack[len(stack)-1]
		res = append(res, node.Val)
		stack = stack[:len(stack)-1]
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
	}
	return ReverseArray(res)
}

func ReverseArray(array []int) []int {
	for i, n := 0, len(array); i < n/2; i++ {
		array[i], array[n-1-i] = array[n-1-i], array[i]
	}
	return array
}

/*
1.2进阶 589 N叉树的前序遍历
定一个N叉树，返回其节点值的前序遍历 。
N叉树在输入中按层序遍历进行序列化表示，每组子节点由空值null分隔（请参见示例）。
   			1
		/   \  \
	    3   2   4
      /  \
      5   6

最后应返回[1,3,5,6,2,4]
*/

// PreorderNTrees 递归解法
func PreorderNTrees(root *entity.Node) []int {
	res := []int{}
	if root == nil {
		return res
	}
	res = append(res, root.Val)
	for _, node := range root.Children {
		res = append(res, PreorderNTrees(node)...)
	}
	return res
}

// PreOrderOfnTress node节点Children中的子节点逆序入栈，出栈时先进后出依次添加到结果集中
func PreOrderOfnTress(root *entity.Node) []int {
	var res []int
	if root == nil {
		return res
	}
	stack := []*entity.Node{root}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		res = append(res, node.Val)
		stack = stack[:len(stack)-1]
		for i := len(node.Children) - 1; i >= 0; i-- {
			stack = append(stack, node.Children[i])
		}
	}
	return res
}

/*
1.3 进阶，590 N叉树的后序遍历
定一个N叉树，返回其节点值的后序遍历 。
N叉树在输入中按层序遍历进行序列化表示，每组子节点由空值null分隔（请参见示例）。
   			1
		/   \  \
	    3   2   4
      /  \
      5   6

最后应返回[5，6，3，2，4，1]
*/

// PostorderOfNTrees 递归
func PostorderOfNTrees(root *entity.Node) []int {
	res := []int{}
	if root == nil {
		return res
	}
	for _, node := range root.Children {
		res = append(res, PostorderOfNTrees(node)...)
	}
	res = append(res, root.Val)
	return res
}

// PostOrderOfnTress 与1.2类似，只是node节点Children中的子节点是顺序入栈，最后对结果集逆序即可。
func PostOrderOfnTress(root *entity.Node) []int {
	var res []int
	if root == nil {
		return res
	}
	stack := []*entity.Node{root}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		res = append(res, node.Val)
		stack = stack[:len(stack)-1]
		stack = append(stack, node.Children...)
	}
	return ReverseArray(res)
}

/*
2 层序遍历，利用BFS(广度优先遍历，迭代)解决
*/

/*
2.1 剑指Offer32 - I. 从上到下打印二叉树
从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
以上面的示例二叉树为例，最后应返回[5,4,8,11,13,4,7,2,5,1]
*/

func LevelOrder(root *entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	queue := []*entity.TreeNode{root}
	for len(queue) != 0 {
		node := queue[0]
		queue = queue[1:]
		res = append(res, node.Val)
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
2.2 剑指Offer32 - II. 从上到下打印二叉树II
从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
以上面的示例二叉树为例，最后应返回[[5],[4,8],[11,13,4],[7,2,5,1]]
*/

func LevelOrderComplex(root *entity.TreeNode) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	queue := []*entity.TreeNode{root}
	for len(queue) != 0 {
		levelSize := len(queue)
		curLevel := make([]int, levelSize)
		for i := range queue {
			node := queue[i]
			curLevel[i] = node.Val
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		queue = queue[levelSize:]
		res = append(res, curLevel)
	}
	return res
}

/*
leetcode 107. 二叉树的层序遍历II
2.3 给定一个二叉树，返回其节点值自底向上的层序遍历。（即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
以上面的示例二叉树为例，最后应返回[[7,2,5,1],[11,13,4],[4,8],[5]]
*/

// LevelOrderBottom 与2.2类似，将得到的结果逆序即可满足要求
func LevelOrderBottom(root *entity.TreeNode) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	queue := []*entity.TreeNode{root}
	for len(queue) != 0 {
		levelSize := len(queue)
		curLevel := make([]int, levelSize)
		for i := 0; i < levelSize; i++ {
			node := queue[i]
			curLevel[i] = node.Val
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		queue = queue[levelSize:]
		res = append(res, curLevel)
	}
	return ReverseComplexArray(res)
}

func ReverseComplexArray(nums [][]int) [][]int {
	for i, n := 0, len(nums); i < n/2; i++ {
		nums[i], nums[n-1-i] = nums[n-1-i], nums[i]
	}
	return nums
}

/*
剑指Offer32 - III. 从上到下打印二叉树III
2.4 leetcode 103 二叉树的锯齿形层序遍历
给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，
层与层之间交替进行）。以上面的示例二叉树为例，最后应返回[[5],[8,4],[11,13,4],[1,5,2,7]]
*/

/*
思路:BFS+队列解决
根节点为第一层，根节点孩子节点为第二层... 以此类推。所以问题就转化为奇数层正序，偶数层逆序。
*/

// ZigzagLevelOrder 与2.2类似，所不同的是要加入层数(level)的判断，根据层数来确定是否要对该层的节点值进行逆序
func ZigzagLevelOrder(root *entity.TreeNode) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	queue := []*entity.TreeNode{root}
	level := 0
	for len(queue) != 0 {
		levelSize := len(queue)
		// 每一层都新建一个slice来存储该层所有节点值
		curLevel := make([]int, levelSize)
		// 队列queue始终存储同一层节点，循环levelSize次，意味着该层节点遍历完毕
		for i := 0; i < levelSize; i++ {
			// 满足队列先进先出特性
			node := queue[i]
			curLevel[i] = node.Val
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		// 循环levelSize次后，意味着该层节点遍历完毕，剩下的节点属于下一层，level累加1
		level++
		queue = queue[levelSize:]
		// 奇数层将curLevel原样添加到结果集合中
		if level%2 == 1 {
			res = append(res, curLevel)
		} else {
			// 偶数层将curLevel反转后添加到结果集合中
			res = append(res, utils.ReverseArray(curLevel))
		}
	}
	return res
}

/*
leetcode 637. 二叉树的层平均值
2.5 给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。
*/

func AverageOfBinaryTree(root *entity.TreeNode) []float64 {
	var res []float64
	if root == nil {
		return res
	}
	queue := []*entity.TreeNode{root}
	for len(queue) != 0 {
		levelSize := len(queue)
		sum := 0
		for i := 0; i < levelSize; i++ {
			node := queue[0]
			queue = queue[1:]
			sum += node.Val
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		res = append(res, float64(sum)/float64(levelSize))
	}
	return res
}

/*
429. N叉树的层序遍历
2.6 给定一个N叉树，返回其节点值的层序遍历。（即从左到右，逐层遍历）。
树的序列化输入是用层序遍历，每组子节点都由null值分隔（参见示例）。
			1
		/   \  \
	    3   2   4
      /  \
      5   6
譬如对以上N叉树，最终应返回[[1],[3,2,4],[5,6]]
*/

// LevelOrderOfNTress 解决思路与二叉树的层序遍历一样，BFS解决即可
func LevelOrderOfNTress(root *entity.Node) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	queue := []*entity.Node{root}
	for len(queue) != 0 {
		levelSize := len(queue)
		var curLevel []int
		for i := 0; i < levelSize; i++ {
			node := queue[0]
			queue = queue[1:]
			curLevel = append(curLevel, node.Val)
			queue = append(queue, node.Children...)
		}
		res = append(res, curLevel)
	}
	return res
}

/*
进阶
3 根据两种遍历结果构造二叉树
*/

/*
剑指 Offer 07. 重建二叉树
leetcode 105. 从前序与中序遍历序列构造二叉树
3.1 给定一棵二叉树的前序遍历preorder与中序遍历inorder结果集。请构造二叉树并返回其根节点。
preorder和inorder均无重复元素.
*/

/*
方案1
buildTreeFromPreAndIn
从preorder找到根节点的值，即preorder[0],然后利用哈希表到inorder中找到其对应的位置index
而且不管是preorder还是inorder，其左右子树的长度都是相等的，所以根节点的左子树范围为
preorder[1:index+1], inorder[:index]; 而根节点的右子树范围为preorder[index+1:], inorder[index+1:]
*/

func buildTreeFromPreAndIn(preorder []int, inorder []int) *entity.TreeNode {
	if len(preorder) <= 0 || len(inorder) <= 0 || len(preorder) != len(inorder) {
		return nil
	}
	hashTable := make(map[int]int)
	for i, v := range inorder {
		hashTable[v] = i
	}
	index := hashTable[preorder[0]]
	root := &entity.TreeNode{preorder[0], nil, nil}
	root.Left = buildTreeFromPreAndIn(preorder[1:index+1], inorder[:index])
	root.Right = buildTreeFromPreAndIn(preorder[index+1:], inorder[index+1:])
	return root
}

/*
方案2
buildTreeFromPreAndInSimple
方案1时间和空间复杂度都太高，不推荐，这里推荐方案2，与方案1不同，从根节点开始递归确定节点的左右子节点的过程
只依赖于中序遍历结果集的左右子树范围。
*/

func BuildTreeFromPreAndInSimple(preorder []int, inorder []int) *entity.TreeNode {
	if len(preorder) <= 0 || len(inorder) <= 0 || len(preorder) != len(inorder) {
		return nil
	}
	hashTable := make(map[int]int)
	// 构建中序遍历序列inorder中节点值与位置的映射关系
	for i, v := range inorder {
		hashTable[v] = i
	}
	preIndex := 0
	var dfs func(int, int) *entity.TreeNode
	dfs = func(left, right int) *entity.TreeNode {
		// 递归终止条件，left>right
		if left > right {
			return nil
		}
		// 根节点的值一定是前序遍历序列的第一个元素
		val := preorder[preIndex]
		preIndex++
		root := &entity.TreeNode{Val: val}
		// 找到根节点值在中序遍历序列inorder中的位置index
		index := hashTable[val]
		// 则中序遍历序列inorder左子树范围为[:index]
		root.Left = dfs(left, index-1)
		// 中序遍历序列inorder右子树范围为[index+1:]
		root.Right = dfs(index+1, right)
		return root
	}
	return dfs(0, len(inorder)-1)
}

/*
leetcode 106. 从中序与后序遍历序列构造二叉树
3.2 给定一棵二叉树的后序遍历postorder与中序遍历inorder结果集。请构造二叉树并返回其根节点。
postorder和inorder均无重复元素
*/

/*
本题与上一题从前序与中序遍历序列构造二叉树其实是一样的，解法基本相同，所不同的是在单层递归逻辑中需要先设置
右子节点，然后再设置左子节点。
以题目中的二叉树为例，后序遍历序列postorder[9,15,7,20,3]末尾元素3一定是指向根节点的，在创建根节点后的
后序遍历序列[9,15,7,20]末尾元素20一定是指向根节点的右子节点(右子树的根节点)的，如果设置成左子节点显然是错误的。
*/

func BuildTreeFromPostAndIn(inorder []int, postorder []int) *entity.TreeNode {
	if len(postorder) <= 0 || len(inorder) <= 0 || len(postorder) != len(inorder) {
		return nil
	}
	hashTable := make(map[int]int)
	for i, v := range inorder {
		hashTable[v] = i
	}
	posIndex := len(postorder) - 1
	var dfs func(int, int) *entity.TreeNode
	dfs = func(left, right int) *entity.TreeNode {
		if left > right {
			return nil
		}
		val := postorder[posIndex]
		posIndex--
		root := &entity.TreeNode{Val: val}
		index := hashTable[val]
		// 想一想，为什么要先设置右子节点，如果先设置左子节点就会出错，why?
		// 因为此时postorder[len(postorder)-1]，也就是20指向的一定是右子树的根节点
		root.Right = dfs(index+1, right)
		root.Left = dfs(left, index-1)
		return root
	}
	return dfs(0, len(inorder)-1)
}

/*
leetcode 889. 根据前序和后序遍历构造二叉树
3.3 返回与给定的前序和后序遍历匹配的任何二叉树。
思路:前序遍历序列pre和后序遍历序列post长度相等,左右子树的长度也相等
pre:根左右
post:左右根
因此,pre[0]为根节点值,val=pre[1]如果存在,肯定是左子树根节点值,那么我们从post中找出
该值val的pos,标记为index,则post中左子树的范围就是post[:index+1],相应的pre中左子树的
范围就是往后顺延一位(前序是根左右,要去掉根节点的索引0,从1开始,长度与post的左子树相等),
也就是pre[1:index+2],相应的,post中右子树的范围就是post[index+1:len(post)-1](注意要去掉最后一位
,因为post最后一位元素是根节点),pre中右子树的范围就是pre[index+2:]
如此递归构建即可.
*/

func ConstructFromPrePost(preorder, postorder []int) *entity.TreeNode {
	if len(preorder) == 0 || len(postorder) == 0 || len(preorder) != len(postorder) {
		return nil
	}
	root := &entity.TreeNode{Val: preorder[0]}
	if len(preorder) == 1 {
		return root
	}
	mark := make(map[int]int)
	for i, v := range postorder {
		mark[v] = i
	}
	// 在postorder中找到左子树根节点的位置pos
	pos := mark[preorder[1]]
	// 根据pos确定左子树的范围
	root.Left = ConstructFromPrePost(preorder[1:pos+2], postorder[:pos+1])
	// 根据pos确定右子树的范围
	root.Right = ConstructFromPrePost(preorder[pos+2:], postorder[pos+1:len(postorder)-1])
	return root
}

/*
在后序遍历序列postorder中找左子树根节点的位置pos可以使用哈希表来优化，将时间复杂度降低至O(1)
*/

func ConstructFromPrePostSimple(preorder []int, postorder []int) *entity.TreeNode {
	hashTable := make(map[int]int)
	// 在后序遍历序列postorder中建立元素和位置的映射关系
	for i, v := range postorder {
		hashTable[v] = i
	}
	// l1,r1,l2,r2分别代表前序遍历序列和后序遍历序列的首尾位置
	var buildTree func(int, int, int, int) *entity.TreeNode
	buildTree = func(l1, r1, l2, r2 int) *entity.TreeNode {
		// 递归终止条件
		if l1 > r1 || l2 > r2 {
			return nil
		}
		if l1 == r1 {
			return &entity.TreeNode{Val: preorder[l1]}
		}
		// l1初始值为0，那么preorder[l1]一定是对应根节点的。
		root := &entity.TreeNode{Val: preorder[l1]}
		// 先序遍历序列preorder是根左右，preorder[l1]对应根节点，那么preorder[l1+1]一定对应左子树根节点
		// 确定左子树根节点在后序遍历序列postorder中的位置
		pos := hashTable[preorder[l1+1]]
		// 求root节点左子树的长度(后序遍历序列postorder是左右根，那么左子树长度即为pos-l2+1, l2初始值为0)
		leftLength := pos - l2 + 1
		// 确定左子树范围
		root.Left = buildTree(l1+1, l1+leftLength, l2, pos)
		// 确定右子树范围，右子树范围可以根据左子树范围推出
		root.Right = buildTree(l1+leftLength+1, r1, pos+1, r2-1)
		return root
	}
	return buildTree(0, len(preorder)-1, 0, len(postorder)-1)
}

/*
leetcode 654. 最大二叉树
3.4 给定一个不含重复元素的整数数组nums 。一个以此数组直接递归构建的最大二叉树定义如下：

二叉树的根是数组nums中的最大元素。
左子树是通过数组中最大值左边部分 递归构造出的最大二叉树。
右子树是通过数组中最大值右边部分 递归构造出的最大二叉树。
返回有给定数组nums构建的最大二叉树 。

示例:
输入：nums = [3,2,1,6,0,5]
输出：[6,3,5,null,2,0,null,null,1]
解释：递归调用如下所示：
- [3,2,1,6,0,5] 中的最大值是 6 ，左边部分是 [3,2,1] ，右边部分是 [0,5] 。
    - [3,2,1] 中的最大值是 3 ，左边部分是 [] ，右边部分是 [2,1] 。
        - 空数组，无子节点。
        - [2,1] 中的最大值是 2 ，左边部分是 [] ，右边部分是 [1] 。
            - 空数组，无子节点。
            - 只有一个元素，所以子节点是一个值为 1 的节点。
    - [0,5] 中的最大值是 5 ，左边部分是 [0] ，右边部分是 [] 。
        - 只有一个元素，所以子节点是一个值为 0 的节点。
        - 空数组，无子节点。
*/

// ConstructMaximumBinaryTree DFS解决
func ConstructMaximumBinaryTree(nums []int) *entity.TreeNode {
	// 递归终止条件
	if len(nums) == 0 {
		return nil
	}
	pos := utils.FindLargestElement(nums)
	root := &entity.TreeNode{Val: nums[pos]}
	root.Left = ConstructMaximumBinaryTree(nums[:pos])
	root.Right = ConstructMaximumBinaryTree(nums[pos+1:])
	return root
}
