package travelProblem

import (
	"AlgorithmPractise/BinaryTree/Entity"
)

/*
二叉树的遍历问题,一般都可以通过DFS(递归)和BFS(迭代)解决
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

func PreOrderTravel(root *Entity.TreeNode) []int {
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
func PreOrderTravelUseIteration(root *Entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	stack := []*Entity.TreeNode{root}
	for len(stack) != 0 {
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

func InOrderTravel(root *Entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	res = append(res, InOrderTravel(root.Left)...)
	res = append(res, root.Val)
	res = append(res, InOrderTravel(root.Right)...)
	return res
}

func InOrderTravelUseIteration(root *Entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	var stack []*Entity.TreeNode
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

func PostOrderTravel(root *Entity.TreeNode) []int {
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
func PostOrderTravelUseIteration(root *Entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	stack := []*Entity.TreeNode{root}
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
	length := len(array)
	for i := 0; i < length/2; i++ {
		temp := array[length-1-i]
		array[length-1-i] = array[i]
		array[i] = temp
	}

	return array
}

/*
1.2进阶，N叉树的前序遍历
定一个N叉树，返回其节点值的前序遍历 。
N叉树在输入中按层序遍历进行序列化表示，每组子节点由空值null分隔（请参见示例）。
   			1
		/   \  \
	    3   2   4
      /  \
      5   6

最后应返回[1,3,5,6,2,4]
*/

// PreOrderOfnTress, node节点Children中的子节点逆序入栈，出栈时先进后出依次添加到结果集中
func PreOrderOfnTress(root *Entity.Node) []int {
	var res []int
	if root == nil {
		return res
	}
	stack := []*Entity.Node{root}
	for len(stack) != 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(res, node.Val)
		if len(node.Children) != 0 {
			stack = append(stack, reverseNodes(node.Children)...)
		}
	}

	return res
}

func reverseNodes(nodes []*Entity.Node) []*Entity.Node {
	length := len(nodes)
	for i := 0; i < length/2; i++ {
		temp := nodes[length-1-i]
		nodes[length-1-i] = nodes[i]
		nodes[i] = temp
	}
	return nodes
}

/*
1.3 进阶，N叉树的后序遍历
定一个N叉树，返回其节点值的后序遍历 。
N叉树在输入中按层序遍历进行序列化表示，每组子节点由空值null分隔（请参见示例）。
   			1
		/   \  \
	    3   2   4
      /  \
      5   6

最后应返回[5，6，3，2，4，1]
*/

// PostOrderOfnTress 与1.2类似，只是node节点Children中的子节点是顺序入栈，最后对结果集逆序即可。
func PostOrderOfnTress(root *Entity.Node) []int {
	var res []int
	if root == nil {
		return res
	}
	stack := []*Entity.Node{root}
	for len(stack) != 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(res, node.Val)
		if len(node.Children) != 0 {
			stack = append(stack, node.Children...)
		}
	}
	return ReverseArray(res)
}

/*
2 层序遍历，利用BFS(广度优先遍历，迭代)解决
*/

/*
2.1 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
以上面的示例二叉树为例，最后应返回[5,4,8,11,13,4,7,2,5,1]
*/

func LevelOrder(root *Entity.TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	queue := []*Entity.TreeNode{root}
	for len(queue) != 0{
		node := queue[0]
		queue = queue[1:]
		res = append(res, node.Val)
		if node.Left != nil{
			queue = append(queue, node.Left)
		}
		if node.Right != nil{
			queue = append(queue, node.Right)
		}
	}
	return res
}

/*
2.2 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
以上面的示例二叉树为例，最后应返回[[5],[4,8],[11,13,4],[7,2,5,1]]
*/

func LevelOrderComplex(root *Entity.TreeNode) [][]int {
	var res [][]int
	if root == nil{
		return res
	}
	queue := []*Entity.TreeNode{root}
	level := 0
	for len(queue) != 0{
		var temp []*Entity.TreeNode
		res = append(res, []int{})
		for _, node := range queue{
			res[level] = append(res[level], node.Val)
			if node.Left != nil{
				temp = append(temp, node.Left)
			}
			if node.Right != nil{
				temp = append(temp, node.Right)
			}
		}
		level++
		queue = temp
	}
	return res
}

/*
2.3 给定一个二叉树，返回其节点值自底向上的层序遍历。（即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
以上面的示例二叉树为例，最后应返回[[7,2,5,1],[11,13,4],[4,8],[5]]
*/

// LevelOrderBottom 与2.2类似，将得到的结果逆序即可满足要求
func LevelOrderBottom(root *Entity.TreeNode) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	queue := []*Entity.TreeNode{root}
	level := 0
	for len(queue) != 0 {
		var temp []*Entity.TreeNode
		res = append(res, []int{})
		for _, node := range queue {
			res[level] = append(res[level], node.Val)
			if node.Left != nil {
				temp = append(temp, node.Left)
			}
			if node.Right != nil {
				temp = append(temp, node.Right)
			}
		}
		level++
		queue = temp
	}

	return ReverseComplexArray(res)
}

func ReverseComplexArray(src [][]int) [][]int {
	length := len(src)
	for i := 0; i < length/2; i++ {
		temp := src[length-1-i]
		src[length-1-i] = src[i]
		src[i] = temp
	}
	return src
}

/*
二叉树的锯齿形层序遍历
2.4 给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，
层与层之间交替进行）。以上面的示例二叉树为例，最后应返回[[5],[8,4],[11,13,4],[1,5,2,7]]
*/

// ZigzagLevelOrder, 与2.2类似，所不同的是要加入层数(level)的判断，根据层数来确定是否要对该层的节点值进行逆序
func ZigzagLevelOrder(root *Entity.TreeNode) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	queue := []*Entity.TreeNode{root}
	level := 1
	for len(queue) != 0 {
		var temp []*Entity.TreeNode
		var curLevel []int
		for _, node := range queue{
			curLevel = append(curLevel, node.Val)
			if node.Left != nil {
				temp = append(temp, node.Left)
			}
			if node.Right != nil {
				temp = append(temp, node.Right)
			}
		}
		if level % 2 == 0 {
			res = append(res, ReverseArray(curLevel))
		} else {
			res = append(res, curLevel)
		}
		level++
		queue = temp
	}
	return res
}

/*
2.5 二叉树的层平均值
给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。
*/

func AverageOfBinaryTree(root *Entity.TreeNode) []float64 {
	var res []float64
	if root == nil {
		return res
	}
	queue := []*Entity.TreeNode{root}
	for len(queue) != 0 {
		levelSize := len(queue)
		var curLevel []int
		for i := 0; i < levelSize; i++ {
			node := queue[0]
			queue = queue[1:]
			curLevel = append(curLevel, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		res = append(res, GetAverageOfArray(curLevel))
	}
	return res
}

func GetAverageOfArray(s []int) float64 {
	var sum int
	for _, v := range s {
		sum += v
	}
	return float64(sum) / float64(len(s))
}

/*
N叉树的层序遍历
2.6 给定一个N叉树，返回其节点值的层序遍历。（即从左到右，逐层遍历）。
树的序列化输入是用层序遍历，每组子节点都由null值分隔（参见示例）。
			1
		/   \  \
	    3   2   4
      /  \
      5   6
譬如对以上N叉树，最终应返回[[1],[3,2,4],[5,6]]
*/

// LevelorderofnTress 解决思路与二叉树的层序遍历一样，BFS解决即可
func LevelorderofnTress(root *Entity.Node) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	queue := []*Entity.Node{root}
	for len(queue) != 0 {
		levelSize := len(queue)
		var curLevel []int
		for i := 0; i < levelSize; i++ {
			node := queue[0]
			queue = queue[1:]
			curLevel = append(curLevel, node.Val)
			if len(node.Children) != 0 {
				for _, v := range node.Children {
					queue = append(queue, v)
				}
			}
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
3.1 给定一棵二叉树的前序遍历preorder与中序遍历inorder结果集。请构造二叉树并返回其根节点。
preorder和inorder均无重复元素
*/

/*
方案1
buildTreeFromPreAndIn
从preorder找到根节点的值，即preorder[0],然后利用哈希表到inorder中找到其对应的位置index
而且不管是preorder还是inorder，其左右子树的长度都是相等的，所以根节点的左子树范围为
preorder[1:index+1], inorder[:index]; 而根节点的右子树范围为preorder[index+1:], inorder[index+1:]
*/

func buildTreeFromPreAndIn(preorder []int, inorder []int) *Entity.TreeNode {
	if len(preorder) <= 0 || len(inorder) <= 0 || len(preorder) != len(inorder) {
		return nil
	}
	hashTable := make(map[int]int)
	for i, v := range inorder {
		hashTable[v] = i
	}
	index := hashTable[preorder[0]]
	root := &Entity.TreeNode{preorder[0], nil, nil}
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

func BuildTreeFromPreAndInSimple(preorder []int, inorder []int) *Entity.TreeNode {
	if len(preorder) <= 0 || len(inorder) <= 0 || len(preorder) != len(inorder) {
		return nil
	}
	hashTable := make(map[int]int)
	for i, v := range inorder {
		hashTable[v] = i
	}
	var dfs func(left, right int) *Entity.TreeNode
	dfs = func(left, right int) *Entity.TreeNode {
		if left > right {
			return nil
		}
		val := preorder[0]
		preorder = preorder[1:]
		root := &Entity.TreeNode{val, nil, nil}
		index := hashTable[val]
		root.Left = dfs(left, index-1)
		root.Right = dfs(index+1, right)
		return root
	}
	return dfs(0, len(inorder)-1)
}

/*
3.2 给定一棵二叉树的后序遍历postorder与中序遍历inorder结果集。请构造二叉树并返回其根节点。
postorder和inorder均无重复元素
*/

func BuildTreeFromPostAndIn(inorder []int, postorder []int) *Entity.TreeNode {
	if len(postorder) <= 0 || len(inorder) <= 0 || len(postorder) != len(inorder) {
		return nil
	}
	hashTable := make(map[int]int)
	for i, v := range inorder {
		hashTable[v] = i
	}
	var dfs func(left, right int) *Entity.TreeNode
	dfs = func(left, right int) *Entity.TreeNode {
		if left > right {
			return nil
		}
		val := postorder[len(postorder)-1]
		postorder = postorder[:len(postorder)-1]
		root := &Entity.TreeNode{val, nil, nil}
		index := hashTable[val]
		// 想一想，为什么要先设置右子节点，如果先设置左子节点就会出错，why?
		root.Right = dfs(index+1, right)
		root.Left = dfs(left, index-1)
		return root
	}
	return dfs(0, len(inorder)-1)
}

/*
3.3 根据前序和后序遍历构造二叉树
返回与给定的前序和后序遍历匹配的任何二叉树。
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

func ConstructFromPrePost(preorder []int, postorder []int) *Entity.TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	root := &Entity.TreeNode{preorder[0], nil, nil}
	if len(preorder) == 1 {
		return root
	}
	// 在postorder中找到左子树根节点的位置pos
	pos := FindPosInArray(postorder, preorder[1])
	// 根据pos确定左子树的范围
	root.Left = ConstructFromPrePost(preorder[1:pos+2], postorder[:pos+1])
	// 根据pos确定右子树的范围
	root.Right = ConstructFromPrePost(preorder[pos+2:], postorder[pos+1:len(postorder)-1])
	return root
}

func FindPosInArray(s []int, target int) int {
	for index, value := range s {
		if value == target {
			return index
		}
	}
	return -1
}