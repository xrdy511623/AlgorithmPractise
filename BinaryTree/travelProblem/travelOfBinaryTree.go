package travelProblem

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

var node10 = &TreeNode{1, nil, nil}
var node9 = &TreeNode{5, nil, nil}
var node8 = &TreeNode{2, nil, nil}
var node7 = &TreeNode{7, nil, nil}
var node6 = &TreeNode{4, node9, node10}
var node5 = &TreeNode{13, nil, nil}
var node4 = &TreeNode{11, node7, node8}
var node3 = &TreeNode{8, node5, node6}
var node2 = &TreeNode{4, node4, nil}
var node1 = &TreeNode{5, node2, node3}

/*
二叉树的遍历问题
*/

/*
1 先序，中序和后序遍历,分别采用DFS(深度优先遍历,递归)和BFS(广度优先遍历，迭代)解决
二叉树示例如下:
		   5
         /  \
        4    8
       / \  / \
      11   13 4
     / \   	 / \
    7  2   	5   1
*/

func PreOrderTravel(root *TreeNode) []int {
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
func PreOrderTravelUseIteration(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	stack := []*TreeNode{root}
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

func InOrderTravel(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	res = append(res, PreOrderTravel(root.Left)...)
	res = append(res, root.Val)
	res = append(res, PreOrderTravel(root.Right)...)
	return res
}

func InOrderTravelUseIteration(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	var stack []*TreeNode
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

func PostOrderTravel(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	res = append(res, PreOrderTravel(root.Left)...)
	res = append(res, PreOrderTravel(root.Right)...)
	res = append(res, root.Val)
	return res
}

// PostOrderTravelUseIteration 入栈根左右，出栈根右左，逆序后即为满足要求的左右根
func PostOrderTravelUseIteration(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	stack := []*TreeNode{root}
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
2 层序遍历，利用BFS(广度优先遍历，迭代)解决
*/

/*
2.1 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
以上面的示例二叉树为例，最后应返回[5,4,8,11,13,4,7,2,5,1]
*/

func LevelOrder(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	queue := []*TreeNode{root}
	for i := 0; i < len(queue); i++ {
		node := queue[i]
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
2.2 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
以上面的示例二叉树为例，最后应返回[[5],[4,8],[11,13,4],[7,2,5,1]]
*/

func LevelOrderComplex(root *TreeNode) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	queue := []*TreeNode{root}
	level := 0
	for len(queue) != 0 {
		temp := []*TreeNode{}
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
	return res
}
