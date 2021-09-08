package Entity

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type Node struct {
	Val      int
	Children []*Node
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
var Node1 = &TreeNode{5, node2, node3}