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
var Node3 = &TreeNode{8, node5, node6}
var Node2 = &TreeNode{4, node4, nil}
var Root = &TreeNode{5, Node2, Node3}



var node55 = &TreeNode{3, nil, nil}
var node44 = &TreeNode{3, nil, nil}
var Node33 = &TreeNode{2,nil, node55}
var Node22 = &TreeNode{2, nil, node44}
var Root1 = &TreeNode{1, Node22, Node33}