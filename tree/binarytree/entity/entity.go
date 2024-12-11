package entity

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
var Node33 = &TreeNode{2, nil, node55}
var Node22 = &TreeNode{2, nil, node44}
var Root1 = &TreeNode{1, Node22, Node33}

type Employee struct {
	Happy int
	Sub   []*Employee
}

var E22 = &Employee{55, nil}
var E21 = &Employee{-90, nil}
var E20 = &Employee{35, nil}
var E19 = &Employee{20, nil}
var E18 = &Employee{40, nil}
var E17 = &Employee{-7, nil}
var E16 = &Employee{6, nil}
var E15 = &Employee{-20, nil}
var E14 = &Employee{45, nil}
var E13 = &Employee{-8, nil}
var E12 = &Employee{16, nil}
var E11 = &Employee{11, []*Employee{E20, E21, E22}}
var E10 = &Employee{12, []*Employee{E18, E19}}
var E9 = &Employee{17, nil}
var E8 = &Employee{50, []*Employee{E16, E17}}
var E7 = &Employee{72, []*Employee{E14, E15}}
var E6 = &Employee{30, []*Employee{E12, E13}}
var E5 = &Employee{-3, nil}
var E4 = &Employee{-1, nil}
var E3 = &Employee{10, []*Employee{E9, E10, E11}}
var E2 = &Employee{20, []*Employee{E6, E7, E8}}
var E1 = &Employee{90, []*Employee{E4, E5}}
var Boss = &Employee{7, []*Employee{E1, E2, E3}}
