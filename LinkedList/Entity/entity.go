package Entity

type ListNode struct {
	Val  int
	Next *ListNode
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

var node6 = &ListNode{3, nil}
var node5 = &ListNode{4, node6}
var node4 = &ListNode{3, node5}
var node3 = &ListNode{2, node4}
var node2 = &ListNode{2, node3}
var Node1 = &ListNode{1, node2}
