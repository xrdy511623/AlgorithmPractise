package feature

import (
	"fmt"
	"reflect"
	"testing"

	"algorithmpractise/tree/binarytree/entity"
)

func TestVerifyPreOrder(t *testing.T) {
	preOrder := []int{8, 5, 1, 7, 10, 12}
	got := VerifyPreOrder(preOrder)
	want := true
	if got != want {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}

func TestTreeToDoublyList(t *testing.T) {
	// 示例：构建一个简单的二叉搜索树
	root := &entity.TreeNode{Val: 4}
	root.Left = &entity.TreeNode{Val: 2}
	root.Right = &entity.TreeNode{Val: 6}
	root.Left.Left = &entity.TreeNode{Val: 1}
	root.Left.Right = &entity.TreeNode{Val: 3}
	root.Right.Left = &entity.TreeNode{Val: 5}
	root.Right.Right = &entity.TreeNode{Val: 7}

	want := []int{1, 2, 3, 4, 5, 6, 7}
	got := []int{}
	head := treeToDoublyList(root)
	if head == nil {
		fmt.Println("List is empty")
		return
	}
	current := head
	for {
		fmt.Print(current.Val, " ")
		got = append(got, current.Val)
		current = current.Right
		if current == head {
			break
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}
