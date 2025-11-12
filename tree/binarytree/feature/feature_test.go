package feature

import (
	"fmt"
	"reflect"
	"testing"

	"algorithmpractise/tree/binarytree/entity"
)

func TestRightSideView(t *testing.T) {
	got := RightSideView(entity.Root)
	want := []int{5, 8, 4, 1}
	if !reflect.DeepEqual(want, got) {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}

func TestIsSymmetric(t *testing.T) {
	got := IsSymmetric(entity.Root1)
	want := false
	if got {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}

func TestIsSymmetricUseBFS(t *testing.T) {
	got := IsSymmetricUseBFS(entity.Root1)
	want := false
	if got {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}

func TestGetMostHappy(t *testing.T) {
	got := GetMostHappy(entity.Boss)
	want := 409
	if got != want {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}

func TestComputeBinaryTree(t *testing.T) {
	inOrder := []int{-3, 12, 6, 8, 9, -10, -7}
	preOrder := []int{8, 12, -3, 6, -10, 9, -7}
	want := []int{0, 3, 0, 7, 0, 2, 0}
	got := computeBinaryTree(preOrder, inOrder)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}
