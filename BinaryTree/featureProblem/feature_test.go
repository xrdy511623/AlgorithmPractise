package featureProblem

import (
	"AlgorithmPractise/BinaryTree/Entity"
	"fmt"
	"reflect"
	"testing"
)

func TestRightSideView(t *testing.T) {
	got := RightSideView(Entity.Root)
	want := []int{5, 8, 4, 1}
	if !reflect.DeepEqual(want, got) {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}

func TestIsSymmetric(t *testing.T) {
	got := IsSymmetric(Entity.Root1)
	want := false
	if got{
		t.Errorf("excepted:%v, got:%v", want, got)
	} else{
		fmt.Println("test pass")
	}
}

func TestIsSymmetricUseBFS(t *testing.T) {
	got := IsSymmetricUseBFS(Entity.Root1)
	want := false
	if got{
		t.Errorf("excepted:%v, got:%v", want, got)
	} else{
		fmt.Println("test pass")
	}
}
