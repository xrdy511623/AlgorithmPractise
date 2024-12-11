package feature

import (
	"algorithm-practise/tree/binarytree/entity"
	"fmt"
	"reflect"
	"testing"
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
