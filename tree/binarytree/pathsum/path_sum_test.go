package pathsum

import (
	"fmt"
	"reflect"
	"testing"

	"algorithmpractise/tree/binarytree/entity"
)

func TestPathSum(t *testing.T) {
	got := PathSum(entity.Root, 22)
	want := [][]int{{5, 4, 11, 2}, {5, 8, 4, 5}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}

func TestPathSumUseDfs(t *testing.T) {
	got := PathSumUseDfs(entity.Root, 22)
	want := [][]int{{5, 4, 11, 2}, {5, 8, 4, 5}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}
