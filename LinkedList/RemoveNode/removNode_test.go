package removenode

import (
	"algorithm-practise/linkedlist/entity"
	"fmt"
	"reflect"
	"testing"
)

func TestRemoveDuplicateNodes(t *testing.T) {
	head := RemoveDuplicateNodes(entity.Node1)
	var got []int
	for head != nil {
		got = append(got, head.Val)
		head = head.Next
	}
	want := []int{1, 2, 3, 4}
	if !reflect.DeepEqual(want, got) {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}
