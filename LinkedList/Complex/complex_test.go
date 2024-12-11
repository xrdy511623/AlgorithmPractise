package complex

import (
	"algorithm-practise/linkedlist/entity"
	"fmt"
	"reflect"
	"testing"
)

func TestSortOddAscEvenDescList(t *testing.T) {
	head := SortOddAscEvenDescList(entity.Node1)
	var got []int
	for head != nil {
		got = append(got, head.Val)
		head = head.Next
	}
	want := []int{1, 2, 3, 4, 5, 6}
	if !reflect.DeepEqual(want, got) {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}
