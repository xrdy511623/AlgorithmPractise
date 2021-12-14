package feature

import (
	"fmt"
	"reflect"
	"testing"
)

func TestTopKFrequent(t *testing.T) {
	nums := []int{1,1,1,1,2,2,3}
	get := TopKFrequent(nums, 2)
	want := []int{1, 2}
	if !reflect.DeepEqual(get, want) {
		t.Errorf("excepted:%v, got:%v", want, get)
		return
	}
	fmt.Println("pass")
}
