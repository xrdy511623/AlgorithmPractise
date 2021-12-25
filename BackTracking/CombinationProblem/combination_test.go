package CombinationProblem

import (
	"fmt"
	"reflect"
	"testing"
)

func TestCombinationSum(t *testing.T){
	candidates := []int{2, 3, 4}
	target := 12
	res := CombinationSum(candidates, target)
	want := len(res)
	got := CompletePack(candidates, target)
	fmt.Println(want, got)
	if !reflect.DeepEqual(want, got){
		t.Errorf("excepted:%v, got:%v", want, got)
	} else{
		fmt.Println("test pass")
	}
}