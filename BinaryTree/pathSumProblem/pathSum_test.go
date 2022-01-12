package pathSumProblem

import (
	"AlgorithmPractise/BinaryTree/Entity"
	"fmt"
	"reflect"
	"testing"
)

func TestPathSum(t *testing.T) {
	got := PathSum(Entity.Root, 22)
	want := [][]int{{5, 4, 11, 2}, {5, 8, 4 ,5}}
	if !reflect.DeepEqual(got, want){
		t.Errorf("excepted:%v, got:%v", want, got)
	} else{
		fmt.Println("test pass")
	}
}
