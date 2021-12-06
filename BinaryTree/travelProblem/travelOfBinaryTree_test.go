package travelProblem

import (
	"AlgorithmPractise/BinaryTree/Entity"
	"fmt"
	"reflect"
	"testing"
)

func TestPreOrderTravel(t *testing.T) {
	got := PreOrderTravel(Entity.Node1)
	want := []int{5, 4, 11, 7, 2, 8, 13, 4, 5, 1}
	if !reflect.DeepEqual(want, got) { // 因为slice不能比较直接，借助反射包中的方法比较
		t.Errorf("excepted:%v, got:%v", want, got) // 测试失败输出错误提示
	} else {
		fmt.Println("test pass")
	}
}
