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
	if !reflect.DeepEqual(want, got) { // 因为slice不能比较直接，借助反射包中的方法比较
		t.Errorf("excepted:%v, got:%v", want, got) // 测试失败输出错误提示
	} else {
		fmt.Println("test pass")
	}
}
