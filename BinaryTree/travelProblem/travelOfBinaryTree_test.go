package travelProblem

import (
	"AlgorithmPractise/BinaryTree/Entity"
	"fmt"
	"reflect"
	"testing"
)

func TestPreOrderTravel(t *testing.T) {
	got := PreOrderTravel(Entity.Root)
	want := []int{5, 4, 11, 7, 2, 8, 13, 4, 5, 1}
	if !reflect.DeepEqual(want, got) { // 因为slice不能比较直接，借助反射包中的方法比较
		t.Errorf("excepted:%v, got:%v", want, got) // 测试失败输出错误提示
	} else {
		fmt.Println("test pass")
	}
}


func TestInOrderTravelUseIteration(t *testing.T) {
	got := InOrderTravelUseIteration(Entity.Root)
	want := []int{7, 11, 2, 4, 5, 13, 8, 5, 4, 1}
	if !reflect.DeepEqual(want, got) {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}

func TestPostOrderTravelUseIteration(t *testing.T) {
	got := PostOrderTravelUseIteration(Entity.Root)
	want := []int{7, 2, 11, 4, 13, 5, 1, 4, 8, 5}
	if !reflect.DeepEqual(want, got) {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}

func TestLevelOrderComplex(t *testing.T) {
	got := LevelOrderComplex(Entity.Root)
	want := [][]int{{5}, {4, 8}, {11, 13, 4}, {7, 2, 5, 1}}
	if !reflect.DeepEqual(want, got) {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}

func TestZigzagLevelOrder(t *testing.T) {
	got := ZigzagLevelOrder(Entity.Root)
	want := [][]int{{5}, {8, 4}, {11, 13, 4}, {1, 5, 2, 7}}
	if !reflect.DeepEqual(want, got) {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}