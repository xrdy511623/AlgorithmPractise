package Medium

import (
	"fmt"
	"testing"
)

func TestBagProblem(t *testing.T) {
	weight := []int{1, 3, 4}
	value := []int{15, 20, 30}
	getMaxValue := BagProblem(weight, value, 4)
	expectedMaxValue := 35
	if getMaxValue != expectedMaxValue {
		t.Errorf("excepted:%v, got:%v", expectedMaxValue, getMaxValue)
	}
	fmt.Println(getMaxValue)
}

func TestBagProblemSimple(t *testing.T) {
	weight := []int{1, 3, 4}
	value := []int{15, 20, 30}
	getMaxValue := BagProblemSimple(weight, value, 4)
	expectedMaxValue := 35
	if getMaxValue != expectedMaxValue {
		t.Errorf("excepted:%v, got:%v", expectedMaxValue, getMaxValue)
	}
	fmt.Println(getMaxValue)
}

func TestCanPartition(t *testing.T) {
	nums := []int{1, 5, 11, 5}
	get := CanPartition(nums)
	want := true
	if get != want {
		t.Errorf("excepted:%v, got:%v", want, get)
	}
}

func TestCompleteBagProblem(t *testing.T) {
	weight := []int{1, 3, 4}
	value := []int{15, 20, 30}
	getMaxValue := CompleteBagProblem(weight, value, 4)
	expectedMaxValue := 60
	if getMaxValue != expectedMaxValue {
		t.Errorf("excepted:%v, got:%v", expectedMaxValue, getMaxValue)
	}
	fmt.Println(getMaxValue)
}

func TestMultiBagProblem(t *testing.T) {
	weight := []int{1, 3, 4}
	value := []int{15, 20, 30}
	nums := []int{2, 3, 3}
	get := MultiBagProblem(weight, value, nums, 10)
	want := 90
	if get != want {
		t.Errorf("excepted:%v, got:%v", want, get)
	}
	fmt.Println("pass")
}


func TestMassageArrangement(t *testing.T) {
	//nums := []int{2,1,4,5,3,1,1,3}
	nums := []int{2,7,9,3,1}
	get := MassageArrangement(nums)
	want := 12
	if get != want {
		t.Errorf("excepted:%v, got:%v", want, get)
	}
	fmt.Println("pass")
}