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
	if getMaxValue != expectedMaxValue{
		t.Errorf("excepted:%v, got:%v", expectedMaxValue, getMaxValue)
	}
	fmt.Println(getMaxValue)
}

func TestBagProblemSimple(t *testing.T) {
	weight := []int{1, 3, 4}
	value := []int{15, 20, 30}
	getMaxValue := BagProblemSimple(weight, value, 4)
	expectedMaxValue := 35
	if getMaxValue != expectedMaxValue{
		t.Errorf("excepted:%v, got:%v", expectedMaxValue, getMaxValue)
	}
	fmt.Println(getMaxValue)
}

func TestCanPartition(t *testing.T) {
	nums := []int{1, 5, 11, 5}
	get := CanPartition(nums)
	want := true
	if get != want{
		t.Errorf("excepted:%v, got:%v", want, get)
	}
}