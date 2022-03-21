package medium

import (
	"fmt"
	"testing"
)

func TestBagProblem(t *testing.T) {
	weight := []int{1, 3, 4}
	value := []int{15, 20, 30}
	getMaxValue := bagProblem(weight, value, 4)
	expectedMaxValue := 35
	if getMaxValue != expectedMaxValue {
		t.Errorf("excepted:%v, got:%v", expectedMaxValue, getMaxValue)
	}
	fmt.Println(getMaxValue)
}

func TestBagProblemSimple(t *testing.T) {
	weight := []int{1, 3, 4}
	value := []int{15, 20, 30}
	getMaxValue := bagProblemSimple(weight, value, 4)
	expectedMaxValue := 35
	if getMaxValue != expectedMaxValue {
		t.Errorf("excepted:%v, got:%v", expectedMaxValue, getMaxValue)
	}
	fmt.Println(getMaxValue)
}

func TestCanPartition(t *testing.T) {
	nums := []int{1, 5, 11, 5}
	got := canPartition(nums)
	want := true
	if got != want {
		t.Errorf("excepted:%v, got:%v", want, got)
	}
}

func TestCompleteBagProblem(t *testing.T) {
	weight := []int{1, 3, 4}
	value := []int{15, 20, 30}
	getMaxValue := completeBagProblem(weight, value, 4)
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
	got := multiBagProblem(weight, value, nums, 10)
	want := 90
	if got != want {
		t.Errorf("excepted:%v, got:%v", want, got)
	}
	fmt.Println("pass")
}


func TestMassageArrangement(t *testing.T) {
	//nums := []int{2,1,4,5,3,1,1,3}
	nums := []int{2,7,9,3,1}
	got := massageArrangement(nums)
	want := 12
	if got != want {
		t.Errorf("excepted:%v, got:%v", want, got)
	}
	fmt.Println("pass")
}