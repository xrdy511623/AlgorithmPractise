package medium

import (
	"fmt"
	"testing"
)

func TestMaxSubarraySumCircular(t *testing.T) {
	nums := []int{5, -3, 5}
	got := maxSubarraySumCircular(nums)
	want := 10
	if got != want {
		t.Errorf("got:%v,expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestNumberOfArithmeticSlices(t *testing.T) {
	nums := []int{1, 3, 5, 7, 9, 10, 13, 15, 17, 19, 20, 21, 22, 23, 24}
	got := numberOfArithmeticSlices(nums)
	want := 19
	if got != want {
		t.Errorf("got:%v,expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestNthUglyNumber(t *testing.T) {
	got := nthUglyNumber(10)
	want := 12
	if got != want {
		t.Errorf("got:%v,expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}
