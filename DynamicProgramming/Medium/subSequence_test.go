package Medium

import (
	"fmt"
	"testing"
)

func TestMaxSubarraySumCircular(t *testing.T) {
	nums := []int{5, -3, 5}
	got := MaxSubarraySumCircular(nums)
	want := 10
	if got != want{
		t.Errorf("got:%v,expected:%v", got, want)
	}else{
		fmt.Println("test pass")
	}
}
