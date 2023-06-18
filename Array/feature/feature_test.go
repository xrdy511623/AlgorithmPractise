package feature

import (
	"fmt"
	"testing"
)

func TestMinSubArrayLen(t *testing.T) {
	nums := []int{5, 1, 3, 5, 10, 7, 4, 9, 2, 8}
	got := MinSubArrayLen(15, nums)
	want := 2
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}
