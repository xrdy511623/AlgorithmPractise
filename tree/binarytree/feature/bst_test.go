package feature

import (
	"fmt"
	"testing"
)

func TestVerifyPreOrder(t *testing.T) {
	preOrder := []int{8, 5, 1, 7, 10, 12}
	got := VerifyPreOrder(preOrder)
	want := true
	if got != want {
		t.Errorf("excepted:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}
