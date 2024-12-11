package stack

import (
	"fmt"
	"testing"
)

func TestEvalRPN(t *testing.T) {
	tokens := []string{"4", "13", "5", "/", "+"}
	got := EvalRPN(tokens)
	want := 6
	if got != want {
		t.Errorf("expected:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}

func TestRemoveDuplicatesComplex(t *testing.T) {
	got := RemoveDuplicatesComplex("deeedbbcccbdaa", 3)
	want := "aa"
	if got != want {
		t.Errorf("expected:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}
}
