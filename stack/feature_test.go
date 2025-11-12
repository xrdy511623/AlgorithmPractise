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

func TestCheckStackSequence(t *testing.T) {
	testCases := []struct {
		name     string
		pushed   []int
		popped   []int
		expected bool
	}{
		{
			name:     "Example Case 1",
			pushed:   []int{1, 2, 3, 4, 5},
			popped:   []int{4, 5, 3, 2, 1},
			expected: true,
		},
		{
			name:     "Example Case 2",
			pushed:   []int{1, 2, 3, 4, 5},
			popped:   []int{4, 3, 5, 1, 2},
			expected: false,
		},
		{
			name:     "Empty Arrays",
			pushed:   []int{},
			popped:   []int{},
			expected: true,
		},
		{
			name:     "Single Element - Valid",
			pushed:   []int{1},
			popped:   []int{1},
			expected: true,
		},
		{
			name:     "Single Element - Invalid (mismatch)",
			pushed:   []int{1},
			popped:   []int{2},
			expected: false,
		},
		{
			name:     "Same Order (all pushed then immediate pop)",
			pushed:   []int{1, 2, 3, 4, 5},
			popped:   []int{1, 2, 3, 4, 5},
			expected: true,
		},
		{
			name:     "Alternate Valid Sequence",
			pushed:   []int{1, 2, 3},
			popped:   []int{2, 1, 3},
			expected: true,
		},
		{
			name:     "Different Lengths",
			pushed:   []int{1, 2, 3},
			popped:   []int{1, 2},
			expected: false,
		},
		{
			name:     "Another Invalid Sequence",
			pushed:   []int{1, 2, 3},
			popped:   []int{3, 1, 2},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := checkStackSequence(tc.pushed, tc.popped)
			if result != tc.expected {
				t.Errorf("Failed %s: expected %v, got %v", tc.name, tc.expected, result)
			}
		})
	}
}
