package greedy

import (
	"fmt"
	"reflect"
	"testing"
)

func TestReconstructQueue(t *testing.T) {
	people := [][]int{{7, 0}, {4, 4}, {7, 1}, {5, 0}, {6, 1}, {5, 2}}
	got := ReconstructQueue(people)
	want := [][]int{{5, 0}, {7, 0}, {5, 2}, {6, 1}, {4, 4}, {7, 1}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("expected:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}

}
