package sum

import (
	"fmt"
	"reflect"
	"testing"
)

func TestFourSum(t *testing.T) {
	array := []int{1, -2, -5, -4, -3, 3, 3, 5}
	got := FourSum(array, -11)
	want := [][]int{[]int{-5, -4, -3, 1}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}
